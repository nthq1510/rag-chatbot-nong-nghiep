#!/usr/bin/env python3
"""Tạo embedding cho các chunks và lưu vào Qdrant.

Input:
- data/chunks/chunks.jsonl: các đoạn tri thức đã chia nhỏ.

Output:
- data/vector_store/embeddings_*.jsonl: bản vector lưu ra file để kiểm tra/lưu trữ.
- data/qdrant_db hoặc Qdrant server: collection vector dùng cho truy xuất RAG.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(ROOT / ".cache" / "huggingface"))
os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", str(ROOT / ".cache" / "sentence_transformers"))
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

INPUT_JSONL = ROOT / "data" / "chunks" / "chunks.jsonl"
OUTPUT_DIR = ROOT / "data" / "vector_store"

MODEL_ALIASES = {
    "sup-simcse-vietnamese-phobert-base": "VoVanPhuc/sup-SimCSE-VietNamese-phobert-base",
    "vietnamese-bi-encoder": "bkai-foundation-models/vietnamese-bi-encoder",
    "bge-m3": "BAAI/bge-m3",
    "multilingual-e5-base": "intfloat/multilingual-e5-base",
}


# Đọc chunks từ data/chunks/chunks.jsonl; limit dùng để test nhanh.
def load_chunks(limit=None):
    """Đọc chunks từ JSONL; limit giúp chạy smoke test nhẹ."""
    chunks = []
    with INPUT_JSONL.open(encoding="utf-8") as file:
        for line in file:
            if line.strip():
                chunks.append(json.loads(line))
            if limit and len(chunks) >= limit:
                break
    return chunks


# Tạo id ổn định cho một chunk khi lưu vào Qdrant.
def qdrant_point_id(chunk_id):
    """Tạo id ổn định cho Qdrant từ chunk_id bằng MD5."""
    digest = hashlib.md5(chunk_id.encode("utf-8")).hexdigest()
    return digest


# Đổi alias ngắn như bge-m3 thành tên model Hugging Face đầy đủ.
def normalize_model_name(model):
    """Cho phép dùng alias ngắn như bge-m3 thay vì tên Hugging Face đầy đủ."""
    return MODEL_ALIASES.get(model, model)


# Chuyển tên model thành slug để đặt tên file vector output.
def model_slug(model_name):
    return model_name.lower().replace("/", "__").replace("-", "_")


# Load SentenceTransformer embedding model.
def load_embedder(model_name, device=None):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency: sentence-transformers. Install requirements first: "
            "pip install -r requirements-vector.txt"
        ) from exc

    kwargs = {}
    if device:
        kwargs["device"] = device
    return SentenceTransformer(model_name, **kwargs)


# Chuẩn bị text trước khi encode; E5 cần prefix "passage:".
def prepare_texts(chunks, model_name):
    """Một số embedding model cần prefix passage khi encode văn bản tài liệu."""
    texts = [chunk["text"] for chunk in chunks]
    if model_name == "intfloat/multilingual-e5-base":
        return [f"passage: {text}" for text in texts]
    return texts


# Encode toàn bộ chunks thành vector embedding.
def embed_chunks(chunks, model_name, batch_size, device):
    """Encode toàn bộ chunks thành vector embedding."""
    embedder = load_embedder(model_name, device=device)
    texts = prepare_texts(chunks, model_name)
    vectors = embedder.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    return vectors.tolist()


# Ghi vector ra file JSONL để có bản lưu/debug ngoài Qdrant.
def write_vectors(chunks, vectors, model_name):
    """Lưu vector ra JSONL để có thể kiểm tra/debug ngoài Qdrant."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"embeddings_{model_slug(model_name)}.jsonl"
    with output_path.open("w", encoding="utf-8") as file:
        for chunk, vector in zip(chunks, vectors):
            row = {
                "id": qdrant_point_id(chunk["chunk_id"]),
                "chunk_id": chunk["chunk_id"],
                "product_id": chunk["product_id"],
                "ten_san_pham": chunk["ten_san_pham"],
                "loai": chunk["loai"],
                "rule_key": chunk["rule_key"],
                "model": model_name,
                "vector": vector,
                "payload": chunk,
            }
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
    return output_path


# Tạo/cập nhật collection Qdrant và upsert vector + payload.
def upload_to_qdrant(chunks, vectors, collection, url, path, api_key, recreate):
    """Tạo/cập nhật collection Qdrant và upsert các vector cùng payload."""
    try:
        from qdrant_client import QdrantClient
        from qdrant_client.http import models
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency: qdrant-client. Install requirements first: "
            "pip install -r requirements-vector.txt"
        ) from exc

    if path:
        client = QdrantClient(path=path)
    else:
        client = QdrantClient(url=url, api_key=api_key)
    vector_size = len(vectors[0])

    if recreate:
        client.recreate_collection(
            collection_name=collection,
            vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
        )
    else:
        collections = [item.name for item in client.get_collections().collections]
        if collection not in collections:
            client.create_collection(
                collection_name=collection,
                vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
            )

    points = []
    for chunk, vector in zip(chunks, vectors):
        points.append(
            models.PointStruct(
                id=qdrant_point_id(chunk["chunk_id"]),
                vector=vector,
                payload=chunk,
            )
        )
    client.upsert(collection_name=collection, points=points)


# Hàm chạy từ terminal: tạo embedding và tùy chọn upload vào Qdrant.
def main():
    parser = argparse.ArgumentParser(description="Embed chunks and optionally index them in Qdrant.")
    parser.add_argument(
        "--model",
        default="bge-m3",
        choices=sorted(MODEL_ALIASES.keys()),
        help="Embedding model alias.",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default=None, help="Example: cpu, cuda, mps.")
    parser.add_argument("--limit", type=int, default=None, help="Embed only the first N chunks.")
    parser.add_argument("--qdrant-url", default=None, help="Example: http://localhost:6333")
    parser.add_argument("--qdrant-path", default=None, help="Local Qdrant path, example: data/qdrant_db")
    parser.add_argument("--qdrant-api-key", default=None)
    parser.add_argument("--collection", default="thuoc_metadata")
    parser.add_argument("--recreate", action="store_true")
    args = parser.parse_args()

    model_name = normalize_model_name(args.model)
    chunks = load_chunks(limit=args.limit)
    if not chunks:
        raise RuntimeError(f"No chunks found. Run scripts/chia_nho_co_so_tri_thuc.py first: {INPUT_JSONL}")

    print(f"Embedding {len(chunks)} chunks with {model_name}")
    vectors = embed_chunks(chunks, model_name, args.batch_size, args.device)
    output_path = write_vectors(chunks, vectors, model_name)
    print(f"Wrote vectors: {output_path.relative_to(ROOT)}")

    if args.qdrant_url or args.qdrant_path:
        upload_to_qdrant(
            chunks=chunks,
            vectors=vectors,
            collection=args.collection,
            url=args.qdrant_url,
            path=args.qdrant_path,
            api_key=args.qdrant_api_key,
            recreate=args.recreate,
        )
        print(f"Indexed {len(chunks)} vectors into Qdrant collection: {args.collection}")


if __name__ == "__main__":
    main()
