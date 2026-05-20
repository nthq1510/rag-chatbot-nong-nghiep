#!/usr/bin/env python3
"""Tạo toàn bộ collection embedding trong Qdrant theo config benchmark.

Script này chạy qua nhiều embedding model trong configs/benchmark_models.yaml
và tạo một collection Qdrant tương ứng cho từng model.
"""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(ROOT / ".cache" / "huggingface"))
os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", str(ROOT / ".cache" / "sentence_transformers"))
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

CHUNKS_JSONL = ROOT / "data" / "chunks" / "chunks.jsonl"


# Chuyển path tương đối thành path tuyệt đối theo thư mục project.
def resolve_path(path):
    """Đổi path tương đối thành path tuyệt đối theo thư mục project."""
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


# Đọc và kiểm tra file YAML chứa danh sách embedding model/collection.
def load_config(path):
    """Đọc và kiểm tra file config YAML chứa danh sách embedding model."""
    try:
        import yaml
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: PyYAML.\n"
            "Run: python3 -m pip install PyYAML"
        ) from exc

    config = yaml.safe_load(resolve_path(path).read_text(encoding="utf-8"))
    models = config.get("models", [])
    if not models:
        raise ValueError("Config must contain a non-empty 'models' list.")

    # alias: tên ngắn của model, ví dụ bge-m3.
    # model_name: tên model trên Hugging Face.
    # collection_name: tên collection trong Qdrant.
    # batch_size: số lượng văn bản encode mỗi lần.
    required = {"alias", "model_name", "collection_name", "batch_size"}
    for index, model_config in enumerate(models, start=1):
        missing = required - set(model_config)
        if missing:
            raise ValueError(f"Model config #{index} is missing keys: {sorted(missing)}")
    return config

# Đọc chunks từ JSONL; limit giúp test nhanh trước khi chạy toàn bộ.
def load_chunks(path=CHUNKS_JSONL, limit=None):
    """Đọc chunks từ JSONL; limit dùng để test nhanh ít dữ liệu."""
    chunks = []
    with path.open(encoding="utf-8") as file:
        for line in file:
            if line.strip():
                chunks.append(json.loads(line))
            if limit and len(chunks) >= limit:
                break
    if not chunks:
        raise RuntimeError(f"No chunks found at {path}")
    return chunks

# Tạo id ổn định cho Qdrant từ model alias và chunk_id.
def qdrant_point_id(model_alias, chunk_id):
    """Tạo id ổn định cho Qdrant từ model alias và chunk_id."""
    digest = hashlib.md5(f"{model_alias}:{chunk_id}".encode("utf-8")).hexdigest()
    return digest

# Tìm model embedding trong cache local trước khi tải từ Hugging Face.
def local_model_path(model_name):
    """Ưu tiên dùng snapshot model đã tải trong cache local."""
    cache_dir = ROOT / ".cache" / "sentence_transformers"
    repo_cache = cache_dir / f"models--{model_name.replace('/', '--')}"
    refs_main = repo_cache / "refs" / "main"
    if refs_main.exists():
        revision = refs_main.read_text(encoding="utf-8").strip()
        snapshot = repo_cache / "snapshots" / revision
        if snapshot.exists():
            return str(snapshot)
    return model_name

# Chọn cách pooling vector token: mean pooling hoặc CLS.
def pooling_strategy(alias, model_name):
    """Chọn cách gom vector token thành vector câu/chunk: CLS hoặc mean."""
    key = f"{alias} {model_name}".lower()
    if "e5" in key or "vietnamese-bi-encoder" in key:
        return "mean"
    return "cls"

# Chọn max_length token phù hợp với từng embedding model.
def max_length_for(alias, model_name):
    """Chọn độ dài token tối đa phù hợp từng embedding model."""
    key = f"{alias} {model_name}".lower()
    if "bge-m3" in key:
        return 8192
    if "phobert" in key or "sup-simcse" in key or "vietnamese-bi-encoder" in key:
        return 256
    return 512


# Chuẩn bị text trước khi encode; E5 cần prefix "passage:".
def prepare_texts(chunks, alias, model_name):
    """Chuẩn bị text trước khi encode; E5 cần prefix `passage:`."""
    texts = [chunk["text"] for chunk in chunks]
    key = f"{alias} {model_name}".lower()
    if "e5" in key:
        return [f"passage: {text}" for text in texts]
    return texts

# Tính trung bình vector token thật, bỏ qua padding.
def mean_pool(last_hidden_state, attention_mask):
    """Tính trung bình vector các token thật, bỏ qua padding token."""
    import torch

    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    summed = torch.sum(last_hidden_state * mask, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counts

# Lớp load tokenizer/model Transformer và encode chunks thành vector.
class TransformersEmbedder:
    """Load tokenizer/model Transformer và encode chunks thành vector chuẩn hóa."""

    def __init__(self, alias, model_name, device=None, max_length=None):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.alias = alias
        self.model_name = model_name
        self.model_path = local_model_path(model_name)
        self.pooling = pooling_strategy(alias, model_name)
        self.max_length = int(max_length or max_length_for(alias, model_name))
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            cache_dir=str(ROOT / ".cache" / "huggingface"),
            trust_remote_code=True,
        )
        self.model = AutoModel.from_pretrained(
            self.model_path,
            cache_dir=str(ROOT / ".cache" / "huggingface"),
            trust_remote_code=True,
        ).to(self.device)
        self.sync_tokenizer_and_model_vocab()
        self.model.eval()

    def sync_tokenizer_and_model_vocab(self):
        """Resize embedding matrix nếu tokenizer có thêm token mới."""
        token_count = len(self.tokenizer)
        embedding_count = self.model.get_input_embeddings().num_embeddings
        if token_count > embedding_count:
            print(
                f"    resizing token embeddings: {embedding_count} -> {token_count}",
                flush=True,
            )
            self.model.resize_token_embeddings(token_count)

    def encode(self, texts, batch_size):
        """Encode danh sách văn bản theo batch để tránh quá tải RAM/VRAM."""
        import torch
        import torch.nn.functional as functional

        vectors = []
        with torch.no_grad():
            for start in range(0, len(texts), batch_size):
                batch = texts[start : start + batch_size]
                encoded = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                max_token_id = int(encoded["input_ids"].max())
                embedding_count = self.model.get_input_embeddings().num_embeddings
                if max_token_id >= embedding_count:
                    raise RuntimeError(
                        f"Tokenizer/model vocab mismatch before CUDA: "
                        f"max token id={max_token_id}, model embeddings={embedding_count}, "
                        f"model={self.model_name}. Try lower max_length or clear/re-download model cache."
                    )
                encoded = {key: value.to(self.device) for key, value in encoded.items()}
                output = self.model(**encoded)
                if self.pooling == "mean":
                    vector = mean_pool(output.last_hidden_state, encoded["attention_mask"])
                else:
                    vector = output.last_hidden_state[:, 0]
                vector = functional.normalize(vector, p=2, dim=1)
                vectors.extend(vector.cpu().tolist())
                print(f"    encoded {min(start + batch_size, len(texts))}/{len(texts)}", flush=True)
        return vectors

# Tạo collection Qdrant mới hoặc xóa/tạo lại nếu --recreate.
def recreate_or_create_collection(client, collection_name, vector_size, recreate):
    """Tạo collection Qdrant mới hoặc xóa/tạo lại nếu --recreate."""
    from qdrant_client.http import models

    exists = client.collection_exists(collection_name)
    if exists and recreate:
        client.delete_collection(collection_name)
        exists = False
    if not exists:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
        )


# Đưa vector và payload chunk vào collection Qdrant.
def upsert_vectors(client, collection_name, model_alias, chunks, vectors):
    """Đưa vector và payload chunk vào collection Qdrant."""
    from qdrant_client.http import models

    points = []
    for chunk, vector in zip(chunks, vectors):
        payload = dict(chunk)
        payload["embedding_model"] = model_alias
        points.append(
            models.PointStruct(
                id=qdrant_point_id(model_alias, chunk["chunk_id"]),
                vector=vector,
                payload=payload,
            )
        )

    client.upsert(collection_name=collection_name, points=points)


# Chạy toàn bộ quy trình tạo index cho một embedding model.
def build_index(client, chunks, model_config, recreate, device):
    """Chạy toàn bộ quy trình index cho một embedding model."""
    alias = model_config["alias"]
    model_name = model_config["model_name"]
    collection_name = model_config["collection_name"]
    batch_size = int(model_config["batch_size"])
    max_length = int(model_config.get("max_length") or max_length_for(alias, model_name))

    print("=" * 72)
    print(f"model alias      : {alias}")
    print(f"huggingface model: {model_name}")
    print(f"collection       : {collection_name}")
    print(f"batch size       : {batch_size}")
    print(f"max length       : {max_length}")
    print(f"total chunks     : {len(chunks)}")

    started = time.perf_counter()
    embedder = TransformersEmbedder(
        alias=alias,
        model_name=model_name,
        device=device,
        max_length=max_length,
    )
    texts = prepare_texts(chunks, alias, model_name)
    vectors = embedder.encode(texts, batch_size=batch_size)
    vector_dimension = len(vectors[0])

    recreate_or_create_collection(
        client=client,
        collection_name=collection_name,
        vector_size=vector_dimension,
        recreate=recreate,
    )
    upsert_vectors(
        client=client,
        collection_name=collection_name,
        model_alias=alias,
        chunks=chunks,
        vectors=vectors,
    )

    elapsed = time.perf_counter() - started
    print(f"vector dimension : {vector_dimension}")
    print(f"elapsed seconds  : {elapsed:.2f}")
    print(f"status           : indexed {len(vectors)} vectors")
    return {
        "alias": alias,
        "model_name": model_name,
        "collection_name": collection_name,
        "batch_size": batch_size,
        "max_length": max_length,
        "vector_dimension": vector_dimension,
        "total_chunks": len(chunks),
        "elapsed_seconds": round(elapsed, 2),
    }


# Hàm chạy từ terminal: tạo tất cả collection embedding theo config.
def main():
    parser = argparse.ArgumentParser(description="Build Qdrant indexes for all benchmark embedding models.")
    parser.add_argument("--config", required=True, help="Path to benchmark_models.yaml")
    parser.add_argument("--qdrant-path", required=True, help="Local Qdrant path, example: data/qdrant_db")
    parser.add_argument("--recreate", action="store_true", help="Delete and rebuild collections if they exist.")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit for a quick smoke test.")
    parser.add_argument("--device", default=None, help="Example: cpu, mps, cuda. Default auto-selects mps/cpu.")
    parser.add_argument("--embedding", default=None, help="Run only one embedding alias from config.")
    args = parser.parse_args()

    try:
        from qdrant_client import QdrantClient
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: qdrant-client.\n"
            "Run: python3 -m pip install qdrant-client"
        ) from exc

    config = load_config(args.config)
    chunks = load_chunks(limit=args.limit)
    client = QdrantClient(path=str(resolve_path(args.qdrant_path)))

    model_configs = config["models"]
    if args.embedding:
        model_configs = [item for item in model_configs if item["alias"] == args.embedding]
        if not model_configs:
            raise ValueError(f"Embedding alias not found in config: {args.embedding}")

    summaries = []
    for model_config in model_configs:
        summaries.append(
            build_index(
                client=client,
                chunks=chunks,
                model_config=model_config,
                recreate=args.recreate,
                device=args.device,
            )
        )

    print("=" * 72)
    print("benchmark summary")
    for summary in summaries:
        print(
            f"{summary['alias']}: dim={summary['vector_dimension']}, "
            f"chunks={summary['total_chunks']}, elapsed={summary['elapsed_seconds']}s, "
            f"collection={summary['collection_name']}"
        )


if __name__ == "__main__":
    main()
