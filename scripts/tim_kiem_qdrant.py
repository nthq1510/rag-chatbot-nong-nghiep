#!/usr/bin/env python3
"""Tìm kiếm thử trong Qdrant bằng một câu query nhập từ terminal."""
import argparse
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(ROOT / ".cache" / "huggingface"))
os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", str(ROOT / ".cache" / "sentence_transformers"))
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

from tao_embedding_va_index_qdrant import MODEL_ALIASES, normalize_model_name


# Tìm embedding model trong cache local để encode query.
def local_model_path(model_name):
    """Tìm embedding model trong cache local; nếu không có thì trả tên model gốc."""
    cache_dir = ROOT / ".cache" / "sentence_transformers"
    repo_cache = cache_dir / f"models--{model_name.replace('/', '--')}"
    refs_main = repo_cache / "refs" / "main"
    if refs_main.exists():
        revision = refs_main.read_text(encoding="utf-8").strip()
        snapshot = repo_cache / "snapshots" / revision
        if snapshot.exists():
            return str(snapshot)
    return model_name


# Encode câu hỏi tìm kiếm thành vector để query Qdrant.
def encode_query(query, model_name):
    """Encode câu query thành vector để gửi vào Qdrant."""
    import torch
    import torch.nn.functional as functional
    from transformers import AutoModel, AutoTokenizer

    text = f"query: {query}" if model_name == "intfloat/multilingual-e5-base" else query
    model_path = local_model_path(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModel.from_pretrained(model_path, local_files_only=True)
    model.eval()

    with torch.no_grad():
        encoded = tokenizer(
            [text],
            padding=True,
            truncation=True,
            max_length=8192 if model_name == "BAAI/bge-m3" else 512,
            return_tensors="pt",
        )
        output = model(**encoded)
        vector = output.last_hidden_state[:, 0]
        vector = functional.normalize(vector, p=2, dim=1)[0]
    return vector.cpu().tolist()


# Hàm chạy từ terminal: nhận query, tìm Qdrant và in top kết quả.
def main():
    parser = argparse.ArgumentParser(description="Search local Qdrant vector database.")
    parser.add_argument("query")
    parser.add_argument("--model", default="bge-m3", choices=sorted(MODEL_ALIASES.keys()))
    parser.add_argument("--qdrant-path", default=str(ROOT / "data" / "qdrant_db"))
    parser.add_argument("--collection", default="thuoc_metadata")
    parser.add_argument("--limit", type=int, default=5)
    args = parser.parse_args()

    try:
        from qdrant_client import QdrantClient
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: qdrant-client.\n"
            "Run this command in the project folder:\n"
            "  python3 -m pip install qdrant-client"
        ) from exc

    model_name = normalize_model_name(args.model)
    vector = encode_query(args.query, model_name)
    client = QdrantClient(path=args.qdrant_path)

    if hasattr(client, "query_points"):
        response = client.query_points(
            collection_name=args.collection,
            query=vector,
            limit=args.limit,
            with_payload=True,
        )
        points = response.points
    else:
        points = client.search(
            collection_name=args.collection,
            query_vector=vector,
            limit=args.limit,
            with_payload=True,
        )

    for index, point in enumerate(points, start=1):
        payload = point.payload or {}
        print(f"{index}. score={point.score:.4f} | {payload.get('product_id')} | {payload.get('ten_san_pham')}")
        print(f"   loai={payload.get('loai')}")
        print(f"   rule_key={payload.get('rule_key')}")
        print(f"   chunk_id={payload.get('chunk_id')}")


if __name__ == "__main__":
    main()
