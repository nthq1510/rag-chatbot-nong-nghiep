#!/usr/bin/env python3
"""Benchmark riêng phần truy xuất top-k từ Qdrant.

Khác với benchmark_rag.py, script này không gọi LLM. Nó chỉ đo embedding
model nào truy xuất chunks/sản phẩm tốt hơn cho cùng một bộ câu hỏi.
"""
import argparse
import json
import os
import re
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(ROOT / ".cache" / "huggingface"))
os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", str(ROOT / ".cache" / "sentence_transformers"))
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

DEFAULT_QDRANT_PATH = ROOT / "data" / "qdrant_db"
DEFAULT_OUTPUT_DIR = ROOT / "results" / "retrieval_benchmark"


# Chuyển path tương đối thành path tuyệt đối theo thư mục project.
def resolve_path(path):
    """Đổi path tương đối thành path tuyệt đối theo thư mục project."""
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


# Chuẩn hóa chuỗi thành slug an toàn để đặt tên file.
def slugify(value):
    """Chuẩn hóa tên thành dạng an toàn để đặt tên file."""
    value = value.lower().replace("-", "_")
    value = re.sub(r"[^a-z0-9_]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return value


# Đổi alias embedding thành tên file kết quả ngắn gọn.
def result_file_stem(alias):
    """Đổi alias embedding thành tên file kết quả ngắn gọn."""
    stems = {
        "sup-simcse-vietnamese-phobert-base": "sup_simcse",
        "vietnamese-bi-encoder": "vietnamese_bi_encoder",
        "bge-m3": "bge_m3",
        "multilingual-e5-base": "multilingual_e5_base",
    }
    return stems.get(alias, slugify(alias))


# Đọc file YAML cấu hình embedding và collection Qdrant.
def load_yaml(path):
    """Đọc file YAML config embedding/collection."""
    try:
        import yaml
    except ImportError as exc:
        raise SystemExit("Missing dependency: PyYAML. Run: python3 -m pip install PyYAML") from exc
    return yaml.safe_load(resolve_path(path).read_text(encoding="utf-8"))


# Đọc danh sách câu hỏi benchmark từ file JSON.
def load_questions(path):
    """Đọc file JSON chứa danh sách câu hỏi benchmark."""
    questions = json.loads(resolve_path(path).read_text(encoding="utf-8"))
    if not questions:
        raise ValueError("Question file is empty.")
    return questions


# Tìm embedding model trong cache local trước khi tải từ Hugging Face.
def local_model_path(model_name):
    """Ưu tiên dùng model embedding đã cache ở local nếu tồn tại."""
    cache_dir = ROOT / ".cache" / "sentence_transformers"
    repo_cache = cache_dir / f"models--{model_name.replace('/', '--')}"
    refs_main = repo_cache / "refs" / "main"
    if refs_main.exists():
        revision = refs_main.read_text(encoding="utf-8").strip()
        snapshot = repo_cache / "snapshots" / revision
        if snapshot.exists():
            return str(snapshot)
    return model_name


# Chọn cách gom vector token thành vector câu/chunk.
def pooling_strategy(alias, model_name):
    """Chọn cách pooling vector: CLS hoặc mean pooling tùy model."""
    key = f"{alias} {model_name}".lower()
    if "e5" in key or "vietnamese-bi-encoder" in key:
        return "mean"
    return "cls"


# Chọn độ dài token tối đa phù hợp từng embedding model.
def max_length_for(alias, model_name):
    """Chọn max_length phù hợp để tránh cắt context quá nhiều hoặc quá tốn RAM."""
    key = f"{alias} {model_name}".lower()
    if "bge-m3" in key:
        return 8192
    if "phobert" in key or "sup-simcse" in key or "vietnamese-bi-encoder" in key:
        return 256
    return 512


# Chuẩn bị câu query trước khi encode; E5 cần prefix "query:".
def prepare_query(question, alias, model_name):
    """Thêm prefix cho query nếu model yêu cầu, ví dụ E5 dùng `query:`."""
    key = f"{alias} {model_name}".lower()
    if "e5" in key:
        return f"query: {question}"
    return question


# Tính mean pooling, chỉ lấy trung bình trên token thật và bỏ qua padding.
def mean_pool(last_hidden_state, attention_mask):
    """Tính vector trung bình có xét attention mask."""
    import torch

    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    summed = torch.sum(last_hidden_state * mask, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counts


# Lớp load embedding model và encode câu hỏi thành vector.
class QueryEncoder:
    """Load embedding model và encode câu hỏi thành vector."""

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
        token_count = len(self.tokenizer)
        embedding_count = self.model.get_input_embeddings().num_embeddings
        if token_count > embedding_count:
            self.model.resize_token_embeddings(token_count)
        self.model.eval()

    def encode(self, question):
        """Encode một câu hỏi thành list float để gửi vào Qdrant."""
        import torch
        import torch.nn.functional as functional

        text = prepare_query(question, self.alias, self.model_name)
        with torch.no_grad():
            encoded = self.tokenizer(
                [text],
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            output = self.model(**encoded)
            if self.pooling == "mean":
                vector = mean_pool(output.last_hidden_state, encoded["attention_mask"])
            else:
                vector = output.last_hidden_state[:, 0]
            vector = functional.normalize(vector, p=2, dim=1)[0]
        return vector.cpu().tolist()


# Gửi vector câu hỏi vào Qdrant để lấy top-k chunks liên quan.
def query_qdrant(client, collection_name, vector, top_k):
    """Truy vấn Qdrant và trả về top-k points."""
    if hasattr(client, "query_points"):
        response = client.query_points(
            collection_name=collection_name,
            query=vector,
            limit=top_k,
            with_payload=True,
        )
        return response.points
    return client.search(
        collection_name=collection_name,
        query_vector=vector,
        limit=top_k,
        with_payload=True,
    )


# Chuyển Qdrant point thành dict gọn để lưu kết quả benchmark.
def format_point(rank, point):
    """Rút gọn Qdrant point thành dict dễ đọc/dễ lưu."""
    payload = point.payload or {}
    return {
        "rank": rank,
        "score": float(point.score),
        "chunk_id": payload.get("chunk_id"),
        "product_id": payload.get("product_id"),
        "ten_san_pham": payload.get("ten_san_pham"),
        "text": payload.get("text"),
    }


# Benchmark một embedding model trên toàn bộ câu hỏi.
def benchmark_model(client, questions, model_config, top_k, device):
    """Benchmark một embedding model trên toàn bộ câu hỏi."""
    alias = model_config["alias"]
    model_name = model_config["model_name"]
    collection_name = model_config["collection_name"]
    encoder = QueryEncoder(
        alias=alias,
        model_name=model_name,
        device=device,
        max_length=model_config.get("max_length"),
    )

    results = []
    for question in questions:
        started = time.perf_counter()
        vector = encoder.encode(question["question"])
        points = query_qdrant(client, collection_name, vector, top_k)
        latency = time.perf_counter() - started
        results.append(
            {
                "question_id": question["id"],
                "question": question["question"],
                "embedding_model": alias,
                "collection": collection_name,
                "retrieved_chunks": [
                    format_point(rank=index, point=point)
                    for index, point in enumerate(points, start=1)
                ],
                "latency_seconds": round(latency, 4),
            }
        )
        print(
            f"{alias} | {question['id']} | top_k={top_k} | latency={latency:.4f}s",
            flush=True,
        )
    return results


# Hàm chạy từ terminal: đọc tham số CLI và benchmark tất cả embedding model.
def main():
    parser = argparse.ArgumentParser(description="Run retrieval-only benchmark across embedding collections.")
    parser.add_argument("--config", required=True, help="Path to configs/benchmark_models.yaml")
    parser.add_argument("--questions", required=True, help="Path to benchmark_questions.json")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--qdrant-path", default=str(DEFAULT_QDRANT_PATH))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--device", default=None, help="Example: cpu, mps, cuda. Default auto-selects mps/cpu.")
    args = parser.parse_args()

    try:
        from qdrant_client import QdrantClient
    except ImportError as exc:
        raise SystemExit("Missing dependency: qdrant-client. Run: python3 -m pip install qdrant-client") from exc

    config = load_yaml(args.config)
    questions = load_questions(args.questions)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    client = QdrantClient(path=str(resolve_path(args.qdrant_path)))
    collections = {item.name for item in client.get_collections().collections}

    for model_config in config["models"]:
        collection_name = model_config["collection_name"]
        alias = model_config["alias"]
        if collection_name not in collections:
            existing = ", ".join(sorted(collections)) or "none"
            raise RuntimeError(
                f"Missing Qdrant collection '{collection_name}' for model '{alias}'. "
                f"Existing collections: {existing}. "
                "Build embedding indexes first with: "
                "python3 scripts/tao_tat_ca_chi_muc_embedding.py "
                "--config configs/benchmark_models.yaml "
                "--qdrant-path data/qdrant_db "
                "--recreate"
            )

        print("=" * 72)
        print(f"benchmarking model: {alias}")
        print(f"collection        : {collection_name}")
        results = benchmark_model(
            client=client,
            questions=questions,
            model_config=model_config,
            top_k=args.top_k,
            device=args.device,
        )
        output_path = output_dir / f"{result_file_stem(alias)}.json"
        output_path.write_text(
            json.dumps(results, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"wrote: {output_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
