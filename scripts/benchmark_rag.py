import argparse
import importlib.machinery
import json
import os
import re
import sys
import time
import types
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(ROOT / ".cache" / "huggingface"))
os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", str(ROOT / ".cache" / "sentence_transformers"))
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

DEFAULT_QDRANT_PATH = ROOT / "data" / "qdrant_db"
DEFAULT_OUTPUT_DIR = ROOT / "results" / "rag_benchmark"

LLM_MODELS = {
    "qwen3_4b": "Qwen/Qwen3-4B-Instruct-2507",
    "qwen25_3b": "Qwen/Qwen2.5-3B-Instruct",
}


# Tạo sklearn giả tối thiểu để một số model Hugging Face không lỗi import phụ.
def install_sklearn_stub():
    """Tránh lỗi import sklearn không cần thiết từ một số model Hugging Face."""
    if "sklearn.metrics" in sys.modules:
        return

    sklearn_module = types.ModuleType("sklearn")
    metrics_module = types.ModuleType("sklearn.metrics")

    def roc_curve(*args, **kwargs):
        raise RuntimeError("roc_curve is not available in this benchmark script.")

    metrics_module.roc_curve = roc_curve
    sklearn_module.metrics = metrics_module
    sklearn_module.__spec__ = importlib.machinery.ModuleSpec("sklearn", loader=None, is_package=True)
    sklearn_module.__path__ = []
    metrics_module.__spec__ = importlib.machinery.ModuleSpec("sklearn.metrics", loader=None)
    sys.modules.setdefault("sklearn", sklearn_module)
    sys.modules.setdefault("sklearn.metrics", metrics_module)


# Chuyển path tương đối thành path tuyệt đối theo thư mục gốc project.
def resolve_path(path):
    """Cho phép truyền cả path tuyệt đối lẫn path tương đối từ thư mục project."""
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


# Đọc file YAML cấu hình, ví dụ configs/benchmark_models.yaml.
def load_yaml(path):
    try:
        import yaml
    except ImportError as exc:
        raise SystemExit("Missing dependency: PyYAML. Run: python3 -m pip install PyYAML") from exc
    return yaml.safe_load(resolve_path(path).read_text(encoding="utf-8"))


# Đọc file câu hỏi benchmark; có thể giới hạn số câu bằng limit để test nhanh.
def load_questions(path, limit=None):
    """Đọc danh sách câu hỏi benchmark, có thể giới hạn số câu để test nhẹ."""
    questions = json.loads(resolve_path(path).read_text(encoding="utf-8"))
    if limit:
        questions = questions[:limit]
    if not questions:
        raise ValueError("Question file is empty.")
    return questions


# Chuẩn hóa một chuỗi thành dạng slug an toàn để đặt tên file.
def slugify(value):
    value = value.lower().replace("-", "_")
    value = re.sub(r"[^a-z0-9_]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return value


# Đổi alias embedding thành tên ngắn dùng cho file output.
def embedding_file_stem(alias):
    stems = {
        "sup-simcse-vietnamese-phobert-base": "sup_simcse",
        "vietnamese-bi-encoder": "vietnamese_bi_encoder",
        "bge-m3": "bge_m3",
        "multilingual-e5-base": "multilingual_e5_base",
    }
    return stems.get(alias, slugify(alias))


# Tìm embedding model trong cache local trước, nếu không có thì dùng tên Hugging Face.
def local_embedding_model_path(model_name):
    """Ưu tiên dùng model embedding đã cache sẵn để chạy ổn hơn trên Kaggle/local."""
    cache_dir = ROOT / ".cache" / "sentence_transformers"
    repo_cache = cache_dir / f"models--{model_name.replace('/', '--')}"
    refs_main = repo_cache / "refs" / "main"
    if refs_main.exists():
        revision = refs_main.read_text(encoding="utf-8").strip()
        snapshot = repo_cache / "snapshots" / revision
        if snapshot.exists():
            return str(snapshot)
    return model_name


# Tìm LLM trong cache local trước, nếu không có thì để Transformers tự tải.
def local_hf_model_path(model_name):
    """Ưu tiên dùng LLM đã cache sẵn, nếu chưa có thì để Transformers tự tải."""
    cache_dir = ROOT / ".cache" / "huggingface" / "hub"
    repo_cache = cache_dir / f"models--{model_name.replace('/', '--')}"
    refs_main = repo_cache / "refs" / "main"
    if refs_main.exists():
        revision = refs_main.read_text(encoding="utf-8").strip()
        snapshot = repo_cache / "snapshots" / revision
        if (snapshot / "config.json").exists():
            return str(snapshot)
    return model_name


# Chọn cách pooling vector cho embedding model: mean pooling hoặc lấy token CLS.
def pooling_strategy(alias, model_name):
    key = f"{alias} {model_name}".lower()
    if "e5" in key or "vietnamese-bi-encoder" in key:
        return "mean"
    return "cls"


# Chọn độ dài token tối đa phù hợp với từng embedding model.
def max_length_for(alias, model_name):
    key = f"{alias} {model_name}".lower()
    if "bge-m3" in key:
        return 8192
    if "phobert" in key or "sup-simcse" in key or "vietnamese-bi-encoder" in key:
        return 256
    return 512


# Chuẩn bị câu query trước khi encode; E5 cần prefix "query:".
def prepare_query(question, alias, model_name):
    """Một số embedding model như E5 cần prefix `query:` khi encode câu hỏi."""
    key = f"{alias} {model_name}".lower()
    if "e5" in key:
        return f"query: {question}"
    return question


# Tính mean pooling: lấy trung bình vector token thật, bỏ qua token padding.
def mean_pool(last_hidden_state, attention_mask):
    import torch

    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    summed = torch.sum(last_hidden_state * mask, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counts


# Lớp encode câu hỏi thành vector embedding để truy xuất Qdrant.
class QueryEncoder:
    """Bọc embedding model để encode câu hỏi thành vector truy xuất Qdrant."""

    def __init__(self, alias, model_name, device=None, max_length=None):
        install_sklearn_stub()

        import torch
        from transformers import AutoModel, AutoTokenizer

        self.alias = alias
        self.model_name = model_name
        self.model_path = local_embedding_model_path(model_name)
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


# Lớp gọi Hugging Face LLM để sinh câu trả lời từ prompt RAG.
class LLMGenerator:
    """Bọc Hugging Face causal LLM để sinh câu trả lời từ prompt RAG."""

    def __init__(self, model_name, device=None, force_download=False):
        install_sklearn_stub()

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_name = model_name
        self.model_path = local_hf_model_path(model_name)
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        dtype = torch.float16 if self.device in {"cuda", "mps"} else torch.float32

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                cache_dir=str(ROOT / ".cache" / "huggingface"),
                trust_remote_code=True,
                use_fast=False,
                force_download=force_download,
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                cache_dir=str(ROOT / ".cache" / "huggingface"),
                trust_remote_code=True,
                torch_dtype=dtype,
                force_download=force_download,
            ).to(self.device)
        except OSError as exc:
            if self.model_path != model_name and not force_download:
                print(
                    f"Local cache snapshot looks incomplete for {model_name}; retrying from Hugging Face.",
                    flush=True,
                )
                self.model_path = model_name
                try:
                    self.tokenizer = AutoTokenizer.from_pretrained(
                        self.model_path,
                        cache_dir=str(ROOT / ".cache" / "huggingface"),
                        trust_remote_code=True,
                        use_fast=False,
                    )
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.model_path,
                        cache_dir=str(ROOT / ".cache" / "huggingface"),
                        trust_remote_code=True,
                        torch_dtype=dtype,
                    ).to(self.device)
                except OSError as retry_exc:
                    exc = retry_exc
                else:
                    self.model.eval()
                    return
            raise RuntimeError(
                f"Cannot load LLM model: {model_name}\n"
                "The model is not fully downloaded or the machine cannot reach Hugging Face.\n"
                "Try redownloading it first with:\n"
                f"  python3 scripts/tai_mo_hinh_llm.py --llm {llm_alias_for_model(model_name)} --force-download\n"
                "Then rerun the RAG benchmark."
            ) from exc
        self.model.eval()

    def generate(self, prompt, max_new_tokens):
        import torch

        messages = [
            {
                "role": "system",
                "content": (
                    "Bạn là trợ lý tư vấn thuốc và phân bón nông nghiệp. "
                    "Chỉ trả lời dựa trên CONTEXT được cung cấp. "
                    "Không bịa tên thuốc, hoạt chất, liều lượng hoặc thông tin không có trong context. "
                    "Nếu context không đủ dữ liệu, hãy nói rõ là chưa đủ dữ liệu."
                ),
            },
            {"role": "user", "content": prompt},
        ]
        if hasattr(self.tokenizer, "apply_chat_template"):
            input_text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            input_text = "\n\n".join(f"{item['role']}: {item['content']}" for item in messages)

        encoded = self.tokenizer(input_text, return_tensors="pt").to(self.device)
        input_length = encoded["input_ids"].shape[-1]
        with torch.no_grad():
            output = self.model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        generated_tokens = output[0][input_length:]
        return self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()


# Truy xuất top-k chunks từ Qdrant bằng vector câu hỏi.
def query_qdrant(client, collection_name, vector, top_k):
    """Truy xuất top-k chunks từ Qdrant cho vector câu hỏi."""
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


# Chuyển Qdrant point thành dict gọn để lưu vào file kết quả.
def format_point(rank, point):
    """Rút gọn Qdrant point thành dict dễ lưu vào JSON kết quả."""
    payload = point.payload or {}
    return {
        "rank": rank,
        "score": float(point.score),
        "chunk_id": payload.get("chunk_id"),
        "product_id": payload.get("product_id"),
        "ten_san_pham": payload.get("ten_san_pham"),
        "text": payload.get("text"),
    }


# Ghép các chunks đã truy xuất thành khối CONTEXT đưa cho LLM.
def build_context(retrieved_chunks):
    """Ghép các chunks truy xuất thành CONTEXT đưa vào prompt của LLM."""
    blocks = []
    for chunk in retrieved_chunks:
        blocks.append(
            "\n".join(
                [
                    f"[Rank {chunk['rank']}]",
                    f"product_id: {chunk.get('product_id')}",
                    f"ten_san_pham: {chunk.get('ten_san_pham')}",
                    f"score: {chunk.get('score'):.4f}",
                    "text:",
                    chunk.get("text") or "",
                ]
            )
        )
    return "\n\n---\n\n".join(blocks)


# Tạo prompt cuối cùng gồm CONTEXT, QUESTION và quy tắc trả lời an toàn.
def build_prompt(question, retrieved_chunks):
    """Tạo prompt RAG: context + câu hỏi + quy tắc chống bịa thông tin."""
    context = build_context(retrieved_chunks)
    return f"""CONTEXT:
{context}

QUESTION:
{question}

Yêu cầu:
- Chỉ dùng thông tin trong CONTEXT.
- Không bịa thuốc, hoạt chất, liều lượng, cây trồng hoặc lưu ý an toàn.
- Nếu CONTEXT không đủ dữ liệu để trả lời chắc chắn, hãy nói rõ.
- Trả lời bằng tiếng Việt.
- Giữ đúng format sau:

1. Chẩn đoán khả năng
2. Sản phẩm gợi ý
3. Lý do phù hợp
4. Cách dùng tóm tắt
5. Lưu ý an toàn
"""


# Encode một câu hỏi rồi truy xuất top-k chunks liên quan trong Qdrant.
def retrieve_for_question(client, encoder, collection_name, question, top_k):
    """Encode một câu hỏi rồi tìm top-k chunks liên quan trong Qdrant."""
    started = time.perf_counter()
    vector = encoder.encode(question)
    points = query_qdrant(client, collection_name, vector, top_k)
    latency = time.perf_counter() - started
    retrieved_chunks = [
        format_point(rank=index, point=point)
        for index, point in enumerate(points, start=1)
    ]
    return retrieved_chunks, latency


# Tìm alias ngắn của LLM từ tên model Hugging Face đầy đủ.
def llm_alias_for_model(model_name):
    for alias, configured_model_name in LLM_MODELS.items():
        if configured_model_name == model_name:
            return alias
    return "all"


# Chạy một pipeline RAG cụ thể: một LLM kết hợp với một embedding collection.
def run_pipeline(
    client,
    questions,
    llm_alias,
    llm_model_name,
    embedding_config,
    top_k,
    device,
    llm_device,
    max_new_tokens,
    force_download_models=False,
):
    """Chạy một pipeline cụ thể: một LLM + một embedding collection."""
    embedding_alias = embedding_config["alias"]
    embedding_model_name = embedding_config["model_name"]
    collection_name = embedding_config["collection_name"]

    encoder = QueryEncoder(
        alias=embedding_alias,
        model_name=embedding_model_name,
        device=device,
        max_length=embedding_config.get("max_length"),
    )
    generator = LLMGenerator(
        model_name=llm_model_name,
        device=llm_device,
        force_download=force_download_models,
    )

    results = []
    for question in questions:
        # Mỗi câu hỏi gồm 2 pha: retrieval lấy context, generation sinh đáp án.
        total_started = time.perf_counter()
        retrieved_chunks, retrieval_seconds = retrieve_for_question(
            client=client,
            encoder=encoder,
            collection_name=collection_name,
            question=question["question"],
            top_k=top_k,
        )
        prompt = build_prompt(question["question"], retrieved_chunks)
        generation_started = time.perf_counter()
        answer = generator.generate(prompt, max_new_tokens=max_new_tokens)
        generation_seconds = time.perf_counter() - generation_started
        total_seconds = time.perf_counter() - total_started
        results.append(
            {
                "question_id": question["id"],
                "question": question["question"],
                "llm_model": llm_alias,
                "embedding_model": embedding_alias,
                "retrieved_chunks": retrieved_chunks,
                "answer": answer,
                "latency": {
                    "retrieval_seconds": round(retrieval_seconds, 4),
                    "generation_seconds": round(generation_seconds, 4),
                    "total_seconds": round(total_seconds, 4),
                },
            }
        )
        print(
            f"{llm_alias} + {embedding_alias} | {question['id']} | "
            f"retrieval={retrieval_seconds:.4f}s | generation={generation_seconds:.4f}s",
            flush=True,
        )
    return results


# Hàm entrypoint: đọc tham số dòng lệnh và chạy toàn bộ benchmark.
def main():
    parser = argparse.ArgumentParser(description="Run RAG benchmark for LLM x embedding pipelines.")
    parser.add_argument("--config", required=True, help="Path to configs/benchmark_models.yaml")
    parser.add_argument("--questions", required=True, help="Path to benchmark_questions.json")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--qdrant-path", default=str(DEFAULT_QDRANT_PATH))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--device", default=None, help="Embedding encoder device: cpu, mps, cuda.")
    parser.add_argument("--llm-device", default=None, help="LLM device: cpu, mps, cuda.")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--limit", type=int, default=None, help="Limit questions for smoke test.")
    parser.add_argument("--llm", choices=sorted(LLM_MODELS), default=None, help="Run only one LLM alias.")
    parser.add_argument("--embedding", default=None, help="Run only one embedding alias from config.")
    parser.add_argument(
        "--force-download-models",
        action="store_true",
        help="Force Hugging Face to redownload LLM files, useful after an interrupted Kaggle download.",
    )
    parser.add_argument(
        "--skip-failed",
        action="store_true",
        help="Continue the benchmark when one LLM/embedding pair cannot be loaded.",
    )
    args = parser.parse_args()

    try:
        from qdrant_client import QdrantClient
    except ImportError as exc:
        raise SystemExit("Missing dependency: qdrant-client. Run: python3 -m pip install qdrant-client") from exc

    config = load_yaml(args.config)
    questions = load_questions(args.questions, limit=args.limit)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    client = QdrantClient(path=str(resolve_path(args.qdrant_path)))
    collections = {item.name for item in client.get_collections().collections}

    embedding_configs = config["models"]
    if args.embedding:
        embedding_configs = [item for item in embedding_configs if item["alias"] == args.embedding]
        if not embedding_configs:
            raise ValueError(f"Embedding alias not found in config: {args.embedding}")

    llm_items = LLM_MODELS.items()
    if args.llm:
        llm_items = [(args.llm, LLM_MODELS[args.llm])]

    for embedding_config in embedding_configs:
        collection_name = embedding_config["collection_name"]
        if collection_name not in collections:
            raise RuntimeError(f"Missing Qdrant collection: {collection_name}")

    for llm_alias, llm_model_name in llm_items:
        for embedding_config in embedding_configs:
            embedding_alias = embedding_config["alias"]
            print("=" * 72)
            print(f"LLM       : {llm_alias} ({llm_model_name})")
            print(f"Embedding : {embedding_alias}")
            print(f"Collection: {embedding_config['collection_name']}")
            try:
                results = run_pipeline(
                    client=client,
                    questions=questions,
                    llm_alias=llm_alias,
                    llm_model_name=llm_model_name,
                    embedding_config=embedding_config,
                    top_k=args.top_k,
                    device=args.device,
                    llm_device=args.llm_device,
                    max_new_tokens=args.max_new_tokens,
                    force_download_models=args.force_download_models,
                )
            except RuntimeError as exc:
                if not args.skip_failed:
                    raise
                print(f"SKIPPED: {llm_alias} + {embedding_alias}: {exc}", flush=True)
                continue
            output_path = output_dir / f"{llm_alias}__{embedding_file_stem(embedding_alias)}.json"
            output_path.write_text(
                json.dumps(results, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            print(f"wrote: {output_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
