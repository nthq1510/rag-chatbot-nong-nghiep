#!/usr/bin/env python3
"""Tải trước các LLM cần dùng để tránh lỗi khi chạy benchmark trên Kaggle."""
import argparse
import importlib.machinery
import os
import sys
import types
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(ROOT / ".cache" / "huggingface"))
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

LLM_MODELS = {
    "qwen3_4b": "Qwen/Qwen3-4B-Instruct-2507",
    "qwen25_3b": "Qwen/Qwen2.5-3B-Instruct",
}


# Tạo sklearn giả tối thiểu để Transformers không lỗi dependency phụ.
def install_sklearn_stub():
    """Tạo sklearn stub tối thiểu để tránh lỗi dependency phụ khi load Transformers."""
    if "sklearn.metrics" in sys.modules:
        return

    sklearn_module = types.ModuleType("sklearn")
    metrics_module = types.ModuleType("sklearn.metrics")

    def roc_curve(*args, **kwargs):
        raise RuntimeError("roc_curve is not available in this download script.")

    metrics_module.roc_curve = roc_curve
    sklearn_module.metrics = metrics_module
    sklearn_module.__spec__ = importlib.machinery.ModuleSpec("sklearn", loader=None, is_package=True)
    sklearn_module.__path__ = []
    metrics_module.__spec__ = importlib.machinery.ModuleSpec("sklearn.metrics", loader=None)
    sys.modules.setdefault("sklearn", sklearn_module)
    sys.modules.setdefault("sklearn.metrics", metrics_module)


# Tải tokenizer và LLM vào cache local của project.
def download_model(model_name, force_download=False):
    """Tải tokenizer và model LLM vào cache Hugging Face của project."""
    install_sklearn_stub()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Downloading tokenizer: {model_name}", flush=True)
    AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=str(ROOT / ".cache" / "huggingface"),
        trust_remote_code=True,
        use_fast=False,
        force_download=force_download,
    )

    print(f"Downloading model: {model_name}", flush=True)
    AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=str(ROOT / ".cache" / "huggingface"),
        trust_remote_code=True,
        force_download=force_download,
    )
    print(f"Done: {model_name}", flush=True)


# Hàm chạy từ terminal: tải một hoặc tất cả LLM theo alias.
def main():
    parser = argparse.ArgumentParser(description="Download LLM models used by RAG benchmark.")
    parser.add_argument(
        "--llm",
        choices=sorted(LLM_MODELS) + ["all"],
        default="all",
        help="LLM alias to download.",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Force Hugging Face to redownload files after an interrupted or corrupt cache.",
    )
    args = parser.parse_args()

    items = LLM_MODELS.items()
    if args.llm != "all":
        items = [(args.llm, LLM_MODELS[args.llm])]

    for alias, model_name in items:
        print("=" * 72)
        print(f"alias: {alias}")
        download_model(model_name, force_download=args.force_download)


if __name__ == "__main__":
    main()
