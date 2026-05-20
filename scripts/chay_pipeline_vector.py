#!/usr/bin/env python3
"""Chạy nhanh pipeline dữ liệu vector: build metadata, chia chunks, tùy chọn embedding."""
import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


# In command ra màn hình rồi chạy command đó trong thư mục project.
def run(command):
    """In command rồi chạy trong thư mục project."""
    print("$ " + " ".join(command))
    subprocess.run(command, cwd=ROOT, check=True)


# Hàm chạy từ terminal: build metadata, chia chunks và tùy chọn tạo embedding.
def main():
    parser = argparse.ArgumentParser(description="Run metadata splitting, chunking, and optional embedding.")
    parser.add_argument("--max-words", type=int, default=220)
    parser.add_argument("--overlap-words", type=int, default=40)
    parser.add_argument("--embed", action="store_true", help="Also run embedding.")
    parser.add_argument("--model", default="bge-m3")
    parser.add_argument("--qdrant-url", default=None)
    parser.add_argument("--qdrant-path", default=None)
    parser.add_argument("--collection", default="thuoc_metadata")
    parser.add_argument("--recreate", action="store_true")
    args = parser.parse_args()

    run([sys.executable, "scripts/xay_dung_co_so_tri_thuc.py"])
    run(
        [
            sys.executable,
            "scripts/chia_nho_co_so_tri_thuc.py",
            "--max-words",
            str(args.max_words),
            "--overlap-words",
            str(args.overlap_words),
        ]
    )

    if args.embed:
        command = [
            sys.executable,
            "scripts/tao_embedding_va_index_qdrant.py",
            "--model",
            args.model,
            "--collection",
            args.collection,
        ]
        if args.qdrant_url:
            command.extend(["--qdrant-url", args.qdrant_url])
        if args.qdrant_path:
            command.extend(["--qdrant-path", args.qdrant_path])
        if args.recreate:
            command.append("--recreate")
        run(command)


if __name__ == "__main__":
    main()
