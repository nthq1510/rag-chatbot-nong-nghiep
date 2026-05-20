#!/usr/bin/env python3
"""Chuyển metadata sản phẩm thành chunks văn bản để tạo embedding.

Input:
- data/Thuoc_Metadata/*.json

Output:
- data/chunks/chunks.jsonl
- data/chunks/chunks.json

Mỗi chunk giữ lại payload như product_id, tên sản phẩm, loại, rule_key
để khi truy xuất Qdrant có thể biết chunk thuộc sản phẩm nào.
"""
import argparse
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = ROOT / "data" / "Thuoc_Metadata"
OUTPUT_DIR = ROOT / "data" / "chunks"
OUTPUT_JSONL = OUTPUT_DIR / "chunks.jsonl"
OUTPUT_JSON = OUTPUT_DIR / "chunks.json"


# Chuẩn hóa một giá trị thành chuỗi sạch, bỏ khoảng trắng thừa.
def clean_text(value):
    return re.sub(r"\s+", " ", str(value or "").strip())


# Ghép list thành chuỗi phân tách bằng dấu chấm phẩy.
def join_list(values):
    if isinstance(values, list):
        return "; ".join(clean_text(value) for value in values if clean_text(value))
    return clean_text(values)


# Format dict thành chuỗi "key: value" để đưa vào text chunk.
def format_dict(value):
    if not isinstance(value, dict):
        return clean_text(value)
    parts = []
    for key, item in value.items():
        if isinstance(item, (dict, list)):
            item = json.dumps(item, ensure_ascii=False, sort_keys=True)
        parts.append(f"{key}: {item}")
    return "; ".join(parts)


# Biến một record metadata thành đoạn text đầy đủ ngữ cảnh cho embedding.
def record_to_text(record):
    """Biến một record metadata thành đoạn văn bản giàu ngữ cảnh cho embedding."""
    lines = [
        f"Sản phẩm: {record['ten_san_pham']}",
        f"Loại: {record['loai']}",
        f"Rule key: {record['rule_key']}",
        f"Thành phần: {format_dict(record['thanh_phan'])}",
        f"Quy cách: {record['quy_cach']}",
        f"Công dụng: {join_list(record['cong_dung'])}",
        f"Triệu chứng: {join_list(record['trieu_chung'])}",
        f"Nguyên nhân: {join_list(record['nguyen_nhan'])}",
        f"Cây trồng phù hợp: {join_list(record['doi_tuong_cay_trong'])}",
        f"Thời điểm xử lý: {join_list(record['thoi_diem_xu_ly'])}",
        f"Hướng dẫn sử dụng: {format_dict(record['huong_dan_su_dung'])}",
        f"An toàn sử dụng: {join_list(record['an_toan_su_dung'])}",
        f"Giá tham khảo: {format_dict(record['gia'])}",
        f"Đặc tính bệnh/hoạt chất: {join_list(record['dac_tinh_benh'])}",
        f"Giai đoạn phù hợp: {join_list(record['giai_doan_phu_hop'])}",
        f"Đất/môi trường phù hợp: {join_list(record['loai_dat_moi_truong'])}",
        f"Gợi ý phối hợp: {join_list(record['goi_y_phoi_hop'])}",
        f"Lý do chuyên gia: {record['ly_do_chuyen_gia']}",
    ]
    return "\n".join(line for line in lines if clean_text(line.split(":", 1)[-1]))


# Chia text thành nhiều chunk theo số từ, có overlap để giữ ngữ cảnh.
def word_chunks(text, max_words, overlap_words):
    """Chia text theo số từ, có overlap để tránh mất ngữ cảnh ở ranh giới chunk."""
    words = text.split()
    if len(words) <= max_words:
        return [text]
    chunks = []
    start = 0
    step = max(1, max_words - overlap_words)
    while start < len(words):
        chunk_words = words[start : start + max_words]
        chunks.append(" ".join(chunk_words))
        if start + max_words >= len(words):
            break
        start += step
    return chunks


# Lấy danh sách file metadata sản phẩm, bỏ qua file bắt đầu bằng "_".
def iter_metadata_files():
    return sorted(path for path in INPUT_DIR.glob("*.json") if not path.name.startswith("_"))


# Tạo toàn bộ chunks từ các file metadata.
def build_chunks(max_words, overlap_words):
    """Tạo danh sách chunks từ toàn bộ file metadata."""
    chunks = []
    for input_path in iter_metadata_files():
        records = json.loads(input_path.read_text(encoding="utf-8"))
        for record in records:
            text = record_to_text(record)
            for index, chunk_text in enumerate(word_chunks(text, max_words, overlap_words)):
                chunk_id = f"{record['product_id']}::chunk_{index + 1}"
                chunks.append(
                    {
                        "chunk_id": chunk_id,
                        "product_id": record["product_id"],
                        "ten_san_pham": record["ten_san_pham"],
                        "loai": record["loai"],
                        "rule_key": record["rule_key"],
                        "chunk_index": index + 1,
                        "source_file": str(input_path.relative_to(ROOT)),
                        "text": chunk_text,
                    }
                )
    return chunks


# Hàm chạy từ terminal: chia metadata thành chunks và ghi JSON/JSONL.
def main():
    parser = argparse.ArgumentParser(description="Chunk product metadata for embedding.")
    parser.add_argument("--max-words", type=int, default=220)
    parser.add_argument("--overlap-words", type=int, default=40)
    args = parser.parse_args()

    if args.overlap_words >= args.max_words:
        raise ValueError("--overlap-words must be smaller than --max-words")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    chunks = build_chunks(args.max_words, args.overlap_words)
    with OUTPUT_JSONL.open("w", encoding="utf-8") as file:
        for chunk in chunks:
            file.write(json.dumps(chunk, ensure_ascii=False) + "\n")
    OUTPUT_JSON.write_text(
        json.dumps(chunks, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Wrote {len(chunks)} chunks: {OUTPUT_JSONL.relative_to(ROOT)}")
    print(f"Wrote readable copy: {OUTPUT_JSON.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
