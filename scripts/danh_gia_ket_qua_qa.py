import argparse
import json
from pathlib import Path


s# Đọc file JSON UTF-8 từ đường dẫn truyền vào.
def load_json(path):
    """Đọc một file JSON UTF-8."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


# Chuẩn hóa text để so khớp không phân biệt chữ hoa/thường.
def normalize(text):
    """Chuẩn hóa text để so khớp không phân biệt hoa/thường."""
    return str(text or "").casefold()


# Kiểm tra câu trả lời có chứa ít nhất một keyword kỳ vọng hay không.
def keyword_hit(answer, keywords):
    """Kiểm tra câu trả lời có chứa ít nhất một keyword kỳ vọng không."""
    if not keywords:
        return True
    answer_text = normalize(answer)
    normalized_keywords = [normalize(item) for item in keywords if str(item).strip()]
    if not normalized_keywords:
        return True
    return any(keyword in answer_text for keyword in normalized_keywords)


# Kiểm tra top-k retrieved chunks có chứa product_id kỳ vọng không.
def product_hit(retrieved_chunks, expected_product_ids):
    """Kiểm tra top-k chunks có lấy đúng product_id kỳ vọng không."""
    expected = {str(item) for item in expected_product_ids if item}
    if not expected:
        return True
    retrieved = {
        str(chunk.get("product_id"))
        for chunk in retrieved_chunks
        if chunk.get("product_id")
    }
    return bool(expected & retrieved)


# Kiểm tra câu trả lời có nhắc đúng tên sản phẩm kỳ vọng không.
def answer_product_hit(answer, question):
    """Kiểm tra câu trả lời có nhắc đúng tên sản phẩm kỳ vọng không."""
    expected_names = [
        question.get("source_product", {}).get("ten_san_pham"),
        *(question.get("expected_keywords") or [])[:1],
    ]
    answer_text = normalize(answer)
    return any(normalize(name) in answer_text for name in expected_names if name)


# Đánh giá từng câu trả lời RAG so với ground truth đã sinh.
def evaluate(expected_questions, results):
    """So sánh từng kết quả RAG với ground truth và tạo bảng đánh giá chi tiết."""
    expected_by_id = {item["id"]: item for item in expected_questions}
    rows = []
    for result in results:
        question = expected_by_id.get(result.get("question_id"))
        if not question:
            continue
        answer = result.get("answer", "")
        row = {
            "question_id": result.get("question_id"),
            "type": question.get("type"),
            "question": question.get("question"),
            "expected_product_ids": question.get("expected_product_ids", []),
            "expected_product_name": question.get("source_product", {}).get("ten_san_pham"),
            "retrieval_product_hit": product_hit(
                result.get("retrieved_chunks", []),
                question.get("expected_product_ids", []),
            ),
            "answer_keyword_hit": keyword_hit(
                answer,
                question.get("expected_keywords", []),
            ),
            "answer_product_name_hit": answer_product_hit(answer, question),
            "answer": answer,
        }
        row["passed"] = (
            row["retrieval_product_hit"]
            and row["answer_keyword_hit"]
            and row["answer_product_name_hit"]
        )
        rows.append(row)
    return rows


# Tính các chỉ số tổng hợp như pass_rate và retrieval_hit_rate.
def summarize(rows):
    """Tính các metric tổng quan từ bảng đánh giá chi tiết."""
    total = len(rows)
    passed = sum(1 for row in rows if row["passed"])
    retrieval_hits = sum(1 for row in rows if row["retrieval_product_hit"])
    keyword_hits = sum(1 for row in rows if row["answer_keyword_hit"])
    name_hits = sum(1 for row in rows if row["answer_product_name_hit"])
    return {
        "total": total,
        "passed": passed,
        "pass_rate": round(passed / total, 4) if total else 0,
        "retrieval_hit_rate": round(retrieval_hits / total, 4) if total else 0,
        "answer_keyword_hit_rate": round(keyword_hits / total, 4) if total else 0,
        "answer_product_name_hit_rate": round(name_hits / total, 4) if total else 0,
    }


# Hàm chạy từ terminal: đọc input, đánh giá, in summary và ghi report.
def main():
    parser = argparse.ArgumentParser(description="Evaluate RAG results against generated QA ground truth.")
    parser.add_argument("--questions", required=True)
    parser.add_argument("--results", required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    questions = load_json(args.questions)
    results = load_json(args.results)
    rows = evaluate(questions, results)
    report = {
        "summary": summarize(rows),
        "items": rows,
    }
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    failed = [row for row in rows if not row["passed"]]
    if failed:
        print("\nFailed question ids:")
        print(", ".join(row["question_id"] for row in failed))
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"wrote: {args.output}")


if __name__ == "__main__":
    main()
