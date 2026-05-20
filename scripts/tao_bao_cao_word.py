"""Tao bao cao Word (.docx) tom tat project RAG chatbot nong nghiep.

Script nay chi dung thu vien chuan Python de tranh phu thuoc vao python-docx.
Ket qua duoc luu tai docs/bao_cao_project_rag_chatbot_nong_nghiep.docx.
"""

from __future__ import annotations

import json
import zipfile
from collections import Counter
from html import escape
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = ROOT / "docs" / "bao_cao_project_rag_chatbot_nong_nghiep.docx"


# Doc du lieu JSON neu file ton tai, neu khong thi tra ve gia tri mac dinh.
def load_json(path: Path, default):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


# Dinh dang so thap phan theo kieu ngan gon de dua vao bang bao cao.
def fmt_number(value, digits: int = 3) -> str:
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


# Dinh dang ti le 0..1 thanh phan tram.
def fmt_percent(value) -> str:
    if isinstance(value, (int, float)):
        return f"{value * 100:.2f}%"
    return str(value)


# Thu thap thong ke tong quan ve du lieu, chunks, cau hoi va ket qua benchmark.
def collect_project_stats() -> dict:
    products = load_json(ROOT / "data" / "dataset" / "data_enriched.json", [])
    chunks = load_json(ROOT / "data" / "chunks" / "chunks.json", [])
    generated_qa = load_json(ROOT / "data" / "eval" / "generated_qa_30.json", [])

    eval_rows = []
    for path in sorted((ROOT / "results" / "generated_qa_30").glob("eval_*.json")):
        data = load_json(path, {})
        summary = data.get("summary", data) if isinstance(data, dict) else {}
        eval_rows.append(
            {
                "pipeline": path.stem.removeprefix("eval_"),
                "total": summary.get("total", ""),
                "passed": summary.get("passed", ""),
                "pass_rate": summary.get("pass_rate", ""),
                "retrieval_hit_rate": summary.get("retrieval_hit_rate", ""),
                "answer_keyword_hit_rate": summary.get("answer_keyword_hit_rate", ""),
                "answer_product_name_hit_rate": summary.get("answer_product_name_hit_rate", ""),
            }
        )

    rag_rows = []
    for path in sorted((ROOT / "results" / "rag_benchmark").glob("*.json")):
        items = load_json(path, [])
        retrieval_times = []
        generation_times = []
        total_times = []
        top1_scores = []
        for item in items:
            latency = item.get("latency", {}) if isinstance(item, dict) else {}
            if isinstance(latency.get("retrieval_seconds"), (int, float)):
                retrieval_times.append(latency["retrieval_seconds"])
            if isinstance(latency.get("generation_seconds"), (int, float)):
                generation_times.append(latency["generation_seconds"])
            if isinstance(latency.get("total_seconds"), (int, float)):
                total_times.append(latency["total_seconds"])

            chunks_found = item.get("retrieved_chunks", []) if isinstance(item, dict) else []
            if chunks_found and isinstance(chunks_found[0], dict):
                score = chunks_found[0].get("score")
                if isinstance(score, (int, float)):
                    top1_scores.append(score)

        def avg(values):
            return sum(values) / len(values) if values else ""

        rag_rows.append(
            {
                "pipeline": path.stem,
                "questions": len(items) if isinstance(items, list) else "",
                "avg_retrieval": avg(retrieval_times),
                "avg_generation": avg(generation_times),
                "avg_total": avg(total_times),
                "avg_top1_score": avg(top1_scores),
            }
        )

    return {
        "products": products,
        "product_type_counts": Counter(item.get("loai", "Khac") for item in products),
        "field_count": len(products[0]) if products else 0,
        "chunks": chunks,
        "generated_qa": generated_qa,
        "qa_type_counts": Counter(
            item.get("loai_cau_hoi") or item.get("type") or "Khac" for item in generated_qa
        ),
        "eval_rows": eval_rows,
        "rag_rows": rag_rows,
    }


# Tao XML cho mot doan van trong file Word.
def paragraph(text: str = "", style: str | None = None) -> str:
    style_xml = f'<w:pStyle w:val="{style}"/>' if style else ""
    return (
        "<w:p>"
        f"<w:pPr>{style_xml}</w:pPr>"
        "<w:r>"
        f"<w:t xml:space=\"preserve\">{escape(text)}</w:t>"
        "</w:r>"
        "</w:p>"
    )


# Tao XML cho mot bang Word don gian.
def table(headers: list[str], rows: list[list[str]]) -> str:
    def cell(value: str) -> str:
        return (
            "<w:tc>"
            "<w:tcPr><w:tcW w:w=\"2400\" w:type=\"dxa\"/></w:tcPr>"
            f"{paragraph(value)}"
            "</w:tc>"
        )

    header_row = "<w:tr>" + "".join(cell(header) for header in headers) + "</w:tr>"
    body_rows = ["<w:tr>" + "".join(cell(str(value)) for value in row) + "</w:tr>" for row in rows]
    return "<w:tbl><w:tblPr><w:tblW w:w=\"0\" w:type=\"auto\"/></w:tblPr>" + header_row + "".join(body_rows) + "</w:tbl>"


# Tao noi dung bao cao theo cau truc de Word hien thi ro rang.
def build_report_body(stats: dict) -> str:
    product_rows = [
        [label, str(count)]
        for label, count in sorted(stats["product_type_counts"].items(), key=lambda item: item[0])
    ]
    qa_rows = [
        [label, str(count)]
        for label, count in sorted(stats["qa_type_counts"].items(), key=lambda item: item[0])
    ]
    eval_rows = [
        [
            row["pipeline"],
            str(row["total"]),
            str(row["passed"]),
            fmt_percent(row["pass_rate"]),
            fmt_percent(row["retrieval_hit_rate"]),
            fmt_percent(row["answer_keyword_hit_rate"]),
            fmt_percent(row["answer_product_name_hit_rate"]),
        ]
        for row in stats["eval_rows"]
    ]
    rag_rows = [
        [
            row["pipeline"],
            str(row["questions"]),
            fmt_number(row["avg_retrieval"]),
            fmt_number(row["avg_generation"]),
            fmt_number(row["avg_total"]),
            fmt_number(row["avg_top1_score"], 4),
        ]
        for row in stats["rag_rows"]
    ]

    parts = [
        paragraph("BÁO CÁO PROJECT", "Title"),
        paragraph("Xây dựng chatbot tư vấn thuốc nông nghiệp sử dụng RAG, Qdrant và Redis", "Subtitle"),
        paragraph(""),
        paragraph("1. Tóm Tắt", "Heading1"),
        paragraph(
            "Project xây dựng hệ thống chatbot hỏi đáp tiếng Việt cho lĩnh vực thuốc, phân bón và sản phẩm nông nghiệp. "
            "Hệ thống sử dụng kiến trúc Retrieval-Augmented Generation (RAG): dữ liệu sản phẩm được chuẩn hóa, làm giàu, "
            "chia nhỏ thành các chunk, mã hóa bằng embedding model, lưu vào Qdrant và được truy xuất để cung cấp ngữ cảnh cho LLM."
        ),
        paragraph(
            "Ngoài pipeline RAG cơ bản, project còn bổ sung Redis conversation memory để chatbot nhớ các lượt hội thoại gần nhất, "
            "giúp xử lý các câu hỏi nối tiếp như 'sản phẩm đó dùng thế nào' hoặc 'giá bao nhiêu'."
        ),
        paragraph("2. Mục Tiêu", "Heading1"),
        paragraph("- Xây dựng cơ sở tri thức từ dữ liệu sản phẩm nông nghiệp."),
        paragraph("- Tạo pipeline truy xuất ngữ nghĩa bằng embedding và Qdrant."),
        paragraph("- Tích hợp LLM để sinh câu trả lời tự nhiên, có căn cứ từ context."),
        paragraph("- Đánh giá nhiều tổ hợp embedding model và LLM."),
        paragraph("- Bổ sung Redis để lưu lịch sử hội thoại theo session_id."),
        paragraph("3. Dữ Liệu", "Heading1"),
        paragraph(
            f"Bộ dữ liệu sau xử lý có {len(stats['products'])} sản phẩm, {stats['field_count']} trường thông tin và "
            f"được chia thành {len(stats['chunks'])} chunk phục vụ embedding."
        ),
        table(["Loại sản phẩm", "Số lượng"], product_rows),
        paragraph(
            "Các trường được làm giàu gồm rule_key, đặc tính bệnh, giai đoạn phù hợp, loại đất/môi trường, gợi ý phối hợp "
            "và lý do chuyên gia. Những trường này giúp dữ liệu giàu ngữ nghĩa hơn, phù hợp cho tìm kiếm và giải thích."
        ),
        paragraph("4. Tiền Xử Lý Và Làm Giàu Dữ Liệu", "Heading1"),
        paragraph(
            "Dữ liệu gốc được đọc từ data/dataset/data.csv, sau đó script lam_giau_du_lieu.py chuẩn hóa nội dung, tách thành phần, "
            "phân tích giá, hướng dẫn sử dụng và bổ sung các trường chuyên gia. Kết quả chính được lưu ở data/dataset/data_enriched.json."
        ),
        paragraph(
            "Tiếp theo, xay_dung_co_so_tri_thuc.py tách dữ liệu theo nhóm sản phẩm vào data/Thuoc_Metadata. "
            "chia_nho_co_so_tri_thuc.py chuyển từng sản phẩm thành văn bản và chia thành chunk có overlap để tránh mất ngữ cảnh."
        ),
        paragraph("5. Kiến Trúc Hệ Thống", "Heading1"),
        paragraph(
            "Luồng xử lý gồm: câu hỏi người dùng -> mã hóa câu hỏi bằng embedding -> truy xuất top-k chunk trong Qdrant -> "
            "ghép context vào prompt -> LLM sinh câu trả lời -> lưu cặp user/assistant vào Redis nếu chạy chế độ chat có bộ nhớ."
        ),
        paragraph(
            "Qdrant được dùng làm vector database. Mỗi embedding model có một collection riêng để benchmark công bằng. "
            "Redis lưu lịch sử theo key rag_chat:<session_id>:messages và có TTL để tự xóa hội thoại cũ."
        ),
        paragraph("6. Mô Hình Và Pipeline Đánh Giá", "Heading1"),
        paragraph(
            "Project thử nghiệm 4 embedding model: sup-SimCSE Vietnamese PhoBERT, Vietnamese Bi-Encoder, BGE-M3 và Multilingual-E5-base. "
            "Hai LLM được dùng là Qwen2.5-3B-Instruct và Qwen3-4B-Instruct-2507, tạo thành 8 pipeline RAG."
        ),
        paragraph("7. Bộ Câu Hỏi Đánh Giá", "Heading1"),
        paragraph(
            f"Bộ generated_qa_30 gồm {len(stats['generated_qa'])} câu hỏi/câu trả lời được tạo dựa trên dữ liệu thật, chia đều theo 5 loại:"
        ),
        table(["Loại câu hỏi", "Số lượng"], qa_rows),
        paragraph("8. Kết Quả Đánh Giá Generated QA 30", "Heading1"),
        table(
            [
                "Pipeline",
                "Tổng",
                "Đạt",
                "Pass rate",
                "Retrieval hit",
                "Keyword hit",
                "Product hit",
            ],
            eval_rows,
        ),
        paragraph(
            "Kết quả cho thấy pipeline qwen3_4b__bge_m3 đạt pass rate cao nhất trong bộ generated_qa_30. "
            "BGE-M3 cũng có retrieval hit rate tốt, cho thấy phù hợp với dữ liệu đã làm giàu của project."
        ),
        paragraph("9. Kết Quả Benchmark RAG", "Heading1"),
        table(
            [
                "Pipeline",
                "Số câu",
                "Retrieval TB (s)",
                "Generation TB (s)",
                "Total TB (s)",
                "Top1 score TB",
            ],
            rag_rows,
        ),
        paragraph(
            "Thời gian truy xuất rất nhỏ so với thời gian sinh câu trả lời. Vì vậy khi tối ưu vận hành, phần cần quan tâm nhất là LLM, "
            "bao gồm kích thước mô hình, thiết bị chạy và giới hạn max_new_tokens."
        ),
        paragraph("10. Bộ Nhớ Hội Thoại Bằng Redis", "Heading1"),
        paragraph(
            "chat_rag_redis.py cho phép chatbot lưu các lượt trò chuyện gần đây vào Redis. Khi người dùng hỏi câu nối tiếp, "
            "hệ thống đưa lịch sử hội thoại gần nhất vào prompt để hiểu ngữ cảnh. Ví dụ sau khi đã tư vấn PHOSPHOROUS ACID 400SL, "
            "người dùng có thể hỏi 'sản phẩm đó dùng thế nào' và chatbot hiểu 'sản phẩm đó' là sản phẩm vừa được nhắc."
        ),
        paragraph("11. Các Notebook Minh Họa", "Heading1"),
        paragraph("- data_pipeline_overview.executed.ipynb: trình bày dữ liệu, tiền xử lý và làm giàu trường."),
        paragraph("- visualize_retrieval.executed.ipynb: trực quan kết quả truy xuất."),
        paragraph("- demo_chat_memory_redis.executed.ipynb: minh họa chatbot có bộ nhớ hội thoại."),
        paragraph("- generated_qa_30_visualization.ipynb: trực quan bộ câu hỏi đánh giá và kết quả pipeline."),
        paragraph("- project_summary_dashboard.executed.ipynb: dashboard tổng hợp toàn project."),
        paragraph("12. Hạn Chế", "Heading1"),
        paragraph(
            "Chất lượng câu trả lời phụ thuộc vào độ đầy đủ của dữ liệu nguồn. Nếu context không có giá hoặc liều lượng của sản phẩm, "
            "chatbot cần nói rõ chưa đủ dữ liệu thay vì suy diễn. Ngoài ra, benchmark tự động chỉ là chỉ báo ban đầu; "
            "các khuyến cáo nông nghiệp quan trọng vẫn cần kiểm tra chuyên gia."
        ),
        paragraph("13. Hướng Phát Triển", "Heading1"),
        paragraph("- Bổ sung giao diện web/chat UI để người dùng cuối dễ sử dụng."),
        paragraph("- Kết hợp Redis Cloud hoặc dịch vụ tương đương khi triển khai thật."),
        paragraph("- Bổ sung đánh giá thủ công bởi chuyên gia nông nghiệp."),
        paragraph("- Tối ưu prompt và cấu trúc chunk để tăng độ chính xác khi hỏi giá, liều lượng và lưu ý an toàn."),
        paragraph("- Triển khai trên GPU cloud để giảm thời gian sinh câu trả lời."),
        paragraph("14. Kết Luận", "Heading1"),
        paragraph(
            "Project đã hoàn thiện một pipeline RAG có dữ liệu riêng, có cơ sở tri thức, vector database, benchmark nhiều mô hình, "
            "bộ câu hỏi đánh giá và cơ chế nhớ hội thoại bằng Redis. Đây là nền tảng tốt để phát triển thành chatbot tư vấn nông nghiệp "
            "có khả năng trả lời tự nhiên nhưng vẫn bám sát dữ liệu."
        ),
    ]
    return "".join(parts)


# Tao cac file XML toi thieu can co trong mot file .docx hop le.
def build_docx_xml(body: str) -> dict[str, str]:
    document_xml = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:body>
    {body}
    <w:sectPr>
      <w:pgSz w:w="11906" w:h="16838"/>
      <w:pgMar w:top="1440" w:right="1440" w:bottom="1440" w:left="1440"/>
    </w:sectPr>
  </w:body>
</w:document>"""

    styles_xml = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:style w:type="paragraph" w:styleId="Title">
    <w:name w:val="Title"/>
    <w:rPr><w:b/><w:sz w:val="36"/></w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Subtitle">
    <w:name w:val="Subtitle"/>
    <w:rPr><w:i/><w:sz w:val="24"/></w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Heading1">
    <w:name w:val="heading 1"/>
    <w:rPr><w:b/><w:sz w:val="28"/></w:rPr>
  </w:style>
</w:styles>"""

    content_types = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
  <Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>
</Types>"""

    rels = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
</Relationships>"""

    doc_rels = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
</Relationships>"""

    return {
        "[Content_Types].xml": content_types,
        "_rels/.rels": rels,
        "word/document.xml": document_xml,
        "word/styles.xml": styles_xml,
        "word/_rels/document.xml.rels": doc_rels,
    }


# Ghi file .docx bang zip package dung chuan OpenXML co ban.
def write_docx(path: Path, files: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as docx:
        for name, content in files.items():
            docx.writestr(name, content)


# Chay tao bao cao Word.
def main() -> None:
    stats = collect_project_stats()
    body = build_report_body(stats)
    files = build_docx_xml(body)
    write_docx(OUTPUT_PATH, files)
    print(f"Da tao bao cao Word: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
