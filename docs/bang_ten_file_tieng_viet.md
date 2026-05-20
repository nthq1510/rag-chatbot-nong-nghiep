# Bảng tên file tiếng Việt trong thư mục scripts

Các file code chính đã được đặt tên lại theo tiếng Việt không dấu để dễ chạy trên terminal và dễ trình bày trong bài luận. Các file tên tiếng Anh cũ đã được loại bỏ để thư mục `scripts/` gọn hơn.

| Tên cũ | Tên tiếng Việt mới | Vai trò |
|---|---|---|
| `enrich_dataset.py` | `lam_giau_du_lieu.py` | Làm sạch và tăng cường dữ liệu CSV thành JSON enriched |
| `build_knowledge_database.py` | `xay_dung_co_so_tri_thuc.py` | Tách dữ liệu enriched theo loại sản phẩm |
| `chunk_knowledge_database.py` | `chia_nho_co_so_tri_thuc.py` | Chia dữ liệu sản phẩm thành các chunks văn bản |
| `embed_and_index_qdrant.py` | `tao_embedding_va_index_qdrant.py` | Tạo embedding và index một model vào Qdrant |
| `build_all_embedding_indexes.py` | `tao_tat_ca_chi_muc_embedding.py` | Tạo 4 collection embedding để benchmark |
| `search_qdrant.py` | `tim_kiem_qdrant.py` | Tìm kiếm thử trong Qdrant bằng một câu hỏi |
| `run_retrieval_benchmark.py` | `benchmark_truy_xuat.py` | Benchmark retrieval-only cho 4 embedding model |
| `run_rag_benchmark.py` | `benchmark_rag.py` | Benchmark RAG đầy đủ cho 2 LLM x 4 embedding |
| `download_llm_models.py` | `tai_mo_hinh_llm.py` | Tải trước các mô hình Qwen |
| `run_vector_pipeline.py` | `chay_pipeline_vector.py` | Chạy nhanh pipeline vector cơ bản |

## Lệnh chạy theo tên mới

```bash
python3 scripts/lam_giau_du_lieu.py
python3 scripts/xay_dung_co_so_tri_thuc.py
python3 scripts/chia_nho_co_so_tri_thuc.py
python3 scripts/tao_tat_ca_chi_muc_embedding.py --config configs/benchmark_models.yaml --qdrant-path data/qdrant_db --recreate
python3 scripts/benchmark_truy_xuat.py --config configs/benchmark_models.yaml --questions data/eval/benchmark_questions.json --top-k 5
python3 scripts/benchmark_rag.py --config configs/benchmark_models.yaml --questions data/eval/benchmark_questions.json --top-k 5
```
