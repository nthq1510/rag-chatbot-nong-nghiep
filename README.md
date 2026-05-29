# Hệ thống Chatbot Tư vấn Thuốc và Vật tư Nông nghiệp (RAG + Qdrant + Redis)

> **Đồ án Nghiên cứu & Thử nghiệm nâng cao về Retrieval-Augmented Generation (RAG)**
>
> Ứng dụng mô hình ngôn ngữ lớn (LLM) và cơ sở dữ liệu Vector (Qdrant) kết hợp bộ nhớ lịch sử hội thoại (Redis) để hỗ trợ nông dân tra cứu và tư vấn thông tin về thuốc bảo vệ thực vật, phân bón, thuốc kích thích sinh trưởng dựa trên kho dữ liệu tri thức chuyên biệt.

---

## 📌 Tổng quan Đề tài

Trong sản xuất nông nghiệp, việc chẩn đoán đúng bệnh và khuyến nghị đúng loại thuốc bảo vệ thực vật có ý nghĩa quyết định đến năng suất cây trồng. Người nông dân thường đặt câu hỏi bằng ngôn ngữ tự nhiên (ví dụ: *"sầu riêng bị vàng lá thối rễ sau mưa thì phun gì?"*) hoặc hỏi các câu liên tiếp (*"sản phẩm đó dùng thế nào?"*, *"giá bao nhiêu?"*). 

Dự án này giải quyết bài toán trên bằng cách xây dựng hệ thống **RAG Chatbot** gồm:
1. **Làm giàu và chuẩn hóa dữ liệu nông nghiệp**: Tăng cường 500 sản phẩm với 20 trường thông tin chi tiết (bao gồm cả lý do chuyên gia khuyên dùng, gợi ý phối hợp thuốc).
2. **Truy xuất ngữ nghĩa (Semantic Search)**: Mã hóa tri thức thành vector và lưu trữ trên **Qdrant Vector Database**.
3. **Sinh câu trả lời thông minh**: Sử dụng các mô hình ngôn ngữ lớn cục bộ (LLMs) dòng **Qwen** để trả lời dựa hoàn toàn trên ngữ cảnh được truy xuất (giảm thiểu tối đa hiện tượng LLM "ảo tưởng" - hallucination).
4. **Bộ nhớ hội thoại ngắn hạn**: Tích hợp **Redis Conversation Memory** để hỗ trợ hiểu ngữ cảnh của các câu hỏi nối tiếp.
5. **Thử nghiệm Benchmark**: Đánh giá so sánh hiệu năng của **8 pipeline RAG** được tạo từ sự kết hợp giữa 2 LLMs và 4 mô hình Embedding khác nhau.

---

## 🛠️ Kiến trúc Hệ thống

```
                           ┌──────────────────────────┐
                           │   Người dùng đặt câu hỏi │
                           └─────────────┬────────────┘
                                         │
                                         ▼
                           ┌──────────────────────────┐
                           │    Mã hóa câu hỏi thành  │
                           │      Vector Embedding    │
                           └─────────────┬────────────┘
                                         │
                                         ▼
┌──────────────────┐       ┌──────────────────────────┐
│  Qdrant Vector   ├──────►│  Truy xuất Top-K Chunks  │
│     Database     │       │     liên quan nhất       │
└──────────────────┘       └─────────────┬────────────┘
                                         │
                                         ▼
┌──────────────────┐       ┌──────────────────────────┐
│   Redis Memory   ├──────►│  Ghép Context + History  │
│  (Lưu lịch sử)   │       │     vào prompt RAG       │
└──────────────────┘       └─────────────┬────────────┘
                                         │
                                         ▼
                           ┌──────────────────────────┐
                           │    LLM sinh câu trả lời  │
                           │   (Qwen2.5-3B / Qwen3-4B)│
                           └─────────────┬────────────┘
                                         │
                                         ▼
                           ┌──────────────────────────┐
                           │ Trả lời người dùng & Lưu │
                           │   lịch sử vào Redis DB   │
                           └──────────────────────────┘
```

---

## 📂 Cấu trúc Thư mục Dự án

```
rag-chatbot-nong-nghiep/
├── configs/
│   └── benchmark_models.yaml       # Cấu hình danh sách mô hình Embedding benchmark
├── data/
│   ├── dataset/
│   │   ├── data.csv                # Bộ dữ liệu thô ban đầu (500 sản phẩm)
│   │   └── data_enriched.json      # Dữ liệu sau khi được làm sạch và làm giàu
│   ├── Thuoc_Metadata/             # Dữ liệu phân nhóm theo loại (phân bón, trừ sâu, trừ cỏ...)
│   ├── chunks/                     # Cơ sở tri thức được chia thành 1000 chunks nhỏ
│   ├── qdrant_db/                  # Cơ sở dữ liệu Vector Qdrant chạy local
│   ├── vector_store/               # Các file vector nhúng đã được tính toán trước (.jsonl)
│   └── eval/                       # Bộ câu hỏi kiểm thử benchmark (demo_questions, generated_qa_30)
├── docs/
│   ├── bai_luan_rag_chatbot_nong_nghiep.md    # Bài luận chi tiết báo cáo kết quả nghiên cứu
│   └── bang_ten_file_tieng_viet.md            # Mô tả vai trò từng file script
├── notebooks/                      # Các file Jupyter Notebook phân tích dữ liệu & trực quan hóa
├── pdf/                            # Biểu đồ phân tích hiệu năng, tốc độ phản hồi và độ chính xác
├── results/                        # Kết quả chạy thực nghiệm benchmark của các pipeline dạng JSON
├── scripts/                        # Thư mục mã nguồn xử lý chính (Python)
└── requirements-vector.txt         # Danh sách các thư viện cần cài đặt
```

---

## 📊 Kết quả Thử nghiệm Benchmark (8 Pipelines)

Hệ thống đánh giá chéo giữa **2 LLMs** (`Qwen2.5-3B-Instruct`, `Qwen3-4B-Instruct-2507`) và **4 mô hình Embedding** (`sup-SimCSE Vietnamese PhoBERT`, `Vietnamese Bi-Encoder`, `BGE-M3`, `Multilingual-E5-base`) trên bộ câu hỏi chuẩn hóa 30 câu (`generated_qa_30`).

### 1. Độ chính xác (Pass Rate) & Tỉ lệ tìm trúng (Retrieval Hit Rate)
| Thứ hạng | Pipeline | Đạt / 30 câu | Pass Rate | Retrieval Hit Rate | Product Name Hit |
| :---: | | :---: | :---: | :---: | :---: |
| **1** | **`qwen3_4b__bge_m3`** | **28** | **93.33%** | **93.33%** | **93.33%** |
| **2** | **`qwen25_3b__bge_m3`** | **26** | **86.67%** | **93.33%** | **86.67%** |
| **3** | `qwen3_4b__multilingual_e5_base` | 22 | 73.33% | 73.33% | 83.33% |
| **4** | `qwen25_3b__multilingual_e5_base` | 20 | 66.67% | 73.33% | 73.33% |
| **5** | `qwen3_4b__vietnamese_bi_encoder` | 12 | 40.00% | 40.00% | 76.67% |
| **6** | `qwen25_3b__vietnamese_bi_encoder` | 11 | 36.67% | 40.00% | 70.00% |
| **7** | `qwen25_3b__sup_simcse` | 4 | 13.33% | 13.33% | 73.33% |
| **8** | `qwen3_4b__sup_simcse` | 4 | 13.33% | 13.33% | 76.67% |

### 2. Tốc độ xử lý trung bình (Latency - 18 câu benchmark)
* **Thời gian Truy xuất (Retrieval)**: Rất nhanh, chỉ khoảng **16ms - 32ms**.
* **Thời gian Sinh văn bản (Generation)**: Chiếm 99% tổng thời gian xử lý:
  * Nhóm LLM **Qwen2.5-3B** hoàn thành phản hồi trung bình trong **10 giây - 11 giây**.
  * Nhóm LLM **Qwen3-4B** hoàn thành phản hồi trung bình trong **16 giây - 21 giây**.

> **Khuyến nghị Lựa chọn**:
> * Chọn **`qwen3_4b__bge_m3`** nếu ưu tiên tối đa độ chính xác (93.33%).
> * Chọn **`qwen25_3b__bge_m3`** nếu muốn cân bằng tốt nhất giữa độ chính xác (86.67%) và tốc độ xử lý nhanh hơn 2 lần.

---

## 🚀 Hướng dẫn Cài đặt & Vận hành

### 1. Chuẩn bị môi trường
Yêu cầu Python 3.9+ và cài đặt các thư viện cần thiết:
```bash
pip install -r requirements-vector.txt
```

### 2. Tiền xử lý dữ liệu và Tạo cơ sở tri thức
Chạy lần lượt các script để làm giàu dữ liệu, chia nhóm sản phẩm và cắt thành các chunk văn bản:
```bash
# 1. Làm sạch và làm giàu dữ liệu từ data.csv -> data_enriched.json
python3 scripts/lam_giau_du_lieu.py

# 2. Tạo cơ sở dữ liệu tri thức phân loại sản phẩm
python3 scripts/xay_dung_co_so_tri_thuc.py

# 3. Chia nhỏ tài liệu sản phẩm thành các text chunks
python3 scripts/chia_nho_co_so_tri_thuc.py
```

### 3. Đánh chỉ mục (Indexing) vào Qdrant DB
Chạy lệnh sau để tạo 4 collection tương ứng với 4 mô hình nhúng trong Qdrant DB cục bộ:
```bash
python3 scripts/tao_tat_ca_chi_muc_embedding.py \
  --config configs/benchmark_models.yaml \
  --qdrant-path data/qdrant_db \
  --recreate
```

### 4. Đánh giá chất lượng RAG (Benchmark)
Bạn có thể tiến hành benchmark khả năng truy xuất độc lập hoặc benchmark tích hợp RAG đầy đủ:
```bash
# Đánh giá khả năng truy xuất (Retrieval-only)
python3 scripts/benchmark_truy_xuat.py \
  --config configs/benchmark_models.yaml \
  --questions data/eval/benchmark_questions.json \
  --top-k 5

# Đánh giá RAG tích hợp đầy đủ (Retrieval + Generation)
python3 scripts/benchmark_rag.py \
  --config configs/benchmark_models.yaml \
  --questions data/eval/benchmark_questions.json \
  --top-k 5

# Đánh giá tự động bộ QA 30 câu bằng mã kiểm thử keyword
python3 scripts/danh_gia_ket_qua_qa.py \
  --questions data/eval/generated_qa_30.json \
  --results-dir results/generated_qa_30
```

### 5. Trò chuyện trực tiếp (Chatbot với Redis Memory)
Để kích hoạt chế độ chatbot lưu giữ ngữ cảnh hội thoại bằng Redis:
1. Đảm bảo dịch vụ Redis đã được khởi động trên máy của bạn (mặc định cổng `6379`).
2. Nếu bạn sử dụng Hugging Face local, hãy tải trước LLM. Hoặc nếu sử dụng Ollama, hãy chạy: `ollama pull qwen2.5:3b`.
3. Chạy script chat:
```bash
# Chạy với Generator local Hugging Face
python3 scripts/chat_rag_redis.py \
  --redis-url redis://localhost:6379/0 \
  --session-id phien_chat_trai_nghiem_1 \
  --embedding bge-m3 \
  --generator hf \
  --llm qwen25_3b \
  --device cpu

# Chạy kết nối qua Ollama local (nếu có cài đặt Ollama)
python3 scripts/chat_rag_redis.py \
  --redis-url redis://localhost:6379/0 \
  --session-id phien_chat_trai_nghiem_1 \
  --embedding bge-m3 \
  --generator ollama \
  --ollama-model qwen2.5:3b \
  --ollama-url http://localhost:11434
```
Trong cửa sổ chat, bạn có thể gõ câu hỏi liên tiếp. Ví dụ:
* *Bạn:* Cây sầu riêng sau mưa rễ yếu và vàng lá thì dùng sản phẩm nào?
* *Bạn:* Cách pha thuốc đó phun thế nào? (Hệ thống dùng Redis để xác định "thuốc đó" ở câu trước).
* *Gõ `/clear` để xóa lịch sử chat, `/exit` để thoát.*
