# Xây dựng hệ thống chatbot tư vấn thuốc nông nghiệp sử dụng RAG, Qdrant và Redis

## Tóm tắt

Trong sản xuất nông nghiệp, việc lựa chọn đúng thuốc bảo vệ thực vật, phân bón hoặc sản phẩm hỗ trợ sinh trưởng có ảnh hưởng trực tiếp đến hiệu quả xử lý sâu bệnh và năng suất cây trồng. Người dùng thường đặt câu hỏi bằng ngôn ngữ tự nhiên, ví dụ: "cây sầu riêng sau mưa bị vàng lá, rễ yếu thì dùng gì?", "sản phẩm đó dùng như thế nào?", hoặc "giá bao nhiêu?". Trong khi đó, dữ liệu sản phẩm thường nằm ở dạng bảng hoặc JSON, gồm nhiều trường như tên sản phẩm, loại, thành phần, công dụng, triệu chứng, nguyên nhân, hướng dẫn sử dụng, giá và lưu ý an toàn.

Đề tài này xây dựng một hệ thống chatbot tư vấn thuốc nông nghiệp dựa trên kiến trúc Retrieval-Augmented Generation (RAG). Hệ thống chuẩn hóa dữ liệu sản phẩm, làm giàu thêm các trường có ý nghĩa chuyên môn, chia dữ liệu thành các đoạn văn bản nhỏ, mã hóa bằng embedding model, lưu vector vào Qdrant và dùng mô hình ngôn ngữ lớn để sinh câu trả lời dựa trên context truy xuất được. Ngoài ra, project bổ sung Redis conversation memory để chatbot có thể ghi nhớ các lượt hội thoại gần nhất theo `session_id`, từ đó hiểu các câu hỏi nối tiếp như "sản phẩm đó dùng thế nào?".

Bộ dữ liệu hiện tại gồm 500 sản phẩm, 20 trường thông tin và 1000 chunk. Project đánh giá 8 pipeline RAG từ 2 mô hình LLM (`Qwen2.5-3B-Instruct`, `Qwen3-4B-Instruct-2507`) kết hợp với 4 embedding model (`sup-SimCSE Vietnamese PhoBERT`, `Vietnamese Bi-Encoder`, `BGE-M3`, `Multilingual-E5-base`). Trên bộ `generated_qa_30`, pipeline `qwen3_4b__bge_m3` đạt pass rate cao nhất 93.33%, trong khi `qwen25_3b__bge_m3` đạt 86.67% và có tốc độ nhanh hơn. Kết quả cho thấy RAG phù hợp với bài toán tư vấn theo dữ liệu riêng, nhưng chất lượng câu trả lời vẫn phụ thuộc mạnh vào chất lượng dữ liệu và độ đúng của bước truy xuất.

## 1. Giới thiệu

### 1.1. Bối cảnh

Trong thực tế, người nông dân hoặc người bán vật tư nông nghiệp thường cần tra cứu nhanh sản phẩm phù hợp với triệu chứng cây trồng. Cùng một vấn đề có thể được mô tả bằng nhiều cách khác nhau, chẳng hạn:

- "lúa bị rầy nâu"
- "lá vàng từng chòm"
- "cháy rầy"
- "rầy bám gốc lúa"
- "sầu riêng sau mưa vàng lá, rễ yếu"

Nếu chỉ tìm kiếm bằng từ khóa chính xác, hệ thống dễ bỏ sót thông tin liên quan. Vì vậy, project sử dụng embedding để tìm kiếm theo ngữ nghĩa. Khi người dùng đặt câu hỏi, câu hỏi được mã hóa thành vector, sau đó hệ thống tìm các chunk gần nhất trong Qdrant và đưa các chunk này vào prompt cho LLM sinh câu trả lời.

### 1.2. Mục tiêu đề tài

Đề tài hướng đến các mục tiêu:

- Xây dựng cơ sở tri thức từ dữ liệu thuốc, phân bón và sản phẩm nông nghiệp.
- Làm giàu dữ liệu bằng các trường hỗ trợ tư vấn như `rule_key`, `dac_tinh_benh`, `giai_doan_phu_hop`, `goi_y_phoi_hop`, `ly_do_chuyen_gia`.
- Chia nhỏ dữ liệu thành chunk phù hợp cho embedding.
- Tạo chỉ mục vector trong Qdrant bằng nhiều embedding model.
- Xây dựng pipeline RAG trả lời câu hỏi bằng tiếng Việt.
- So sánh 8 pipeline gồm 2 LLM và 4 embedding model.
- Bổ sung Redis để lưu lịch sử hội thoại ngắn hạn.
- Tạo notebook trực quan hóa dữ liệu, pipeline và kết quả benchmark.

### 1.3. Phạm vi đề tài

Hệ thống đóng vai trò hỗ trợ tra cứu và tư vấn ban đầu dựa trên dữ liệu có sẵn. Chatbot không thay thế chuyên gia nông nghiệp và không tự đưa ra khuyến cáo ngoài dữ liệu. Nếu context truy xuất không có thông tin, hệ thống cần nói rõ chưa đủ dữ liệu thay vì tự suy diễn.

## 2. Cơ sở lý thuyết

### 2.1. Chatbot hỏi đáp theo dữ liệu riêng

Chatbot hỏi đáp theo dữ liệu riêng là hệ thống trả lời câu hỏi dựa trên một tập dữ liệu cụ thể của người dùng hoặc tổ chức. Trong project này, dữ liệu riêng là bộ sản phẩm nông nghiệp đã được chuẩn hóa và làm giàu. Khác với chatbot tổng quát, hệ thống cần ưu tiên thông tin trong cơ sở tri thức của project.

### 2.2. Embedding

Embedding là kỹ thuật biến đổi văn bản thành vector số. Hai đoạn văn có ý nghĩa gần nhau thường có vector gần nhau trong không gian vector. Nhờ đó, hệ thống có thể tìm các đoạn dữ liệu liên quan đến câu hỏi ngay cả khi người dùng không dùng đúng từ trong dữ liệu.

Ví dụ:

```text
Câu hỏi: Cây sầu riêng sau mưa bị vàng lá, rễ yếu thì dùng sản phẩm nào?
Dữ liệu: Sản phẩm hỗ trợ phòng trị nấm Phytophthora, thối rễ, xì mủ, vàng lá sau mưa.
```

Hai đoạn trên không trùng hoàn toàn từ khóa, nhưng embedding có thể biểu diễn chúng gần nhau về mặt ngữ nghĩa.

### 2.3. Qdrant

Qdrant là vector database dùng để lưu trữ và tìm kiếm vector. Trong project, mỗi embedding model có một collection riêng:

| Embedding alias | Hugging Face model | Collection Qdrant |
|---|---|---|
| `sup-simcse-vietnamese-phobert-base` | `VoVanPhuc/sup-SimCSE-VietNamese-phobert-base` | `thuoc_metadata_sup_simcse` |
| `vietnamese-bi-encoder` | `bkai-foundation-models/vietnamese-bi-encoder` | `thuoc_metadata_vietnamese_bi_encoder` |
| `bge-m3` | `BAAI/bge-m3` | `thuoc_metadata_bge_m3` |
| `multilingual-e5-base` | `intfloat/multilingual-e5-base` | `thuoc_metadata_multilingual_e5_base` |

### 2.4. RAG

RAG là viết tắt của Retrieval-Augmented Generation. Pipeline RAG trong project gồm:

1. Người dùng đặt câu hỏi.
2. Câu hỏi được mã hóa thành vector.
3. Qdrant truy xuất top-k chunk liên quan.
4. Các chunk được ghép vào prompt làm context.
5. LLM sinh câu trả lời dựa trên context.
6. Nếu dùng chế độ chat Redis, lượt hỏi/đáp được lưu vào bộ nhớ hội thoại.

Ưu điểm của RAG là giảm rủi ro LLM bịa thông tin, vì LLM được yêu cầu trả lời dựa trên context truy xuất từ dữ liệu thật.

### 2.5. Redis conversation memory

Redis được dùng để lưu lịch sử hội thoại ngắn hạn. Mỗi phiên chat có một `session_id`, tương ứng với key:

```text
rag_chat:<session_id>:messages
```

Mỗi lượt chat lưu 2 message: `user` và `assistant`. Khi người dùng hỏi câu mới, hệ thống lấy một số lượt gần nhất đưa vào prompt để LLM hiểu ngữ cảnh. Cách này giúp chatbot xử lý các câu hỏi nối tiếp như:

```text
Người dùng: Cây sầu riêng sau mưa bị vàng lá, rễ yếu thì dùng gì?
Chatbot: Gợi ý PHOSPHOROUS ACID 400SL...
Người dùng: Sản phẩm đó dùng như thế nào?
```

Ở câu thứ hai, cụm "sản phẩm đó" được hiểu dựa vào lịch sử hội thoại.

## 3. Dữ liệu

### 3.1. Nguồn dữ liệu

Dữ liệu chính sau xử lý nằm tại:

```text
data/dataset/data_enriched.json
```

Bộ dữ liệu gồm 500 sản phẩm và 20 trường:

- `product_id`
- `ten_san_pham`
- `loai`
- `thanh_phan`
- `quy_cach`
- `cong_dung`
- `trieu_chung`
- `nguyen_nhan`
- `doi_tuong_cay_trong`
- `thoi_diem_xu_ly`
- `huong_dan_su_dung`
- `an_toan_su_dung`
- `gia`
- `url_img`
- `rule_key`
- `dac_tinh_benh`
- `giai_doan_phu_hop`
- `loai_dat_moi_truong`
- `goi_y_phoi_hop`
- `ly_do_chuyen_gia`

### 3.2. Phân bố theo loại sản phẩm

| Loại sản phẩm trong dữ liệu | Tên diễn giải | Số lượng |
|---|---|---:|
| `phan_bon` | Phân bón | 167 |
| `thuoc_tru_benh` | Thuốc trừ bệnh | 117 |
| `thuoc_tru_sau` | Thuốc trừ sâu | 111 |
| `thuoc_tru_co` | Thuốc trừ cỏ | 54 |
| `thuoc_kich_thich_sinh_truong` | Thuốc kích thích sinh trưởng | 51 |
| Tổng cộng |  | 500 |

### 3.3. Các trường làm giàu

Các trường làm giàu giúp dữ liệu không chỉ là thông tin sản phẩm thô, mà có thêm ngữ nghĩa phục vụ tư vấn:

- `rule_key`: khóa nhóm theo loại sản phẩm và triệu chứng/ngữ cảnh chính.
- `dac_tinh_benh`: mô tả đặc tính bệnh, sâu hại, cỏ dại hoặc vấn đề dinh dưỡng.
- `giai_doan_phu_hop`: giai đoạn cây trồng phù hợp để dùng sản phẩm.
- `loai_dat_moi_truong`: điều kiện đất, nước hoặc môi trường liên quan.
- `goi_y_phoi_hop`: gợi ý phối hợp với sản phẩm hoặc nhóm chất khác.
- `ly_do_chuyen_gia`: giải thích bằng ngôn ngữ tự nhiên vì sao sản phẩm phù hợp.

Những trường này được tạo bởi script:

```text
scripts/lam_giau_du_lieu.py
```

## 4. Quy trình xây dựng hệ thống

### 4.1. Tiền xử lý và làm giàu dữ liệu

Pipeline dữ liệu bắt đầu từ file CSV gốc:

```text
data/dataset/data.csv
```

Script `lam_giau_du_lieu.py` thực hiện các bước:

- Làm sạch văn bản.
- Chuẩn hóa một số trường thông tin.
- Phân tích thành phần, giá và hướng dẫn sử dụng.
- Tạo các trường làm giàu như `rule_key`, `goi_y_phoi_hop`, `ly_do_chuyen_gia`.
- Lưu kết quả vào `data/dataset/data_enriched.json`.

### 4.2. Xây dựng cơ sở tri thức

Script:

```text
scripts/xay_dung_co_so_tri_thuc.py
```

Script này đọc `data_enriched.json`, nhóm sản phẩm theo trường `loai` và lưu thành các file JSON trong:

```text
data/Thuoc_Metadata/
```

Các file đầu ra gồm:

- `phan_bon.json`
- `thuoc_kich_thich_sinh_truong.json`
- `thuoc_tru_benh.json`
- `thuoc_tru_co.json`
- `thuoc_tru_sau.json`
- `_summary.json`

### 4.3. Chia nhỏ cơ sở tri thức thành chunk

Script:

```text
scripts/chia_nho_co_so_tri_thuc.py
```

Script này chuyển từng sản phẩm thành đoạn văn bản mô tả gồm tên, loại, thành phần, công dụng, triệu chứng, nguyên nhân, hướng dẫn sử dụng, giá, lưu ý an toàn và lý do chuyên gia. Sau đó văn bản được chia thành chunk.

Kết quả hiện tại:

| Chỉ tiêu | Giá trị |
|---|---:|
| Số sản phẩm | 500 |
| Số chunk | 1000 |
| Trung bình chunk/sản phẩm | 2 |

Các file chunk:

```text
data/chunks/chunks.json
data/chunks/chunks.jsonl
```

### 4.4. Tạo embedding và index Qdrant

Script chính:

```text
scripts/tao_tat_ca_chi_muc_embedding.py
```

Script này đọc cấu hình trong `configs/benchmark_models.yaml`, tải embedding model, mã hóa chunks và tạo các collection trong Qdrant. Mỗi embedding model có một collection riêng để dễ benchmark.

Ngoài ra, project có script:

```text
scripts/tao_embedding_va_index_qdrant.py
```

Script này hỗ trợ tạo embedding/index cho một cấu hình cụ thể.

### 4.5. Truy xuất và sinh câu trả lời

Pipeline RAG nằm trong:

```text
scripts/benchmark_rag.py
```

Script này thực hiện:

- Load câu hỏi đánh giá.
- Mã hóa câu hỏi bằng embedding model tương ứng.
- Truy xuất top-k chunk từ Qdrant.
- Xây dựng prompt RAG.
- Gọi LLM để sinh câu trả lời.
- Lưu kết quả vào thư mục `results/`.

Script chat có bộ nhớ:

```text
scripts/chat_rag_redis.py
```

Script này dùng cùng logic RAG nhưng bổ sung Redis để lưu lịch sử hội thoại.

## 5. Thiết kế thực nghiệm

### 5.1. Các bộ câu hỏi

Project hiện có 3 nhóm câu hỏi:

| File | Mục đích |
|---|---|
| `data/eval/demo_question.json` | Test nhanh 1 câu |
| `data/eval/benchmark_questions.json` | Benchmark RAG/retrieval ban đầu với 18 câu |
| `data/eval/generated_qa_30.json` | Bộ 30 câu hỏi/câu trả lời sinh từ dữ liệu thật |

Bộ `generated_qa_30` được dùng để kiểm tra pipeline trên dữ liệu có nhãn kỳ vọng rõ hơn. Bộ này có 30 câu, chia đều thành 5 loại:

| Loại câu hỏi | Số lượng |
|---|---:|
| `goi_y_san_pham` | 6 |
| `cach_dung` | 6 |
| `gia` | 6 |
| `thanh_phan` | 6 |
| `an_toan` | 6 |

### 5.2. Các pipeline được so sánh

Benchmark gồm 8 pipeline:

| Pipeline | LLM | Embedding |
|---|---|---|
| `qwen25_3b__bge_m3` | Qwen2.5-3B-Instruct | BGE-M3 |
| `qwen25_3b__multilingual_e5_base` | Qwen2.5-3B-Instruct | Multilingual-E5-base |
| `qwen25_3b__sup_simcse` | Qwen2.5-3B-Instruct | Sup-SimCSE Vietnamese PhoBERT |
| `qwen25_3b__vietnamese_bi_encoder` | Qwen2.5-3B-Instruct | Vietnamese Bi-Encoder |
| `qwen3_4b__bge_m3` | Qwen3-4B-Instruct-2507 | BGE-M3 |
| `qwen3_4b__multilingual_e5_base` | Qwen3-4B-Instruct-2507 | Multilingual-E5-base |
| `qwen3_4b__sup_simcse` | Qwen3-4B-Instruct-2507 | Sup-SimCSE Vietnamese PhoBERT |
| `qwen3_4b__vietnamese_bi_encoder` | Qwen3-4B-Instruct-2507 | Vietnamese Bi-Encoder |

### 5.3. Chỉ số đánh giá

Với bộ `generated_qa_30`, project dùng các chỉ số:

- `pass_rate`: tỷ lệ câu đạt theo tiêu chí đánh giá.
- `retrieval_hit_rate`: tỷ lệ truy xuất đúng sản phẩm/kỳ vọng.
- `answer_keyword_hit_rate`: tỷ lệ câu trả lời chứa keyword kỳ vọng.
- `answer_product_name_hit_rate`: tỷ lệ câu trả lời có nhắc đúng tên sản phẩm.

Với benchmark RAG trên 18 câu, project ghi nhận:

- `retrieval_seconds`: thời gian truy xuất.
- `generation_seconds`: thời gian sinh câu trả lời.
- `total_seconds`: tổng thời gian.
- `top1_score`: điểm similarity của chunk top-1.

## 6. Kết quả thực nghiệm

### 6.1. Kết quả `generated_qa_30`

| Pipeline | Tổng câu | Đạt | Pass rate | Retrieval hit | Keyword hit | Product name hit |
|---|---:|---:|---:|---:|---:|---:|
| `qwen3_4b__bge_m3` | 30 | 28 | 93.33% | 93.33% | 96.67% | 93.33% |
| `qwen25_3b__bge_m3` | 30 | 26 | 86.67% | 93.33% | 90.00% | 86.67% |
| `qwen3_4b__multilingual_e5_base` | 30 | 22 | 73.33% | 73.33% | 93.33% | 83.33% |
| `qwen25_3b__multilingual_e5_base` | 30 | 20 | 66.67% | 73.33% | 83.33% | 73.33% |
| `qwen3_4b__vietnamese_bi_encoder` | 30 | 12 | 40.00% | 40.00% | 80.00% | 76.67% |
| `qwen25_3b__vietnamese_bi_encoder` | 30 | 11 | 36.67% | 40.00% | 80.00% | 70.00% |
| `qwen25_3b__sup_simcse` | 30 | 4 | 13.33% | 13.33% | 86.67% | 73.33% |
| `qwen3_4b__sup_simcse` | 30 | 4 | 13.33% | 13.33% | 83.33% | 76.67% |

Kết quả cho thấy tổ hợp BGE-M3 có hiệu quả tốt nhất trên bộ `generated_qa_30`. Pipeline `qwen3_4b__bge_m3` đạt điểm cao nhất, trong khi `qwen25_3b__bge_m3` cũng đạt tốt và có lợi thế tốc độ hơn.

### 6.2. Kết quả tốc độ RAG benchmark

| Pipeline | Số câu | Retrieval TB | Generation TB | Total TB | Top1 score TB |
|---|---:|---:|---:|---:|---:|
| `qwen25_3b__sup_simcse` | 18 | 0.021s | 10.134s | 10.155s | 0.7792 |
| `qwen25_3b__vietnamese_bi_encoder` | 18 | 0.021s | 10.659s | 10.680s | 0.4770 |
| `qwen25_3b__multilingual_e5_base` | 18 | 0.022s | 10.830s | 10.852s | 0.8714 |
| `qwen25_3b__bge_m3` | 18 | 0.028s | 11.376s | 11.404s | 0.7007 |
| `qwen3_4b__vietnamese_bi_encoder` | 18 | 0.027s | 16.375s | 16.403s | 0.4770 |
| `qwen3_4b__sup_simcse` | 18 | 0.023s | 16.389s | 16.411s | 0.7792 |
| `qwen3_4b__multilingual_e5_base` | 18 | 0.030s | 20.761s | 20.790s | 0.8714 |
| `qwen3_4b__bge_m3` | 18 | 0.032s | 21.609s | 21.641s | 0.7007 |

Nhận xét:

- Thời gian retrieval rất nhỏ, chỉ khoảng vài chục mili-giây.
- Thời gian generation chiếm gần như toàn bộ tổng thời gian trả lời.
- Qwen2.5-3B nhanh hơn Qwen3-4B.
- Qwen3-4B có thể cho chất lượng trả lời tốt hơn ở một số pipeline, nhưng đổi lại chậm hơn.

### 6.3. Kết quả retrieval-only

| Embedding model | Số câu | Latency TB | Top1 score TB |
|---|---:|---:|---:|
| `bge_m3` | 18 | 0.025s | 0.7007 |
| `multilingual_e5_base` | 18 | 0.016s | 0.8714 |
| `sup_simcse` | 18 | 0.028s | 0.7792 |
| `vietnamese_bi_encoder` | 18 | 0.016s | 0.4770 |

Điểm similarity giữa các embedding model không nên xem là thước đo tuyệt đối vì mỗi mô hình có phân bố vector khác nhau. Tuy nhiên, bảng này vẫn hữu ích để tham khảo khả năng truy xuất tương đối trong cùng bộ thử nghiệm.

## 7. Redis memory và kiểm tra khả năng ghi nhớ

Script `chat_rag_redis.py` hỗ trợ chatbot ghi nhớ hội thoại bằng Redis. Lịch sử được lưu theo `session_id`, giúp hệ thống phân biệt từng người dùng hoặc từng phiên chat.

Ví dụ luồng kiểm tra:

```text
User: Cây sầu riêng sau mưa bị vàng lá, rễ yếu và chậm phục hồi thì dùng sản phẩm nào?
Assistant: Gợi ý PHOSPHOROUS ACID 400SL...
User: Sản phẩm đó dùng như thế nào?
Assistant: Hiểu "sản phẩm đó" là PHOSPHOROUS ACID 400SL và trả lời cách dùng.
```

Redis giúp hệ thống phản hồi tự nhiên hơn vì câu hỏi sau không bị xử lý độc lập hoàn toàn. Tuy nhiên, Redis chỉ lưu bộ nhớ ngắn hạn, không thay thế cơ sở tri thức Qdrant. Thông tin chuyên môn vẫn phải đến từ dữ liệu truy xuất.

## 8. Notebook trực quan hóa

Project có các notebook phục vụ trình bày và kiểm tra kết quả:

| Notebook | Vai trò |
|---|---|
| `data_pipeline_overview.executed.ipynb` | Trình bày dữ liệu thô, tiền xử lý, làm giàu trường và chunking |
| `visualize_retrieval.executed.ipynb` | Trực quan kết quả retrieval |
| `generated_qa_30_visualization.ipynb` | Trực quan bộ câu hỏi `generated_qa_30` và kết quả đánh giá |
| `demo_chat_memory_redis.executed.ipynb` | Minh họa Redis conversation memory |
| `project_summary_dashboard.executed.ipynb` | Dashboard tổng hợp toàn project, gồm bảng so sánh tốc độ trả lời câu hỏi |
| `demo_rag.ipynb` | Demo RAG một câu hỏi |

Notebook `project_summary_dashboard.executed.ipynb` là notebook tổng hợp quan trọng nhất vì trình bày cả dữ liệu, kết quả 8 pipeline, lỗi còn lại, ví dụ câu trả lời RAG, Redis memory và hướng phát triển.

## 9. Đề xuất lựa chọn pipeline

Nếu ưu tiên chất lượng trên bộ `generated_qa_30`, lựa chọn tốt nhất hiện tại là:

```text
qwen3_4b__bge_m3
```

Pipeline này đạt pass rate 93.33% và retrieval hit rate 93.33%.

Nếu ưu tiên cân bằng giữa chất lượng và tốc độ, lựa chọn phù hợp là:

```text
qwen25_3b__bge_m3
```

Pipeline này đạt pass rate 86.67%, retrieval hit rate 93.33% và có tốc độ nhanh hơn Qwen3-4B.

Nếu chỉ cần demo nhanh hoặc chạy trên môi trường tài nguyên hạn chế, có thể cân nhắc Qwen2.5-3B với embedding nhẹ hơn. Tuy nhiên, kết quả thực nghiệm cho thấy BGE-M3 đang phù hợp nhất với bộ generated QA của project.

## 10. Hạn chế

Project vẫn còn một số hạn chế:

- Bộ `generated_qa_30` mới có 30 câu, chưa đủ lớn để đại diện toàn bộ tình huống nông nghiệp.
- Dữ liệu giá, liều lượng hoặc lưu ý an toàn có thể chưa đầy đủ cho mọi sản phẩm.
- Đánh giá tự động chỉ kiểm tra keyword, tên sản phẩm và truy xuất; chưa thay thế được đánh giá thủ công bởi chuyên gia.
- Hệ thống chưa khai thác ảnh sản phẩm hoặc ảnh triệu chứng cây trồng.
- Khi chạy LLM trên Kaggle hoặc máy local, tốc độ phụ thuộc mạnh vào GPU và dung lượng bộ nhớ.
- Redis memory phù hợp cho hội thoại ngắn hạn, chưa phải bộ nhớ dài hạn có khả năng tổng hợp hồ sơ người dùng.

## 11. Hướng phát triển

Các hướng phát triển tiếp theo:

- Mở rộng bộ câu hỏi đánh giá từ 30 lên 100 hoặc 200 câu.
- Gán nhãn expected product rõ hơn cho nhiều tình huống thực tế.
- Bổ sung đánh giá thủ công bởi người có chuyên môn nông nghiệp.
- Thử reranker để sắp xếp lại chunk sau bước truy xuất.
- Tối ưu prompt để trả lời ngắn gọn hơn, nhất là khi dùng trên giao diện chat.
- Xây dựng giao diện web hoàn chỉnh cho người dùng cuối.
- Triển khai Redis Cloud hoặc dịch vụ tương đương khi demo online.
- Thử nghiệm GPU cloud để cải thiện tốc độ sinh câu trả lời.
- Nghiên cứu thêm hướng dùng ảnh triệu chứng cây trồng nếu có dữ liệu ảnh được gán nhãn.

## 12. Kết luận

Đề tài đã xây dựng được một hệ thống chatbot tư vấn thuốc nông nghiệp dựa trên kiến trúc RAG. Dữ liệu 500 sản phẩm được chuẩn hóa, làm giàu thành 20 trường thông tin, chia thành 1000 chunk và lưu trong Qdrant thông qua nhiều embedding model. Hệ thống có khả năng truy xuất context liên quan và dùng LLM để sinh câu trả lời tiếng Việt dựa trên dữ liệu.

Kết quả thực nghiệm trên `generated_qa_30` cho thấy pipeline `qwen3_4b__bge_m3` đạt chất lượng tốt nhất, trong khi `qwen25_3b__bge_m3` là lựa chọn cân bằng hơn giữa chất lượng và tốc độ. Phần Redis memory giúp chatbot xử lý hội thoại tự nhiên hơn bằng cách ghi nhớ các lượt hỏi đáp gần nhất theo `session_id`.

Nhìn chung, project đã hình thành đầy đủ các thành phần quan trọng của một hệ thống RAG thực tế: dữ liệu, tiền xử lý, làm giàu thông tin, chunking, embedding, vector database, LLM generation, benchmark, notebook trực quan và bộ nhớ hội thoại. Đây là nền tảng tốt để tiếp tục phát triển thành một chatbot hỗ trợ tra cứu và tư vấn nông nghiệp có tính ứng dụng cao.

## Phụ lục: Các lệnh chính

### Làm giàu dữ liệu

```bash
python3 scripts/lam_giau_du_lieu.py
```

### Xây dựng cơ sở tri thức và chia chunk

```bash
python3 scripts/xay_dung_co_so_tri_thuc.py
python3 scripts/chia_nho_co_so_tri_thuc.py
```

### Tạo 4 embedding index trong Qdrant

```bash
python3 scripts/tao_tat_ca_chi_muc_embedding.py \
  --config configs/benchmark_models.yaml \
  --qdrant-path data/qdrant_db \
  --recreate
```

### Chạy retrieval benchmark

```bash
python3 scripts/benchmark_truy_xuat.py \
  --config configs/benchmark_models.yaml \
  --questions data/eval/benchmark_questions.json \
  --top-k 5
```

### Chạy RAG benchmark

```bash
python3 scripts/benchmark_rag.py \
  --config configs/benchmark_models.yaml \
  --questions data/eval/benchmark_questions.json \
  --top-k 5
```

### Đánh giá generated QA 30

```bash
python3 scripts/danh_gia_ket_qua_qa.py \
  --questions data/eval/generated_qa_30.json \
  --results-dir results/generated_qa_30
```

### Chạy chat RAG có Redis memory

```bash
python3 scripts/chat_rag_redis.py \
  --redis-url redis://localhost:6379/0 \
  --session-id demo-memory-1 \
  --embedding bge-m3 \
  --generator hf \
  --llm qwen3_4b \
  --device cuda \
  --llm-device cuda
```
