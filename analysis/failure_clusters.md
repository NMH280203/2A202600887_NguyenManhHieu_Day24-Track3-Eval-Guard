# Failure Cluster Analysis — Phase A

**Sinh viên:** Nguyễn Mạnh Hiếu — 2A202600887

**Ngày:** 2026-06-30
**Evaluation backend:** lexical fallback (môi trường chạy không truy cập được OpenAI; code vẫn hỗ trợ RAGAS bằng `RAGAS_MODE=auto`)

## 1. Aggregate scores theo distribution

| Metric | factual | multi_hop | adversarial |
|---|---:|---:|---:|
| faithfulness | 1.0000 | 1.0000 | 1.0000 |
| answer_relevancy | 0.6219 | 0.4877 | 0.4553 |
| context_precision | 0.3210 | 0.3065 | 0.2519 |
| context_recall | 0.9624 | 0.7986 | 0.6356 |
| **avg_score** | **0.7263** | **0.6482** | **0.5857** |

Điểm trung bình toàn bộ 50 câu là **0.6669**. Adversarial thấp hơn factual 0.1406 điểm, đúng với kỳ vọng của bộ stress test.

## 2. Bottom 10 questions

| Rank | ID | Distribution | Question (rút gọn) | Avg | Worst metric |
|---:|---:|---|---|---:|---|
| 1 | 41 | adversarial | Số ngày phép năm hiện hành | 0.5258 | context_precision |
| 2 | 40 | multi_hop | Thử việc tháng 3 gặp vi phạm bảo mật | 0.5430 | context_precision |
| 3 | 21 | multi_hop | Senior 9 năm: phép và khoảng lương | 0.5493 | context_precision |
| 4 | 18 | factual | Cơ cấu điểm đánh giá hiệu suất | 0.5503 | answer_relevancy |
| 5 | 50 | adversarial | Manager dùng VPN cá nhân khi WFH | 0.5535 | context_precision |
| 6 | 44 | adversarial | Chu kỳ đổi mật khẩu | 0.5660 | context_precision |
| 7 | 48 | adversarial | Bảo hiểm PVI trong thời gian thử việc | 0.5730 | context_precision |
| 8 | 42 | adversarial | Chu kỳ cộng phép theo thâm niên | 0.5824 | context_precision |
| 9 | 49 | adversarial | So sánh phép v2023 và bản hiện hành | 0.5844 | context_precision |
| 10 | 45 | adversarial | Yêu cầu kích hoạt MFA | 0.5873 | context_precision |

## 3. Failure cluster matrix

Mỗi ô là số câu có metric tương ứng thấp nhất.

| Worst metric | factual | multi_hop | adversarial | Total |
|---|---:|---:|---:|---:|
| faithfulness | 0 | 0 | 0 | 0 |
| answer_relevancy | 1 | 2 | 0 | 3 |
| context_precision | 19 | 18 | 10 | 47 |
| context_recall | 0 | 0 | 0 | 0 |

## 4. Dominant failure analysis

Theo số lượng tuyệt đối, **factual** là distribution dominant (20 failure) vì nó có 20 câu, trong khi adversarial chỉ có 10. Theo tỷ lệ, adversarial đáng lo hơn: 10/10 câu có context precision là metric thấp nhất và điểm trung bình chỉ 0.5857. Metric dominant chung là **context_precision** với 47/50 câu.

BM25 offline thường lấy đúng tài liệu nhưng top-3 còn chứa đoạn không trực tiếp trả lời câu hỏi. Vấn đề rõ nhất ở các cặp tài liệu nhiều phiên bản như phép năm v2023/v2024 và mật khẩu v1/v2. Multi-hop còn phải lấy dữ kiện từ nhiều policy nên precision giảm khi context budget cố định.

## 5. Suggested fixes

| Metric yếu | Root cause | Suggested fix |
|---|---|---|
| faithfulness | Câu trả lời có thể thêm claim ngoài context khi dùng LLM | Bắt buộc citation theo chunk, temperature 0 và verify numeric claims |
| context_recall | Multi-hop cần nhiều tài liệu nhưng top-k nhỏ | Query decomposition, tăng candidate pool và hợp nhất theo sub-query |
| context_precision | BM25 trả về version cũ hoặc đoạn cùng từ khóa | Metadata filter `effective_date/status`, dense retrieval và cross-encoder rerank |
| answer_relevancy | Extractive answer chứa câu phụ | Prompt trả lời theo schema và trim câu không khớp intent |

## 6. Adversarial distribution

Adversarial có avg 0.5857, thấp hơn multi-hop 0.6482 và factual 0.7263. Có 7/10 vị trí bottom-10 thuộc nhóm này. Các câu 41, 44 và 49 cho thấy retrieval dựa trên từ khóa chưa đủ để giải quyết version conflict; câu 50 cần hiểu phủ định của chính sách VPN thay vì ưu tiên cụm “tăng bảo mật”. Production pipeline nên gắn metadata phiên bản và loại tài liệu hết hiệu lực trước reranking.
