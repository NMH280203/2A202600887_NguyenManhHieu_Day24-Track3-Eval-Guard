# LLM Judge Bias Report — Phase B

**Sinh viên:** Nguyễn Mạnh Hiếu — 2A202600887

**Ngày:** 2026-06-30
**Judge cấu hình:** gpt-4o-mini; lần chạy report dùng deterministic fallback do không có network

## 1. Pairwise judge results

Answer A là model answer trong `human_labels_10q.json`; Answer B là ground truth tương ứng.

| ID | Nội dung | Final winner | Nhận định |
|---:|---|---|---|
| 1 | Nghỉ kết hôn | tie | Dữ kiện chính tương đương |
| 5 | Duyệt mua thiết bị 55 triệu | B | B nêu đúng cấp CEO và ngưỡng |
| 12 | Thưởng Tết tối thiểu | tie | A là tập dữ kiện cốt lõi của B |
| 21 | Senior 9 năm | tie | Cùng kết quả phép và lương |
| 23 | Hoàn đào tạo sau 8 tháng | B | Fallback thiên về lời giải chi tiết hơn |
| 29 | Tạm ứng 8 triệu quá hạn | B | B đủ người duyệt và số tiền phạt |
| 33 | Manager 12 năm | tie | Cùng hai kết quả cần hỏi |
| 41 | Phép năm v2023/v2024 | B | B ưu tiên policy hiện hành |
| 46 | Phép trong thử việc | tie | Cùng kết luận phủ định và phương án thay thế |
| 50 | VPN cá nhân khi WFH | B | B nêu đúng lệnh cấm và VPN bắt buộc |

## 2. Swap-and-average

| ID | Pass 1 | Pass 2 (đã map về thứ tự gốc) | Final | Consistent |
|---:|---|---|---|---|
| 1 | tie | tie | tie | Yes |
| 5 | B | B | B | Yes |
| 12 | tie | tie | tie | Yes |
| 21 | tie | tie | tie | Yes |
| 23 | B | B | B | Yes |
| 29 | B | B | B | Yes |
| 33 | tie | tie | tie | Yes |
| 41 | B | B | B | Yes |
| 46 | tie | tie | tie | Yes |
| 50 | B | B | B | Yes |

**Position bias rate:** 0/10 = **0%**. Swap-and-average không phát hiện đảo winner trong lần chạy này.

## 3. Cohen's κ

| Question ID | Human | Judge | Agree |
|---:|---:|---:|---|
| 1 | 1 | 1 | Yes |
| 5 | 0 | 0 | Yes |
| 12 | 1 | 1 | Yes |
| 21 | 1 | 1 | Yes |
| 23 | 1 | 0 | No |
| 29 | 0 | 0 | Yes |
| 33 | 1 | 1 | Yes |
| 41 | 0 | 0 | Yes |
| 46 | 1 | 1 | Yes |
| 50 | 0 | 0 | Yes |

**Cohen's κ = 0.8000 — substantial agreement.** Sai khác duy nhất là câu 23: human chấp nhận câu trả lời ngắn vì đã đủ số tiền, còn fallback judge ưu tiên ground truth giải thích đủ tỷ lệ hoàn trả.

## 4. Verbosity bias

Trong 5 case có winner rõ ràng, winner đều là câu dài hơn: A thắng và dài hơn 0/5; B thắng và dài hơn 5/5. **Verbosity bias = 100%**. Chỉ số này không chứng minh độ dài gây ra winner vì các câu B cũng chính xác/đầy đủ hơn, nhưng cho thấy cần kiểm soát length bias bằng rubric cụ thể và swap order.

## 5. Kết luận

κ đạt 0.8 và position bias bằng 0, nhưng tập 10 câu quá nhỏ để xem judge là ground truth. Verbosity bias 100% là rủi ro rõ ràng: judge có thể phạt câu đúng nhưng ngắn như ID 23. Trong production nên dùng judge làm quality signal theo batch, giữ human calibration định kỳ, log cả hai pass và đưa case bất nhất hoặc sát ngưỡng vào hàng review thủ công.
