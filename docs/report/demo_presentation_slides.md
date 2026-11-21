# Slide Thuyết trình & Kịch bản Demo — AntiDeepFake-Face

**Đề tài 10 — Bài tập lớn Xử lý ảnh**  
**Người tổng hợp & trình bày phần UI/Demo:** M6 — Lý Trọng Hiếu  

---

## Slide 1: Trang bìa
- **Tiêu đề:** AntiDeepFake-Face: Hệ thống Phát hiện Gian lận Khuôn mặt
- **Phân loại 4+1 lớp:** `Real`, `Printed`, `On-screen`, `Deepfake`, `No_Face`
- **Nhóm thực hiện (6 thành viên):** M1 (Nguyễn Phông), M2 (Triệu Nguyễn Cát Tường), M3 (Khoa Phạm), M4 (Nguyễn Thanh Hảo), M5 (Lê Minh Thành, Nguyễn Phông), M6 (Lý Trọng Hiếu).

---

## Slide 2: Bài toán & Động lực
- Xác thực khuôn mặt bị đe dọa bởi **2 lớp tấn công**:
  1. **Presentation Attack (Spoof):** Ảnh in giấy (`Printed`), phát lại qua màn hình (`On-screen`).
  2. **Digital Manipulation (`Deepfake`):** Khuôn mặt hoán đổi/sinh bằng AI.
- **Thách thức thực tế:** Cần hệ thống tích hợp gọn nhẹ chạy được trên **laptop CPU**, chuẩn hóa dữ liệu/giao diện, không lưu dữ liệu cá nhân người dùng.

---

## Slide 3: Kiến trúc Tổng thể & Hợp đồng Module
- Kiến trúc module hóa theo `src/schemas.py` (đóng băng sau Ngày 3):
  - **M3 (`video_io`, `face_detection`):** Đầu ra chuẩn 100% **RGB `uint8`**, crop `(224, 224, 3)`, nhiều mặt chọn mặt lớn nhất.
  - **M4 (`anti_spoof`):** MiniFASNet pretrained (hai mức: Real/Spoof → Printed/On-screen).
  - **M5 (`deepfake`):** Deepfake detector pretrained (GĐ 2, bật/tắt qua config).
  - **M1 (`decision`, `pipeline`):** Tổng hợp `N` frame + quy tắc ưu tiên **DD6**.
  - **M2 (`evaluation`):** Đo `APCER`, `BPCER`, `ACER`, `AUC`, Confusion Matrix.
  - **M6 (`app`, `tests`, `docs/report`):** Web App Streamlit, bộ kiểm thử & báo cáo.

---

## Slide 4: Quy tắc Quyết định Đa nhánh (DD6 & DD7)
- **Thứ tự ưu tiên DD6:**
  1. Không có khuôn mặt → `No_Face`
  2. Điểm spoof > `τ_s` → `Printed` hoặc `On-screen`
  3. Điểm deepfake > `τ_d` → `Deepfake`
  4. Còn lại → `Real` (nếu điểm sát ngưỡng → bật cảnh báo *"Kết quả chưa chắc chắn"*)
- **Tổng hợp nhiều frame (DD7):** Trung bình xác suất qua `N` frame (`N ∈ {1, 5, 10, 30}`) giúp giảm nhiễu đơn khung hình.

---

## Slide 5: Thiết kế Giao diện Web App (M6)
- **3 chế độ đầu vào:** Upload Ảnh (JPG/PNG), Upload Video ngắn, Webcam (`st.camera_input` + bộ đệm trượt / OpenCV).
- **Tương tác tham số:** Thanh trượt `N` frame, `sample_fps`, ngưỡng `τ_s`, `τ_d`, công tắc GĐ 2.
- **Trực quan hóa kết quả:** Bounding box + nhãn trên khung hình, thẻ kết luận, thanh xác suất 4 lớp, biểu đồ dòng thời gian nhãn theo frame, FPS & độ trễ.
- **An toàn & Minh bạch:** Luôn hiển thị banner giới hạn học thuật; 6 thông báo lỗi chuẩn hóa (Mục II.6); xóa ngay tệp tạm sau phiên xử lý (DD8).

---

## Slide 6: Kết quả Thực nghiệm GĐ 1 — Anti-Spoof (E1, E2, E3)
- **E1 (Real vs Spoof):** `APCER = [TBD]`, `BPCER = [TBD]`, `ACER = [TBD]` tại ngưỡng `τ_s = [TBD]`.
- **E2 (3 lớp Real / Printed / On-screen):** Hiển thị Confusion Matrix `[Chèn ảnh từ results/]`.
- **E3 (Chéo miền):** So sánh kết quả trên dataset công khai vs dữ liệu tự thu thập (theo thiết bị & ánh sáng).

---

## Slide 7: Ảnh hưởng Số frame `N` & Nhánh Deepfake (E4, E5)
- **E4 (`N ∈ {1, 5, 10, 30}`):** Biểu đồ đánh đổi giữa độ ổn định (`ACER`) và thời gian xử lý `[Chèn biểu đồ từ results/]`.
- **E5 (GĐ 2 Deepfake & 4 lớp):** `AUC = [TBD]`, Confusion Matrix 4 lớp `[Chèn biểu đồ từ results/]`.

---

## Slide 8: Hiệu năng Hệ thống & Kiểm thử (E6 & Mục II.7)
- **E6 (CPU Performance):** Bảng FPS và độ trễ (ms/frame) theo từng chế độ đầu vào `[TBD từ M2]`.
- **Kiểm thử (M6):** Unit test `pytest` cho 6 module + Integration test `run_pipeline` + 11 kịch bản kiểm thử UI (file hỏng, video dài, 0 mặt, nhiều mặt, lỗi webcam, sát ngưỡng).

---

## Slide 9: Kịch bản Demo Trực tiếp (Live Demo Script — M6)
1. **Bước 1 (Real):** Đưa khuôn mặt thật qua Webcam/Ảnh → hệ thống trả `Real`, độ tin cậy cao, giải thích thanh xác suất.
2. **Bước 2 (Printed & On-screen):** Đưa ảnh in giấy hoặc quay màn hình điện thoại → hệ thống bắt đúng `Printed` / `On-screen` nhờ ngữ cảnh viền mặt.
3. **Bước 3 (Video Deepfake):** Upload video deepfake ngắn (`N = 10`) → quan sát thanh tiến trình và **dòng thời gian nhãn theo frame**.
4. **Bước 4 (Edge Cases):** Thử ảnh 2 khuôn mặt (cảnh báo chọn mặt lớn nhất) và tệp hỏng (thông báo lỗi chuẩn II.6).

---

## Slide 10: Giới hạn Hệ thống
1. **Ưu tiên DD6:** Video Deepfake chiếu qua màn hình rồi quay lại sẽ bị gán **`On-screen`** (đúng bản chất tấn công vật lý qua màn hình nhưng hạn chế phân tách đồng thời).
2. **Lệch miền (Domain shift):** Nhạy với ánh sáng yếu, ngược sáng, độ phân giải webcam.
3. **Ngữ cảnh crop:** Camera quá sát làm mất viền giấy/màn hình giảm độ chính xác phân biệt `Printed`/`On-screen`.
4. **Hiệu năng CPU:** Chạy đồng thời hai nhánh nặng ở `N = 30` làm giảm FPS.

---

## Slide 11: Hướng Phát triển Tương lai — NullSwap
- **Vấn đề:** Mô hình sinh Deepfake mới liên tục xuất hiện khiến bộ phát hiện thụ động dễ bị vượt qua.
- **Định hướng NullSwap:** Chuyển từ phát hiện thụ động sang **bảo vệ chủ động (Proactive Anti-Swapping)**:
  - Tạo nhiễu bảo vệ vô hình lên ảnh chân dung gốc trước khi công khai, làm mô hình hoán đổi khuôn mặt không trích xuất được đặc trưng danh tính (hoặc biến dạng rõ rệt).
  - Tích hợp thêm tab **"Bảo vệ ảnh (NullSwap)"** trực tiếp trên Streamlit App.
- **Cải tiến khác:** Xuất nhãn song song (Physical + Digital), đặc trưng thời gian/rPPG, tối ưu ONNX INT8.

---

## Slide 12: Tổng kết & Q&A
- Hoàn thành hệ thống tích hợp End-to-End **AntiDeepFake-Face** theo đúng quy ước kiến trúc 6 thành viên.
- Mã nguồn module hóa sạch, dễ thay thế model, kèm bộ kiểm thử tự động & tài liệu thực nghiệm đầy đủ.
- **Cảm ơn Thầy/Cô và các bạn đã lắng nghe — Mời đặt câu hỏi!**
