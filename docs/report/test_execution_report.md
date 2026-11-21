# Báo cáo Thực thi Kiểm thử (Test Execution Report) — M6

**Phụ trách:** M6 — Lý Trọng Hiếu  
**Tham chiếu:** Tài liệu Design Phần II — Mục II.7 (Kế hoạch kiểm thử và thực nghiệm)

---

## 1. Phạm vi kiểm thử

Hệ thống kiểm thử được tổ chức trong thư mục `tests/` gồm 3 cấp độ theo Mục II.7:
1. **Unit test (`pytest`):**
   - `tests/test_video_io.py`: Tệp hỏng (`InvalidInputError`), tệp không tồn tại, ảnh đơn RGB `uint8`, video ngắn kiểm soát `sample_fps` & `max_frames`.
   - `tests/test_face_detection.py`: 0 khuôn mặt (`No_Face`), 1 khuôn mặt (`224x224x3` RGB `uint8`), nhiều khuôn mặt (chọn bbox diện tích lớn nhất).
   - `tests/test_anti_spoof_deepfake.py`: Hợp đồng đầu ra `SpoofScore` (`Real`, `Printed`, `On-screen`, tổng = `1.0`) và `DeepfakeScore` (`fake_prob ∈ [0.0, 1.0]`).
   - `tests/test_decision.py`: Thứ tự ưu tiên DD6 (Spoof ưu tiên trước Deepfake), ca sát ngưỡng, chế độ GĐ 1 (`df_scores=None`), chuẩn 5 chuỗi nhãn.
   - `tests/test_evaluation.py`: Kiểm chứng công thức `APCER`, `BPCER`, `ACER` và `confusion_matrix` trên dữ liệu giả có đáp án biết trước.
   - `tests/test_app_ui_logic.py`: Kiểm tra toàn bộ `app/components/` (6 thông báo lỗi II.6, validate ảnh/video, vẽ bbox RGB, dòng thời gian, FPS, xóa tệp tạm DD8, bộ đệm trượt DD1).
2. **Integration test:**
   - `tests/test_pipeline_integration.py`: Chạy `run_pipeline` đầu-cuối trên ảnh mẫu và video ngắn ở cả GĐ 1 và GĐ 2.
3. **Kiểm thử giao diện (Manual UI Test):**
   - Thực hiện theo 11 kịch bản trong `tests/ui_test_scenarios.md`.

---

## 2. Nhật ký thực thi kiểm thử tự động (`pytest`)

> **Hướng dẫn chạy lại toàn bộ bộ test sau khi M1–M5 merge code vào `src/`:**
> ```bash
> pytest -v tests/
> ```

| Tệp kiểm thử | Module mục tiêu | Số ca kiểm thử | Kết quả sau tích hợp | Ghi chú |
|---|---|---:|---|---|
| `tests/test_app_ui_logic.py` | `app/components/*` (M6) | 8 | `[TBD - Chạy pytest trên môi trường đầy đủ]` | Bao gồm 6 thông báo lỗi II.6 & DD8 |
| `tests/test_video_io.py` | `src/video_io` (M3) | 4 | `[TBD - Chờ merge M3]` | Tự động skip nếu chưa có `src/video_io` |
| `tests/test_face_detection.py` | `src/face_detection` (M3) | 3 | `[TBD - Chờ merge M3]` | Tự động skip nếu chưa có `src/face_detection` |
| `tests/test_anti_spoof_deepfake.py` | `src/anti_spoof`, `src/deepfake` (M4, M5) | 2 | `[TBD - Chờ merge M4, M5]` | Kiểm tra hợp đồng xác suất |
| `tests/test_decision.py` | `src/decision` (M1) | 6 | `[TBD - Chờ merge M1]` | Kiểm tra thứ tự ưu tiên DD6 |
| `tests/test_evaluation.py` | `src/evaluation` (M2) | 1 | `[TBD - Chờ merge M2]` | Kiểm chứng APCER/BPCER/ACER |
| `tests/test_pipeline_integration.py` | `src/pipeline.py` (M1) | 2 | `[TBD - Chờ merge M1]` | End-to-end ảnh & video |

---

## 3. Nhật ký kiểm thử giao diện thủ công (`tests/ui_test_scenarios.md`)

| Mã TC | Mô tả ngắn | Kết quả (Pass/Fail) | Người kiểm tra | Ngày |
|---|---|---|---|---|
| UI-01 | Upload ảnh `Real` | `[TBD]` | M6 | `[TBD]` |
| UI-02 | Upload video & timeline | `[TBD]` | M6 | `[TBD]` |
| UI-03 | Webcam & bộ đệm trượt | `[TBD]` | M6 | `[TBD]` |
| UI-04 | Thanh trượt `N`, `τ_s`, `τ_d`, GĐ 2 toggle | `[TBD]` | M6 | `[TBD]` |
| UI-05 | Tệp hỏng / sai định dạng | `[TBD]` | M6 | `[TBD]` |
| UI-06 | Video vượt giới hạn dung lượng/độ dài | `[TBD]` | M6 | `[TBD]` |
| UI-07 | Ảnh không có khuôn mặt (`No_Face`) | `[TBD]` | M6 | `[TBD]` |
| UI-08 | Ảnh nhiều khuôn mặt | `[TBD]` | M6 | `[TBD]` |
| UI-09 | Lỗi không truy cập được webcam | `[TBD]` | M6 | `[TBD]` |
| UI-10 | Cảnh báo điểm sát ngưỡng | `[TBD]` | M6 | `[TBD]` |
| UI-11 | Xác minh không lưu file người dùng | `[TBD]` | M6 | `[TBD]` |
