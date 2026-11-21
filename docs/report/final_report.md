# BÁO CÁO TỔNG KẾT ĐỀ TÀI 10 — ANTI-DEEPFAKE (FACE)

**Môn học:** Xử lý ảnh — Bài tập lớn  
**Nhóm thực hiện (6 thành viên):**
- **M1 — Nguyễn Phông:** Integrator (`src/decision/`, `src/pipeline.py`, `configs/`)
- **M2 — Triệu Nguyễn Cát Tường:** Dữ liệu & Đánh giá (`src/evaluation/`, `data/`, `scripts/`, `results/`)
- **M3 — Khoa Phạm:** Lấy frame & Phát hiện mặt (`src/video_io/`, `src/face_detection/`)
- **M4 — Nguyễn Thanh Hảo:** Anti-spoof (`src/anti_spoof/`, `models/`)
- **M5 — Lê Minh Thành, Nguyễn Phông:** Deepfake Detector (`src/deepfake/`, `docs/research/`)
- **M6 — Lý Trọng Hiếu:** UI/UX, Tester & Báo cáo (`app/`, `tests/`, `docs/report/`)

> **Lưu ý nguyên tắc NFR07:** Mọi số liệu đánh giá định lượng trong báo cáo này phải lấy trực tiếp từ kết quả chạy script của M2 (`results/`) và kiểm thử thực tế của M6 (`tests/`). Các ô `[TBD - Số liệu từ M2]` sẽ được điền sau khi hoàn tất thực nghiệm E1–E6 ở Ngày 12.

---

## 1. Đặt vấn đề & Mục tiêu

Hệ thống xác thực khuôn mặt hiện đại thường đối mặt với hai nhóm tấn công chính:
1. **Tấn công trình diễn (Presentation Attack / Spoofing):**
   - **Print attack (`Printed`):** Dùng ảnh in trên giấy đưa trước camera.
   - **Replay attack (`On-screen`):** Hiển thị ảnh hoặc video trên màn hình điện thoại/máy tính rồi quay lại.
2. **Tấn công tổng hợp khuôn mặt (`Deepfake`):** Hoán đổi hoặc tạo khuôn mặt bằng mô hình sinh.

**Mục tiêu đề tài:** Xây dựng ứng dụng Web end-to-end (`AntiDeepFake-Face`) chạy được trên laptop CPU, tích hợp các mô hình pretrained theo lộ trình giai đoạn:
- **Giai đoạn 1 (GĐ 1):** Phân loại `Real`, `Printed`, `On-screen` (+ `No_Face`).
- **Giai đoạn 2 (GĐ 2):** Mở rộng thêm nhánh `Deepfake` (có thể bật/tắt qua cấu hình).

---

## 2. Kiến trúc Hệ thống & Quy ước Interface

### 2.1 Luồng xử lý (Pipeline)
```text
Đầu vào (Ảnh / Video / Webcam)
  └─> [M3] video_io.load_frames() ──────────> List[FramePacket] (RGB uint8)
        └─> [M3] face_detection.detect_and_crop() ──> FaceCropResult (224x224x3 RGB uint8)
              ├──> [M4] anti_spoof.predict_spoof() ────> SpoofScore {Real, Printed, On-screen}
              └──> [M5] deepfake.predict_deepfake() ──> DeepfakeScore {fake_prob} (nếu bật GĐ 2)
                    └─> [M1] decision.aggregate_decision() ──> FinalDecision
                          ├──> [M2] evaluation.evaluate_dataset() ──> results/
                          └──> [M6] app/app.py (Streamlit) ──> Kết quả, Bbox, Timeline, FPS
```

### 2.2 Quy ước đóng băng (Interface Freeze — sau Ngày 3)
- **Hệ màu & kích thước:** Thống nhất 100% `RGB` `np.uint8`. `FramePacket.image_rgb` có shape `(H, W, 3)`; `FaceCropResult.face_rgb` có shape `(224, 224, 3)`. Chuẩn hóa `/255.0` hoặc ImageNet mean/std được đóng gói riêng bên trong wrapper M4, M5.
- **5 hằng số nhãn chuẩn:** `"Real"`, `"Printed"`, `"On-screen"`, `"Deepfake"`, `"No_Face"`.
- **Quy tắc quyết định DD6:**
  1. Không có khuôn mặt → `"No_Face"`.
  2. Điểm spoof vượt `τ_s` → `"Printed"` hoặc `"On-screen"` theo mức 2.
  3. Ngược lại, điểm `fake_prob` vượt `τ_d` → `"Deepfake"`.
  4. Còn lại → `"Real"`. Các ca nằm trong vùng biên quanh ngưỡng `τ` được cảnh báo *"Kết quả chưa chắc chắn"*.

---

## 3. Dữ liệu & Chính sách Bảo mật

- **Cấu trúc chuẩn:** `data/metadata.csv` gồm đúng 7 cột (`sample_id`, `filepath`, `label`, `subject_id`, `split`, `source`, `device`).
- **Chia tập theo `subject_id`:** Tuyệt đối không để trùng người giữa tập `val` (chọn ngưỡng `τ_s`, `τ_d`) và tập `test`.
- **Bảo mật dữ liệu người dùng (DD8):**
  - Không đẩy ảnh/video cá nhân thô lên repository Git.
  - Web App không dùng cơ sở dữ liệu; dữ liệu tải lên chỉ tồn tại trong bộ nhớ phiên (`temporary_memory_file`) và bị xóa ngay sau khi phân tích.

---

## 4. Triển khai Giao diện & Kiểm thử (M6)

- **Web App Streamlit (`app/app.py`, `app/components/`, `app/assets/`):**
  - 3 chế độ đầu vào: Upload ảnh (`.jpg`, `.png`), Upload video ngắn (giới hạn dung lượng & độ dài), Webcam (hỗ trợ cả `st.camera_input` kèm `SlidingFrameBuffer` và OpenCV `VideoCapture`).
  - Điều chỉnh trực tiếp tham số `N ∈ {1..30}` (có nút chọn nhanh `{1, 5, 10, 30}`), `sample_fps`, ngưỡng `τ_s`, `τ_d`, biên độ sát ngưỡng, và công tắc bật/tắt nhánh GĐ 2.
  - Hiển thị khung hình có bounding box + nhãn, thẻ kết luận, thanh xác suất 4 lớp, biểu đồ dòng thời gian nhãn theo frame, FPS/độ trễ và banner giới hạn học thuật luôn hiện diện.
  - Chuẩn hóa đủ 6 thông báo lỗi/cảnh báo theo Mục II.6.
- **Kiểm thử (`tests/`):** Unit test cho 6 module (`video_io`, `face_detection`, `anti_spoof`, `deepfake`, `decision`, `evaluation`, `app/components`), Integration test cho `run_pipeline`, và 11 kịch bản kiểm thử giao diện (`tests/ui_test_scenarios.md`).

---

## 5. Tổng hợp Kết quả Thực nghiệm (E1–E6 — Số liệu từ M2)

### 5.1 E1 & E2 — Đánh giá Giai đoạn 1 (Anti-Spoof)
- **E1 (Real vs Spoof trên dữ liệu tự thu thập, chọn ngưỡng trên tập `val`):**
  - Ngưỡng tối ưu `τ_s`: `[TBD - Số liệu từ M2]`
  - `APCER`: `[TBD]` | `BPCER`: `[TBD]` | `ACER`: `[TBD]`
- **E2 (Phân loại 3 lớp `Real`, `Printed`, `On-screen` trên tập `test`):**
  - Confusion Matrix 3×3: `[Chèn bảng/ảnh từ results/ của M2]`

### 5.2 E3 — Kiểm tra Chéo miền (Cross-Domain: Dataset công khai vs Tự thu thập)
| Nguồn dữ liệu / Thiết bị / Ánh sáng | APCER | BPCER | ACER | Nhận xét độ lệch miền |
|---|---:|---:|---:|---|
| Dataset công khai | `[TBD]` | `[TBD]` | `[TBD]` | `[TBD - Phân tích M2 & M4]` |
| Dữ liệu tự thu thập (Laptop webcam) | `[TBD]` | `[TBD]` | `[TBD]` | `[TBD]` |
| Dữ liệu tự thu thập (Điện thoại / Ánh sáng yếu) | `[TBD]` | `[TBD]` | `[TBD]` | `[TBD]` |

### 5.3 E4 — Ảnh hưởng của Số frame Tổng hợp `N ∈ {1, 5, 10, 30}`
| Số frame `N` | ACER | Độ trễ tổng (s) | Nhận xét đánh đổi tốc độ – ổn định |
|---:|---:|---:|---|
| `N = 1` | `[TBD]` | `[TBD]` | Nhạy với nhiễu chuyển động/chớp mắt đơn frame |
| `N = 5` | `[TBD]` | `[TBD]` | `[TBD]` |
| `N = 10` | `[TBD]` | `[TBD]` | `[TBD]` |
| `N = 30` | `[TBD]` | `[TBD]` | `[TBD]` |

### 5.4 E5 — Đánh giá Giai đoạn 2 (Deepfake & Hệ thống 4 lớp)
- **Real vs Deepfake:** `AUC = [TBD]`, `Accuracy = [TBD]`.
- **Hệ thống đầy đủ 4 lớp:** Confusion Matrix 4×4 `[Chèn từ results/ của M2]`.

### 5.5 E6 — Hiệu năng Hệ thống trên CPU
| Chế độ đầu vào | Cấu hình (`N`, GĐ 1/2) | FPS trung bình | Độ trễ / frame (ms) | Đạt mục tiêu demo (`≥ 5 FPS`) |
|---|---|---:|---:|---|
| Ảnh đơn | `N = 1` | `[TBD]` | `[TBD]` | `[TBD]` |
| Video ngắn | `N = 10`, GĐ 1 | `[TBD]` | `[TBD]` | `[TBD]` |
| Video ngắn | `N = 10`, GĐ 1+2 | `[TBD]` | `[TBD]` | `[TBD]` |
| Webcam | Bộ đệm trượt | `[TBD]` | `[TBD]` | `[TBD]` |

---

## 6. Giới hạn Hệ thống (System Limitations)

1. **Xung đột ưu tiên nhánh khi Deepfake được quay lại qua màn hình (Quy tắc DD6):**
   - Theo thiết kế DD6, nhánh Anti-Spoof luôn được kiểm tra trước nhánh Deepfake. Nếu một video Deepfake được phát trên màn hình điện thoại/máy tính rồi đưa trước webcam, hệ thống sẽ phát hiện dấu hiệu viền màn hình / hiệu ứng moiré và gán nhãn **`On-screen`** thay vì **`Deepfake`**. Đây là hành vi đúng về mặt vật lý của phiên xác thực camera trực tiếp (Presentation Attack tại điểm thu), nhưng cần được nêu rõ để tránh hiểu nhầm khi thử nghiệm demo.
2. **Độ nhạy với độ lệch miền (Domain Shift):**
   - Kết quả mô hình pretrained chịu ảnh hưởng đáng kể bởi cảm biến camera (webcam laptop vs camera điện thoại), điều kiện ánh sáng ngược sáng/thiếu sáng và nén video (compression artifacts).
3. **Phụ thuộc vào bước phát hiện khuôn mặt & tỉ lệ crop (DD4):**
   - Với `Printed` và `On-screen`, mô hình Anti-Spoof cần ngữ cảnh vùng biên quanh mặt (mép giấy, khung màn hình). Nếu camera đặt quá gần làm mất viền xung quanh, khả năng phân biệt `Printed` và `On-screen` suy giảm.
4. **Giới hạn suy luận thời gian thực trên CPU:**
   - Khi bật đồng thời cả GĐ 1 (Anti-Spoof) và GĐ 2 (Deepfake) với `N = 30` trên laptop không có GPU, tổng thời gian xử lý tăng lên đáng kể, đòi hỏi giới hạn `sample_fps = 5` và độ dài video tối đa.

---

## 7. Hướng Phát triển Tương lai — NullSwap & Mở rộng Hệ thống

Bên cạnh hướng phát hiện thụ động (passive detection) đang triển khai, nhóm đề xuất lộ trình phát triển tiếp theo xoay quanh **NullSwap** và nâng cao độ bền vững mô hình:

1. **Nghiên cứu & Tích hợp hướng NullSwap (Phòng vệ chủ động / Chống hoán đổi khuôn mặt):**
   - **Bối cảnh:** Các mô hình hoán đổi khuôn mặt (face-swapping) ngày càng chân thực khiến các bộ phát hiện hậu kiểm (post-hoc deepfake detector) dễ bị suy giảm hiệu năng khi gặp công nghệ sinh mới (unseen generators).
   - **Định hướng NullSwap:** Thay vì chỉ phát hiện sau khi ảnh đã bị giả mạo, hướng tiếp cận **NullSwap** tập trung vào cơ chế bảo vệ danh tính khuôn mặt theo hướng chủ động (proactive defense / anti-swapping perturbation hoặc vô hiệu hóa đặc trưng hoán đổi khuôn mặt trong không gian tiềm ẩn — latent space), khiến mô hình deepfake khi lấy ảnh đầu vào được bảo vệ sẽ tạo ra kết quả bị triệt tiêu đặc trưng nhận dạng hoặc biến dạng dễ phát hiện.
   - **Ứng dụng vào hệ thống:** Bổ sung tab **"Bảo vệ ảnh trước Deepfake (NullSwap)"** trên giao diện Streamlit (M6), cho phép người dùng tải ảnh chân dung để nhúng nhiễu bảo vệ vô hình trước khi chia sẻ lên mạng xã hội, kết hợp song song với pipeline phát hiện hiện có.
2. **Cải thiện phân tách đa nhánh không xung đột (Multi-label / Hierarchical Output):**
   - Thay vì chỉ trả về 1 nhãn đơn theo thứ tự DD6, cho phép hiển thị đồng thời hai chiều thông tin: *Tính chân thực vật lý (Physical: Real / Printed / On-screen)* và *Tính chân thực nội dung (Digital: Authentic / Deepfake)* khi phát hiện tình huống video deepfake phát qua màn hình.
3. **Phân tích nhất quán theo thời gian (Temporal Consistency) & Sinh học (rPPG):**
   - Khai thác đặc trưng chớp mắt, chuyển động vi mô và tín hiệu nhịp tim từ da mặt (rPPG) trên chuỗi `N` frame thay vì chỉ lấy trung bình xác suất từng frame độc lập.
4. **Tối ưu hóa triển khai (ONNX / Quantization INT8):**
   - Chuyển đổi toàn bộ nhánh phát hiện mặt, Anti-Spoof và Deepfake sang ONNX Runtime lượng tử hóa INT8 để đạt ổn định `≥ 15 FPS` trên CPU phổ thông.
