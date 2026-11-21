# Wireframe Giao diện, Thông báo lỗi & Kiểm chứng Webcam (GĐ 0 — M6)

**Phụ trách:** M6 — Lý Trọng Hiếu (UI/UX, Tester & Báo cáo)  
**Tham chiếu:** Tài liệu Design Phần II — Mục II.6, II.12, DD1, DD6, DD8

---

## 1. Wireframe Giao diện Web App (Streamlit)

```text
+======================================================================================+
| [Tiêu đề] AntiDeepFake-Face — Hệ thống Phát hiện Gian lận Khuôn mặt                  |
+--------------------------------------------------------------------------------------+
| [Banner Giới hạn — Luôn hiển thị]                                                    |
| Lưu ý: Prototype học thuật; kết quả phụ thuộc camera, ánh sáng và dữ liệu.           |
| Giới hạn DD6: Video Deepfake quay lại qua màn hình được ưu tiên gán nhãn On-screen.  |
+========================+=============================================================+
| SIDEBAR (Cấu hình)     | KHU VỰC CHÍNH (Đầu vào & Kết quả)                           |
|                        |                                                             |
| [Radio] Chế độ đầu vào |  (1) Chọn nguồn đầu vào theo chế độ:                        |
|  (o) Upload ảnh        |      - Upload ảnh: FileUploader (.jpg, .png) + Preview        |
|  ( ) Upload video      |      - Upload video: FileUploader (.mp4, .avi, .mov <=50MB)   |
|  ( ) Webcam            |      - Webcam: st.camera_input / OpenCV ID 0 + Buffer N     |
|                        |                                                             |
| [Select] Chọn nhanh N  |  (2) Nút hành động: [ Phân tích ] (Primary Button)          |
|  {Tùy chỉnh, 1,5,10,30}|      -> Khi xử lý video: hiển thị Progress Bar %            |
| [Slider] Số frame N    |                                                             |
|  1 [====|======] 30    |  (3) Khu thông báo lỗi / cảnh báo (nếu có theo Mục II.6)    |
| [Slider] sample_fps    |                                                             |
|  1 [===|=======] 15    |  (4) Khu Kết quả (2 cột):                                   |
|                        |      +-------------------------+-------------------------+  |
| [Slider] τ_spoof (τ_s) |      | Khung hình đại diện     | Thẻ Kết luận            |  |
|  0.10 [====|===] 0.95  |      | + Bounding Box (x1..y2) | Nhãn: Real / Printed /  |  |
| [Toggle] Bật GĐ 2 (DF) |      | + Nhãn & Độ tin cậy     | On-screen / Deepfake /  |  |
| [Slider] τ_deepfake    |      |                         | No_Face                 |  |
|  0.10 [====|===] 0.95  |      |                         | Độ tin cậy (%) & Lý do  |  |
| [Slider] Biên độ sát   |      |                         | Thanh xác suất 4 lớp    |  |
|  ngưỡng (0.00 - 0.20)  |      |                         | Số frame | FPS | ms/fr  |  |
|                        |      +-------------------------+-------------------------+  |
|                        |                                                             |
|                        |  (5) Dòng thời gian nhãn theo frame:                        |
|                        |      - Biểu đồ Line Chart xác suất theo từng frame_id       |
|                        |      - Expander: Bảng log chi tiết từng frame               |
+========================+=============================================================+
```

---

## 2. Chuẩn hóa Hệ thống Thông báo lỗi Người dùng (Mục II.6)

| STT | Tình huống | Điều kiện kích hoạt trong code (`app/components/`) | Thông báo chuẩn (Nguyên văn Mục II.6) |
|---|---|---|---|
| 1 | Tệp không mở được hoặc sai định dạng | Sai đuôi tệp, kích thước 0 byte, PIL/OpenCV không giải mã được, hoặc `InvalidInputError` từ `src/video_io`. | `Tệp không hợp lệ. Vui lòng chọn ảnh JPG, PNG hoặc video hợp lệ.` |
| 2 | Video quá dài hoặc quá lớn | Dung lượng `> 50 MB` hoặc thời lượng `> 30 giây`. | `Video vượt giới hạn cho phép. Vui lòng chọn video ngắn hơn.` |
| 3 | Không tìm thấy khuôn mặt | `FinalDecision.final_label == "No_Face"`. | `Không tìm thấy khuôn mặt. Hãy thử ảnh hoặc video rõ mặt hơn.` |
| 4 | Nhiều khuôn mặt | Có `face_count > 1` hoặc `multi_face == True` trong `frame_logs`. | `Phát hiện nhiều khuôn mặt. Hệ thống phân tích khuôn mặt lớn nhất.` |
| 5 | Không mở được webcam | `cv2.VideoCapture` không mở được / không đọc được frame hoặc trình duyệt chặn quyền camera. | `Không truy cập được webcam. Hãy kiểm tra quyền truy cập hoặc dùng chế độ upload.` |
| 6 | Điểm sát ngưỡng | `|spoof_score - τ_s| <= margin` hoặc `|fake_prob - τ_d| <= margin`. | `Kết quả chưa chắc chắn. Hãy thử thêm frame hoặc cải thiện ánh sáng.` |

---

## 3. Yêu cầu Chức năng & Phi chức năng phía Giao diện (I.7, I.8, NFR07)

- **FR-UI-01:** Hỗ trợ đủ 3 phương thức nhập: Ảnh (JPG/PNG), Video ngắn, Webcam.
- **FR-UI-02:** Cho phép điều chỉnh trực tiếp số frame tổng hợp `N` (`1..30`, hỗ trợ preset `{1, 5, 10, 30}` cho thí nghiệm E4), ngưỡng `τ_s`, `τ_d`, và công tắc bật/tắt nhánh Deepfake (GĐ 2).
- **FR-UI-03:** Hiển thị bounding box `(x1, y1, x2, y2)` trên khung hình RGB, thẻ kết luận với 5 nhãn chuẩn (`Real`, `Printed`, `On-screen`, `Deepfake`, `No_Face`), thanh xác suất từng nhánh, dòng thời gian nhãn theo frame và FPS thực tế.
- **NFR-UI-01 (Quyền riêng tư - DD8):** Không sử dụng database; toàn bộ ảnh/video xử lý qua bộ đệm phiên (`temporary_memory_file`) và xóa ngay sau khi `run_pipeline` hoàn tất.
- **NFR-UI-02 (Hiệu năng - DD8):** Dùng `@st.cache_resource` để nạp/cache tài nguyên pipeline một lần duy nhất trong suốt vòng đời server.
- **NFR-UI-03 (Tính trung thực - NFR07):** Mọi con số FPS, độ trễ (ms) và xác suất đều lấy trực tiếp từ đồng hồ đo thời gian thực và `FinalDecision` trả về bởi code.

---

## 4. Báo cáo Kiểm chứng Luồng Webcam trên Streamlit (GĐ 0)

Trong GĐ 0, M6 đánh giá 2 phương án tích hợp webcam cho Streamlit:

1. **`st.camera_input` (Native Streamlit + `SlidingFrameBuffer` trong `session_state`)**:
   - **Ưu điểm:** Hoạt động ổn định qua HTTPS/trình duyệt mà không cần cài thêm thư viện WebRTC/ICE/STUN phức tạp, tương thích tốt khi chạy demo cục bộ hoặc qua reverse proxy/iframe; xử lý lỗi cấp quyền trực tiếp ở trình duyệt.
   - **Hạn chế:** Chụp từng khung hình rời rạc; để tổng hợp nhiều frame (`DD1`), người dùng chụp liên tiếp vào `SlidingFrameBuffer` (giữ tối đa `N` frame gần nhất).
2. **`cv2.VideoCapture(device_id)` (OpenCV cục bộ)**:
   - **Ưu điểm:** Đọc trực tiếp luồng video liên tục từ webcam máy tính với tốc độ cao, kết hợp trực tiếp với `src.video_io.reader.load_frames(0, ...)`.
   - **Hạn chế:** Chỉ dùng được khi Streamlit server chạy cùng máy với người dùng (localhost demo); trên môi trường server từ xa sẽ không thấy thiết bị `/dev/video0`.

**Quyết định triển khai (M6):** Hỗ trợ song song cả 2 chế độ trong `app/components/webcam_handler.py` và `app/app.py` để vừa demo mượt trên laptop cá nhân (OpenCV Device ID 0), vừa dự phòng an toàn qua trình duyệt (`st.camera_input`), kèm hàm `check_webcam_available` phát hiện lỗi và xuất đúng thông báo chuẩn Mục II.6.
