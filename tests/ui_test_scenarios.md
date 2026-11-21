# Kịch bản Kiểm thử Giao diện (UI Test Scenarios) — M6

**Phụ trách:** M6 (Lý Trọng Hiếu)  
**Tham chiếu:** Tài liệu Design Mục II.6 (Giao diện & Thông báo lỗi), Mục II.7 (Kiểm thử giao diện)

---

## 1. Điều kiện tiên quyết
- Chạy ứng dụng từ thư mục gốc repository:
  ```bash
  streamlit run app/app.py
  ```
- Xác nhận banner giới hạn học thuật luôn hiển thị ở đầu trang (*"Lưu ý: Đây là prototype học thuật; kết quả phụ thuộc vào chất lượng camera, điều kiện ánh sáng và dữ liệu huấn luyện..."*).

---

## 2. Bảng kịch bản kiểm thử theo chế độ & tình huống lỗi

| Mã TC | Nhóm kiểm thử | Thao tác người dùng | Kết quả mong đợi | Trạng thái |
|---|---|---|---|---|
| **UI-01** | Upload ảnh hợp lệ (`Real`) | Chọn chế độ `Upload ảnh`, tải ảnh khuôn mặt rõ (`.jpg`/`.png`), nhấn **Phân tích**. | Hiển thị khung hình có bounding box + nhãn, thẻ kết luận `Real`, thanh xác suất 4 lớp, FPS và log 1 frame. | [ ] Chưa chạy |
| **UI-02** | Upload video hợp lệ (`Printed`/`On-screen`/`Deepfake`) | Chọn `Upload video`, tải video `.mp4` ngắn (< 30s, < 50MB), chỉnh `N = 10`, nhấn **Phân tích**. | Thanh tiến trình chạy trong khi xử lý; xuất hiện biểu đồ dòng thời gian nhãn theo frame (`line_chart`) và bảng log chi tiết. | [ ] Chưa chạy |
| **UI-03** | Webcam trình duyệt | Chọn `Webcam` -> `Camera trình duyệt`, cấp quyền camera, chụp 1 hoặc nhiều ảnh vào bộ đệm trượt, nhấn **Phân tích**. | Bộ đệm tăng số lượng frame, phân tích trả nhãn tổng hợp và độ tin cậy, hỗ trợ nút `Xóa bộ đệm webcam`. | [ ] Chưa chạy |
| **UI-04** | Thay đổi tham số `N` và ngưỡng `τ` | Thử `N ∈ {1, 5, 10, 30}`, kéo thanh trượt `τ_s`, `τ_d`, bật/tắt `Bật nhánh Deepfake (GĐ 2)`. | Khi tắt Deepfake: thanh trượt `τ_d` mờ đi và kết quả ghi chú GĐ 2 đã tắt; thay đổi ngưỡng làm thay đổi quyết định/cảnh báo sát ngưỡng. | [ ] Chưa chạy |
| **UI-05** | Lỗi tệp hỏng / sai định dạng | Đổi đuôi `.txt` thành `.jpg` hoặc upload tệp hỏng. | Hiển thị đúng thông báo: *"Tệp không hợp lệ. Vui lòng chọn ảnh JPG, PNG hoặc video hợp lệ."* | [ ] Chưa chạy |
| **UI-06** | Lỗi video quá dài / quá lớn | Upload video > 50MB hoặc thời lượng > 30s. | Hệ thống từ chối trước khi chạy model và hiển thị: *"Video vượt giới hạn cho phép. Vui lòng chọn video ngắn hơn."* | [ ] Chưa chạy |
| **UI-07** | Lỗi không tìm thấy khuôn mặt | Upload ảnh phong cảnh hoặc che toàn bộ mặt. | Thẻ kết quả hiển thị nhãn `No_Face` và thông báo: *"Không tìm thấy khuôn mặt. Hãy thử ảnh hoặc video rõ mặt hơn."* | [ ] Chưa chạy |
| **UI-08** | Cảnh báo nhiều khuôn mặt | Upload ảnh có 2–3 khuôn mặt cùng lúc. | Xuất hiện cảnh báo: *"Phát hiện nhiều khuôn mặt. Hệ thống phân tích khuôn mặt lớn nhất."* và bounding box vẽ quanh mặt lớn nhất. | [ ] Chưa chạy |
| **UI-09** | Lỗi không truy cập được webcam | Chặn quyền camera trên trình duyệt hoặc chọn Device ID không tồn tại ở chế độ OpenCV. | Hiển thị thông báo: *"Không truy cập được webcam. Hãy kiểm tra quyền truy cập hoặc dùng chế độ upload."* | [ ] Chưa chạy |
| **UI-10** | Cảnh báo điểm sát ngưỡng | Thử mẫu có điểm spoof/deepfake nằm trong vùng `[τ - margin, τ + margin]`. | Xuất hiện cảnh báo: *"Kết quả chưa chắc chắn. Hãy thử thêm frame hoặc cải thiện ánh sáng."* | [ ] Chưa chạy |
| **UI-11** | Không lưu dữ liệu người dùng (DD8) | Kiểm tra thư mục tạm hệ thống / thư mục dự án sau khi nhấn **Phân tích**. | Không có tệp ảnh/video người dùng nào bị lưu lại trên đĩa sau khi kết thúc phiên xử lý. | [ ] Chưa chạy |
