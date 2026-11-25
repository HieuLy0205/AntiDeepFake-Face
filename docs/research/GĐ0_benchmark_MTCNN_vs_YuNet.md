# BÁO CÁO NGHIÊN CỨU & BENCHMARK GIAI ĐOẠN 0: SO SÁNH MTCNN VÀ YUNET
**Người thực hiện:** Khoa Phạm (Thành viên M3 - Module Video I/O & Face Detection)  
**Ngày bàn giao:** Ngày 3 (GĐ 0)  
**Mục tiêu:** So sánh tốc độ và độ ổn định của MTCNN và YuNet để chốt lựa chọn Face Detector cho toàn bộ pipeline Anti-Deepfake.

---

## 1. Phương pháp Thực nghiệm
- **Môi trường đo đạc:** Chạy trực tiếp trên CPU (máy cá nhân).
- **Kích thước ảnh thử nghiệm:** $640 \times 480$ (độ phân giải chuẩn của webcam và video mẫu).
- **Số lần lặp đo (iterations):** 50 lần sau giai đoạn khởi động (warmup).
- **Thư viện:** 
  - `YuNet`: OpenCV DNN (`cv2.FaceDetectorYN`) với trọng số ONNX `face_detection_yunet_2023mar.onnx`.
  - `MTCNN`: `facenet-pytorch` (3 tầng mạng: P-Net, R-Net, O-Net) chạy trên CPU PyTorch.

---

## 2. Kết quả Đo đạc Định lượng (Thực nghiệm)

| Tiêu chí | MTCNN (facenet-pytorch) | YuNet (OpenCV Zoo) | Đánh giá / Ưu thế |
| :--- | :--- | :--- | :--- |
| **Kích thước trọng số (Size)** | ~2,500 KB (2.5 MB) | **227.1 KB** | YuNet nhẹ hơn **~11 lần** |
| **Thời gian xử lý TB (Latency)** | 25.54 ms / frame | **10.76 ms / frame** | YuNet nhanh hơn **2.4 lần** |
| **Tốc độ khung hình (FPS CPU)** | 39.2 FPS | **92.9 FPS** | YuNet vượt trội |
| **Thư viện phụ thuộc (Dependencies)** | Yêu cầu `torch`, `torchvision` | **Chỉ cần `opencv-python`** | YuNet không cần PyTorch, giảm xung đột môi trường |
| **Hỗ trợ góc nghiêng & landmark** | Tốt khi mặt thẳng, giảm khi quay nghiêng >45° | **Rất tốt** (hỗ trợ 5 landmarks mắt, mũi, miệng ngay cả góc nghiêng) |

---

## 3. Phân tích Chi tiết Kỹ thuật

1. **Hiệu năng và Ngân sách Thời gian (Latency Budget):**
   - Chỉ tiêu của toàn bộ pipeline khi demo trên Streamlit là $\ge 5\text{ FPS}$ (tổng thời gian xử lý toàn hệ thống $\le 200\text{ ms/frame}$).
   - Trong đó, các model phía sau (Anti-spoofing MiniFASNet và Deepfake detector) sẽ tiêu tốn từ $50\text{ ms} - 120\text{ ms}$.
   - Nếu dùng YuNet ($10.76\text{ ms}$), bước phát hiện mặt chỉ chiếm khoảng **5%** ngân sách thời gian, để lại 95% thời gian cho các model phân loại của M4 và M5.
2. **Khả năng đóng gói và triển khai:**
   - YuNet là một file ONNX độc lập 227 KB, nhúng thẳng vào thư mục `models/`.
   - Tránh được rủi ro xung đột phiên bản CUDA/PyTorch giữa các thành viên.

---

## 4. Quyết định & Đề xuất Bàn giao (GĐ 0 Decision)
- **CHỐT:** Chọn **YuNet** làm model Face Detector chính thức cho dự án Anti-Deepfake.
- **Trọng số đã tải:** `models/face_detection_yunet_2023mar.onnx`.
- **Interface xuất ra:** `src/face_detection/detector.py` sẽ đóng gói YuNet theo đúng chữ ký hàm đã cam kết trong `schemas.py`.
