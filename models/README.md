# Model Anti-Spoofing – Level 1

## Thông tin model

- Tên model: MiniFASNetV2.
- Nguồn: Silent-Face-Anti-Spoofing.
- Tác giả: Minivision AI.
- Repository: https://github.com/minivision-ai/Silent-Face-Anti-Spoofing
- Giấy phép mã nguồn: Apache License 2.0.
- Điều khoản sử dụng riêng của trọng số: chưa được xác minh.
- File giấy phép: models/anti_spoof/LICENSE.

## Trọng số sử dụng

- File: 2.7_80x80_MiniFASNetV2.pth.
- Đường dẫn: models/anti_spoof/.
- Loại: Pretrained model.
- Phương pháp: Integrated, không huấn luyện lại.
- Giữ nguyên file trọng số gốc.

## Cấu hình sử dụng

- Framework: PyTorch.
- Thiết bị: CPU hoặc CUDA GPU.
- Đầu vào module: RGB, uint8, 224 × 224 × 3.
- Tỉ lệ crop khuyến nghị: 2.7.
- Kích thước đầu vào model: 80 × 80.
- Giá trị pixel: 0–255, chuyển sang float32.
- Hàm sử dụng: predict_level1(crop_rgb).
- Đầu ra: (p_real, p_spoof).
- Tổng xác suất: 1.0 (trong giới hạn sai số dấu phẩy động).
- Model được nạp một lần bằng cache.

## Tương thích hệ màu RGB

Model gốc sử dụng ảnh BGR. Để tuân thủ quy ước RGB của nhóm, thứ tự kênh trọng số tại lớp convolution đầu tiên được hoán đổi khi nạp model vào bộ nhớ.

Phương pháp này không thay đổi kiến trúc model, không huấn luyện lại, không chỉnh sửa file weights gốc và không chuyển ảnh đầu vào từ RGB sang BGR.

## Mục đích

Phân biệt khuôn mặt thật (Real) và khuôn mặt giả mạo (Spoof) tại Level 1 của module Anti-Spoofing.

## Kết quả kiểm thử

Kiểm thử trên CPU với ba ảnh mẫu từ Silent-Face-Anti-Spoofing.

| Ảnh | Loại ảnh | Real | Spoof | Thời gian | Kết quả |
|---|---|---:|---:|---:|---|
| image_T1.jpg | Real | 99.16% | 0.84% | 109.70 ms | PASS |
| image_F1.jpg | Printed | 0.16% | 99.84% | 14.75 ms | PASS |
| image_F2.jpg | On-screen | 0.05% | 99.95% | 19.18 ms | PASS |

- Số ảnh dự đoán đúng: 3/3.
- Không phát sinh lỗi runtime.
- Tổng xác suất của mỗi ảnh: 1.0.
- Thời gian trung bình sau lần nạp model đầu tiên: 16.97 ms/ảnh.

## Giới hạn

- Kết quả trên ba ảnh mẫu chỉ phục vụ kiểm thử ban đầu, không đại diện cho độ chính xác tổng thể.
- Chưa kiểm thử thực tế trên GPU.
- Chưa kiểm thử với ảnh tự chuẩn bị và đầu ra thực tế của M3.
- Chất lượng dự đoán phụ thuộc vào vùng crop khuôn mặt và điều kiện ảnh đầu vào.
- Level 1 chỉ phân biệt Real và Spoof. Việc phân biệt Printed và On-screen do Level 2 đảm nhiệm.