# AntiDeepFake-Face

Hệ thống bảo vệ danh tính khuôn mặt lấy cảm hứng từ NullSwap sử dụng Xử lý ảnh, Thị giác máy tính và Trí tuệ nhân tạo.

## Cấu trúc thư mục:

```text
AntiDeepFake-Face/
│
├── README.md
├── .gitignore
├── requirements.txt
│
├── docs/
│   ├── define/
│   ├── design/
│   ├── research/
│   └── report/
│
├── src/
│   ├── preprocessing/
│   ├── face_detection/
│   ├── feature_extraction/
│   ├── protection/
│   ├── evaluation/
│   └── utils/
│
├── app/
│   ├── app.py
│   ├── components/
│   └── assets/
│
├── models/
│
├── data/
│   └── samples/
│
├── tests/
│
├── notebooks/
│
└── project_management/
```

## Quy tắc thư mục

| Thư mục | Mục đích |
|----------|----------|
| `docs/` | Chứa tài liệu phân tích, thiết kế, bài báo nghiên cứu, tài liệu kiểm thử và báo cáo |
| `src/` | Chứa các thuật toán xử lý ảnh, phát hiện khuôn mặt, trích xuất đặc trưng và bảo vệ danh tính |
| `app/` | Chứa ứng dụng Web dùng để trình diễn hệ thống |
| `models/` | Chứa các mô hình huấn luyện sẵn, checkpoint và cấu hình mô hình |
| `data/` | Chứa dữ liệu mẫu và ảnh phục vụ kiểm thử |
| `tests/` | Chứa các bài kiểm thử đơn vị và kiểm thử tích hợp |
| `notebooks/` | Chứa các notebook phục vụ thực nghiệm và nghiên cứu thuật toán |
| `project_management/` | Chứa tài liệu phân công công việc, tiến độ và biên bản họp |

## Tổng quan dự án
AntiDeepFake-Face là một dự án học thuật được phát triển trong học phần Xử lý ảnh và Thị giác máy tính.
Với sự phát triển nhanh chóng của công nghệ DeepFake, việc bảo vệ danh tính khuôn mặt cá nhân đang trở thành một thách thức quan trọng. Dự án tập trung nghiên cứu các phương pháp bảo vệ danh tính khuôn mặt một cách chủ động trước các cuộc tấn công hoán đổi khuôn mặt bằng DeepFake.
Lấy cảm hứng từ bài báo nghiên cứu *NullSwap: Proactive Identity Cloaking Against Deepfake Face Swapping*, dự án kết hợp các kỹ thuật xử lý ảnh và học sâu nhằm xây dựng một nguyên mẫu có khả năng bảo vệ danh tính khuôn mặt trong khi vẫn duy trì chất lượng thị giác của ảnh gốc.

## Bối cảnh nghiên cứu
Những tiến bộ gần đây trong công nghệ DeepFake Face Swapping cho phép chuyển danh tính khuôn mặt từ một người sang một hình ảnh khác với độ chân thực rất cao. Mặc dù công nghệ này có nhiều ứng dụng hợp pháp, nó cũng làm gia tăng các vấn đề liên quan đến quyền riêng tư, việc giả mạo danh tính và sự lan truyền thông tin sai lệch.
Thay vì chỉ phát hiện DeepFake sau khi chúng đã được tạo ra, dự án này nghiên cứu một hướng phòng vệ chủ động lấy cảm hứng từ NullSwap. Mục tiêu là tìm hiểu cách chỉnh sửa ảnh khuôn mặt theo một phương pháp có kiểm soát nhằm giảm khả năng bị khai thác bởi các cuộc tấn công hoán đổi khuôn mặt trong tương lai, đồng thời vẫn giữ được vẻ tự nhiên của ảnh.
Các ý tưởng và khái niệm được trình bày trong bài báo NullSwap đóng vai trò là nền tảng nghiên cứu chính cho dự án.

## Mục tiêu dự án
- Phát triển một ứng dụng web phục vụ xử lý ảnh khuôn mặt.
- Phát hiện và tiền xử lý các vùng khuôn mặt.
- Nghiên cứu phương pháp trích xuất đặc trưng khuôn mặt bằng học sâu.
- Xây dựng nguyên mẫu bảo vệ danh tính lấy cảm hứng từ NullSwap.
- So sánh ảnh gốc và ảnh đã được bảo vệ.
- Đánh giá chất lượng hình ảnh và hiệu năng xử lý.

## Thành viên nhóm
Dự án được phát triển bởi nhóm gồm 06 sinh viên:
| Thành viên | Nhiệm vụ |
|----------|----------|
| Thành viên 1 | Trưởng nhóm & Tích hợp hệ thống |
| Thành viên 2 | Xử lý ảnh |
| Thành viên 3 | Phát hiện khuôn mặt |
| Thành viên 4 | Học sâu & Trích xuất đặc trưng |
| Thành viên 5 | Bảo vệ danh tính |
| Thành viên 6 | Ứng dụng Web, Đánh giá & Kiểm thử |

## Phạm vi dự án
### Trong phạm vi
- Xử lý ảnh khuôn mặt tĩnh.
- Phát hiện và cắt khuôn mặt.
- Trích xuất đặc trưng khuôn mặt.
- Xây dựng nguyên mẫu bảo vệ danh tính.
- So sánh và đánh giá hình ảnh.
- Minh họa hệ thống thông qua ứng dụng Web.
### Ngoài phạm vi
- Xử lý webcam theo thời gian thực.
- Bảo vệ DeepFake trên video.
- Tái hiện hoàn chỉnh framework NullSwap gốc.
- Huấn luyện mô hình quy mô lớn từ đầu.

## Sản phẩm đầu ra mong đợi
- Tài liệu dự án (Define & Design).
- Các mô-đun xử lý ảnh.
- Nguyên mẫu bảo vệ danh tính khuôn mặt.
- Ứng dụng Web minh họa.
- Kết quả kiểm thử và đánh giá.
- Báo cáo cuối cùng của dự án.

## Công nghệ sử dụng
| Công nghệ | Mục đích |
|----------|----------|
| Python | Ngôn ngữ lập trình chính |
| OpenCV | Xử lý ảnh |
| PyTorch | Học sâu |
| MTCNN | Phát hiện khuôn mặt |
| NumPy | Các phép tính số học |
| Streamlit | Xây dựng ứng dụng Web |
| Matplotlib | Trực quan hóa dữ liệu |

## Kiến trúc hệ thống
Hệ thống đề xuất bao gồm các mô-đun sau:
```text
Ảnh đầu vào
    ↓
Giao diện Web
    ↓
Tiền xử lý ảnh
    ↓
Phát hiện khuôn mặt
    ↓
Trích xuất đặc trưng
    ↓
Bảo vệ danh tính
    ↓
Đánh giá
    ↓
Ảnh đầu ra đã được bảo vệ
```

## Tài liệu tham khảo
Wang et al., *NullSwap: Proactive Identity Cloaking Against Deepfake Face Swapping*, ICCV 2025.
Bài báo nghiên cứu: https://arxiv.org/html/2503.18678v1
Wang et al., *NullSwap: Proactive Identity Cloaking Against Deepfake Face Swapping*, ICCV 2025.

Bài báo nghiên cứu: https://arxiv.org/html/2503.18678v1
