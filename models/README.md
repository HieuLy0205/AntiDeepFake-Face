# Models Directory

Tài liệu ghi nguồn và thông tin model phục vụ module `src/face_detection/`:

## 1. YuNet (Face Detection)
- **Model:** `face_detection_yunet_2023mar.onnx`
- **Nguồn:** [OpenCV Zoo - YuNet](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet)
- **Kích thước file:** ~230 KB
- **Kiến trúc:** YuNet (ShiqiYu/libfacedetection) được tối ưu hóa cho edge devices và CPU.
- **Giấy phép (License):** Apache-2.0 License.
- **Đầu vào mặc định:** Ảnh kích thước linh hoạt, hỗ trợ scale kích thước qua `setInputSize((w, h))`.
- **Đầu ra:** Bounding box `[x, y, w, h]`, 5 điểm landmark mắt-mũi-miệng, và điểm số tin cậy (confidence score).
