# AntiDeepFake-Face

Hệ thống **Phát hiện Gian lận Khuôn mặt Đa tầng (Multi-Stage Face Anti-Spoofing & Deepfake Detection)** sử dụng Xử lý ảnh, Thị giác máy tính và Trí tuệ nhân tạo.

## Cấu trúc thư mục

```text
AntiDeepFake-Face/
├── README.md
├── README_VN.md
├── requirements.txt
├── .streamlit/               # Cấu hình theme tối & giao diện Streamlit
├── configs/                  # Tham số mặc định (YAML: ngưỡng, tỉ lệ crop, model)
├── docs/
│   ├── define/               # Phần I: Định nghĩa yêu cầu
│   ├── design/               # Phần II: Thiết kế hệ thống
│   ├── research/             # Khảo sát model và dataset
│   └── report/               # Báo cáo & kịch bản kiểm thử
├── src/
│   ├── video_io/             # M3: Đọc ảnh/video/webcam, tách cảnh (scene/uniform/fps)
│   ├── face_detection/       # M3: Phát hiện đa khuôn mặt YuNet & cắt chuẩn 224x224 RGB
│   ├── anti_spoof/           # M4 (GĐ 1): MiniFASNetV2 + phân tích vật lý FFT/Glare Level 2
│   ├── deepfake/             # M5 (GĐ 2): SigLIP2 / ViT / Ensemble Deepfake Detector (Batch)
│   ├── decision/             # M1: Tổng hợp đa khung hình & quy tắc ưu tiên DD6/DD7
│   ├── evaluation/           # M2: Đánh giá APCER / BPCER / ACER / ROC-AUC
│   ├── schemas.py            # Hợp đồng dữ liệu chuẩn giữa các module
│   └── pipeline.py           # Luồng suy luận đầu-cuối tích hợp toàn hệ thống
├── app/
│   ├── app.py                # Giao diện Web Streamlit (M6)
│   ├── components/           # Cầu nối pipeline, validate đầu vào, vẽ HUD & webcam
│   └── assets/               # Stylesheet CSS giao diện
├── models/                   # Trọng số model pretrained & cache HuggingFace cục bộ
├── data/                     # Dữ liệu mẫu & metadata kiểm thử
├── scripts/                  # Tải model, xử lý dữ liệu, huấn luyện & benchmark
├── benchmark/                # Đo hiệu năng các bộ phát hiện khuôn mặt
├── results/                  # Kết quả benchmark & đánh giá định lượng
├── tests/                    # Bộ kiểm thử đơn vị & tích hợp (pytest)
├── notebooks/                # Notebook thực nghiệm
└── project_management/       # Tài liệu phân công & tiến độ nhóm
```

## Tổng quan & Mục tiêu dự án

**AntiDeepFake-Face** là dự án học thuật thuộc học phần *Xử lý ảnh và Thị giác máy tính*, tập trung giải quyết đồng thời hai nhóm tấn công giả mạo khuôn mặt phổ biến:
1. **Tấn công trình diễn vật lý (Presentation Attack — GĐ 1):** Dùng ảnh in trên giấy (`Printed`) hoặc phát lại video/ảnh qua màn hình điện thoại, máy tính (`On-screen`) đưa trước camera.
2. **Tấn công tổng hợp kỹ thuật số (Digital Manipulation — GĐ 2):** Dùng AI hoán đổi hoặc tạo sinh khuôn mặt giả (`Deepfake`).

*(Ghi chú bối cảnh nghiên cứu: Nhóm cũng khảo sát hướng phòng vệ chủ động như NullSwap [Wang et al., ICCV 2025] trong tài liệu nghiên cứu; nguyên mẫu phần mềm hoàn chỉnh hiện tại triển khai hệ thống phát hiện đa tầng theo thời gian thực qua Web App.)*

## Kiến trúc hệ thống thực tế

```text
[Ảnh / Video / Webcam (Browser & Live OpenCV)]
                     │
                     ▼
   M3: Video I/O (Scene / Uniform / FPS)
     + YuNet Multi-Face Detection
     + Cắt kép: 2.7× (ngữ cảnh rộng cho GĐ 1) & 1.2× (sát mặt cho GĐ 2)
            │                          │
            ▼                          ▼
  GĐ 1 (M4): Anti-Spoofing     GĐ 2 (M5): Deepfake AI
  • Level 1: MiniFASNetV2      • SigLIP2 / ViT / Ensemble
  • Level 2: 2D FFT + Glare    • Suy luận theo lô (Batch)
  → Real / Printed / On-screen → xác suất Deepfake [0..1]
            │                          │
            └────────────┬─────────────┘
                         ▼
   M1: Tổng hợp đa khung hình & nhiều người (DD6 / DD7)
   Ưu tiên: No_Face → Printed / On-screen → Deepfake → Real
                         │
                         ▼
   M6: Streamlit Cyber-Forensic Studio
   • Thẻ kết luận & Xác suất kèm mốc ngưỡng
   • Lưới Storyboard từng khung cảnh & Soi từng khuôn mặt (#1, #2...)
   • Biểu đồ dòng thời gian Altair & Xuất báo cáo JSON / CSV (RAM-Only DD8)
```

## Khởi động nhanh trên Terminal

```powershell
# 1. Di chuyển vào thư mục dự án và cài thư viện
cd "AntiDeepFake-Face"
pip install -r requirements.txt

# 2. (Tùy chọn) Tải trước trọng số YuNet & model HuggingFace để chạy offline
python scripts/download_yunet.py
python scripts/download_m5_models.py

# 3. Khởi chạy Web App
streamlit run app/app.py

# 4. Chạy toàn bộ bộ kiểm thử tự động
pytest -v
```

## Công nghệ sử dụng

| Công nghệ | Vai trò |
|---|---|
| Python 3 | Ngôn ngữ lập trình chính |
| OpenCV (YuNet + DNN/Image Processing) | Đọc video/webcam, phát hiện đa khuôn mặt, phân tích FFT/Glare Level 2 |
| PyTorch & Transformers (HuggingFace) | Suy luận `MiniFASNetV2` (GĐ 1) và `SigLIP2` / `ViT` (GĐ 2) |
| NumPy & Pandas | Xử lý tensor, tính xác suất tổng hợp & xuất báo cáo |
| Streamlit & Altair | Giao diện Web tương tác, Live Webcam HUD & biểu đồ dòng thời gian |
| Pytest | Kiểm thử đơn vị, kiểm thử tích hợp & kiểm thử UI không đầu (`AppTest`) |

## Thành viên nhóm (06 sinh viên)

| ID | Nhiệm vụ |
|---|---|
| M1 | Tích hợp hệ thống & Bộ tổng hợp quyết định (`src/pipeline.py`, `src/decision/`) |
| M2 | Dữ liệu & Đánh giá (`data/`, `src/evaluation/`) |
| M3 | Lấy frame & Phát hiện khuôn mặt (`src/video_io/`, `src/face_detection/`) |
| M4 | Chống giả mạo ảnh in & màn hình (`src/anti_spoof/`) |
| M5 | Phát hiện khuôn mặt Deepfake (`src/deepfake/`) |
| M6 | Thiết kế UI/UX Streamlit, Kiểm thử & Báo cáo (`app/`, `tests/`) |

## Tài liệu tham khảo
- Minivision AI, *Silent-Face-Anti-Spoofing* (`2.7_80x80_MiniFASNetV2`).
- OpenCV Zoo, *YuNet Face Detection (`face_detection_yunet_2023mar.onnx`)*.
- HuggingFace Pretrained Models: `king1oo1/deepfake-model`, `prithivMLmods/Deep-Fake-Detector-v2-Model`.
- Wang et al., *NullSwap: Proactive Identity Cloaking Against Deepfake Face Swapping*, ICCV 2025 (https://arxiv.org/html/2503.18678v1).
