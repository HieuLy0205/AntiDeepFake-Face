# Model Weights Registry — AntiDeepFake-Face

> **Lưu ý Git (DD8):** Không commit trực tiếp tệp trọng số nặng (`.safetensors`, `.bin`, `.pt`, `.pth`, `.onnx`) hoặc thư mục cache `models/hf_cache/` lên GitHub. Toàn bộ tệp `.pt` và `models/hf_cache/` đã được khai báo trong `.gitignore`.

---

## Nhánh Giai đoạn 2 — Deepfake Detector (M5)

### 1. Trọng số đã Fine-tune trên RTFS-800 (Google Colab NVIDIA L4 — 0 rò rỉ `subject_id`)

Đặt các tệp `.pt` vào thư mục `models/` — `src/deepfake/predictor.py` sẽ **tự động phát hiện và nạp**.

| Tệp trọng số (`models/`) | Backbone gốc | Dung lượng | Ngưỡng `τ_d` (`val`) | TEST AUC (200 video) | EER | ACER | CPU Apple M2 (Batch=1 / Batch=10) |
|---|---|---:|---:|---:|---:|---:|---|
| **`m5_rtfs_siglip2_adapter.pt`** *(Mặc định)* | `king1oo1/deepfake-model` (`SigLIP2-Base`) | 54.2 MB | `0.6693` | **0.9656** `[0.929, 0.993]` | **6.67%** | **9.00%** | 80.4 ms (**12.4 FPS**) / 61.9 ms (**16.2 FPS**) |
| **`m5_rtfs_vit_adapter.pt`** *(Bổ trợ Ensemble)* | `prithivMLmods/Deep-Fake-Detector-v2-Model` (`ViT-Base`) | 54.2 MB | `0.8345` | 0.9280 `[0.893, 0.961]` | 17.67% | 16.00% | 77.1 ms (**13.0 FPS**) / 55.6 ms (**18.0 FPS**) |
| **`m5_rtfs_efficientnet_b0.pt`** *(CNN siêu nhẹ)* | `timm/efficientnet_b0` (4.0M params) | **15.6 MB** | `0.5861` | 0.9529 `[0.926, 0.977]` | 11.67% | 12.67% | 119.2 ms (8.4 FPS) / **37.2 ms (26.9 FPS)** |
| **Ensemble 50/50** (`SigLIP2` + `ViT` adapters) | Kết hợp 2 adapter trên | 108.4 MB | `0.5769` | 0.9643 `[0.929, 0.989]` | **6.67%** | **7.33%** | 169.4 ms (5.9 FPS) / 118.5 ms (**8.4 FPS**) |

### 2. Danh sách mô hình Pretrained gốc trên Hugging Face

| Hugging Face Model ID | Kiến trúc / Backbone | Số tham số | `id2label` gốc | Index lớp `Fake` | Nguồn URL |
|---|---|---:|---|---:|---|
| `king1oo1/deepfake-model` | SigLIP2-Base (`siglip2-base-patch16-224`) | 92.9M | `{0: 'Real', 1: 'Fake'}` | `1` | https://huggingface.co/king1oo1/deepfake-model |
| `prithivMLmods/Deep-Fake-Detector-v2-Model` | ViT-Base (`vit-base-patch16-224-in21k`) | 85.8M | `{0: 'Realism', 1: 'Deepfake'}` | `1` | https://huggingface.co/prithivMLmods/Deep-Fake-Detector-v2-Model |
| `prithivMLmods/Deepfake-Detect-Siglip2` | SigLIP2-Base (`siglip2-base-patch16-224`) | 92.9M | `{0: 'Fake', 1: 'Real'}` | `0` *(đảo chiều)* | https://huggingface.co/prithivMLmods/Deepfake-Detect-Siglip2 |
| `dima806/deepfake_vs_real_image_detection` | ViT-Base (`vit-base-patch16-224-in21k`) | 85.8M | `{0: 'Real', 1: 'Fake'}` | `1` | https://huggingface.co/dima806/deepfake_vs_real_image_detection |

### 3. Cách tải trọng số nền & sử dụng

```bash
# 1. Tải trọng số nền từ HuggingFace vào models/hf_cache/ (dùng offline)
.venv/bin/python scripts/download_m5_models.py \
  --models king1oo1/deepfake-model prithivMLmods/Deep-Fake-Detector-v2-Model

# 2. Đặt m5_rtfs_siglip2_adapter.pt (và tùy chọn vit/efficientnet_b0) vào models/
# 3. Kiểm chứng hoạt động thật & tốc độ CPU:
.venv/bin/python scripts/verify_m5_e2e_demo.py
```

Ví dụ gọi trong code:
```python
from src.deepfake import predict_deepfake, predict_deepfake_batch, LightweightCNNDetector, EnsembleDeepfakeDetector

# Cách chuẩn cho M1 (tự động nạp king1oo1/deepfake-model + m5_rtfs_siglip2_adapter.pt):
score = predict_deepfake(face_crop)

# Hoặc dùng CNN nhẹ EfficientNet-B0 (15.6 MB, ~27 FPS batch-10 trên CPU):
cnn_det = LightweightCNNDetector("models/m5_rtfs_efficientnet_b0.pt")
score_cnn = cnn_det.predict(face_crop)
```

---

## Nhánh Giai đoạn 1 — Anti-Spoof (M4)

| Tệp trọng số | Kiến trúc / Nguồn | Kích thước | Giấy phép | Ghi chú kỹ thuật |
|---|---|---:|---|---|
| `models/minifasnet_onnx/minifasnet_v2.onnx` | Upstream **MiniFASNetV2 (`2.7_80x80`)** từ `minivision-ai/Silent-Face-Anti-Spoofing` (ONNX export: `garciafido/minifasnet-v2-anti-spoofing-onnx`) | **1.70 MB** | Apache-2.0 | Crop vùng mặt mở rộng `2.7×` về `80×80` BGR float32 `[0..255]`; kết hợp phân tích phổ 2D FFT Moiré + không gian màu HSV/YCbCr (Boulkenafet et al. / Patel et al., IEEE TIFS) để tách mức 2 `Printed` vs `On-screen`. |

Lệnh tải tự động trọng số Anti-Spoof:
```bash
.venv/bin/python scripts/download_minifasnet.py
```
