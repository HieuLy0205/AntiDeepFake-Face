# Khảo sát Nghiên cứu, Dữ liệu & Benchmark Mô hình Phát hiện Deepfake (M5)

**Phụ trách:** M5 — Lê Minh Thành, Nguyễn Phông  
**Phạm vi:** Giai đoạn 2 (`src/deepfake/`) — Phát hiện khuôn mặt Deepfake chạy trên laptop CPU  
**Nguyên tắc báo cáo (NFR07):** Toàn bộ số liệu định lượng trong tài liệu này được đo trực tiếp từ script thực nghiệm ([scripts/prepare_rtfs_metadata.py](../../scripts/prepare_rtfs_metadata.py), [scripts/benchmark_m5_deepfake.py](../../scripts/benchmark_m5_deepfake.py)) và lưu trữ tái lập tại [results/m5_val_benchmark.json](../../results/m5_val_benchmark.json) và [results/m5_val_predictions.csv](../../results/m5_val_predictions.csv).

---

## 1. Đặt vấn đề & Vị trí của M5 trong Kiến trúc AntiDeepFake-Face

Trong hệ thống **AntiDeepFake-Face**, hai lớp tấn công khuôn mặt được tách thành 2 nhánh độc lập và tổng hợp theo quy tắc ưu tiên **DD6**:
1. **Giai đoạn 1 — Anti-Spoofing (M4):** Bắt tấn công trình diễn vật lý tại camera (`Printed` — ảnh in giấy; `On-screen` — phát lại qua màn hình).
2. **Giai đoạn 2 — Deepfake Detection (M5):** Bắt tấn công giả mạo kỹ thuật số (`Deepfake` — hoán đổi khuôn mặt / khuôn mặt tổng hợp AI) khi đầu vào vượt qua bộ lọc Anti-Spoof.

### Hợp đồng giao tiếp đóng băng của M5 (`src/schemas.py`)
- **Đầu vào:** `FaceCropResult` từ M3 (`src/face_detection/`), chứa ảnh khuôn mặt lớn nhất đã crop về chuẩn `(224, 224, 3)`, hệ màu **RGB**, kiểu `np.uint8`. Toàn bộ bước chuẩn hóa tensor và ánh xạ nhãn được đóng gói kín bên trong `src/deepfake/`.
- **Đầu ra:** `DeepfakeScore(frame_id: int, fake_prob: float)` với `0.0 <= fake_prob <= 1.0`.
- **Tổng hợp nhiều frame (DD7):** M1 lấy trung bình `fake_prob` qua `N` khung hình (mặc định `N = 10`, khảo sát `N ∈ {1, 5, 10, 30}`), rồi so với ngưỡng `τ_d` (hiệu chuẩn trên tập `val`).
- **Phân biệt với hướng NullSwap:** Nghiên cứu *NullSwap* (Wang et al., ICCV 2025) là hướng **phòng vệ chủ động** (*proactive identity cloaking* — nhúng nhiễu bảo vệ ảnh trước khi bị hoán đổi mặt), nằm ở mục *Hướng phát triển tương lai* của đồ án. Nhiệm vụ M5 hiện tại triển khai **phát hiện thụ động** (*passive deepfake detection*) tương thích trực tiếp với Web App Streamlit và bộ kiểm thử `tests/test_anti_spoof_deepfake.py`.

---

## 2. Khảo sát Các Bài báo Khoa học & Bộ dữ liệu Liên quan

### 2.1 Bộ dữ liệu & Khung Benchmark chuẩn

| Nghiên cứu / Bộ dữ liệu | Tác giả & Hội nghị | Nguồn | Ý nghĩa đối với M5 |
|---|---|---|---|
| **FaceForensics++ (FF++)** | Rössler et al., ICCV 2019 | https://arxiv.org/abs/1901.08971 | Bộ dữ liệu nền tảng gồm 4 phương pháp giả mạo (*Deepfakes*, *Face2Face*, *FaceSwap*, *NeuralTextures*) ở 3 mức nén (`raw`, `c23`, `c40`); đặt baseline **XceptionNet**. |
| **Celeb-DF (v2)** | Li et al., CVPR 2020 | https://arxiv.org/abs/1909.12962 | Deepfake chất lượng thị giác cao, ít lỗi biên thô; thước đo chuẩn kiểm tra suy giảm hiệu năng khi đánh giá chéo tập (*cross-dataset*). |
| **DFDC** | Dolhansky et al., 2020 | https://arxiv.org/abs/2006.07397 | >100.000 video quay đa dạng điều kiện ánh sáng/góc mặt *in-the-wild*; khẳng định sức mạnh của **EfficientNet** + augmentation đa dạng. |
| **DeepfakeBench** | Yan et al., NeurIPS 2023 | https://arxiv.org/abs/2307.01426 (`SCLBD/DeepfakeBench`) | Chuẩn hóa tiền xử lý crop mặt và giao thức đánh giá công bằng giữa các detector trên kịch bản *cross-dataset* và *cross-manipulation*. |
| **RTFS-10k** | Kontur AI / `key0303/rtfs-10k` | https://huggingface.co/datasets/kontur-ai/rtfs-10k | Bộ dữ liệu video-call thực tế gồm 1.636 clip gốc YouTube (CC) và 9.772 video face-swap tạo bởi **InSwapper** và **UniFace**, có nhãn ánh sáng/chuyển động đầu và hậu xử lý `faceenchancer`. |

### 2.2 Hướng tiếp cận Phát hiện Deepfake (2018–2026)

1. **CNN gọn nhẹ & Khai thác miền tần số (gần với Xử lý ảnh cổ điển):**
   - **MesoNet** (Afchar et al., WIFS 2018 — https://arxiv.org/abs/1809.00888): Mạng CNN nông (`Meso-4`, `MesoInception-4`) học đặc trưng trung mô (*mesoscopic*), rất nhẹ cho CPU nhưng tổng quát hóa kém với bộ sinh hiện đại.
   - **F3-Net** (*Thinking in Frequency*, Qian et al., ECCV 2020 — https://arxiv.org/abs/2007.09355): Kết hợp nhánh không gian với biến đổi **DCT (Discrete Cosine Transform)** để bắt dị thường tần số cao bị che khuất dưới nén video.
2. **Tổng quát hóa chéo bộ sinh bằng tổng hợp mẫu biên (*Data-Centric Generalization*):**
   - **SBI — Self-Blended Images** (Shiohara & Yamasaki, CVPR 2022 — https://arxiv.org/abs/2204.08376): Tự trộn chính ảnh thật với bản biến đổi của nó để ép mô hình (như EfficientNet-B4) học dấu vết ranh giới ghép & lệch màu tổng quát thay vì học thuộc 1 công cụ tạo deepfake cụ thể.
3. **Khai thác Vision Foundation Models (CLIP / SigLIP / ViT) & Tinh chỉnh tiết kiệm tham số (2023–2026):**
   - **UniversalFakeDetect** (Ojha et al., CVPR 2023 — https://arxiv.org/abs/2303.11413): Dùng đặc trưng đóng băng từ **CLIP-ViT** giúp phát hiện cả GAN lẫn Diffusion chưa từng gặp trong lúc huấn luyện.
   - **Effort** (*Efficient Orthogonal Modeling*, Yan et al., ICML — https://arxiv.org/abs/2411.15633): Dùng phân rã **SVD** tách không gian trọng số thành không gian ngữ nghĩa (đóng băng) và không gian trực giao bắt giả mạo (~0.19M tham số học).
   - **GenD** (*Deepfake Detection that Generalizes Across Benchmarks*, WACV 2026 — `yermandy/GenD_CLIP_L_14`): Tinh chỉnh tối giản lớp LayerNorm (LN-tuning) kết hợp chuẩn hóa siêu cầu L2 trên Vision Foundation Models nhằm tối đa hóa độ tổng quát hóa chéo tập.
4. **Phòng vệ chủ động (*Proactive Defense* — Hướng mở rộng tương lai):**
   - **NullSwap** (Wang et al., ICCV 2025 — https://arxiv.org/html/2503.18678v1): Nhúng nhiễu bảo vệ vô hình lên ảnh chân dung gốc nhằm vô hiệu hóa các mô hình face-swapping trước khi tấn công xảy ra.

---

## 3. Phân tích Dữ liệu Thực nghiệm (`dataset_RTFS_800_Subset`) & Chống Rò rỉ Dữ liệu

### 3.1 Đặc điểm tập con 800 video
Tập dữ liệu làm việc cục bộ `dataset_RTFS_800_Subset/` (1.43 GB, rút mẫu `seed=42` từ `rtfs-10k`) đã được kiểm tra toàn vẹn bằng OpenCV trên đủ **800/800 video** (0 file lỗi, 0 file 0-byte):
- **Phân bố nhãn (tỷ lệ lệch lớp 1:3):**
  - `Real` (`original_videos/`): **200 video** (25.0%).
  - `Deepfake` (`inswapper/`): **300 video** (37.5%, gồm **139** có `faceenchancer` và **161** không có).
  - `Deepfake` (`uniface/`): **300 video** (37.5%, gồm **115** có `faceenchancer` và **185** không có).
- **Thông số kỹ thuật:** Chủ yếu Full HD `1920×1080` (567/800) và HD `1280×720` (205/800); FPS `15–60`; thời lượng `7.0–40.0s` (trung bình ~18.5s; có **79/800 video > 30s** vượt ngưỡng upload UI M6, cần chọn video ≤ 30s khi demo trực tiếp).
- **Bối cảnh:** ~91% quay `indoor`, ~79% có xoay đầu (`head rotation`), ~39% có cử chỉ tay (`gesticulation`), đủ 3 điều kiện ánh sáng (`bright`, `average`, `dark`).

### 3.2 Phát hiện & Xử lý Rò rỉ Danh tính (`subject_id` Leakage)
Phân tích sâu cho thấy do rút mẫu độc lập từng CSV nên có **trùng video gốc YouTube (`yt_id`)** giữa 3 thư mục:
- `original_videos` ∩ `inswapper`: **46** `yt_id` trùng (38 clip trùng hệt).
- `original_videos` ∩ `uniface`: **45** `yt_id` trùng (41 clip trùng hệt).
- `inswapper` ∩ `uniface`: **63** `yt_id` trùng (56 clip trùng hệt).

Script [scripts/prepare_rtfs_metadata.py](../../scripts/prepare_rtfs_metadata.py) đã ánh xạ chính xác `target_clip → videos.csv['id']` làm `subject_id` (`yt_<id>`) và chia tập phân tầng theo nhóm (`seed=42`, tỷ lệ chuẩn 3 phần `60% train / 15% val / 25% test` phục vụ cả đánh giá zero-shot lẫn fine-tune không rò rỉ) vào đúng 7 cột chuẩn của [data/metadata.csv](../../data/metadata.csv):

| Tập | Tổng video | Số `subject_id` duy nhất | `Real` | `Deepfake` (`inswapper` / `+enh`) | `Deepfake` (`uniface` / `+enh`) | Trùng `subject_id` với các tập khác |
|---|---:|---:|---:|---:|---:|---:|
| **`train`** (huấn luyện / fine-tune) | **480** | 350 | 120 | 180 (89 / 91) | 180 (105 / 75) | **0** |
| **`val`** (chọn model & hiệu chuẩn `τ_d`) | **120** | 100 | 30 | 45 (29 / 16) | 45 (27 / 18) | **0** |
| **`test`** (đánh giá cuối E5 giữ kín) | **200** | 164 | 50 | 75 (43 / 32) | 75 (53 / 22) | **0** |

*(Ghi chú: ở vòng khảo sát Zero-shot sơ bộ Mục 4.2, nhóm cũng đã chạy đối chứng 4 mô hình cộng đồng trên tập chia 2 phần 240 `val` lưu tại [results/m5_val_benchmark.json](../../results/m5_val_benchmark.json); sau đó cố định tập chia 3 phần ở trên để huấn luyện trên Google Colab GPU L4 và đánh giá công bằng Zero-shot vs. Fine-tuned trên cùng 200 video `test` chưa từng thấy ở Mục 5.)*

---

## 4. Kết quả Benchmark Pretrained Models trên CPU (`val` — 240 video)

### 4.1 Thiết lập thực nghiệm
- **Phần cứng:** Laptop CPU **Apple M2** (8 nhân, 16 GB RAM), chạy hoàn toàn trên `device="cpu"` (PyTorch 2.15, Transformers 5.24).
- **Trích xuất khuôn mặt:** Lấy đều `5 frame/video` → phát hiện mặt bằng **MTCNN** (`facenet-pytorch`, chọn khuôn mặt lớn nhất + lề ngữ cảnh 25%) → crop về `(224, 224, 3)` RGB `uint8`. Tỷ lệ phát hiện mặt đạt **100%** (1.200/1.200 crop trên 240 video `val`, 0 lần fallback).
- **Tổng hợp video:** Lấy trung bình `fake_prob` của 5 frame theo quy tắc DD7.
- **Lưu ý kỹ thuật phát hiện khi tích hợp:** Mỗi checkpoint HuggingFace quy ước `id2label` khác nhau (đặc biệt `prithivMLmods/Deepfake-Detect-Siglip2` đặt `{0: 'Fake', 1: 'Real'}` **ngược** với 3 mô hình còn lại). Hàm `resolve_fake_label_index()` trong [src/deepfake/predictor.py](../../src/deepfake/predictor.py) đã tự động hóa việc ánh xạ để loại bỏ hoàn toàn lỗi đảo nhãn.

### 4.2 Bảng so sánh tổng quan 4 mô hình đơn & Tổ hợp 2 mô hình bổ trợ (tập `val`, 240 video)

| Mô hình | Kiến trúc (Params) | ROC-AUC (95% CI) | PR-AUC | EER | Ngưỡng tối ưu `τ_d` | Accuracy @ `τ_d` | F1-macro @ `τ_d` | APCER ↓ | BPCER ↓ | ACER ↓ | CPU Batch=1 (ms / FPS) | CPU Batch=10 (ms/fr / FPS) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **`king1oo1/deepfake-model`** *(Winner đơn)* | SigLIP2-Base (92.9M) | **0.6203** `[0.539, 0.703]` | **0.8310** | **0.4194** | `0.5935` | **64.17%** | **0.5889** | **0.3333** | 0.4333 | **0.3833** | 87.9 ms (**11.4 FPS**) | 72.9 ms (**13.7 FPS**) |
| **`prithivMLmods/Deep-Fake-Detector-v2-Model`** | ViT-Base (85.8M) | 0.5936 `[0.513, 0.668]` | 0.8302 | 0.4667 | `0.6376` | 45.83% | 0.4583 | 0.6889 | **0.1000** | 0.3944 | **66.7 ms (15.0 FPS)** | **52.6 ms (19.0 FPS)** |
| `prithivMLmods/Deepfake-Detect-Siglip2` | SigLIP2-Base (92.9M) | 0.4819 `[0.396, 0.565]` | 0.7465 | 0.5417 | `0.0000` | 67.08% | 0.5311 | 0.1889 | 0.7500 | 0.4694 | 79.3 ms (12.6 FPS) | 62.8 ms (15.9 FPS) |
| `dima806/deepfake_vs_real_image_detection` | ViT-Base (85.8M) | 0.4306 `[0.353, 0.512]` | 0.7198 | 0.4806 | `0.7814` | 50.00% | 0.4773 | 0.5278 | 0.4167 | 0.4722 | 74.2 ms (13.5 FPS) | 62.9 ms (15.9 FPS) |
| **Ensemble 50/50** (`king1oo1` + `Deep-Fake-v2`) | SigLIP2 + ViT | **0.6610** `[0.587, 0.734]` | **0.8631** | 0.4361 | `0.6233` | 47.50% | 0.4750 | 0.6778 | **0.0667** | **0.3722** | ~154.6 ms (~6.5 FPS) | ~125.5 ms (**~8.0 FPS**) |

*(Ghi chú: Do tập `val` lệch lớp 25% Real / 75% Deepfake, **ROC-AUC**, **Balanced Accuracy** và **ACER** là các chỉ số đáng tin cậy hơn Accuracy thuần. Ở ngưỡng `0.50`, Ensemble 50/50 đạt `Accuracy = 61.25%`, `F1-macro = 0.5599`, `APCER = 0.3611`, `BPCER = 0.4667`, `ACER = 0.4139`.)*

### 4.3 Phân tích lỗi theo lát cắt (*Slice-based Error Analysis* — ROC-AUC theo Generator, Face Enhancer & Ánh sáng)

| Lát cắt dữ liệu (`val`) | `king1oo1/deepfake-model` | `Deep-Fake-Detector-v2` | `Deepfake-Detect-Siglip2` | `dima806/deepfake_vs_real` | **Ensemble 50/50** |
|---|---:|---:|---:|---:|---:|
| **Real vs InSwapper (tổng)** | 0.5278 | **0.6683** | 0.4221 | 0.3369 | 0.6402 |
| ↳ InSwapper **không** `faceenchancer` | 0.5522 | **0.7173** | 0.3934 | 0.2781 | 0.6917 |
| ↳ InSwapper **có** `faceenchancer` | 0.4912 | **0.5949** | 0.4653 | 0.4250 | 0.5630 |
| **Real vs UniFace (tổng)** | **0.7128** | 0.5189 | 0.5403 | 0.5243 | 0.6819 |
| ↳ UniFace **không** `faceenchancer` | **0.6937** | 0.5199 | 0.5293 | 0.4982 | 0.6720 |
| ↳ UniFace **có** `faceenchancer` | **0.7653** | 0.5160 | 0.5705 | 0.5958 | 0.7090 |
| **Tất cả Fake — không enhancer** | **0.6300** | 0.6088 | 0.4681 | 0.3992 | **0.6808** |
| **Tất cả Fake — có enhancer** | **0.6008** | 0.5633 | 0.5074 | 0.4933 | **0.6214** |
| Ánh sáng: `bright` / `average` / `dark` | **0.678** / 0.548 / **0.674** | 0.632 / **0.582** / 0.554 | 0.413 / 0.514 / 0.583 | 0.457 / 0.425 / 0.411 | — |
| Giới tính: `man` / `woman` | **0.640** / 0.596 | 0.565 / **0.648** | 0.523 / 0.431 | 0.534 / 0.342 | — |

### 4.4 Bốn phát hiện khoa học rút ra từ thực nghiệm
1. **Sụp đổ chéo miền (*Cross-Domain Collapse*) ở các mô hình huấn luyện trên ảnh AI toàn mặt:** Dù công bố độ chính xác >92–99% trên tập gốc (Kaggle 190k / StyleGAN / Diffusion), `dima806` (`AUC = 0.4306`) và `Deepfake-Detect-Siglip2` (`AUC = 0.4819`) gần như không phân biệt được face-swap trong video-call RTFS-10k (vốn chỉ thay vùng mặt trong và giữ nguyên nền, tóc, nhiễu camera thực tế).
2. **Hiệu quả của dữ liệu huấn luyện đúng bài toán face-swap:** `king1oo1/deepfake-model` (SigLIP2 huấn luyện trên **FF++ + DFDC + Celeb-DF**) vượt trội rõ rệt so với `Deepfake-Detect-Siglip2` dù dùng **cùng hệt kiến trúc SigLIP2-Base 92.9M** (`0.6203` vs `0.4819`, chênh `+0.1384 AUC`), chứng minh dữ liệu huấn luyện quyết định lớn hơn kiến trúc thuần.
3. **Tác động hai chiều của hậu xử lý `faceenchancer`:** Trên `InSwapper`, bật `faceenchancer` làm giảm AUC của `Deep-Fake-Detector-v2` tới **`-0.1224`** (`0.7173 → 0.5949`) do bộ tăng nét làm mờ ranh giới ghép thô. Ngược lại, trên `UniFace`, `king1oo1` phát hiện bản có `faceenchancer` tốt hơn (`0.7653` vs `0.6937`), nhiều khả năng do bắt được kết cấu da mịn bất thường từ bộ làm nét.
4. **Tính bổ trợ theo công cụ sinh (*Generator Complementarity*):** `Deep-Fake-Detector-v2` mạnh trên **InSwapper** (`0.6683`, đặc biệt `0.7173` không enhancer) nhưng gần ngẫu nhiên trên **UniFace** (`0.5189`); ngược lại `king1oo1` mạnh trên **UniFace** (`0.7128`) nhưng yếu trên **InSwapper** (`0.5278`). Kết hợp 50/50 hai mô hình đẩy **ROC-AUC tổng lên `0.6610`** và vẫn đạt **~8.0 FPS** (batch-10 CPU) — vượt mục tiêu demo `≥ 5 FPS`.

---

## 5. Kết quả Fine-tuning trên GPU L4 & Đánh giá Cuối E5 trên Tập `test` (200 video chưa thấy)

Để khắc phục hiện tượng sụp đổ chéo miền ở Mục 4, nhóm triển khai huấn luyện có kiểm soát trên **Google Colab NVIDIA L4 (BF16)** bằng notebook [notebooks/m5_train_deepfake_colab_L4.ipynb](../../notebooks/m5_train_deepfake_colab_L4.ipynb) (4.800 ảnh crop `224×224` MTCNN, 0 rò rỉ `subject_id`, áp dụng **GenD-style All LayerNorms + Top-4 Transformer Blocks**, class-balanced loss `3:1`, label smoothing `0.05` và Forensic Augmentation gồm lật ngang, random crop, đổi sáng/màu, blur và nén JPEG `Q=55..95`). Toàn bộ ngưỡng quyết định `τ_d` được hiệu chuẩn **duy nhất trên `val`** rồi chấm 1 lần trên **`test` (200 video, 164 người chưa từng thấy)**:

### 5.1 Bảng so sánh Zero-shot vs. Fine-tuned trên Tập `test` giữ kín (Thí nghiệm E5)

| Mô hình | Val AUC | **TEST AUC** (95% CI) | EER ↓ | `τ_d` (`val`) | Accuracy | F1-macro | APCER ↓ | BPCER ↓ | ACER ↓ | InSwapper / +Enh | UniFace / +Enh | CPU Apple M2 (Batch=1 / Batch=10) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| **`finetuned_siglip2`** *(Mặc định)* | 0.9689 | **0.9656** `[0.929, 0.993]` | **0.0667** | `0.6693` | 88.5% | 0.8624 | 0.1400 | 0.0400 | 0.0900 | 0.9592 / 0.9431 | **0.9720 / 0.9727** | 80.4 ms (**12.4 FPS**) / 61.9 ms (**16.2 FPS**) |
| **`finetuned_ensemble`** *(50/50)* | 0.9711 | 0.9643 `[0.929, 0.989]` | **0.0667** | `0.5769` | **92.0%** | **0.8997** | **0.0867** | 0.0600 | **0.0733** | **0.9616 / 0.9487** | 0.9669 / 0.9655 | 169.4 ms (5.9 FPS) / 118.5 ms (**8.4 FPS**) |
| **`finetuned_efficientnet_b0`** *(CNN 4M)* | **0.9715** | 0.9529 `[0.926, 0.977]` | 0.1167 | `0.5861` | 82.0% | 0.7980 | 0.2333 | **0.0200** | 0.1267 | 0.9357 / 0.9038 | 0.9701 / 0.9691 | 119.2 ms (8.4 FPS) / **37.2 ms (26.9 FPS)** |
| **`finetuned_vit`** *(ViT-Base)* | 0.8978 | 0.9280 `[0.893, 0.961]` | 0.1767 | `0.8345` | 79.0% | 0.7660 | 0.2600 | 0.0600 | 0.1600 | 0.9261 / 0.9181 | 0.9299 / 0.9200 | 77.1 ms (**13.0 FPS**) / 55.6 ms (**18.0 FPS**) |
| `zeroshot_siglip2` *(Baseline)* | 0.5400 | 0.7172 `[0.638, 0.795]` | 0.3567 | `0.5730` | 65.0% | 0.6128 | 0.3600 | 0.3200 | 0.3400 | 0.6547 / 0.6138 | 0.7797 / 0.8009 | 80.4 ms / 61.9 ms |
| `zeroshot_ensemble` *(Baseline)* | 0.5444 | 0.7108 `[0.628, 0.793]` | 0.3500 | `0.6808` | 39.0% | 0.3838 | 0.8067 | 0.0200 | 0.4133 | 0.6995 / 0.6469 | 0.7221 / 0.7509 | 169.4 ms / 118.5 ms |
| `zeroshot_vit` *(Baseline)* | 0.5656 | 0.5661 `[0.472, 0.661]` | 0.4600 | `0.4472` | 54.0% | 0.5009 | 0.4533 | 0.4800 | 0.4667 | 0.6365 / 0.6150 | 0.4957 / 0.5564 | 77.1 ms / 55.6 ms |

### 5.2 Nhận xét & Quyết định chốt mô hình cho `src/deepfake/`
- **Bước nhảy vượt trội sau fine-tune có kiểm soát:** `finetuned_siglip2` tăng **`+24.84%` TEST ROC-AUC** (`0.7172 → 0.9656`), giảm **EER từ `35.67% → 6.67%`** và **ACER từ `34.00% → 9.00%`**, bám sát `Val AUC = 0.9689` (chứng tỏ **không overfit** và tổng quát hóa tốt sang người mới). Khả năng bắt cả **InSwapper + `faceenchancer` (`0.9431`)** lẫn **UniFace + `faceenchancer` (`0.9727`)** đều ổn định >0.94 AUC.
- **Lựa chọn triển khai trong [src/deepfake/predictor.py](../../src/deepfake/predictor.py):**
  - **Mặc định (`predict_deepfake`):** `king1oo1/deepfake-model` + adapter `models/m5_rtfs_siglip2_adapter.pt` (`τ_d = 0.6693`), đạt **TEST AUC = 0.9656**, **12.4 FPS** (đơn frame) / **16.2 FPS** (batch `N = 10`) trên CPU Apple M2.
  - **Tùy chọn chính xác tối đa (`EnsembleDeepfakeDetector`):** Kết hợp 50/50 `SigLIP2` + `ViT` adapters đạt **Accuracy = 92.0%**, **F1-macro = 0.8997**, **ACER = 7.33%** (~8.4 FPS batch-10 CPU).
  - **Tùy chọn siêu nhẹ (`LightweightCNNDetector`):** `models/m5_rtfs_efficientnet_b0.pt` (15.6 MB, 4.0M tham số), đạt **TEST AUC = 0.9529**, **BPCER = 2.0%**, **26.9 FPS** (batch-10 CPU) cho máy cấu hình yếu.

---

## 6. Lệnh Tái lập & Kiểm chứng Luồng Demo Thật

```bash
# 1. Sinh tập chia 3 phần train/val/test chống rò rỉ (0 trùng subject_id)
.venv/bin/python scripts/prepare_rtfs_metadata.py --train-ratio 0.60 --val-ratio 0.15 --seed 42

# 2. Trích xuất 4.800 ảnh khuôn mặt 224x224 bằng MTCNN & đóng gói ZIP cho Google Colab L4
.venv/bin/python scripts/extract_rtfs_crops.py --frames-per-video 6 --workers 4
.venv/bin/python scripts/export_colab_dataset.py

# 3. Huấn luyện trên Colab L4 bằng notebooks/m5_train_deepfake_colab_L4.ipynb, đặt 3 file .pt vào models/

# 4. Kiểm chứng luồng demo thật trên 5 video test chưa thấy & đo tốc độ CPU
.venv/bin/python scripts/verify_m5_e2e_demo.py

# 5. Chạy bộ unit test M5 và kiểm tra hợp đồng giao tiếp M6
.venv/bin/python -m pytest -v tests/test_deepfake_m5.py tests/test_anti_spoof_deepfake.py
```
