# AntiDeepFake-Face

A **Multi-Stage Face Anti-Spoofing & Deepfake Detection** system built with Image Processing, Computer Vision, and Deep Learning.

## Folder Structure

```text
AntiDeepFake-Face/
├── README.md
├── README_VN.md
├── requirements.txt
├── .streamlit/               # Dark forensic theme & minimal toolbar config
├── configs/                  # Default YAML parameters (thresholds, crop scales, models)
├── docs/                     # Requirements, design docs, surveys, and reports
├── src/
│   ├── video_io/             # M3: Image/video/webcam reader (scene / uniform / fps sampling)
│   ├── face_detection/       # M3: YuNet multi-face detector & 224x224 RGB cropping
│   ├── anti_spoof/           # M4 (Stage 1): MiniFASNetV2 + FFT/Glare Level 2 physical cues
│   ├── deepfake/             # M5 (Stage 2): SigLIP2 / ViT / Ensemble Deepfake Detector
│   ├── decision/             # M1: Multi-frame & DD6 precedence decision aggregator
│   ├── evaluation/           # M2: APCER / BPCER / ACER / ROC-AUC metrics
│   ├── schemas.py            # Shared inter-module data contracts
│   └── pipeline.py           # End-to-end dual-scale batched inference pipeline
├── app/
│   ├── app.py                # Streamlit Web UI (M6)
│   ├── components/           # Pipeline bridge, validators, HUD visualizer, webcam handler
│   └── assets/               # Custom CSS theme
├── models/                   # Pretrained weights & local HuggingFace cache
├── data/                     # Sample data & test metadata
├── scripts/                  # Model downloaders, training, and benchmarking scripts
├── benchmark/                # Detector speed/accuracy benchmarks
├── results/                  # Verified benchmark & evaluation outputs
├── tests/                    # Unit, integration, and headless UI tests (pytest)
├── notebooks/                # Experimental notebooks
└── project_management/       # Team planning & progress tracking
```

## Project Overview

**AntiDeepFake-Face** is an academic Computer Vision & Image Processing project addressing two major facial fraud vectors in a unified pipeline:
1. **Physical Presentation Attacks (Stage 1 — Anti-Spoofing):** Printed photographs (`Printed`) and screen replay attacks (`On-screen`) presented in front of a camera.
2. **Digital AI Manipulation (Stage 2 — Deepfake Detection):** Synthetic or face-swapped faces (`Deepfake`) in images, videos, or webcam feeds.

*(Research context: The team also surveyed proactive defense literature such as NullSwap [Wang et al., ICCV 2025]; the implemented software prototype delivers an end-to-end real-time detection pipeline and interactive Streamlit web dashboard.)*

## System Architecture

```text
[Image / Video / Webcam (Browser & Live OpenCV)]
                     │
                     ▼
   M3: Video I/O (Scene / Uniform / FPS)
     + YuNet Multi-Face Detection
     + Dual-Scale Crop: 2.7× context (Stage 1) & 1.2× face (Stage 2)
            │                          │
            ▼                          ▼
  Stage 1 (M4): Anti-Spoofing  Stage 2 (M5): Deepfake AI
  • Level 1: MiniFASNetV2      • SigLIP2 / ViT / Ensemble
  • Level 2: 2D FFT + Glare    • Batched CPU/GPU Inference
  → Real / Printed / On-screen → Deepfake probability [0..1]
            │                          │
            └────────────┬─────────────┘
                         ▼
   M1: Multi-Frame & Multi-Face Aggregator (DD6 / DD7)
   Precedence: No_Face → Printed / On-screen → Deepfake → Real
                         │
                         ▼
   M6: Streamlit Cyber-Forensic Studio
   • Verdict Card & Threshold-Marked Probability Bars
   • Storyboard Scene Gallery & Per-Person Face Magnifier (#1, #2...)
   • Altair Timeline Chart & RAM-Only JSON / CSV Export (DD8)
```

## Quickstart (Terminal)

```powershell
# 1. Enter project directory & install dependencies
cd "AntiDeepFake-Face"
pip install -r requirements.txt

# 2. (Optional) Prefetch YuNet and HuggingFace weights for offline runs
python scripts/download_yunet.py
python scripts/download_m5_models.py

# 3. Launch Web App
streamlit run app/app.py

# 4. Run test suite
pytest -v
```

## Technology Stack

| Technology | Purpose |
|---|---|
| Python 3 | Core programming language |
| OpenCV (YuNet + Image Processing) | Video/webcam I/O, multi-face detection, FFT/Glare Level 2 cues |
| PyTorch & Transformers | `MiniFASNetV2` (Stage 1) and `SigLIP2` / `ViT` (Stage 2) inference |
| NumPy & Pandas | Tensor preprocessing, multi-frame aggregation, report export |
| Streamlit & Altair | Web UI, Storyboard Scene Gallery, Live Webcam HUD, interactive charts |
| Pytest | Unit, integration, and headless `AppTest` verification |

## Team Members

| Member | Responsibility |
|---|---|
| Member 1 | System Integration & Decision Aggregator |
| Member 2 | Data & Evaluation |
| Member 3 | Frame Extraction & Face Detection |
| Member 4 | Anti-Spoofing (Printed, On-screen) |
| Member 5 | Deepfake Detector |
| Member 6 | UI/UX Design, Testing & Reporting |

## References
- Minivision AI, *Silent-Face-Anti-Spoofing* (`2.7_80x80_MiniFASNetV2`).
- OpenCV Zoo, *YuNet Face Detection (`face_detection_yunet_2023mar.onnx`)*.
- HuggingFace Models: `king1oo1/deepfake-model`, `prithivMLmods/Deep-Fake-Detector-v2-Model`.
- Wang et al., *NullSwap: Proactive Identity Cloaking Against Deepfake Face Swapping*, ICCV 2025 (https://arxiv.org/html/2503.18678v1).
