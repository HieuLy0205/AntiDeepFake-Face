# AntiDeepFake-Face

NullSwap-inspired Face Identity Protection using Image Processing, Computer Vision and AI.

## Folder Tree

```text
AntiDeepFake-Face/
├── README.md
├── .gitignore
├── requirements.txt
├── configs/              # Configuration parameters (k, N, thresholds, seed)
├── docs/
│   ├── requirements/     # Part I: Requirements Definition
│   ├── design/           # Part II: System Design
│   ├── research/         # Model and Dataset Survey
│   └── report/
├── src/
│   ├── video_io/         # Video input and frame extraction
│   ├── face_detection/
│   ├── anti_spoof/       # Phase 1: Anti-Spoofing Detection
│   ├── deepfake/         # Phase 2: Deepfake Detection
│   ├── decision/
│   ├── evaluation/
│   ├── utils/
│   └── pipeline.py
├── app/
│   ├── app.py
│   ├── components/
│   └── assets/
├── models/              # Model weights (sources documented in README)
├── data/
│   ├── raw/
│   ├── test/            # Real, Printed, On-screen, Deepfake
│   └── metadata.csv
├── scripts/
```

## Folder Rules

| Folder | Purpose |
|----------|----------|
| `docs/` | Define, design, research papers, testing documents, and reports |
| `src/` | Image processing, face detection, feature extraction, and identity protection algorithms |
| `app/` | Web application for demonstrating the system |
| `models/` | Pretrained models, checkpoints, and model configuration files |
| `data/` | Sample datasets and testing images |
| `tests/` | Unit tests and integration tests |
| `notebooks/` | Experiments and algorithm prototyping |
| `project_management/` | Task assignments, schedules, and meeting minutes |

## Project Overview
AntiDeepFake-Face is an academic project developed as part of the Image Processing and Computer Vision course.
With the rapid growth of DeepFake technologies, protecting personal facial identity has become an increasingly important challenge. This project focuses on exploring proactive approaches to facial identity protection against DeepFake face-swapping attacks.
Inspired by the research paper *NullSwap: Proactive Identity Cloaking Against Deepfake Face Swapping*, the project combines image processing techniques and deep learning methods to build a prototype capable of protecting facial identity while preserving the visual quality of the original image.
## Research Background
Recent advances in DeepFake face-swapping technology make it possible to transfer a person's facial identity onto another image with high realism. While these technologies have many legitimate applications, they also raise concerns about privacy, identity misuse, and the creation of misleading content.
Rather than detecting DeepFakes after they have been generated, this project investigates a proactive defense strategy inspired by *NullSwap*. The goal is to explore how facial images can be modified in a controlled manner to reduce their vulnerability to future face-swapping attacks while maintaining their natural appearance.
The concepts and ideas presented in the NullSwap paper serve as the primary research foundation for this project.

## Project Objectives
- Develop a web application for facial image processing.
- Detect and preprocess facial regions.
- Explore facial feature extraction using deep learning.
- Implement a NullSwap-inspired identity protection prototype.
- Compare original and protected images.
- Evaluate image quality and processing performance.

## Team Members
This project is developed by a team of six students:

| Member | Responsibility|
|----------|----------|
| Member 1 | System Integration |
| Member 2 | Data & Evaluation |
| Member 3 | Frame Extraction & Face Detection |
| Member 4 | Anti-spoof (Printed, On-screen) |
| Member 5 | Deepfake Detector |
| Member 6 | UI/UX Designer, Tester & Report Writer |

##Technology Stack
| Technology | Purpose |
| Python | Main programming language |
| OpenCV | Image processing |
| PyTorch | Deep learning |
| MTCNN | Face detection |
| NumPy | Numerical operations |
| Streamlit | Web application |
| Matplotlib | Visualization |

## System Architecture
The proposed system consists of the following modules:
```text

```

## Research Reference
Wang et al., NullSwap: Proactive Identity Cloaking Against Deepfake Face Swapping, ICCV 2025.
Research paper: https://arxiv.org/html/2503.18678v1
