# AntiDeepFake-Face

NullSwap-inspired Face Identity Protection using Image Processing, Computer Vision and AI.

## Folder Tree

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
| Member 1 | Project Leader & System Integration |
| Member 2 | Image Processing |
| Member 3 | Face Detection |
| Member 4 | Deep Learning & Feature Extraction|
| Member 5 | Identity Protection |
| Member 6 | Web Application, Evaluation & Testing |

## Project Scope
### In Scope
- Static facial image processing.
- Face detection and cropping.
- Facial feature extraction.
- Prototype identity protection.
- Image comparison and evaluation.
- Web-based demonstration.
### Out of Scope
- Real-time webcam processing.
- Video DeepFake protection.
- Full reproduction of the original NullSwap framework.
- Large-scale model training from scratch.

## Expected Deliverables:
Project documentation (Define & Design).
Image processing modules.
Face identity protection prototype.
Web application demonstration.
Testing and evaluation results.
Final project report.

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
User Image
    ↓
Web Interface
    ↓
Image Preprocessing
    ↓
Face Detection
    ↓
Feature Extraction
    ↓
Identity Protection
    ↓
Evaluation
    ↓
Protected Image Output
```

## Research Reference
Wang et al., NullSwap: Proactive Identity Cloaking Against Deepfake Face Swapping, ICCV 2025.
Research paper: https://arxiv.org/html/2503.18678v1
