import time
import os
import cv2
import numpy as np
import torch
from facenet_pytorch import MTCNN

def create_synthetic_face_image(width=640, height=480):
    """Tạo ảnh nhân tạo có vẽ khuôn mặt đơn giản để benchmark."""
    img = np.full((height, width, 3), 200, dtype=np.uint8)
    # Vẽ mặt tròn đơn giản
    cv2.circle(img, (width // 2, height // 2), 100, (255, 220, 190), -1)
    # Vẽ 2 mắt
    cv2.circle(img, (width // 2 - 35, height // 2 - 20), 12, (50, 50, 50), -1)
    cv2.circle(img, (width // 2 + 35, height // 2 - 20), 12, (50, 50, 50), -1)
    # Vẽ miệng
    cv2.ellipse(img, (width // 2, height // 2 + 40), (40, 20), 0, 0, 180, (50, 50, 200), 4)
    return img

def benchmark():
    img_bgr = create_synthetic_face_image()
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w, _ = img_bgr.shape
    num_runs = 50

    print("==================================================")
    print("BENCHMARK SO SÁNH MTCNN VÀ YUNET TRÊN CPU")
    print("==================================================")

    # 1. Benchmark YuNet
    yunet_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "face_detection_yunet_2023mar.onnx")
    if not os.path.exists(yunet_path):
        from scripts.download_yunet import download_model
        yunet_path = download_model()

    yunet = cv2.FaceDetectorYN.create(
        model=yunet_path,
        config="",
        input_size=(w, h),
        score_threshold=0.5,
        nms_threshold=0.3,
        top_k=5000
    )

    # Warmup
    for _ in range(5):
        yunet.setInputSize((w, h))
        yunet.detect(img_bgr)

    yunet_times = []
    for _ in range(num_runs):
        t0 = time.perf_counter()
        yunet.setInputSize((w, h))
        yunet.detect(img_bgr)
        t1 = time.perf_counter()
        yunet_times.append((t1 - t0) * 1000)

    avg_yunet_ms = np.mean(yunet_times)
    fps_yunet = 1000.0 / avg_yunet_ms
    yunet_size_kb = os.path.getsize(yunet_path) / 1024

    print(f"\n[YuNet - OpenCV DNN]")
    print(f"- Kích thước model : {yunet_size_kb:.1f} KB")
    print(f"- Thời gian TB     : {avg_yunet_ms:.2f} ms / frame")
    print(f"- Tốc độ ước tính  : {fps_yunet:.1f} FPS (CPU)")

    # 2. Benchmark MTCNN
    device = torch.device('cpu')
    mtcnn = MTCNN(keep_all=True, device=device)

    # Warmup
    for _ in range(3):
        mtcnn.detect(img_rgb)

    mtcnn_times = []
    for _ in range(num_runs):
        t0 = time.perf_counter()
        mtcnn.detect(img_rgb)
        t1 = time.perf_counter()
        mtcnn_times.append((t1 - t0) * 1000)

    avg_mtcnn_ms = np.mean(mtcnn_times)
    fps_mtcnn = 1000.0 / avg_mtcnn_ms

    print(f"\n[MTCNN - facenet-pytorch]")
    print(f"- Kích thước model : ~2.5 MB (P-Net, R-Net, O-Net)")
    print(f"- Thời gian TB     : {avg_mtcnn_ms:.2f} ms / frame")
    print(f"- Tốc độ ước tính  : {fps_mtcnn:.1f} FPS (CPU)")

    print("\n==================================================")
    print(f"KẾT LUẬN: YuNet nhanh gấp {avg_mtcnn_ms / avg_yunet_ms:.1f} lần so với MTCNN trên CPU!")
    print("==================================================")

    return {
        "yunet": {"latency_ms": avg_yunet_ms, "fps": fps_yunet, "size_kb": yunet_size_kb},
        "mtcnn": {"latency_ms": avg_mtcnn_ms, "fps": fps_mtcnn}
    }

if __name__ == "__main__":
    benchmark()
