import os
import sys
import time
import tempfile
import cv2
import numpy as np

# Thêm thư mục gốc vào PYTHONPATH
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.video_io.reader import load_frames, load_single_frame
from src.face_detection.detector import FaceDetector

FIXTURE_FACE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tests", "fixtures", "sample_face.jpg")

def generate_test_video(output_path: str, num_frames: int = 60, fps: float = 30.0):
    """Tạo video giả lập 60 frames (2 giây) từ ảnh khuôn mặt chuẩn."""
    if not os.path.exists(FIXTURE_FACE):
        raise FileNotFoundError(f"Không tìm thấy fixture {FIXTURE_FACE}")
    
    face_bgr = cv2.imread(FIXTURE_FACE)
    h, w, _ = face_bgr.shape
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    for _ in range(num_frames):
        out.write(face_bgr)
    out.release()

def run_demo():
    print("==================================================================")
    print("DEMO END-TO-END MODULE M3: VIDEO_IO & FACE_DETECTION (KHOA PHẠM)")
    print("==================================================================")

    # 1. Khởi tạo model YuNet
    detector = FaceDetector()
    print("[1] Khởi tạo FaceDetector (YuNet ONNX): THÀNH CÔNG")

    # 2. Tạo video test giả lập
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        test_video_path = tmp.name
    
    try:
        generate_test_video(test_video_path, num_frames=60, fps=30.0)
        print(f"[2] Tạo video giả lập (60 frames @ 30 FPS): THÀNH CÔNG")

        # 3. Đo đạc thời gian video_io (lấy mẫu 10 frame @ sample_fps=5)
        t_io_start = time.perf_counter()
        packets = load_frames(test_video_path, sample_fps=5, max_frames=10)
        t_io_end = time.perf_counter()
        io_latency_ms = (t_io_end - t_io_start) * 1000
        print(f"[3] Đọc {len(packets)} frames qua load_frames(): {io_latency_ms:.2f} ms")

        # 4. Đo đạc thời gian face_detection trên từng frame
        print("[4] Bắt đầu phát hiện mặt và cắt chuẩn (224, 224, 3) RGB uint8...")
        det_times = []
        crop_results = []
        for pkt in packets:
            t0 = time.perf_counter()
            res = detector.detect_and_crop(pkt, target_size=(224, 224), scale_ratio=1.3)
            t1 = time.perf_counter()
            det_times.append((t1 - t0) * 1000)
            crop_results.append(res)

        avg_det_ms = np.mean(det_times)
        fps_det = 1000.0 / avg_det_ms
        total_fps_m3 = len(packets) / ((t_io_end - t_io_start) + sum(det_times) / 1000.0)

        # 5. Kiểm tra tính hợp lệ của kết quả
        all_valid = all(
            r.has_face and r.face_rgb.shape == (224, 224, 3) and r.face_rgb.dtype == np.uint8
            for r in crop_results
        )

        print("------------------------------------------------------------------")
        print("KẾT QUẢ ĐO ĐẠC HIỆU NĂNG TRÊN CPU:")
        print(f"- Số frame đã xử lý: {len(crop_results)}")
        print(f"- Thời gian phát hiện mặt TB: {avg_det_ms:.2f} ms / frame")
        print(f"- Tốc độ Face Detection CPU : {fps_det:.1f} FPS")
        print(f"- Tốc độ End-to-End M3 (I/O + Detection): {total_fps_m3:.1f} FPS")
        print(f"- Kiểm tra chuẩn đầu ra (224x224 RGB uint8): {'ĐẠT (PASS)' if all_valid else 'KHÔNG ĐẠT'}")
        print(f"- Yêu cầu đồ án (≥ 5 FPS trên CPU): {'VƯỢT CHỈ TIÊU (PASS)' if total_fps_m3 >= 5.0 else 'FAIL'}")
        print("==================================================================")
        
    finally:
        if os.path.exists(test_video_path):
            os.remove(test_video_path)

if __name__ == "__main__":
    run_demo()
