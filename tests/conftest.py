"""Fixtures dùng chung cho bộ kiểm thử pytest của AntiDeepFake-Face (M6)."""

from pathlib import Path
import sys
from typing import Tuple
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def sample_rgb_frame():
    """Tạo ảnh giả lập RGB uint8 kích thước (480, 640, 3) trong bộ nhớ."""
    np = pytest.importorskip("numpy")
    rng = np.random.default_rng(42)
    return rng.integers(0, 255, size=(480, 640, 3), dtype=np.uint8)


@pytest.fixture
def sample_face_crop_224():
    """Tạo ảnh crop khuôn mặt giả lập chuẩn (224, 224, 3) RGB uint8."""
    np = pytest.importorskip("numpy")
    rng = np.random.default_rng(7)
    return rng.integers(0, 255, size=(224, 224, 3), dtype=np.uint8)


@pytest.fixture
def temp_valid_image_path(tmp_path: Path, sample_rgb_frame) -> Path:
    """Tạo tệp ảnh JPG hợp lệ tạm thời trong tmp_path (không lưu dữ liệu thật)."""
    Image = pytest.importorskip("PIL.Image")
    img_path = tmp_path / "sample_face.jpg"
    Image.fromarray(sample_rgb_frame, mode="RGB").save(img_path, format="JPEG")
    return img_path


@pytest.fixture
def temp_corrupt_file_path(tmp_path: Path) -> Path:
    """Tạo tệp hỏng giả mạo đuôi .mp4 để kiểm tra lỗi InvalidInputError."""
    bad_path = tmp_path / "corrupt_video.mp4"
    bad_path.write_bytes(b"not-a-valid-video-header-0000")
    return bad_path


@pytest.fixture
def temp_short_video_path(tmp_path: Path) -> Tuple[Path, int, int]:
    """Tạo video MP4 ngắn tổng hợp (2 giây, 10 FPS = 20 frames) bằng OpenCV."""
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")

    video_path = tmp_path / "short_synthetic.mp4"
    fps = 10
    total_frames = 20
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_path), fourcc, float(fps), (320, 240))
    if not writer.isOpened():
        pytest.skip("OpenCV không hỗ trợ ghi mp4v trong môi trường test hiện tại.")

    try:
        for i in range(total_frames):
            frame_bgr = np.full((240, 320, 3), fill_value=(i * 10) % 255, dtype=np.uint8)
            writer.write(frame_bgr)
    finally:
        writer.release()

    return video_path, fps, total_frames
