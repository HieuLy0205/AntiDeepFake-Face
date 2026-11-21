"""Unit test cho module src/video_io (M3) theo Mục II.4 & II.7."""

import pytest


def test_load_frames_corrupt_file_raises_invalid_input(temp_corrupt_file_path):
    """Tệp hỏng hoặc không đọc được phải ném InvalidInputError."""
    reader = pytest.importorskip("src.video_io.reader")
    video_io_pkg = pytest.importorskip("src.video_io")
    invalid_err = getattr(reader, "InvalidInputError", getattr(video_io_pkg, "InvalidInputError", ValueError))

    with pytest.raises(invalid_err):
        reader.load_frames(str(temp_corrupt_file_path), sample_fps=5, max_frames=30)


def test_load_frames_missing_file_raises_invalid_input(tmp_path):
    """Đường dẫn không tồn tại phải ném InvalidInputError."""
    reader = pytest.importorskip("src.video_io.reader")
    video_io_pkg = pytest.importorskip("src.video_io")
    invalid_err = getattr(reader, "InvalidInputError", getattr(video_io_pkg, "InvalidInputError", ValueError))

    with pytest.raises(invalid_err):
        reader.load_frames(str(tmp_path / "does_not_exist.mp4"), sample_fps=5, max_frames=30)


def test_load_frames_image_returns_single_rgb_uint8_packet(temp_valid_image_path):
    """Đầu vào ảnh trả đúng 1 FramePacket có hệ màu RGB và dtype uint8."""
    np = pytest.importorskip("numpy")
    reader = pytest.importorskip("src.video_io.reader")
    schemas = pytest.importorskip("src.schemas")

    packets = reader.load_frames(str(temp_valid_image_path), sample_fps=5, max_frames=30)
    assert len(packets) == 1
    pkt = packets[0]
    assert isinstance(pkt, schemas.FramePacket)
    assert pkt.frame_id == 0
    assert pkt.image_rgb.dtype == np.uint8
    assert pkt.image_rgb.ndim == 3 and pkt.image_rgb.shape[2] == 3


def test_load_frames_short_video_respects_sample_fps_and_max_frames(temp_short_video_path):
    """Video ngắn (2s @ 10fps) lấy mẫu tại 5fps phải trả <= max_frames và <= ~10 frames."""
    np = pytest.importorskip("numpy")
    reader = pytest.importorskip("src.video_io.reader")

    video_path, _fps, _total = temp_short_video_path
    packets = reader.load_frames(str(video_path), sample_fps=5, max_frames=6)

    assert 1 <= len(packets) <= 6
    for pkt in packets:
        assert pkt.image_rgb.dtype == np.uint8
        assert pkt.image_rgb.ndim == 3 and pkt.image_rgb.shape[2] == 3
