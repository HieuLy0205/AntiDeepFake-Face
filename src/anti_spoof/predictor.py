"""Cầu nối giữa crop của M3, model M4 và hợp đồng pipeline hiện tại."""

from time import perf_counter

from src.schemas import FaceCropResult, SpoofScore

from .model import AntiSpoofModel

_model = AntiSpoofModel()
# Dùng chung một instance để không tạo lại wrapper ở mỗi frame.

def predict_spoof(crop: FaceCropResult) -> SpoofScore:
    """Nhận FaceCropResult từ M3, trả SpoofScore mà pipeline đang cần.

    M3 giữ ảnh trong crop.face_rgb; adapter đổi key chữ thường của M4 sang
    định dạng chữ hoa hiện được dùng trong schemas.py và pipeline.py.
    """
    if not isinstance(crop, FaceCropResult):
        raise TypeError("crop must be a FaceCropResult.")
    if not crop.has_face or crop.face_rgb is None:
        raise ValueError("crop must contain a detected face image.")

    # Đo thời gian suy luận để pipeline ghi vào latency_ms.
    started_at = perf_counter()
    probabilities = _model.predict(crop.face_rgb)
    latency_ms = (perf_counter() - started_at) * 1000.0

    # Giữ frame_id và tên key tương thích với SpoofScore hiện có.
    return SpoofScore(
        frame_id=crop.frame_id,
        probs={
            "Real": probabilities["real"],
            "Printed": probabilities["printed"],
            "On-screen": probabilities["onscreen"],
        },
        latency_ms=latency_ms,
    )
