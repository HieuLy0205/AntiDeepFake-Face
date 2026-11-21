"""Hằng số chuẩn hóa theo Quy ước Interface (Mục II.4) và Thông báo lỗi UI (Mục II.6)."""

from typing import Dict, Tuple

# 5 nhãn chuẩn hóa (Quy ước mục 5.3 - tuyệt đối không thay đổi chính tả)
LABEL_REAL = "Real"
LABEL_PRINTED = "Printed"
LABEL_ONSCREEN = "On-screen"
LABEL_DEEPFAKE = "Deepfake"
LABEL_NO_FACE = "No_Face"

VALID_FINAL_LABELS: Tuple[str, ...] = (
    LABEL_REAL,
    LABEL_PRINTED,
    LABEL_ONSCREEN,
    LABEL_DEEPFAKE,
    LABEL_NO_FACE,
)

# 4 lớp xác suất tổng hợp trong FinalDecision.aggregated_probs
AGGREGATED_CLASSES: Tuple[str, ...] = (
    LABEL_REAL,
    LABEL_PRINTED,
    LABEL_ONSCREEN,
    LABEL_DEEPFAKE,
)

# 3 lớp của nhánh Anti-Spoof (M4)
SPOOF_CLASSES: Tuple[str, ...] = (
    LABEL_REAL,
    LABEL_PRINTED,
    LABEL_ONSCREEN,
)

# Tên hiển thị tiếng Việt thân thiện trên UI (giữ nguyên key gốc khi giao tiếp)
LABEL_DISPLAY_VI: Dict[str, str] = {
    LABEL_REAL: "Người thật (Real)",
    LABEL_PRINTED: "Ảnh in giấy (Printed)",
    LABEL_ONSCREEN: "Phát lại qua màn hình (On-screen)",
    LABEL_DEEPFAKE: "Khuôn mặt tổng hợp (Deepfake)",
    LABEL_NO_FACE: "Không phát hiện khuôn mặt (No_Face)",
}

LABEL_BADGE_COLORS: Dict[str, str] = {
    LABEL_REAL: "#15803d",
    LABEL_PRINTED: "#b45309",
    LABEL_ONSCREEN: "#c2410c",
    LABEL_DEEPFAKE: "#b91c1c",
    LABEL_NO_FACE: "#475569",
}

# Định dạng tệp cho phép (II.3 & II.6)
ALLOWED_IMAGE_EXTENSIONS: Tuple[str, ...] = (".jpg", ".jpeg", ".png")
ALLOWED_VIDEO_EXTENSIONS: Tuple[str, ...] = (".mp4", ".avi", ".mov", ".mkv")

# Tham số mặc định theo DD1 & thiết kế
DEFAULT_SAMPLE_FPS = 5
DEFAULT_MAX_FRAMES = 30
DEFAULT_N_FRAMES = 10
DEFAULT_TAU_SPOOF = 0.60
DEFAULT_TAU_DEEPFAKE = 0.60
DEFAULT_UNCERTAIN_MARGIN = 0.08

# Giới hạn tải lên để đảm bảo hiệu năng chạy laptop (II.6)
MAX_IMAGE_SIZE_MB = 15
MAX_VIDEO_SIZE_MB = 50
MAX_VIDEO_DURATION_SEC = 30

# Banner giới hạn luôn hiển thị (Mục II.6)
ACADEMIC_LIMITATION_BANNER = (
    "Lưu ý: Đây là nguyên mẫu thử nghiệm; độ chính xác phụ thuộc vào chất lượng camera, "
    "điều kiện ánh sáng và góc quay. Hệ thống không lưu trữ ảnh hoặc video của bạn."
)

DD6_PRECEDENCE_NOTE = (
    "Mẹo sử dụng: Nếu video Deepfake được phát trên màn hình rồi quay lại bằng camera khác, "
    "hệ thống sẽ ưu tiên cảnh báo theo dấu hiệu phát lại qua màn hình (On-screen)."
)

# Hệ thống thông báo lỗi UI chuẩn hóa (Mục II.6 - giữ nguyên văn)
UI_ERRORS: Dict[str, str] = {
    "invalid_file": "Tệp không hợp lệ. Vui lòng chọn ảnh JPG, PNG hoặc video hợp lệ.",
    "video_too_large_or_long": "Video vượt giới hạn cho phép. Vui lòng chọn video ngắn hơn.",
    "no_face": "Không tìm thấy khuôn mặt. Hãy thử ảnh hoặc video rõ mặt hơn.",
    "multi_face": "Phát hiện nhiều khuôn mặt. Hệ thống phân tích khuôn mặt lớn nhất.",
    "webcam_unavailable": (
        "Không truy cập được webcam. Hãy kiểm tra quyền truy cập hoặc dùng chế độ upload."
    ),
    "uncertain_score": "Kết quả chưa chắc chắn. Hãy thử thêm frame hoặc cải thiện ánh sáng.",
}
