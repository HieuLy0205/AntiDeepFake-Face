import cv2
import torch
import numpy as np

from pathlib import Path
from functools import lru_cache
from .MiniFASNet import MiniFASNetV2


# Đường dẫn trọng số pretrained
MODEL_PATH = (
    Path(__file__).resolve().parents[2]
    / "models"
    / "anti_spoof"
    / "2.7_80x80_MiniFASNetV2.pth"
)

# Tự động sử dụng GPU nếu có, nếu không dùng CPU
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# Hàm nạp model pretrained và lưu cache
# Model chỉ được nạp một lần trong quá trình sử dụng
@lru_cache(maxsize=1)
def load_model():

    # Kiểm tra file trọng số
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Không tìm thấy model: {MODEL_PATH}"
        )

    # Khởi tạo MiniFASNetV2 cho ảnh 80x80
    model = MiniFASNetV2(
        conv6_kernel=(5, 5)
    ).to(DEVICE)

    # Đọc trọng số pretrained
    state_dict = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True
    )

    # Loại bỏ tiền tố module. nếu có
    state_dict = {
        key.removeprefix("module."): value
        for key, value in state_dict.items()
    }

    # Điều chỉnh thứ tự kênh trọng số BGR sang RGB
    # Chỉ điều chỉnh lớp convolution đầu tiên
    # Không thay đổi hệ màu ảnh đầu vào
    state_dict["conv1.conv.weight"] = (
        state_dict["conv1.conv.weight"][:, [2, 1, 0], :, :]
        .contiguous()
    )

    # Nạp trọng số vào model
    model.load_state_dict(state_dict)

    # Chuyển sang chế độ dự đoán
    model.eval()

    return model


# Hàm tiền xử lý ảnh RGB nhận từ M3
def preprocess_image(crop_rgb: np.ndarray):

    # Kiểm tra kiểu dữ liệu đầu vào
    if not isinstance(crop_rgb, np.ndarray):
        raise ValueError("Ảnh phải là numpy array")

    # Kiểm tra kích thước theo quy ước nhóm
    if crop_rgb.shape != (224, 224, 3):
        raise ValueError("Ảnh phải có kích thước 224x224x3")

    # Kiểm tra kiểu pixel
    if crop_rgb.dtype != np.uint8:
        raise ValueError("Ảnh phải có kiểu uint8")

    # Resize về kích thước yêu cầu của MiniFASNet
    image = cv2.resize(crop_rgb, (80, 80))

    # Giữ nguyên hệ màu RGB theo quy ước nhóm

    # Chuyển sang float32, giữ khoảng pixel 0-255
    image = image.astype(np.float32)

    # Chuyển từ HWC sang CHW
    image = np.transpose(image, (2, 0, 1))

    # Chuyển ảnh thành tensor
    tensor = torch.from_numpy(image)

    # Thêm chiều batch và chuyển sang CPU/GPU
    tensor = tensor.unsqueeze(0).to(DEVICE)

    return tensor


# Hàm dự đoán Level 1: Real và Spoof
# Trả về tuple gồm p_real và p_spoof
def predict_level1(
    crop_rgb: np.ndarray
) -> tuple[float, float]:

    # Tiền xử lý ảnh
    tensor = preprocess_image(crop_rgb)

    # Lấy model đã được cache
    model = load_model()

    # Tắt tính gradient để tăng hiệu năng
    with torch.inference_mode():

        # Chạy dự đoán
        output = model(tensor)

        # Chuyển kết quả thành xác suất
        probs = torch.softmax(output, dim=1)

    probs = probs[0].cpu().numpy()

    # Nhãn 1 là khuôn mặt thật
    p_real = float(probs[1])

    # Nhãn 0 và 2 thuộc nhóm giả mạo
    p_spoof = float(probs[0] + probs[2])

    # Đảm bảo tổng xác suất bằng 1
    total = p_real + p_spoof

    return p_real / total, p_spoof / total