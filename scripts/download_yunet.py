import os
import urllib.request

YUNET_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "face_detection_yunet_2023mar.onnx")

def download_model():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if os.path.exists(OUTPUT_PATH):
        print(f"Model already exists at: {OUTPUT_PATH}")
        return OUTPUT_PATH
    
    print(f"Downloading YuNet model to {OUTPUT_PATH}...")
    req = urllib.request.Request(YUNET_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp, open(OUTPUT_PATH, "wb") as f:
        f.write(resp.read())
    print("Download completed successfully!")
    return OUTPUT_PATH

if __name__ == "__main__":
    download_model()
