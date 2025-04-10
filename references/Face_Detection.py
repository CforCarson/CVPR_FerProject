import os
from pathlib import Path
from PIL import Image
import torch
from huggingface_hub import hf_hub_download
from ultralytics import YOLO
from ultralytics.utils import LOGGER
from supervision import Detections
import face_recognition
from tqdm import tqdm
from torchvision import models, transforms
import torch.nn as nn

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 初始化模型
LOGGER.setLevel("ERROR")  # 关闭 YOLO 日志
yolo = YOLO(
    hf_hub_download(repo_id="arnabdhar/YOLOv8-Face-Detection", filename="model.pt")
)


# 加载分类模型
def load_classifier_model(path: str):
    model = models.resnet18()
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(nn.Dropout(0.8), nn.Linear(num_ftrs, 2))
    model.load_state_dict(torch.load(path, map_location=device))
    return model.to(device).eval()


classifier_model = load_classifier_model("./face_classifier.pth")

# 图像预处理
transform = transforms.Compose(
    [
        transforms.Grayscale(),
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ]
)

folder_path = Path(r"generated_images_by_class")


# 工具方法
def get_image_files_recursive(folder: Path):
    return list(folder.rglob("*.jpg"))


def open_image(image_path: Path):
    try:
        return Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"⚠️ 无法读取图片 `{image_path.name}`，错误：{e}")
        return None


# 检测函数
def detect_faces(image, method: str):
    if method == "YOLO":
        output = yolo(image)
        results = Detections.from_ultralytics(output[0])
        return len(results)
    elif method == "face_recognition":
        image_np = (
            image
            if isinstance(image, (list, tuple))
            else face_recognition.load_image_file(image)
        )
        face_locations = face_recognition.face_locations(image_np)
        return len(face_locations)
    raise ValueError(f"Unsupported detection method: {method}")


# 主处理流程
def process_images(folder: Path, methods: list):
    for method in methods:
        print(f"\n📌 使用 {method} 检测中...")
        for image_path in tqdm(get_image_files_recursive(folder)):
            img = (
                str(image_path)
                if method == "face_recognition"
                else open_image(image_path)
            )
            if img is None:
                continue
            try:
                num_faces = detect_faces(img, method)
                if num_faces == 0:
                    os.remove(image_path)
            except Exception as e:
                print(f"⚠️ 检测失败 `{image_path.name}`，错误: {e}")


# 执行所有流程
process_images(folder_path, ["face_recognition", "YOLO"])
