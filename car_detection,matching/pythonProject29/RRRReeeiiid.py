'''
import os
import csv
import cv2
import numpy as np
import torch
import sys
import json

# -------------------- ReID 모델 세팅 --------------------
reid_path = os.path.join(os.getcwd(), "VehicleX", "Re-ID")
sys.path.append(reid_path)
from reid import models

# 모델 생성 (CPU용)
reid_model = models.create('ide', num_features=256, norm=True, num_classes=1000, arch='resnet50')
reid_model.eval()  # eval 모드

# 폴더 세팅
images_folder = "images_"
detect_csv_folder = "detect_csv"
reid_feature_folder = "reid_features"
json_folder = "output_json"

os.makedirs(reid_feature_folder, exist_ok=True)
os.makedirs(detect_csv_folder, exist_ok=True)


# -------------------- 유틸 함수 --------------------
def extract_feature(crop):
    """ReID feature 추출 (CPU, RGB 변환 + 정규화 + resize 포함)"""
    if crop is None or crop.size == 0:
        return np.zeros((256,), dtype=np.float32)  # 빈 이미지 처리

    # 모델 입력에 맞게 resize (예: 256x128)
    crop = cv2.resize(crop, (128, 256))

    # BGR → RGB
    crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

    # 0~1로 정규화
    crop = crop.astype(np.float32) / 255.0

    # 채널별 정규화
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    crop = (crop - mean) / std

    # Tensor 변환
    tensor = torch.tensor(crop).permute(2, 0, 1).unsqueeze(0).float()  # (1, C, H, W)

    with torch.no_grad():
        output = reid_model(tensor)
        feat = output[0] if isinstance(output, tuple) else output
        return feat.cpu().numpy().flatten()


def save_detect_csv(image_file, bboxes):
    """탐지 CSV 저장"""
    img_name = os.path.splitext(image_file)[0]
    csv_path = os.path.join(detect_csv_folder, img_name + "_detect.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "pair_id", "x1", "y1", "x2", "y2", "width", "height", "area"])
        for idx, bbox in enumerate(bboxes, start=1):
            x1, y1, x2, y2 = bbox
            w, h = x2 - x1, y2 - y1
            area = w * h
            writer.writerow([img_name, idx, x1, y1, x2, y2, w, h, area])


# -------------------- JSON → CSV 변환 --------------------
json_files = [f for f in os.listdir(json_folder) if f.endswith(".json")]
for json_file in json_files:
    json_path = os.path.join(json_folder, json_file)
    with open(json_path, "r") as f:
        data = json.load(f)
    bboxes = [d["bbox"] for d in data.get("detections", [])]
    if not bboxes:
        continue
    save_detect_csv(json_file, bboxes)
    print(f"[CSV 생성 완료] {json_file}")

# -------------------- CSV → ReID feature 추출 --------------------
csv_files = [f for f in os.listdir(detect_csv_folder) if f.endswith("_detect.csv")]
for csv_file in csv_files:
    csv_path = os.path.join(detect_csv_folder, csv_file)
    features_per_image = []

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        image_name = None
        img = None
        for row in reader:
            if image_name is None:
                image_name = row["image"]
                img_path = os.path.join(images_folder, image_name)
                if not os.path.exists(img_path):
                    print(f"[경고] 이미지 파일 없음: {image_name}")
                    break
                img = cv2.imread(img_path)

            x1 = int(row["x1"])
            y1 = int(row["y1"])
            x2 = int(row["x2"])
            y2 = int(row["y2"])

            crop = img[y1:y2, x1:x2]
            feat = extract_feature(crop)
            features_per_image.append({
                "image": image_name,
                "bbox": [x1, y1, x2, y2],
                "feature": feat
            })

    # npy 저장
    base_name = os.path.splitext(csv_file)[0].replace("_detect", "")
    save_path = os.path.join(reid_feature_folder, base_name + "_features.npy")
    np.save(save_path, features_per_image)
    print(f"[ReID feature 완료] {base_name} → {save_path}, bbox 수: {len(features_per_image)}")
'''

import os
import csv
import cv2
import numpy as np
import torch
import sys
import json
import logging

# -------------------- fast-reid 모델 세팅 --------------------
# fast-reid 폴더 경로를 sys.path에 추가 (ModuleNotFoundError 해결)
# Pycharm의 "Source Root" 설정이 터미널에 반영되지 않을 경우를 대비
fast_reid_path = "C:/Users/82103/PycharmProjects/pythonProject29/fast-reid"
if fast_reid_path not in sys.path:
    sys.path.append(fast_reid_path)

from fastreid.config import get_cfg
from fastreid.modeling import build_model
from fastreid.utils.checkpoint import Checkpointer

# 설정 파일 로드
cfg = get_cfg()
cfg.merge_from_file(os.path.join(fast_reid_path, "configs", "VehicleID", "bagtricks_R50-ibn.yml"))
# 이 줄을 추가하여 모델을 CPU에서 실행하도록 설정합니다.
cfg.MODEL.DEVICE = "cpu"
cfg.freeze()

# 모델 생성
reid_model = build_model(cfg)

# 모델 가중치 로드
Checkpointer(reid_model).load(os.path.join(fast_reid_path, "checkpoints", "vehicleid_bot_R50-ibn.pth"))
reid_model.eval()  # eval 모드

# 폴더 세팅
images_folder = "images"
detect_csv_folder = "detect_csv"
reid_feature_folder = "reid_features"
json_folder = "output_json"

os.makedirs(reid_feature_folder, exist_ok=True)
os.makedirs(detect_csv_folder, exist_ok=True)


# -------------------- 유틸 함수 --------------------
def extract_feature(crop):
    """ReID feature 추출 (CPU, RGB 변환 + 정규화 + resize 포함)"""
    if crop is None or crop.size == 0:
        return np.zeros((256,), dtype=np.float32)

    # 모델 입력에 맞게 resize (256x128)
    crop = cv2.resize(crop, (128, 256))

    # BGR → RGB
    crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

    # 0~1로 정규화 및 채널별 정규화 (mean, std는 fast-reid 기본값)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    crop = (crop.astype(np.float32) / 255.0 - mean) / std

    # Tensor 변환
    tensor = torch.tensor(crop).permute(2, 0, 1).unsqueeze(0).float()

    with torch.no_grad():
        output = reid_model(tensor)
        feat = output[0] if isinstance(output, tuple) else output
        return feat.cpu().numpy().flatten()


def save_detect_csv(image_file, bboxes):
    """탐지 CSV 저장"""
    img_name = os.path.splitext(image_file)[0]
    csv_path = os.path.join(detect_csv_folder, img_name + "_detect.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "pair_id", "x1", "y1", "x2", "y2", "width", "height", "area"])
        for idx, bbox in enumerate(bboxes, start=1):
            x1, y1, x2, y2 = bbox
            w, h = x2 - x1, y2 - y1
            area = w * h
            writer.writerow([img_name, idx, x1, y1, x2, y2, w, h, area])


# -------------------- JSON → CSV 변환 --------------------
json_files = [f for f in os.listdir(json_folder) if f.endswith(".json")]
for json_file in json_files:
    json_path = os.path.join(json_folder, json_file)
    with open(json_path, "r") as f:
        data = json.load(f)
    bboxes = [d["bbox"] for d in data.get("detections", [])]
    if not bboxes:
        continue
    save_detect_csv(json_file, bboxes)
    logging.info(f"[CSV 생성 완료] {json_file}")

# -------------------- CSV → ReID feature 추출 --------------------
csv_files = [f for f in os.listdir(detect_csv_folder) if f.endswith("_detect.csv")]
for csv_file in csv_files:
    csv_path = os.path.join(detect_csv_folder, csv_file)
    features_per_image = []

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        image_name = None
        img = None
        for row in reader:
            if image_name is None:
                image_name = row["image"]
                img_path = os.path.join(images_folder, image_name)
                if not os.path.exists(img_path):
                    logging.warning(f"[경고] 이미지 파일 없음: {image_name}")
                    break
                img = cv2.imread(img_path)

            x1 = int(row["x1"])
            y1 = int(row["y1"])
            x2 = int(row["x2"])
            y2 = int(row["y2"])

            crop = img[y1:y2, x1:x2]
            feat = extract_feature(crop)
            features_per_image.append({
                "image": image_name,
                "bbox": [x1, y1, x2, y2],
                "feature": feat
            })

    # npy 저장
    base_name = os.path.splitext(csv_file)[0].replace("_detect", "")
    save_path = os.path.join(reid_feature_folder, base_name + "_features.npy")
    np.save(save_path, features_per_image)
    logging.info(f"[ReID feature 완료] {base_name} → {save_path}, bbox 수: {len(features_per_image)}")