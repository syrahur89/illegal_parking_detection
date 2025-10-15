from ultralytics import YOLO
import os
import json
import cv2
import subprocess
import sys

# -------------------- 경로 세팅 --------------------
model_path = "best.pt"
img_folder = "images"
output_img_folder = "output_images"  #탐지 bbox이미지저장
output_json_folder = "output_json"  #탐지bbox좌표.json

os.makedirs(output_img_folder, exist_ok=True)
os.makedirs(output_json_folder, exist_ok=True)

# -------------------- 모델 로드 --------------------
model = YOLO(model_path)

# -------------------- 이미지 파일 목록 --------------------
img_files = [f for f in os.listdir(img_folder) if f.lower().endswith((".jpg", ".jpeg", ".png"))]

# -------------------- 탐지 및 저장 --------------------
for img_file in img_files:
    img_path = os.path.join(img_folder, img_file)
    results = model(img_path)
    img = cv2.imread(img_path)

    detections = []
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0])

            detections.append({
                "class": "vehicle",
                "bbox": [x1, y1, x2, y2],
                "confidence": conf
            })

            # 이미지에 박스 그리기
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, f"vehicle {conf:.2f}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # -------------------- JSON 저장 --------------------
    json_data = {
        "image": img_file,
        "detections": detections
    }
    json_path = os.path.join(output_json_folder, img_file + ".json")  # 이미지 이름 그대로 + .json
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=4)

    # -------------------- 이미지 저장 --------------------
    output_img_path = os.path.join(output_img_folder, img_file)
    cv2.imwrite(output_img_path, img)

    print(f"[완료] {img_file} → JSON: {json_path}, 이미지: {output_img_path}")

