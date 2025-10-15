import cv2
import os
import pandas as pd
import numpy as np
import sys

# -------------------- 경로 설정 --------------------
image_folder = "images"
stop_csv_path = "stop_results.csv"
matching_csv_path = "matching_csv/matching_records.csv"

# -------------------- CSV 불러오기 --------------------
try:
    stop_df = pd.read_csv(stop_csv_path)
    match_df = pd.read_csv(matching_csv_path)


    image_files = {os.path.splitext(f.lower())[0]: f for f in os.listdir(image_folder)}
except FileNotFoundError as e:
    print(f"[오류] 파일 경로를 확인해주세요: {e}")
    sys.exit()


# -------------------- 이미지 파일 찾기 --------------------
def find_image_file(base_name):
    if pd.isna(base_name) or not base_name:
        return None


    name = str(base_name).lower()


    name = name.replace("_detect.csv", "")


    for ext in ['.png', '.jpg', '.jpeg']:
        if name.endswith(ext):
            name = name[:-len(ext)]
            break


    processed_name = name
    for tag in ['_b', '_c', '_a']:

        if tag + '_top' in processed_name:
            processed_name = processed_name.replace(tag, '', 1)
            break


    search_names = [name, processed_name]

    for search_key in search_names:
        if search_key in image_files:
            return os.path.join(image_folder, image_files[search_key])


    for disk_name_key, full_filename in image_files.items():
        if disk_name_key.startswith(name) or name.startswith(disk_name_key):
            return os.path.join(image_folder, full_filename)

    return None


# -------------------- 시각화  --------------------
cv2.namedWindow("Vehicle Stop Visualization", cv2.WINDOW_NORMAL)

for idx, stop_row in stop_df.iterrows():
    vehicle_id = stop_row["vehicle_id"]
    stop_status = stop_row["stop"]

    match_row = match_df[match_df["vehicle_id"] == vehicle_id]

    if match_row.empty:
        continue
    match_row = match_row.iloc[0]

    t0_name = match_row["t0_image"]
    t1_name = match_row["t1_image"]

    t0_path = find_image_file(t0_name)
    t1_path = find_image_file(t1_name)

    if t0_path is None or t1_path is None:
        print(f"이미지 로드 실패 (파일 이름 불일치): ID {vehicle_id} - T0:{t0_name}, T1:{t1_name}")
        continue

    img0 = cv2.imread(t0_path)
    img1 = cv2.imread(t1_path)
    if img0 is None or img1 is None:
        print(f"이미지 로드 실패 (파일 손상): ID {vehicle_id} - {t0_path} 또는 {t1_path}")
        continue

    # -------------------- bbox 좌표 --------------------
    try:
        t0_bbox = [int(match_row["t0_orig_x1"]), int(match_row["t0_orig_y1"]),
                   int(match_row["t0_orig_x2"]), int(match_row["t0_orig_y2"])]
        t1_bbox = [int(match_row["t1_orig_x1"]), int(match_row["t1_orig_y1"]),
                   int(match_row["t1_orig_x2"]), int(match_row["t1_orig_y2"])]
    except (KeyError, ValueError):
        print(f"bbox 좌표 오류/누락 (NaN): ID {vehicle_id}")
        continue

    # -------------------- Stop/Move 색상 --------------------
    color = (255, 0, 0) if stop_status == "Yes" else (0, 255, 255)
    label = f"ID:{int(vehicle_id)} {stop_status}"

    for img, bbox in zip([img0, img1], [t0_bbox, t1_bbox]):
        if not all(isinstance(coord, int) for coord in bbox):
            continue

        x1, y1, x2, y2 = bbox

        if x2 > x1 and y2 > y1:
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, label, (x1, max(y1 - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # -------------------- 두 이미지 합치기 --------------------
    height = max(img0.shape[0], img1.shape[0])
    combined = cv2.hconcat([cv2.resize(img0, (img0.shape[1], height)),
                            cv2.resize(img1, (img1.shape[1], height))])
    cv2.imshow("Vehicle Stop Visualization", combined)

    # -------------------- 키 입력 대기 --------------------
    while True:
        key = cv2.waitKey(0) & 0xFF
        if key == 13:  # Enter → 다음 차량
            break
        elif key == 27:  # ESC → 종료
            cv2.destroyAllWindows()
            sys.exit()

cv2.destroyAllWindows()

