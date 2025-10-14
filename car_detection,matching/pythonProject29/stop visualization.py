import cv2
import os
import pandas as pd

# -------------------- 경로 설정 --------------------
image_folder = "images"  # 실제 이미지 폴더
stop_csv_path = "stop_results.csv"
matching_csv_path = "matching_csv/all_matching_vehicles_refined_final.csv"

# -------------------- CSV 불러오기 --------------------
stop_df = pd.read_csv(stop_csv_path)
match_df = pd.read_csv(matching_csv_path)
image_files = os.listdir(image_folder)

# -------------------- 이미지 파일 찾기 --------------------
def find_image_file(base_name):
    if not base_name:
        return None
    # _detect.csv 제거, 확장자 제거 후 소문자 비교
    base = base_name.replace("_detect.csv", "").replace(".jpg", "").replace(".png", "").lower()
    for fname in image_files:
        name_no_ext = os.path.splitext(fname)[0].lower()
        if base == name_no_ext:
            return os.path.join(image_folder, fname)
    return None

# -------------------- 시각화 --------------------
cv2.namedWindow("Vehicle Stop Visualization", cv2.WINDOW_NORMAL)

for idx, stop_row in stop_df.iterrows():
    vehicle_id = stop_row["vehicle_id"]
    stop_status = stop_row["stop"]

    # matching_csv에서 bbox 정보 가져오기
    match_row = match_df[match_df["vehicle_id"] == vehicle_id]
    if match_row.empty:
        print(f"매칭 정보 없음: ID {vehicle_id}")
        continue
    match_row = match_row.iloc[0]

    t0_name = match_row["t0_image"]
    t1_name = match_row["t1_image"]

    t0_path = find_image_file(t0_name)
    t1_path = find_image_file(t1_name)

    if t0_path is None or t1_path is None:
        print(f"이미지 없음: ID {vehicle_id}")
        continue

    img0 = cv2.imread(t0_path)
    img1 = cv2.imread(t1_path)
    if img0 is None or img1 is None:
        print(f"이미지 로드 실패: ID {vehicle_id}")
        continue

    # -------------------- bbox 좌표 --------------------
    try:
        t0_bbox = [int(match_row["t0_x1"]), int(match_row["t0_y1"]),
                   int(match_row["t0_x2"]), int(match_row["t0_y2"])]
        t1_bbox = [int(match_row["t1_x1"]), int(match_row["t1_y1"]),
                   int(match_row["t1_x2"]), int(match_row["t1_y2"])]
    except KeyError:
        print(f"bbox 정보 없음: ID {vehicle_id}")
        continue

    # -------------------- Stop/Move 색상 --------------------
    color = (255, 0, 0) if stop_status == "Yes" else (0, 255, 255)
    label = f"ID:{vehicle_id} {stop_status}"

    for img, bbox in zip([img0, img1], [t0_bbox, t1_bbox]):
        x1, y1, x2, y2 = bbox
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, label, (x1, max(y1 - 10, 0)),
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
            exit()

cv2.destroyAllWindows()


