'''
import cv2
import csv
import os


# 경로 설정
match_csv_path = "matching_csv/all_matching_vehicles.csv"

image_folder = "images_"  # 이미지가 있는 폴더 경로


# 이미지 파일 스캔
image_files = os.listdir(image_folder)
image_files_lower = [f.lower() for f in image_files]  # 대소문자 통일


# 이미지 찾기 함수
def find_image_file(base_name):
    if not base_name:
        return None
    base = base_name.replace("_detect.csv", "").replace(".json", "").lower()
    for fname in image_files:
        if base in fname.lower():  # 이름 일부 포함되면 선택
            return os.path.join(image_folder, fname)
    return None


# CSV 불러오기 (matched만)
matched_pairs = []
with open(match_csv_path, "r", newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row["matched"].lower() == "yes":
            t0_bbox = [float(row["t0_x1"]), float(row["t0_y1"]),
                       float(row["t0_x2"]), float(row["t0_y2"])]
            t1_bbox = [float(row["t1_x1"]), float(row["t1_y1"]),
                       float(row["t1_x2"]), float(row["t1_y2"])]
            matched_pairs.append({
                "vehicle_id": row["vehicle_id"],
                "t0_image": row["t0_image"],
                "t1_image": row["t1_image"],
                "t0_bbox": t0_bbox,
                "t1_bbox": t1_bbox
            })

cv2.namedWindow("Matched Vehicle", cv2.WINDOW_NORMAL)

for pair in matched_pairs:
    t0_path = find_image_file(pair["t0_image"])
    t1_path = find_image_file(pair["t1_image"])

    if t0_path is None or t1_path is None:
        print(f"이미지 없음: ID {pair['vehicle_id']}")
        continue

    img0 = cv2.imread(t0_path)
    img1 = cv2.imread(t1_path)

    if img0 is None or img1 is None:
        print(f"이미지 로드 실패: ID {pair['vehicle_id']}")
        continue

    # 박스와 ID 표시
    x1, y1, x2, y2 = map(int, pair["t0_bbox"])
    cv2.rectangle(img0, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(img0, f'ID: {pair["vehicle_id"]}', (x1, y1-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    x1, y1, x2, y2 = map(int, pair["t1_bbox"])
    cv2.rectangle(img1, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(img1, f'ID: {pair["vehicle_id"]}', (x1, y1-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # 두 이미지 합치기
    combined = cv2.hconcat([img0, img1])
    cv2.imshow("Matched Vehicle", combined)

    # 키 입력 대기
    while True:
        key = cv2.waitKey(0) & 0xFF
        if key == 13:  # Enter 키
            break  # 다음 차량으로
        elif key == 27:  # ESC 키
            cv2.destroyAllWindows()
            exit()

cv2.destroyAllWindows()
'''


'''
import cv2
import csv
import os

# -------------------- 경로 설정 --------------------

match_csv_path = "all_matching_vehicles_t1_based"
image_folder = "images_"

# -------------------- 이미지 스캔 --------------------
image_files = os.listdir(image_folder)


def find_image_file(base_name):
    if not base_name:
        return None
    base = base_name.replace("_detect.csv", "").replace(".json", "").lower()
    for fname in image_files:
        if base in fname.lower():
            return os.path.join(image_folder, fname)
    return None


# -------------------- CSV 불러오기 --------------------
matched_pairs = []
with open(match_csv_path, "r", newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        # bbox 값이 비어있으면 건너뛰기
        if row.get("t0_x1") == "" or row.get("t1_x1") == "":
            continue
        t0_bbox = [float(row["t0_x1"]), float(row["t0_y1"]),
                   float(row["t0_x2"]), float(row["t0_y2"])]
        t1_bbox = [float(row["t1_x1"]), float(row["t1_y1"]),
                   float(row["t1_x2"]), float(row["t1_y2"])]
        matched_pairs.append({
            "vehicle_id": row["vehicle_id"],
            "t0_image": row["t0_image"],
            "t1_image": row["t1_image"],
            "t0_bbox": t0_bbox,
            "t1_bbox": t1_bbox,
            "t0_note": row.get("note", "new"),  # note 컬럼
            "t1_note": row.get("note", "new"),
            "iou": float(row.get("iou", 0)),
            "reid_similarity": float(row.get("reid_similarity", 0))
        })

# -------------------- 시각화 --------------------
cv2.namedWindow("Matched Vehicle", cv2.WINDOW_NORMAL)

for pair in matched_pairs:
    t0_path = find_image_file(pair["t0_image"])
    t1_path = find_image_file(pair["t1_image"])

    if t0_path is None or t1_path is None:
        print(f"이미지 없음: ID {pair['vehicle_id']}")
        continue

    img0 = cv2.imread(t0_path)
    img1 = cv2.imread(t1_path)

    if img0 is None or img1 is None:
        print(f"이미지 로드 실패: ID {pair['vehicle_id']}")
        continue

    # -------------------- 박스와 ID 표시 --------------------
    for img, bbox, note in zip([img0, img1], [pair["t0_bbox"], pair["t1_bbox"]], [pair["t0_note"], pair["t1_note"]]):
        x1, y1, x2, y2 = map(int, bbox)

        # note가 'new'가 아니면 매칭된 차량 → 빨간색, 'new'이면 새 차량 → 초록색
        color = (0, 0, 255) if note != "new" else (0, 255, 0)

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, f'ID:{pair["vehicle_id"]} IoU:{pair["iou"]:.2f} ReID:{pair["reid_similarity"]:.2f}',
                    (x1, max(y1 - 10, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # -------------------- 두 이미지 합치기 --------------------
    height = max(img0.shape[0], img1.shape[0])
    combined = cv2.hconcat([cv2.resize(img0, (img0.shape[1], height)),
                            cv2.resize(img1, (img1.shape[1], height))])
    cv2.imshow("Matched Vehicle", combined)

    # -------------------- 키 입력 대기 --------------------
    while True:
        key = cv2.waitKey(0) & 0xFF
        if key == 13:  # Enter → 다음 차량
            break
        elif key == 27:  # ESC → 종료
            cv2.destroyAllWindows()
            exit()

cv2.destroyAllWindows()
'''


import cv2
import csv
import os


# -------------------- 경로 설정 --------------------
match_csv_path = "matching_csv/all_matching_vehicles_refined_final.csv"  # 수정
image_folder = "images"  # 실제 이미지 폴더 경로

image_files = os.listdir(image_folder)

# -------------------- 이미지 파일 찾기 --------------------
def find_image_file(base_name):
    if not base_name:
        return None
    # "_detect.csv" 제거, 소문자로 비교
    base = base_name.replace("_detect.csv", "").lower()
    for fname in image_files:
        if base in fname.lower():
            return os.path.join(image_folder, fname)
    return None

# -------------------- CSV 불러오기 --------------------
matched_pairs = []
with open(match_csv_path, "r", newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        # t0_bbox 또는 t1_bbox 비어있으면 건너뛰기
        if row.get("t0_x1") == "" or row.get("t1_x1") == "":
            continue
        t0_bbox = [int(float(row["t0_x1"])), int(float(row["t0_y1"])),
                   int(float(row["t0_x2"])), int(float(row["t0_y2"]))]
        t1_bbox = [int(float(row["t1_x1"])), int(float(row["t1_y1"])),
                   int(float(row["t1_x2"])), int(float(row["t1_y2"]))]
        matched_pairs.append({
            "vehicle_id": row["vehicle_id"],
            "t0_image": row["t0_image"],
            "t1_image": row["t1_image"],
            "t0_bbox": t0_bbox,
            "t1_bbox": t1_bbox,
            "note": row.get("note", "new"),
            "iou": float(row.get("iou", 0)),
            "reid_similarity": float(row.get("reid_similarity", 0))
        })

# -------------------- 시각화 --------------------
cv2.namedWindow("Matched Vehicle", cv2.WINDOW_NORMAL)

for pair in matched_pairs:
    t0_path = find_image_file(pair["t0_image"])
    t1_path = find_image_file(pair["t1_image"])

    if t0_path is None or t1_path is None:
        print(f"이미지 없음: ID {pair['vehicle_id']}")
        continue

    img0 = cv2.imread(t0_path)
    img1 = cv2.imread(t1_path)

    if img0 is None or img1 is None:
        print(f"이미지 로드 실패: ID {pair['vehicle_id']}")
        continue

    # -------------------- 박스와 ID 표시 --------------------
    for img, bbox in zip([img0, img1], [pair["t0_bbox"], pair["t1_bbox"]]):
        x1, y1, x2, y2 = bbox
        # note가 'new'가 아니면 매칭된 차량 → 빨간색, 'new'이면 새 차량 → 초록색
        color = (0, 0, 255) if pair["note"] != "new" else (0, 255, 0)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, f'ID:{pair["vehicle_id"]} IoU:{pair["iou"]:.2f} ReID:{pair["reid_similarity"]:.2f}',
                    (x1, max(y1 - 10, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # -------------------- 두 이미지 합치기 --------------------
    height = max(img0.shape[0], img1.shape[0])
    combined = cv2.hconcat([cv2.resize(img0, (img0.shape[1], height)),
                            cv2.resize(img1, (img1.shape[1], height))])
    cv2.imshow("Matched Vehicle", combined)

    # -------------------- 키 입력 대기 --------------------
    while True:
        key = cv2.waitKey(0) & 0xFF
        if key == 13:  # Enter → 다음 차량
            break
        elif key == 27:  # ESC → 종료
            cv2.destroyAllWindows()
            exit()

cv2.destroyAllWindows()
