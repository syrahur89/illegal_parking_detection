import cv2
import csv
import os
import numpy as np
import sys

# -------------------- 경로 설정 --------------------
match_csv_path = "matching_csv/matching_records.csv"
image_folder = "images"  # 실제 이미지 폴더 경로


# 시각화할 이미지 저장할 폴더 및 로그 파일 경로 설정
output_folder = "visual_results"
os.makedirs(output_folder, exist_ok=True)
log_csv_path = os.path.join(output_folder, "visual_log.csv")

# 이미지 폴더 내 모든 파일 목록을 미리 로드하여 효율성을 높임
all_image_filenames = [f for f in os.listdir(image_folder) if os.path.isfile(os.path.join(image_folder, f))]


MIN_VISUAL_SCORE = 0.0


# -------------------- 이미지 파일 찾기  --------------------
def find_image_file(csv_base_name):
    #csv파일명기반으로 T0,T1이미지찾음
    if not csv_base_name:
        return None

    cleaned_name = csv_base_name.lower().replace("_detect.csv", "")
    extensions = [".jpg", ".jpeg", ".png"]

    for ext in extensions:
        search_name_with_ext = (cleaned_name + ext).lower()
        for fname in all_image_filenames:
            if fname.lower() == search_name_with_ext:
                return os.path.join(image_folder, fname)

            if cleaned_name in fname.lower() and fname.lower().endswith(ext):
                return os.path.join(image_folder, fname)

    parts = cleaned_name.split('_')
    final_base_name = '_'.join([p for p in parts if p not in ['t0', 't1']])

    t_suffix = ''
    if '_t0' in cleaned_name:
        t_suffix = '_t0'
    elif '_t1' in cleaned_name:
        t_suffix = '_t1'

    for ext in extensions:
        search_pattern = (final_base_name + t_suffix + ext).lower()
        for fname in all_image_filenames:
            if search_pattern in fname.lower():
                return os.path.join(image_folder, fname)

    return None


# -------------------- 파일명 텍스트 추가 함수 정의 --------------------
def add_filename_to_image(img, csv_name):
    if img is None or not csv_name:
        return img

    filename = os.path.splitext(csv_name)[0]
    filename = filename.replace("_detect", "")

    h, w = img.shape[:2]
    text_color = (255, 255, 0)  # 청록색
    font_scale = 0.7
    font_thickness = 2

    (text_w, text_h), baseline = cv2.getTextSize(filename, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
    text_x = (w - text_w) // 2
    text_y = text_h + 10

    cv2.putText(img, filename, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, font_thickness)
    return img


# -------------------- CSV 불러오기 --------------------
all_records = []
try:
    with open(match_csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        # 'is_matched' 컬럼이 없으면 로드할 수 없으므로 확인
        if "is_matched" not in reader.fieldnames:
            print(f"[오류] CSV 파일에 'is_matched' 컬럼이 없습니다. 매칭 필터링 불가.")
            sys.exit(1)

        # T0의 원본 BBox 좌표 (시각화를 위해 사용)를 파싱
        for row in reader:
            t0_bbox_orig = [
                int(float(row["t0_orig_x1"])) if row.get("t0_orig_x1") and row["t0_orig_x1"] != "" else None,
                int(float(row["t0_orig_y1"])) if row.get("t0_orig_y1") and row["t0_orig_y1"] != "" else None,
                int(float(row["t0_orig_x2"])) if row.get("t0_orig_x2") and row["t0_orig_x2"] != "" else None,
                int(float(row["t0_orig_y2"])) if row.get("t0_orig_y2") and row["t0_orig_y2"] != "" else None
            ]
            # T1의 원본 BBox 좌표 (시각화를 위해 사용)를 파싱
            t1_bbox_orig = [
                int(float(row["t1_orig_x1"])) if row.get("t1_orig_x1") and row["t1_orig_x1"] != "" else None,
                int(float(row["t1_orig_y1"])) if row.get("t1_orig_y1") and row["t1_orig_y1"] != "" else None,
                int(float(row["t1_orig_x2"])) if row.get("t1_orig_x2") and row["t1_orig_x2"] != "" else None,
                int(float(row["t1_orig_y2"])) if row.get("t1_orig_y2") and row["t1_orig_y2"] != "" else None
            ]
            # 레코드 저장
            all_records.append({
                "vehicle_id": row["vehicle_id"],
                "t0_image": row["t0_image"],
                "t1_image": row["t1_image"],
                "t0_bbox_orig": t0_bbox_orig,
                "t1_bbox_orig": t1_bbox_orig,
                "note": row.get("note", "new"),
                "iou": float(row.get("iou", 0)),
                "reid_similarity": float(row.get("reid_similarity", 0)),
                "final_score": float(row.get("final_score", 0.0)),
                "is_matched": int(row.get("is_matched", 0))  # 매칭 여부 로드
            })
except FileNotFoundError:
    print(f"[오류] 매칭 CSV 파일이 경로에 없습니다: {match_csv_path}")
    sys.exit(1)
except Exception as e:
    print(f"[오류] CSV 파일을 읽는 중 문제 발생: {e}. 좌표 값 형식을 확인하세요.")
    sys.exit(1)

# -------------------- 필터링 로직 (모든 매칭 타입 포함) --------------------
# is_matched=1 이므로 헝가리안, REID 매칭성공만 시각화
# MIN_VISUAL_SCORE=0.0 으로 설정했으므로, new가 아닌 모든 매칭이 포함.
matched_pairs = [
    pair for pair in all_records
    if pair["is_matched"] == 1 and pair["final_score"] >= MIN_VISUAL_SCORE
]

print(f"총 레코드 수: {len(all_records)} / 시각화 대상 매칭된 레코드 수: {len(matched_pairs)}")

# -------------------- 시각화 및 저장 --------------------
cv2.namedWindow("Matched Vehicle", cv2.WINDOW_NORMAL)

# 로그 CSV 파일을 설정 및 헤더 작성
log_file_exists = os.path.exists(log_csv_path) and os.path.getsize(log_csv_path) > 0
log_file = open(log_csv_path, "a", newline="")
log_writer = csv.writer(log_file)

if not log_file_exists:
    log_writer.writerow(["vehicle_id", "t0_image_csv", "t1_image_csv", "note", "iou", "reid_similarity", "final_score",
                         "saved_filename"])

for pair in matched_pairs:
    t0_csv_name = pair["t0_image"]
    t1_csv_name = pair["t1_image"]
    vehicle_id = pair["vehicle_id"]
    note = pair["note"]
    final_score = pair["final_score"]
    iou = pair["iou"]
    reid_sim = pair["reid_similarity"]
    # 이미지 파일 로드
    t0_path = find_image_file(t0_csv_name)
    t1_path = find_image_file(t1_csv_name)

    img0 = cv2.imread(t0_path) if t0_path else None
    img1 = cv2.imread(t1_path) if t1_path else None

    if img0 is None and img1 is None:
        continue

    # 🚨 매칭 타입별 색상 설정
    color = (0, 0, 255)  # Default: Red
    if note == "hungarian":
        color = (255, 0, 0)  # Blue
    elif note == "reid_assist":
        color = (0, 255, 255)  # Yellow
    elif note == "fallback_nearest":
        color = (0, 255, 0)  # Green

    # -------------------- T0/T1 파일명, 박스, 텍스트 추가 --------------------
    img0 = add_filename_to_image(img0, t0_csv_name)
    img1 = add_filename_to_image(img1, t1_csv_name)

    # T0 이미지에 박스 표시 (t0_bbox_orig 사용)
    if img0 is not None and None not in pair["t0_bbox_orig"]:
        x1, y1, x2, y2 = [int(v) for v in pair["t0_bbox_orig"]]

        #  ID, Score, IOU, ReID 유사도를 포함한 텍스트 생성
        text_id_score = f'ID:{vehicle_id} S:{final_score:.2f} IOU:{iou:.2f} REID:{reid_sim:.2f}'
        text_note = f'({note})'
        # BBox 그리기 및 텍스트 출력
        (text_w, text_h), _ = cv2.getTextSize(text_id_score, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        text_y1 = max(y1 - 10, text_h + 5)
        text_y2 = text_y1 + text_h + 5

        cv2.rectangle(img0, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img0, text_id_score, (x1, text_y1), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.putText(img0, text_note, (x1, text_y2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    # T1 이미지에 BBox 및 정보 표시 (T0와 동일한 정보 표시)
    if img1 is not None and None not in pair["t1_bbox_orig"]:
        x1, y1, x2, y2 = [int(v) for v in pair["t1_bbox_orig"]]


        text_id_score = f'ID:{vehicle_id} S:{final_score:.2f} IOU:{iou:.2f} REID:{reid_sim:.2f}'
        text_note = f'({note})'

        (text_w, text_h), _ = cv2.getTextSize(text_id_score, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        text_y1 = max(y1 - 10, text_h + 5)
        text_y2 = text_y1 + text_h + 5

        cv2.rectangle(img1, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img1, text_id_score, (x1, text_y1), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.putText(img1, text_note, (x1, text_y2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    # -------------------- 두 이미지 합치기 --------------------
    combined = None
    if img0 is None:
        combined = img1
    elif img1 is None:
        combined = img0
    else:
        h0, w0 = img0.shape[:2]
        h1, w1 = img1.shape[:2]

        if h0 != h1:
            if h0 > h1:
                img1 = cv2.resize(img1, (int(w1 * h0 / h1), h0))
            else:
                img0 = cv2.resize(img0, (int(w0 * h1 / h0), h1))

        combined = cv2.hconcat([img0, img1])

    if combined is not None:
        # 이미지 파일 이름 생성 (ID, 타입, Final Score, IOU, ReID 유사도 포함)

        saved_filename = f"ID_{vehicle_id}_{note.upper()}_S{final_score:.2f}_I{iou:.2f}_R{reid_sim:.2f}.jpg"
        saved_path = os.path.join(output_folder, saved_filename)

        # 합쳐진 이미지 파일 저장
        cv2.imwrite(saved_path, combined)

        # CSV 로그 기록
        log_writer.writerow([
            vehicle_id, t0_csv_name, t1_csv_name, note, iou, reid_sim, final_score, saved_filename
        ])

        cv2.imshow("Matched Vehicle", combined)

    # -------------------- 키 입력 대기--------------------
    while True:
        key = cv2.waitKey(0) & 0xFF
        if key == 13:  # Enter → 다음 차량 시각화
            break
        elif key == 27:  # ESC → 시각화 종료
            log_file.close()
            cv2.destroyAllWindows()
            exit()

# 루프 완료 후 로그 파일 닫기
log_file.close()
cv2.destroyAllWindows()
print(f"\n[완료] 합쳐진 이미지 파일이 '{output_folder}'에 저장되었고, 로그가 '{log_csv_path}'에 기록되었습니다.")