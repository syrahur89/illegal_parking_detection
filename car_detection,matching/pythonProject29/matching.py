import os
import csv
import numpy as np
from scipy.optimize import linear_sum_assignment
import subprocess
import sys
from collections import defaultdict

# -------------------- 경로 설정 --------------------
detect_csv_folder = "detect_csv"
reid_feature_folder = "reid_features"
match_csv_folder = "matching_csv"
PAIR_CSV_PATH = "t0t1_two.csv"  # 페어 파일 경로
os.makedirs(match_csv_folder, exist_ok=True)

# -------------------- (튜닝 가능한) 매칭 조건 --------------------
IOU_THRESH = 0.03
CENTER_RATIO = 1.5
SIZE_SIM_THRESH = 0.35
REID_COSINE_THRESH = 0.20
MIN_BBOX_SIZE = 12
ALPHA = 0.2
BETA = 0.8
TOP_N_REID = 3
NEAREST_PIXELS = 200
FINAL_SCORE_THRESH = 0.18


# -------------------- 유틸 함수 --------------------

def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = max(0, (box1[2] - box1[0])) * max(0, (box1[3] - box1[1]))
    area2 = max(0, (box2[2] - box2[0])) * max(0, (box2[3] - box2[1]))
    union_area = area1 + area2 - inter_area
    return inter_area / union_area if union_area > 0 else 0.0


def center_distance(box1, box2):
    cx1, cy1 = (box1[0] + box1[2]) / 2.0, (box1[1] + box1[3]) / 2.0
    cx2, cy2 = (box2[0] + box2[2]) / 2.0, (box2[1] + box2[3]) / 2.0
    return float(np.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2))


def size_similarity(box1, box2):
    w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
    w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
    area1, area2 = max(0, w1 * h1), max(0, w2 * h2)
    return (min(area1, area2) / max(area1, area2)) if max(area1, area2) > 0 else 0.0


def cosine_similarity(f1, f2):
    if f1 is None or f2 is None:
        return 0.0
    n1 = np.linalg.norm(f1)
    n2 = np.linalg.norm(f2)
    if n1 == 0 or n2 == 0:
        return 0.0
    return float(np.dot(f1, f2) / (n1 * n2))


def load_features(csv_file):
    base_name = os.path.splitext(csv_file)[0]
    if base_name.lower().endswith('_detect'):
        base_name = base_name[:-7]

    feature_path = os.path.join(reid_feature_folder, base_name + "_features.npy")

    if not os.path.exists(feature_path):
        return []
    return np.load(feature_path, allow_pickle=True)


def read_detect_csv(csv_file):
    bboxes = []
    file_path = os.path.join(detect_csv_folder, csv_file)

    if not os.path.exists(file_path):
        return bboxes

    try:
        with open(file_path, "r", newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    x1, y1, x2, y2 = int(float(row["x1"])), int(float(row["y1"])), int(float(row["x2"])), int(
                        float(row["y2"]))
                except:
                    continue
                bboxes.append([x1, y1, x2, y2])
    except Exception as e:
        print(f"[오류] CSV 파일 읽기 중 문제 발생: {file_path} - {e}")
        return bboxes

    return bboxes


#  T0/T1 접미사를 사용하여 실제 파일명을 유추하도록 수정
def complete_filename(base_name, detect_csv_folder, is_t0):
    """
    t0t1_augmented.csv의 불완전한 이름(0001_top)을 기반으로 실제 detect_csv 파일을 찾음.
    is_t0=True는 t0 파일을, False는 t1 파일을 찾아야 함을 의미.
    """

    # 1. 이미 완성된 이름이라면 바로 반환
    if base_name.lower().endswith('_detect.csv'):
        file_path = os.path.join(detect_csv_folder, base_name)
        if os.path.exists(file_path):
            return base_name

    # 2. base_name을 접두사로 가지고, t0/t1 정보가 포함된 파일을 찾음

    # 파일명은 '0001_top_20250618_t0.jpg_detect.csv' 형식이므로,
    # base_name인 '0001_top'이 접두사가 됨.

    if is_t0:
        suffix_filter = '_t0.'
    else:
        suffix_filter = '_t1.'

    for filename in os.listdir(detect_csv_folder):
        # filename이 '0001_top'으로 시작하고, '_t0.' 또는 '_t1.'을 포함하며, '_detect.csv'로 끝나는지 확인
        if filename.startswith(base_name) and filename.endswith('_detect.csv') and suffix_filter in filename:
            return filename

    return ""


# -------------------- 매칭 메인 --------------------
vehicle_global_id = 1
all_vehicle_records = []
t0_to_t1_map = defaultdict(list)
unique_t0_files = set()

# 1. t0t1_augmented.csv 파일을 읽어와 매핑 생성 및 파일명 완성
if not os.path.exists(PAIR_CSV_PATH):
    print(f"[오류] 페어 파일 없음: {PAIR_CSV_PATH}")
    sys.exit(1)

found_pair_count = 0
try:
    with open(PAIR_CSV_PATH, "r", newline="") as f:
        reader = csv.DictReader(f)
        if "t0_filename" not in reader.fieldnames or "t1_filename" not in reader.fieldnames:
            print("[오류] t0t1_pair.csv에 't0_filename' 또는 't1_filename' 컬럼이 없습니다.")
            sys.exit(1)

        for row in reader:
            t0_base_name = row["t0_filename"].strip()
            t1_base_name = row["t1_filename"].strip()

            # 🚨 수정된 complete_filename 함수 사용
            t0_filename = complete_filename(t0_base_name, detect_csv_folder, is_t0=True)
            t1_filename = complete_filename(t1_base_name, detect_csv_folder, is_t0=False)

            if t0_filename and t1_filename:
                t0_to_t1_map[t0_filename].append(t1_filename)
                unique_t0_files.add(t0_filename)
                found_pair_count += 1
            # else:
            #     print(f"[경고] 매칭되는 실제 파일명을 찾지 못함: T0={t0_base_name}, T1={t1_base_name}")

except Exception as e:
    print(f"[오류] t0t1_augmented.csv 파일을 읽는 중 문제 발생: {e}")
    sys.exit(1)

if found_pair_count == 0:
    print(f"[오류] t0t1_augmented.csv에서 유효한 파일쌍을 하나도 찾지 못했습니다. detect_csv 폴더와 파일명을 확인하세요. 파일명 형식 문제일 가능성이 높습니다.")
    # sys.exit(1) # 오류 메시지를 출력하고 계속 진행 (0% 매칭률 예상)

# 2. T0 파일 순회 및 T1 매칭
total_t1_count = 0
matched_t1_count = 0

for t0_file in sorted(list(unique_t0_files)):

    t0_bboxes = read_detect_csv(t0_file)
    t0_feats = load_features(t0_file)

    if len(t0_bboxes) == 0:
        continue

    # T0 박스 ID 할당 리스트 (ID 중복 해결)
    t0_to_global_id = [-1] * len(t0_bboxes)

    # T0 박스 할당 추적 Set (T0 박스가 여러 T1 파일에 중복 매칭되는 것을 방지)
    t0_assigned_in_t0_file = set()

    # T0에 연결된 모든 T1 파일 순회
    for t1_file in t0_to_t1_map[t0_file]:

        t1_bboxes = read_detect_csv(t1_file)
        t1_feats = load_features(t1_file)

        if len(t1_bboxes) == 0:
            continue

        total_t1_count += len(t1_bboxes)

        t1_assigned = set()
        matched_pairs_t1 = []

        # -------------------- 매칭 로직 --------------------

        # 1) 후보 매트릭스 생성 & Hungarian 매칭
        # T0 박스가 이미 매칭되었다면 cost matrix 후보에서 제외 (1e6)
        cost_matrix = np.full((len(t0_bboxes), len(t1_bboxes)), 1e6, dtype=float)
        metrics = {}

        for i, b0 in enumerate(t0_bboxes):
            if i in t0_assigned_in_t0_file:
                continue  # 이미 매칭된 T0 박스는 이번 T1과의 매칭 후보에서 제외

            candidate_list = []
            for j, b1 in enumerate(t1_bboxes):
                if min(b1[2] - b1[0], b1[3] - b1[1]) < MIN_BBOX_SIZE: continue
                iou = compute_iou(b0, b1)
                center_dist = center_distance(b0, b1)
                if iou < IOU_THRESH and center_dist > CENTER_RATIO * ((b0[2] - b0[0] + b0[3] - b0[1]) / 2.0): continue

                f0 = t0_feats[i].get("feature") if (i < len(t0_feats) and isinstance(t0_feats[i], dict)) else (
                    t0_feats[i] if i < len(t0_feats) and not isinstance(t0_feats[i], dict) else None)
                f1 = t1_feats[j].get("feature") if (j < len(t1_feats) and isinstance(t1_feats[j], dict)) else (
                    t1_feats[j] if j < len(t1_feats) and not isinstance(t1_feats[j], dict) else None)
                cos_sim = cosine_similarity(f0, f1)

                score = ALPHA * iou + BETA * cos_sim
                candidate_list.append((j, score,
                                       {"iou": iou, "center_dist": center_dist, "size_sim": size_similarity(b0, b1),
                                        "reid": cos_sim}))

            if candidate_list:
                candidate_list.sort(key=lambda x: x[1], reverse=True)
                for (j, score, metric) in candidate_list[:TOP_N_REID]:
                    cost_matrix[i, j] = 1.0 - score
                    metrics[(i, j)] = metric

        # 2) Hungarian 1:1 매칭 시도
        try:
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
        except Exception:
            row_ind, col_ind = np.array([], dtype=int), np.array([], dtype=int)

        for r, c in zip(row_ind, col_ind):
            if r in t0_assigned_in_t0_file or c in t1_assigned or cost_matrix[r, c] >= 1e6: continue

            m = metrics.get((r, c), None)
            if m is None: continue
            final_score = ALPHA * m["iou"] + BETA * m["reid"]
            if final_score < FINAL_SCORE_THRESH: continue

            # **ID 할당/재사용 로직**
            current_id = t0_to_global_id[r]
            if current_id == -1:
                current_id = vehicle_global_id
                t0_to_global_id[r] = current_id
                vehicle_global_id += 1

            rec = {
                "vehicle_id": current_id,
                "t0_image": t0_file, "t1_image": t1_file,
                "t0_bbox": t0_bboxes[r], "t1_bbox": t1_bboxes[c],
                "iou": m["iou"], "center_dist": m["center_dist"], "size_sim": m["size_sim"],
                "reid_similarity": m["reid"], "final_score": final_score, "note": "hungarian"
            }
            t1_assigned.add(c)
            t0_assigned_in_t0_file.add(r)  # T0 박스 사용 완료 처리
            matched_pairs_t1.append(rec)
            matched_t1_count += 1

        # 3) ReID 보조 (거리 무시, 높은 ReID 기준)
        for j, b1 in enumerate(t1_bboxes):
            if j in t1_assigned: continue
            best_idx, best_cos = -1, 0.0
            for i, b0 in enumerate(t0_bboxes):
                if i in t0_assigned_in_t0_file: continue  # 이미 매칭된 T0 박스 제외

                f0 = t0_feats[i].get("feature") if (i < len(t0_feats) and isinstance(t0_feats[i], dict)) else (
                    t0_feats[i] if i < len(t0_feats) and not isinstance(t0_feats[i], dict) else None)
                f1 = t1_feats[j].get("feature") if (j < len(t1_feats) and isinstance(t1_feats[j], dict)) else (
                    t1_feats[j] if j < len(t1_feats) and not isinstance(t1_feats[j], dict) else None)
                cos_sim = cosine_similarity(f0, f1)

                if cos_sim >= REID_COSINE_THRESH and cos_sim > best_cos:
                    best_cos = cos_sim
                    best_idx = i

            if best_idx >= 0:
                m_iou = compute_iou(t0_bboxes[best_idx], b1)

                # **ID 할당/재사용 로직**
                current_id = t0_to_global_id[best_idx]
                if current_id == -1:
                    current_id = vehicle_global_id
                    t0_to_global_id[best_idx] = current_id
                    vehicle_global_id += 1

                rec = {
                    "vehicle_id": current_id,
                    "t0_image": t0_file, "t1_image": t1_file, "t0_bbox": t0_bboxes[best_idx], "t1_bbox": b1,
                    "iou": m_iou, "center_dist": center_distance(t0_bboxes[best_idx], b1),
                    "size_sim": size_similarity(t0_bboxes[best_idx], b1),
                    "reid_similarity": best_cos, "final_score": ALPHA * m_iou + BETA * best_cos, "note": "reid_assist"
                }
                t1_assigned.add(j)
                t0_assigned_in_t0_file.add(best_idx)  # T0 박스 사용 완료 처리
                matched_pairs_t1.append(rec)
                matched_t1_count += 1

        # Fallback A: 가장 가까운 박스 (거리 NEAREST_PIXELS 이내)
        for j, b1 in enumerate(t1_bboxes):
            if j in t1_assigned: continue
            best_idx, best_dist = -1, float('inf')
            for i, b0 in enumerate(t0_bboxes):
                if i in t0_assigned_in_t0_file: continue  # 이미 매칭된 T0 박스 제외

                dist = center_distance(b0, b1)
                if dist < best_dist:
                    best_idx, best_dist = i, dist

            if best_idx >= 0 and best_dist <= NEAREST_PIXELS:
                m_iou = compute_iou(t0_bboxes[best_idx], b1)

                # **ID 할당/재사용 로직**
                current_id = t0_to_global_id[best_idx]
                if current_id == -1:
                    current_id = vehicle_global_id
                    t0_to_global_id[best_idx] = current_id
                    vehicle_global_id += 1

                rec = {
                    "vehicle_id": current_id,
                    "t0_image": t0_file, "t1_image": t1_file, "t0_bbox": t0_bboxes[best_idx], "t1_bbox": b1,
                    "iou": m_iou, "center_dist": best_dist, "size_sim": size_similarity(t0_bboxes[best_idx], b1),
                    "reid_similarity": 0.0, "final_score": 0.0, "note": "fallback_nearest"
                }
                t1_assigned.add(j)
                t0_assigned_in_t0_file.add(best_idx)  # T0 박스 사용 완료 처리
                matched_pairs_t1.append(rec)
                matched_t1_count += 1

        # 6) 아직 매칭 안 된 t1은 새 vehicle로 처리
        for j, b1 in enumerate(t1_bboxes):
            if j not in t1_assigned:
                rec = {
                    "vehicle_id": vehicle_global_id, "t0_image": "", "t1_image": t1_file,
                    "t0_bbox": [], "t1_bbox": b1, "iou": 0.0, "center_dist": 0.0, "size_sim": 0.0,
                    "reid_similarity": 0.0, "final_score": 0.0, "note": "new"
                }
                matched_pairs_t1.append(rec)
                vehicle_global_id += 1

        all_vehicle_records.extend(matched_pairs_t1)

# -------------------- CSV 저장 --------------------
single_csv_path = os.path.join(match_csv_folder, "all_matching_vehicles_refined_final.csv")
with open(single_csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "vehicle_id", "t0_image", "t1_image",
        "t0_x1", "t0_y1", "t0_x2", "t0_y2",
        "t1_x1", "t1_y1", "t1_x2", "t1_y2",
        "iou", "reid_similarity", "final_score", "note"
    ])
    for rec in all_vehicle_records:
        t0_bbox = rec["t0_bbox"] if rec["t0_bbox"] else ["", "", "", ""]
        t1_bbox = rec["t1_bbox"] if rec["t1_bbox"] else ["", "", "", ""]
        writer.writerow([
                            rec["vehicle_id"], rec["t0_image"], rec["t1_image"]
                        ] + t0_bbox + t1_bbox + [
                            rec.get("iou", 0.0), rec.get("reid_similarity", 0.0), rec.get("final_score", 0.0),
                            rec.get("note", "")
                        ])

# -------------------- 결과 요약 출력 --------------------
match_rate = (matched_t1_count / total_t1_count * 100.0) if total_t1_count > 0 else 0.0
print(f"[완료] CSV 저장: {single_csv_path}")
print(f"총 t1 박스 수: {total_t1_count}, 매칭된 t1 수(신뢰기준 통과 포함): {matched_t1_count}, 매칭률: {match_rate:.2f}%")

subprocess.run([sys.executable, "matching visualization.py"])