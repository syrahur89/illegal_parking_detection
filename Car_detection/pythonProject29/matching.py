import os
import os
import csv
import numpy as np
from scipy.optimize import linear_sum_assignment
import subprocess
import sys
from collections import defaultdict

# -------------------- 데이터 설정  --------------------
detect_csv_folder = "aligned_csv"   #이미지별 정렬 bbox담긴 csv
reid_feature_folder = "reid_features"   #이미지 별 reid특징벡터
match_csv_folder = "matching_csv"     #매칭결과 담긴 csv

PAIR_CSV_PATH = "t0t1_pair.csv"   # T0 ↔ T1 이미지 쌍 매핑 정보
os.makedirs(match_csv_folder, exist_ok=True)

# -------------------- 하이퍼파라미터 설정 --------------------
IOU_THRESH = 0.03  #매칭 최소 IoU 기준
CENTER_RATIO = 1.5  # 중심 거리 제한 비율
SIZE_SIM_THRESH = 0.85  # ReID 보조 매칭시 차량 크기 유사도 최소 기준
REID_COSINE_THRESH = 0.80  # ReID 보조매칭시 코사인 유사도 최소 기준
MIN_BBOX_SIZE = 12   # 너무 작은 박스 제거
ALPHA = 1.0  # 최종 score 계산 시 IoU 가중치
BETA = 0.0  # ReID 가중치 (1단계에서 미적용)
TOP_N_REID = 3  # Hungarian 시 고려할 T1 후보 수
FINAL_SCORE_THRESH = 0.30  # 최종 매칭 점수 최소 기준
MAX_REID_DIST = 200   # ReID Assist 시 최대 중심 거리(픽셀)


# -------------------- 유틸 함수 (변경 없음) --------------------

"""두 박스 간 IoU 계산"""
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

"""박스 중심 좌표 간 거리 계산"""
def center_distance(box1, box2):
    cx1, cy1 = (box1[0] + box1[2]) / 2.0, (box1[1] + box1[3]) / 2.0
    cx2, cy2 = (box2[0] + box2[2]) / 2.0, (box2[1] + box2[3]) / 2.0
    return float(np.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2))

"""두 박스의 면적 비율로 크기 유사도 계산"""
def size_similarity(box1, box2):
    w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
    w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
    area1, area2 = max(0, w1 * h1), max(0, w2 * h2)
    return (min(area1, area2) / max(area1, area2)) if max(area1, area2) > 0 else 0.0

"""ReID 특징벡터 간 코사인 유사도 계산"""
def cosine_similarity(f1, f2):
    if f1 is None or f2 is None:
        return 0.0
    if isinstance(f1, dict):
        f1 = f1.get("feature")
    if isinstance(f2, dict):
        f2 = f2.get("feature")

    if f1 is None or f2 is None:
        return 0.0

    n1 = np.linalg.norm(f1)
    n2 = np.linalg.norm(f2)
    if n1 == 0 or n2 == 0:
        return 0.0
    return float(np.dot(f1, f2) / (n1 * n2))

"""탐지 CSV 이름 기반으로 대응되는 ReID 특징벡터 .npy 로드"""
def load_features(csv_file):
    base_name = os.path.splitext(csv_file)[0]
    if base_name.lower().endswith('_detect'):
        base_name = base_name[:-7]

    feature_path = os.path.join(reid_feature_folder, base_name + "_features.npy")

    if not os.path.exists(feature_path):
        return []

    return np.load(feature_path, allow_pickle=True).tolist()

"""탐지 결과 CSV 읽어서 id,정렬bbox,정렬전 bbox 정보 리스트로 반환"""
def read_detect_csv(csv_file):
    bbox_records = []
    file_path = os.path.join(detect_csv_folder, csv_file)
    if not os.path.exists(file_path): return bbox_records
    try:
        with open(file_path, "r", newline='') as f:
            reader = csv.DictReader(f)
            if "pair_id" not in reader.fieldnames:
                print(f"[오류!] '{csv_file}'에 'pair_id' 컬럼이 없습니다.")
                return bbox_records
            for row in reader:
                try:
                    pair_id = int(row["pair_id"])
                    x1, y1, x2, y2 = int(float(row["x1"])), int(float(row["y1"])), int(float(row["x2"])), int(
                        float(row["y2"]))
                    x1_orig, y1_orig, x2_orig, y2_orig = \
                        int(float(row["x1_original"])), int(float(row["y1_original"])), \
                            int(float(row["x2_original"])), int(float(row["y2_original"]))
                    bbox_records.append({
                        "pair_id": pair_id,
                        "aligned_bbox": [x1, y1, x2, y2],
                        "original_bbox": [x1_orig, y1_orig, x2_orig, y2_orig],
                        "feature": None
                    })
                except Exception:
                    continue
    except Exception as e:
        print(f"[오류] CSV 파일 읽기 중 문제 발생: {file_path} - {e}")
        return bbox_records
    return bbox_records

"""T0/T1 기본 파일명에 맞는 실제 CSV 파일명을 찾아서 반환"""
def complete_filename(base_name, detect_csv_folder, is_t0):
    if base_name.lower().endswith('_detect.csv'):
        file_path = os.path.join(detect_csv_folder, base_name)
        if os.path.exists(file_path): return base_name
    if is_t0:
        suffix_filter = '_t0.'
    else:
        suffix_filter = '_t1.'
    for filename in os.listdir(detect_csv_folder):
        if filename.startswith(base_name) and filename.endswith('_detect.csv') and suffix_filter in filename:
            return filename
    return ""

 #탐지 bbox에 ReID 특징벡터 연결 (pair_id 기준)
def attach_features_by_id(records, feats):
    if not feats or not records: return records
    id_to_feat = {}
    for f in feats:
        if isinstance(f, dict) and 'bbox_id' in f:
            id_to_feat[f['bbox_id']] = f['feature']
    valid_records = []
    for r in records:
        bbox_id = r.get('pair_id')
        if bbox_id is not None and bbox_id in id_to_feat:
            r['feature'] = id_to_feat[bbox_id]
            valid_records.append(r)
    return valid_records


# -------------------- 매칭 메인 (ID 기반 매칭 적용) --------------------
vehicle_global_id = 1      #차량고유ID(새 차량 발견시 증가)
all_vehicle_records = []     # 전체 매칭 결과 저장
t0_to_t1_map = defaultdict(list)   # t0 이미지별 대응되는 t1 이미지 리스트
unique_t0_files = set()    # 중복 방지용 집합

# 1. t0t1_pair.csv 파일 읽어 사진 매칭쌍 불러오기
if not os.path.exists(PAIR_CSV_PATH):
    print(f"[오류] 페어 파일 없음: {PAIR_CSV_PATH}")
    sys.exit(1)

found_pair_count = 0
try:
    with open(PAIR_CSV_PATH, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            t0_base_name = row["t0_filename"].strip()
            t1_base_name = row["t1_filename"].strip()
            t0_filename = complete_filename(t0_base_name, detect_csv_folder, is_t0=True)
            t1_filename = complete_filename(t1_base_name, detect_csv_folder, is_t0=False)

            if t0_filename and t1_filename:
                t0_to_t1_map[t0_filename].append(t1_filename)
                unique_t0_files.add(t0_filename)
                found_pair_count += 1
except Exception as e:
    print(f"[오류] t0t1_two.csv 파일을 읽는 중 문제 발생: {e}")
    sys.exit(1)

# 2. T0 파일 순회 및 T1 매칭
total_t1_count = 0
matched_t1_count = 0

for t0_file in sorted(list(unique_t0_files)):
    # t0 CSV + 특징 로드
    t0_raw_records = read_detect_csv(t0_file)
    t0_feats = load_features(t0_file)
    t0_records = attach_features_by_id(t0_raw_records, t0_feats)

    if len(t0_records) == 0:
        continue

    t0_to_global_id = [-1] * len(t0_records)
    t0_assigned_in_t0_file = set()

    for t1_file in t0_to_t1_map[t0_file]:
        # t1 CSV + 특징 로드
        t1_raw_records = read_detect_csv(t1_file)
        t1_feats = load_features(t1_file)
        t1_records = attach_features_by_id(t1_raw_records, t1_feats)

        if len(t1_records) == 0:
            continue

        total_t1_count += len(t1_records)
        t1_assigned = set()   # 현재 T1 파일에서 이미 매칭된 T1 박스 인덱스
        matched_pairs_t1 = []   # T0-T1 쌍 매칭 결과 임시 저장


        # -------------------- 1단계: Hungarian Matching (IoU) --------------------

        cost_matrix = np.full((len(t0_records), len(t1_records)), 1e6, dtype=float)
        metrics = {}  # 각 쌍의 IoU, ReID 등의 상세 지표 저장

        for i, r0 in enumerate(t0_records):
            if i in t0_assigned_in_t0_file: continue  # 이미 매칭된 T0는 건너뜀
            b0 = r0["aligned_bbox"]

            candidate_list = []
            for j, r1 in enumerate(t1_records):
                b1 = r1["aligned_bbox"]
                # 너무 작은 박스 제외
                if min(b1[2] - b1[0], b1[3] - b1[1]) < MIN_BBOX_SIZE: continue  # 최소 크기 미달 제외
                iou = compute_iou(b0, b1)
                iou = compute_iou(b0, b1)
                center_dist = center_distance(b0, b1)
                # 초기 필터링: IOU가 낮고 (0.03), 중심 거리가 멀면 (1.5배) 후보에서 제외
                if iou < IOU_THRESH and center_dist > CENTER_RATIO * ((b0[2] - b0[0] + b0[3] - b0[1]) / 2.0): continue

                f0 = r0.get("feature")
                f1 = r1.get("feature")
                cos_sim = cosine_similarity(f0, f1)

                if f0 is None or f1 is None: continue  # ReID 특징이 없으면 제외

                score = ALPHA * iou + BETA * cos_sim  #매칭 점수 계산
                candidate_list.append((j, score,
                                       {"iou": iou, "center_dist": center_dist, "size_sim": size_similarity(b0, b1),
                                        "reid": cos_sim}))

            if candidate_list:
                candidate_list.sort(key=lambda x: x[1], reverse=True)
                for (j, score, metric) in candidate_list[:TOP_N_REID]: # 상위 N개 후보만 비용 행렬에 반영
                    cost_matrix[i, j] = 1.0 - score # 비용 = 1.0 - 점수
                    metrics[(i, j)] = metric
        # 헝가리안 알고리즘 실행 (최소 비용 매칭)
        try:
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
        except Exception:
            row_ind, col_ind = np.array([], dtype=int), np.array([], dtype=int)
        # 이미 매칭되었거나 비용이 너무 높으면 제외
        for r, c in zip(row_ind, col_ind):
            if r in t0_assigned_in_t0_file or c in t1_assigned or cost_matrix[r, c] >= 1e6: continue

            m = metrics.get((r, c), None)
            if m is None: continue

            final_score = ALPHA * m["iou"] + BETA * m["reid"]
            if final_score < FINAL_SCORE_THRESH: continue
            # Global ID 부여 및 기록
            current_id = t0_to_global_id[r]
            if current_id == -1:
                current_id = vehicle_global_id
                t0_to_global_id[r] = current_id
                vehicle_global_id += 1
            # 매칭 결과 저장
            rec = {
                "vehicle_id": current_id, "t0_image": t0_file, "t1_image": t1_file,
                "t0_bbox": t0_records[r]["aligned_bbox"], "t1_bbox": t1_records[c]["aligned_bbox"],
                "t0_bbox_orig": t0_records[r]["original_bbox"], "t1_bbox_orig": t1_records[c]["original_bbox"],
                "iou": m["iou"], "center_dist": m["center_dist"], "size_sim": m["size_sim"],
                "reid_similarity": m["reid"], "final_score": final_score, "note": "hungarian",
                "is_matched": 1
            }
            t1_assigned.add(c)
            t0_assigned_in_t0_file.add(r)
            matched_pairs_t1.append(rec)
            matched_t1_count += 1

        # -------------------- 2단계: ReID Assist (극단적 엄격화) --------------------

        for j, r1 in enumerate(t1_records):
            if j in t1_assigned: continue   # 이미 매칭된 T1은 건너뜀
            b1 = r1["aligned_bbox"]
            if min(b1[2] - b1[0], b1[3] - b1[1]) < MIN_BBOX_SIZE: continue

            best_idx, best_cos = -1, 0.0

            for i, r0 in enumerate(t0_records):
                if i in t0_assigned_in_t0_file: continue  # 이미 매칭된 T0는 건너뜀
                b0 = r0["aligned_bbox"]

                # 크기 유사도 제약 (SIZE_SIM_THRESH = 0.85)
                m_size_sim = size_similarity(b0, b1)
                if m_size_sim < SIZE_SIM_THRESH: continue
                # IoU가 너무 작으면 제외
                m_iou = compute_iou(b0, b1)
                if m_iou < 0.001: continue

                f0 = r0.get("feature")
                f1 = r1.get("feature")
                if f0 is None or f1 is None: continue

                cos_sim = cosine_similarity(f0, f1)

                # ReID 코사인 유사도 제약 (REID_COSINE_THRESH = 0.80)
                if cos_sim >= REID_COSINE_THRESH and cos_sim > best_cos:
                    best_cos = cos_sim
                    best_idx = i

            if best_idx >= 0:   # 매칭 후보가 있다면
                b0 = t0_records[best_idx]["aligned_bbox"]
                current_dist = center_distance(b0, b1)

                # 최대 거리 제약 (MAX_REID_DIST = 200)
                if current_dist > MAX_REID_DIST: continue

                m_iou_final = compute_iou(b0, b1)
                m_size_sim_final = size_similarity(b0, b1)
                # Global ID 부여 및 기록
                current_id = t0_to_global_id[best_idx]
                if current_id == -1:
                    current_id = vehicle_global_id
                    t0_to_global_id[best_idx] = current_id
                    vehicle_global_id += 1
                # 매칭 결과 저장 (note: reid_assist)
                rec = {
                    "vehicle_id": current_id, "t0_image": t0_file, "t1_image": t1_file,
                    "t0_bbox": b0, "t1_bbox": b1,
                    "t0_bbox_orig": t0_records[best_idx]["original_bbox"],
                    "t1_bbox_orig": t1_records[j]["original_bbox"],
                    "iou": m_iou_final, "center_dist": current_dist,
                    "size_sim": m_size_sim_final,
                    "reid_similarity": best_cos, "final_score": ALPHA * m_iou_final + BETA * best_cos,
                    "note": "reid_assist",
                    "is_matched": 1
                }
                t1_assigned.add(j)
                t0_assigned_in_t0_file.add(best_idx)
                matched_pairs_t1.append(rec)
                matched_t1_count += 1



        # -------------------- 4단계: 신규 차량 처리 (미매칭된 T1) --------------------

        for j, r1 in enumerate(t1_records):
            # 1, 2단계에서 매칭되지 않은 모든 T1 박스를 'new'로 처리
            if j not in t1_assigned:
                rec = {
                    "vehicle_id": vehicle_global_id, "t0_image": "", "t1_image": t1_file,
                    "t0_bbox": [], "t1_bbox": r1["aligned_bbox"],
                    "t0_bbox_orig": [], "t1_bbox_orig": r1["original_bbox"],
                    "iou": 0.0, "center_dist": 0.0, "size_sim": 0.0,
                    "reid_similarity": 0.0, "final_score": 0.0, "note": "new",
                    "is_matched": 0
                }
                matched_pairs_t1.append(rec)
                vehicle_global_id += 1

        all_vehicle_records.extend(matched_pairs_t1)

# -------------------- 매칭CSV 저장 (경로명 변경) --------------------
#
single_csv_path = os.path.join(match_csv_folder, "matching_records.csv")
with open(single_csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "vehicle_id", "t0_image", "t1_image",
        "t0_x1", "t0_y1", "t0_x2", "t0_y2",  # 정렬된 좌표
        "t1_x1", "t1_y1", "t1_x2", "t1_y2",  # 정렬된 좌표
        "t0_orig_x1", "t0_orig_y1", "t0_orig_x2", "t0_orig_y2",  # 원본 좌표
        "t1_orig_x1", "t1_orig_y1", "t1_orig_x2", "t1_orig_y2",  # 원본 좌표
        "iou", "reid_similarity", "final_score", "note",
        "is_matched"
    ])
    for rec in all_vehicle_records:
        t0_bbox = rec["t0_bbox"] if rec["t0_bbox"] else ["", "", "", ""]
        t1_bbox = rec["t1_bbox"] if rec["t1_bbox"] else ["", "", "", ""]
        t0_bbox_orig = rec.get("t0_bbox_orig") if rec.get("t0_bbox_orig") else ["", "", "", ""]
        t1_bbox_orig = rec.get("t1_bbox_orig") if rec.get("t1_bbox_orig") else ["", "", "", ""]

        writer.writerow([
                            rec["vehicle_id"], rec["t0_image"], rec["t1_image"]
                        ] + t0_bbox + t1_bbox + t0_bbox_orig + t1_bbox_orig + [
                            rec.get("iou", 0.0), rec.get("reid_similarity", 0.0), rec.get("final_score", 0.0),
                            rec.get("note", ""),
                            rec.get("is_matched", 0)
                        ])

# -------------------- 결과 요약 출력 --------------------
match_rate = (matched_t1_count / total_t1_count * 100.0) if total_t1_count > 0 else 0.0
print(f"[완료] CSV 저장: {single_csv_path}")
print(f"총 t1 박스 수: {total_t1_count}, 매칭된 t1 수(신뢰기준 통과 포함): {matched_t1_count}, 매칭률: {match_rate:.2f}%")


