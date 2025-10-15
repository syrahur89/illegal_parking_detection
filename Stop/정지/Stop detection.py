import pandas as pd
import numpy as np
import math
import os
import sys

# 파일 경로
matching_csv = "matching_csv/matching_records.csv" #매칭결과 CSV
pair_csv = "t0t1_pair.csv"   # T0/T1 파일 쌍 및 시간 간격(delta_min) 정보 포함 CSV
output_csv = "stop_results.csv" #정지결과 CSV


# -------------------- 튜닝 파라미터  --------------------
MOVE_RATIO_THRESHOLD = 0.60  # 정지 차량으로 판정하기 위한 최대 이동 비율 (이동 거리 / T0 BBox 대각선)
DELTA_TIME_THRESHOLD = 5     # 최소 시간 간격(5분)
DIAG_RATIO_MIN = 0.50        # T1 BBox 대각선 길이가 T0의 최소 50%를 유지해야 함 (급격한 크기 변화 방지)
DIAG_RATIO_MAX = 1.50        # T1 BBox 대각선 길이가 T0의 최대 150%를 초과하지 않아야 함

# 데이터 불러오기
try:
    df = pd.read_csv(matching_csv)  # 매칭 결과 로드 (vehicle_id, BBox, note 등)
    pairs = pd.read_csv(pair_csv)  # 파일 쌍 및 시간 정보 로드
except FileNotFoundError as e:
    print(f"[오류] 파일 경로를 확인해주세요: {e}")
    sys.exit()

# zone_id 존재 여부 체크
has_zone_id = "zone_id" in pairs.columns

# 결과 저장용 리스트
results = []

for _, row in df.iterrows():
    # -------------------- 1. 매칭 조건 (hungarian과 reid_assist 모두 처리) --------------------
    if row["note"] not in ["hungarian", "reid_assist"]:
        continue

    # -------------------- 2. 좌표 및 크기 계산 --------------------
    #T0/T1의 정렬된 BBox 좌표를 가져옴 (차량 이동 거리 계산에 사용)
    try:
        x1_0, y1_0, x2_0, y2_0 = row["t0_x1"], row["t0_y1"], row["t0_x2"], row["t0_y2"]
        x1_1, y1_1, x2_1, y2_1 = row["t1_x1"], row["t1_y1"], row["t1_x2"], row["t1_y2"]
    except (KeyError, TypeError):
        continue

    # # T0/T1의 원본 BBox 좌표를 가져옴 (정지결과 CSV에 기록)
    try:
        orig_x1_0, orig_y1_0, orig_x2_0, orig_y2_0 = row["t0_orig_x1"], row["t0_orig_y1"], row["t0_orig_x2"], row[
            "t0_orig_y2"]
        orig_x1_1, orig_y1_1, orig_x2_1, orig_y2_1 = row["t1_orig_x1"], row["t1_orig_y1"], row["t1_orig_x2"], row[
            "t1_orig_y2"]
    except KeyError:
        orig_x1_0, orig_y1_0, orig_x2_0, orig_y2_0 = np.nan, np.nan, np.nan, np.nan
        orig_x1_1, orig_y1_1, orig_x2_1, orig_y2_1 = np.nan, np.nan, np.nan, np.nan
    # 원본 BBox의 너비와 높이 계산
    orig_w0 = abs(orig_x2_0 - orig_x1_0) if not np.isnan(orig_x1_0) else np.nan
    orig_h0 = abs(orig_y2_0 - orig_y1_0) if not np.isnan(orig_y1_0) else np.nan
    orig_w1 = abs(orig_x2_1 - orig_x1_1) if not np.isnan(orig_x1_1) else np.nan
    orig_h1 = abs(orig_y2_1 - orig_y1_1) if not np.isnan(orig_y1_1) else np.nan

    # 중심 좌표 (정렬 좌표 사용)
    cx0, cy0 = (x1_0 + x2_0) / 2, (y1_0 + y2_0) / 2
    cx1, cy1 = (x1_1 + x2_1) / 2, (y1_1 + y2_1) / 2

    # 이동 거리 (유클리드 거리)
    move_dist = math.sqrt((cx1 - cx0) ** 2 + (cy1 - cy0) ** 2)

    # t0/t1 bbox 너비/높이 및 대각선 길이 (정렬 좌표 기반)
    w0, h0 = x2_0 - x1_0, y2_0 - y1_0
    w1, h1 = x2_1 - x1_1, y2_1 - y1_1
    diag0 = math.sqrt(w0 ** 2 + h0 ** 2)  # T0 BBox 대각선 길이 (이동 비율의 기준 크기)
    diag1 = math.sqrt(w1 ** 2 + h1 ** 2)  # T1 BBox 대각선 길이


    # BBox 크기 변화율 (T1 대각선 / T0 대각선) 계산
    diag_ratio = diag1 / diag0 if diag0 > 0 else np.nan

    # T0/T1 이미지 이름 처리(t0t1_pair.csv 파일명과 매칭시키기 위해)
    t0_name = os.path.splitext(row["t0_image"].replace("_detect.csv", ""))[0]
    t1_name = os.path.splitext(row["t1_image"].replace("_detect.csv", ""))[0]

    # delta_min & zone_id 찾기
    delta_row = pairs[(pairs["t0_filename"] == t0_name) & (pairs["t1_filename"] == t1_name)]
    if len(delta_row) > 0:
        delta_min = delta_row.iloc[0]["delta_min"]
        zone_id = delta_row.iloc[0]["zone_id"] if has_zone_id else ""
    else:
        continue

    # 이동 비율(정렬된 BBox 중심점 이동 거리 / T0 BBox 대각선 길이)
    move_ratio = move_dist / diag0 if diag0 > 0 else 0

    # -------------------- 3. 정지 조건 체크 (5분 간격 기준 적용) --------------------
    cond1 = move_ratio <= MOVE_RATIO_THRESHOLD  #조건 1: 이동 비율이 임계값(0.60) 이하
    cond2 = delta_min >= DELTA_TIME_THRESHOLD  # 조건 2: T0/T1 간 시간 간격이 최소 임계값(5분) 이상
    cond3 = (diag_ratio >= DIAG_RATIO_MIN) and (diag_ratio <= DIAG_RATIO_MAX) # 조건 3: T0와 T1의 BBox 크기 변화율이 허용 범위(0.50 ~ 1.50) 내에 있음

    # 최종 판정: (낮은 이동량 AND 충분한 시간) AND (낮은 크기 변화)를 모두 만족하면 'Yes'(정지)
    stop_flag = "Yes" if (cond1 and cond2 and cond3) else "No"

    # -------------------- 4. 결과 리스트에 추가 --------------------
    results.append({
        "vehicle_id": row["vehicle_id"],
        "zone_id": zone_id,
        "t0_image": t0_name,
        "t1_image": t1_name,

        # 원본 좌표 및 크기 추가
        "t0_orig_x1": orig_x1_0, "t0_orig_y1": orig_y1_0,
        "t0_orig_x2": orig_x2_0, "t0_orig_y2": orig_y2_0,
        "t0_orig_w": round(orig_w0, 3) if not np.isnan(orig_w0) else np.nan,
        "t0_orig_h": round(orig_h0, 3) if not np.isnan(orig_h0) else np.nan,

        "t1_orig_x1": orig_x1_1, "t1_orig_y1": orig_y1_1,
        "t1_orig_x2": orig_x2_1, "t1_orig_y2": orig_y2_1,
        "t1_orig_w": round(orig_w1, 3) if not np.isnan(orig_w1) else np.nan,
        "t1_orig_h": round(orig_h1, 3) if not np.isnan(orig_h1) else np.nan,

        # 정지 판정 관련 계산 값
        "move_dist": round(move_dist, 3),  # T0/T1 중심점 간의 정렬된 거리
        "diag0": round(diag0, 3),  # T0 BBox 대각선 길이 (기준)
        "diag1": round(diag1, 3),  # T1 BBox 대각선 길이
        "diag_ratio": round(diag_ratio, 3),  # BBox 크기 변화율 (diag1 / diag0)
        "delta_min": delta_min, # T0/T1 간의 시간 간격 (분)
        "move_ratio": round(move_ratio, 3),   # 이동 비율 (move_dist / diag0)
        "stop": stop_flag  # 최종 정지 판정
    })

# 결과 저장
out_df = pd.DataFrame(results)
out_df.to_csv(output_csv, index=False)
print(f"정지 판정 결과 저장 완료: {output_csv}")
print(f"총 {len(results)}개의 Hungarian 및 ReID Assist 매칭 차량에 대해 정지 판정을 수행했습니다.")