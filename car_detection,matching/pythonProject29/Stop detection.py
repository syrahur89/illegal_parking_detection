import pandas as pd
import numpy as np
import math
import os

# 파일 경로
matching_csv = "matching_csv/all_matching_vehicles_refined_final.csv"
pair_csv = "t0t1_two.csv"
output_csv = "stop_results.csv"

# 데이터 불러오기
df = pd.read_csv(matching_csv)
pairs = pd.read_csv(pair_csv)

# zone_id 존재 여부 체크
has_zone_id = "zone_id" in pairs.columns

# 결과 저장용 리스트
results = []

for _, row in df.iterrows():
    # note가 hungarian 또는 reid_assist인 경우만 사용 (매칭 성공 차량)
    if row["note"] not in ["hungarian", "reid_assist"]:
        continue

    # t0/t1 bbox 좌표
    x1_0, y1_0, x2_0, y2_0 = row["t0_x1"], row["t0_y1"], row["t0_x2"], row["t0_y2"]
    x1_1, y1_1, x2_1, y2_1 = row["t1_x1"], row["t1_y1"], row["t1_x2"], row["t1_y2"]

    # 중심 좌표
    cx0, cy0 = (x1_0 + x2_0) / 2, (y1_0 + y2_0) / 2
    cx1, cy1 = (x1_1 + x2_1) / 2, (y1_1 + y2_1) / 2

    # 이동 거리
    move_dist = math.sqrt((cx1 - cx0) ** 2 + (cy1 - cy0) ** 2)

    # t0 bbox 대각선 길이
    w0, h0 = x2_0 - x1_0, y2_0 - y1_0
    diag0 = math.sqrt(w0 ** 2 + h0 ** 2)

    # t0/t1 이미지 이름에서 detect.csv 제거
    t0_name = os.path.splitext(row["t0_image"].replace("_detect.csv",""))[0]
    t1_name = os.path.splitext(row["t1_image"].replace("_detect.csv",""))[0]

    # delta_min & zone_id 찾기
    delta_row = pairs[(pairs["t0_filename"] == t0_name) & (pairs["t1_filename"] == t1_name)]
    if len(delta_row) > 0:
        delta_min = delta_row.iloc[0]["delta_min"]
        zone_id = delta_row.iloc[0]["zone_id"] if has_zone_id else ""
    else:
        delta_min = 0
        zone_id = ""

    # 이동 비율
    move_ratio = move_dist / diag0 if diag0 > 0 else 0

    # 조건 체크
    cond1 = move_ratio <= 0.15   # 이동량 ≤ bbox 대각선의 15%
    cond2 = delta_min >= 5       # Δt ≥ 5분
    stop_flag = "Yes" if (cond1 and cond2) else "No"

    results.append({
        "vehicle_id": row["vehicle_id"],
        "zone_id": zone_id,
        "t0_image": t0_name,
        "t1_image": t1_name,
        "move_dist": round(move_dist, 3),
        "diag0": round(diag0, 3),
        "delta_min": delta_min,
        "move_ratio": round(move_ratio, 3),
        "stop": stop_flag
    })

# 결과 저장
out_df = pd.DataFrame(results)
out_df.to_csv(output_csv, index=False)
print(f"정지 판정 결과 저장 완료: {output_csv}")
