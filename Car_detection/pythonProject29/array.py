import os
import csv
import numpy as np
import cv2
import sys

detect_csv_folder = "detect_csv" #탐지 bbox csv
align_folder = "H"  #이미지별 변환행렬
output_folder = "aligned_csv" #정렬후 좌표저장
os.makedirs(output_folder, exist_ok=True)

'''BBox좌표에 호모그래피 변환적용'''
def apply_homography(x1, y1, x2, y2, H):

    # bbox 꼭짓점 4개 정의
    pts = np.array([
        [x1, y1],
        [x2, y1],
        [x2, y2],
        [x1, y2]
    ], dtype=np.float32).reshape(-1, 1, 2)

    # 변환
    transformed = cv2.perspectiveTransform(pts, H).reshape(-1, 2)

    # 새로운 bbox 좌표
    new_x1, new_y1 = transformed[:, 0].min(), transformed[:, 1].min()
    new_x2, new_y2 = transformed[:, 0].max(), transformed[:, 1].max()
    return int(new_x1), int(new_y1), int(new_x2), int(new_y2)


for csv_file in os.listdir(detect_csv_folder):
    if not csv_file.endswith(".csv"):
        continue

    csv_path = os.path.join(detect_csv_folder, csv_file)
    out_path = os.path.join(output_folder, csv_file)

    rows = []

    # 처리 메시지 출력
    print(f"처리 중: {csv_file}")

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames_list = reader.fieldnames

        for row in reader:
            img_name = row["image"]

            # BBox 좌표를 안전하게 정수형으로 변환
            try:
                x1, y1, x2, y2 = map(int, [row["x1"], row["y1"], row["x2"], row["y2"]])
            except ValueError:
                print(f"[경고] {csv_file} 파일의 레코드에서 BBox 좌표가 유효하지 않음. 건너뜀.")
                continue

            # 대응되는 npy 파일 경로 생성
            npy_name = os.path.splitext(img_name)[0] + ".npy"
            npy_path = os.path.join(align_folder, npy_name)

            H = None
            if os.path.exists(npy_path):
                # npy 파일이 있으면 변환 행렬 로드
                H = np.load(npy_path)
            else:
                print(f"[알림] 변환행렬 없음: {npy_path}. 원본 좌표를 정렬 좌표로 사용합니다.")

            # 원본 좌표 저장
            row["x1_original"] = x1
            row["y1_original"] = y1
            row["x2_original"] = x2
            row["y2_original"] = y2

            if H is not None:
                # 1. 변환 행렬이 있는 경우: bbox 변환 적용
                new_x1, new_y1, new_x2, new_y2 = apply_homography(x1, y1, x2, y2, H)
            else:
                # 2. 변환 행렬이 없는 경우: 원본 좌표를 정렬 좌표로 사용
                new_x1, new_y1, new_x2, new_y2 = x1, y1, x2, y2

            # width, height, area 다시 계산
            width = new_x2 - new_x1
            height = new_y2 - new_y1
            area = width * height

            # 변환된/대체된 좌표와 값 덮어쓰기
            row["x1"] = new_x1
            row["y1"] = new_y1
            row["x2"] = new_x2
            row["y2"] = new_y2
            row["width"] = width
            row["height"] = height
            row["area"] = area

            rows.append(row)

    # 저장
    if rows:
        # 필드 순서: 원래 필드명 + 추가된 original 컬럼
        # 원본 CSV의 필드명을 기반으로 하되, 새로운 필드를 끝에 추가합니다.
        new_fieldnames = fieldnames_list + ["x1_original", "y1_original", "x2_original", "y2_original"]

        # 실제 데이터의 필드명 목록을 사용하여 최종 fieldnames를 결정
        final_fieldnames = list(rows[0].keys())

        with open(out_path, "w", newline="", encoding="utf-8") as f:
            # 원본 필드 + 추가된 필드를 포함하는 fieldnames를 사용
            writer = csv.DictWriter(f, fieldnames=final_fieldnames)
            writer.writeheader()
            writer.writerows(rows)

print("\n변환된 CSV 저장 완료:", output_folder)

