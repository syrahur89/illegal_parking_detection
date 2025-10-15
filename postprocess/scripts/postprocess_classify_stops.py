#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
차량 박스 → per-image 폴리곤 분류기
- stop_results CSV(필수): t1_img, xc, yc, t1_w, t1_h
- name_alias_for_t1 CSV(필수): image_name, template_key (예: DJI_0581)
- per_image 폴더(필수): DJI_05XX.geojson 파일들 (한 파일에 불법/합법/나무 등 구역이 모두 포함)
- 규칙: 폴리곤 속성의 문자열에 따라 클래스 매핑 (ILLEGAL/LEGAL/TREE/OTHER)
- 결과: classified_stops.csv (+ 옵션: overlays/*.png)

주의:
- 좌표계는 이미지 픽셀 좌표라고 가정(폴리곤도 같은 기준).
- 박스 중심점이 다중 폴리곤에 겹치면 우선순위: ILLEGAL > TREE > LEGAL > OTHER
"""

import os
import re
import argparse
import pandas as pd

# 지오패키지
import geopandas as gpd
from shapely.geometry import Point, Polygon, box

# --- 폴리곤 속성의 문자열에서 클래스 감지 규칙 ---
# key: 정규식(소문자), value: 표준 클래스명
CLASS_PATTERNS = [
    (r"(no[_\-\s]*parking|illegal|불법|금지|금정차|no[\s]*stop)", "ILLEGAL"),
    (r"(tree|나무|수목|교목|수관)", "TREE"),
    (r"(parking|합법|허용|allowed|legal)", "LEGAL"),
]

CLASS_PRIORITY = {"ILLEGAL": 3, "TREE": 2, "LEGAL": 1, "OTHER": 0}

def detect_class_from_props(props: dict) -> str:
    # 폴리곤의 속성들 중 문자열 항목을 훑어 규칙 매칭
    for k, v in props.items():
        if isinstance(v, str):
            s = v.strip().lower()
            for pat, cls in CLASS_PATTERNS:
                if re.search(pat, s):
                    return cls
    return "OTHER"

def load_per_image_geo(per_img_dir: str, template_key: str):
    """per_image 폴더에서 해당 DJI_XXXX.geojson 로드"""
    # 파일명은 대소문자 혼재 가능 → 소문자 비교
    target = None
    for fn in os.listdir(per_img_dir):
        if fn.lower() == f"{template_key.lower()}.geojson":
            target = os.path.join(per_img_dir, fn)
            break
    if target is None:
        return None, "NOT_FOUND"

    try:
        gdf = gpd.read_file(target)
        if gdf.empty:
            return gdf, "EMPTY"
        # 클래스 열이 명시되어 있지 않다면 속성 스캔으로 생성
        if "zone_class" not in gdf.columns:
            gdf["zone_class"] = gdf.apply(lambda r: detect_class_from_props(r.to_dict()), axis=1)
        return gdf, "OK"
    except Exception as e:
        return None, f"LOAD_ERROR:{e}"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stop-results", required=True)  # stop_results_clean.filtered.csv
    ap.add_argument("--t1-col", default="t1_img")
    ap.add_argument("--xc-col", default="xc")
    ap.add_argument("--yc-col", default="yc")
    ap.add_argument("--w-col", default="t1_w")
    ap.add_argument("--h-col", default="t1_h")

    ap.add_argument("--alias-csv", required=True)     # name_alias_for_t1.csv (image_name, template_key)
    ap.add_argument("--per-image-dir", required=True) # data\polygons\per_image
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--make-overlays", type=int, default=0)  # n장 생성 (0이면 미생성)
    ap.add_argument("--image-root")  # 오버레이용 원본 이미지 폴더 (선택)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # 1) 데이터 로드
    df = pd.read_csv(args.stop_results)
    miss_cols = [c for c in [args.t1_col, args.xc_col, args.yc_col, args.w_col, args.h_col] if c not in df.columns]
    if miss_cols:
        raise SystemExit(f"[FATAL] stop_results에 컬럼 없음: {miss_cols}")

    alias = pd.read_csv(args.alias_csv)
    if not {"image_name", "template_key"}.issubset(alias.columns):
        raise SystemExit("[FATAL] alias_csv는 'image_name, template_key' 컬럼이 필요")

    # 2) alias 조인으로 template_key 붙이기
    df = df.copy()
    df["__key__"] = df[args.t1_col].astype(str).str.strip()
    alias = alias.copy()
    alias["__key__"] = alias["image_name"].astype(str).str.strip()
    df = df.merge(alias[["__key__", "template_key"]], on="__key__", how="left")
    if df["template_key"].isna().any():
        na_cnt = int(df["template_key"].isna().sum())
        print(f"[WARN] template_key 누락 {na_cnt}건 → 해당 행은 'NO_ALIAS' 처리")

    # 3) 중심점 포인트 생성 (픽셀 좌표 가정)
    gdf_pts = gpd.GeoDataFrame(
        df,
        geometry=[Point(xy) for xy in zip(df[args.xc_col], df[args.yc_col])],
        crs=None  # 픽셀좌표이므로 명시 CRS 없음
    )

    # 4) 이미지별 분류
    results = []
    unique_keys = sorted(gdf_pts["template_key"].dropna().unique().tolist())
    for key in unique_keys:
        per_gdf, status = load_per_image_geo(args.per_image_dir, key)
        if per_gdf is None or per_gdf.empty:
            sub = gdf_pts[gdf_pts["template_key"] == key].copy()
            for _, r in sub.iterrows():
                results.append({
                    "image_name": r[args.t1_col],
                    "template_key": key,
                    "xc": r[args.xc_col], "yc": r[args.yc_col],
                    "t1_w": r[args.w_col], "t1_h": r[args.h_col],
                    "classification": "NO_POLYGON",
                    "detail": status
                })
            continue

        # 동일한 좌표계(픽셀) 가정: 직접 spatial join
        # polygon 클래스를 표준화
        per_gdf["zone_class"] = per_gdf["zone_class"].fillna("OTHER").astype(str)

        # 해당 key의 포인트만
        pts = gdf_pts[gdf_pts["template_key"] == key].copy()

        # 각 포인트가 포함되는 모든 폴리곤 후보 찾기
        join = gpd.sjoin(pts, per_gdf[["zone_class", "geometry"]], how="left", predicate="within")

        # 포인트별 다중매치 우선순위 적용
        for img_name, grp in join.groupby(by=args.t1_col):
            # 원본 행들
            rows = grp.sort_index()
            for idx, row in rows.iterrows():
                cand = grp.loc[idx]
                classes = []
                if isinstance(cand, pd.Series):
                    classes = [cand.get("zone_class")]
                else:
                    classes = cand["zone_class"].tolist()
                classes = [c for c in classes if isinstance(c, str)]
                if not classes or (len(classes) == 1 and pd.isna(classes[0])):
                    final = "OUTSIDE"
                    detail = ""
                else:
                    # 우선순위 최대
                    final = max(classes, key=lambda c: CLASS_PRIORITY.get(c, 0))
                    detail = ";".join(sorted(set([c for c in classes if pd.notna(c)])))

                src = df.loc[idx]  # 원본 행
                results.append({
                    "image_name": src[args.t1_col],
                    "template_key": key,
                    "xc": src[args.xc_col], "yc": src[args.yc_col],
                    "t1_w": src[args.w_col], "t1_h": src[args.h_col],
                    "classification": final,
                    "detail": detail
                })

    # alias가 없던 행 처리
    no_alias = gdf_pts[gdf_pts["template_key"].isna()]
    for _, r in no_alias.iterrows():
        results.append({
            "image_name": r[args.t1_col],
            "template_key": "",
            "xc": r[args.xc_col], "yc": r[args.yc_col],
            "t1_w": r[args.w_col], "t1_h": r[args.h_col],
            "classification": "NO_ALIAS",
            "detail": ""
        })

    out_df = pd.DataFrame(results)
    out_csv = os.path.join(args.out_dir, "classified_stops.csv")
    out_df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"[OK] 저장: {out_csv} (rows={len(out_df)})")

    # 5) (옵션) 오버레이 생성
    if args.make_overlays and args.image_root:
        try:
            from PIL import Image, ImageDraw, ImageFont
            import random

            ov_dir = os.path.join(args.out_dir, "overlays")
            os.makedirs(ov_dir, exist_ok=True)

            # 간단 색상 매핑
            COLORS = {
                "ILLEGAL": (255, 0, 0),
                "LEGAL": (0, 200, 0),
                "TREE": (0, 160, 255),
                "OTHER": (200, 200, 200),
                "OUTSIDE": (120, 120, 120),
                "NO_ALIAS": (180, 0, 180),
                "NO_POLYGON": (255, 128, 0),
            }

            # 이미지별 n장
            sample = out_df.groupby("image_name", as_index=False).count().head(args.make_overlays)["image_name"].tolist()
            for img_name in sample:
                # 이미지 찾기 (확장자/대소문자 섞임 가정)
                base = os.path.splitext(img_name)[0].lower()
                found = None
                for root, _, files in os.walk(args.image_root):
                    for fn in files:
                        if os.path.splitext(fn)[0].lower() == base:
                            found = os.path.join(root, fn); break
                    if found: break
                if not found:
                    continue

                im = Image.open(found).convert("RGB")
                draw = ImageDraw.Draw(im, "RGBA")

                # 해당 이미지의 결과들
                sub = out_df[out_df["image_name"] == img_name]
                for _, rr in sub.iterrows():
                    x, y, w, h = rr["xc"], rr["yc"], rr["t1_w"], rr["t1_h"]
                    x1, y1 = x - w/2, y - h/2
                    x2, y2 = x + w/2, y + h/2
                    color = COLORS.get(rr["classification"], (255, 255, 0))
                    draw.rectangle([x1, y1, x2, y2], outline=color + (255,), width=3)
                    draw.text((x1+3, y1+3), rr["classification"], fill=color + (200,))

                im.save(os.path.join(ov_dir, f"{img_name}.png"))
            print(f"[OK] 오버레이 생성: {ov_dir}")
        except Exception as e:
            print(f"[WARN] 오버레이 생성 실패: {e}")

if __name__ == "__main__":
    main()
