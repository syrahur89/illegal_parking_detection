#!/usr/bin/env python
# -*- coding: utf-8 -*-
# VER: proxy_quality_metrics_vFINAL_rowid
"""
Proxy 성능 지표 산출 (정답 라벨 없음)
입력:  classified_stops.csv  (image_name, template_key, xc, yc, t1_w, t1_h, classification)
보조:  per_image\DJI_05XX.geojson
출력:  proxy_per_box.csv, proxy_kpi.csv, proxy_agreement_by_image.csv
"""

import os
import argparse
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon
from shapely.ops import unary_union

def box_poly(cx, cy, w, h):
    return Polygon([(cx-w/2,cy-h/2),(cx+w/2,cy-h/2),(cx+w/2,cy+h/2),(cx-w/2,cy+h/2)])

def find_geojson(per_image_dir: str, template_key: str):
    want = f"{str(template_key).lower()}.geojson"
    for fn in os.listdir(per_image_dir):
        if fn.lower() == want:
            return os.path.join(per_image_dir, fn)
    return None

def safe_union(geoms):
    fixed=[]
    for g in geoms:
        try:
            if g is None or g.is_empty:
                continue
            if not g.is_valid:
                g = g.buffer(0)
            if not g.is_empty:
                fixed.append(g)
        except Exception:
            continue
    if not fixed:
        return None
    try:
        return unary_union(fixed)
    except Exception:
        return unary_union([g.buffer(0) for g in fixed])

def main():
    print("[START] proxy_quality_metrics_vFINAL_rowid")
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--per-image-dir", required=True)
    ap.add_argument("--overlap_thr", type=float, default=0.10)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_csv(args.pred)
    need = {"image_name","template_key","xc","yc","t1_w","t1_h","classification"}
    miss = sorted(list(need - set(df.columns)))
    if miss:
        raise SystemExit(f"[FATAL] classified_stops.csv 컬럼 누락: {miss}")

    # 고유 ID 부여(원본 index 사용)
    df = df.copy()
    df = df[df["template_key"].notna()]
    df["row_id"] = df.index.astype(int)

    rows=[]
    for key, sub in df.groupby("template_key"):
        gpath = find_geojson(args.per_image_dir, key)
        if not gpath:
            for _,r in sub.iterrows():
                rows.append((int(r["row_id"]), r["image_name"], key, "NO_POLYGON", 0.0, None))
            continue
        try:
            gdf = gpd.read_file(gpath)
        except Exception:
            for _,r in sub.iterrows():
                rows.append((int(r["row_id"]), r["image_name"], key, "LOAD_ERR", 0.0, None))
            continue
        if gdf.empty or gdf.geometry.isna().all():
            for _,r in sub.iterrows():
                rows.append((int(r["row_id"]), r["image_name"], key, "NO_POLYGON", 0.0, None))
            continue

        union_all = safe_union(gdf.geometry)
        if union_all is None or union_all.is_empty:
            for _,r in sub.iterrows():
                rows.append((int(r["row_id"]), r["image_name"], key, "NO_POLYGON", 0.0, None))
            continue

        for _,r in sub.iterrows():
            cx,cy,w,h = float(r["xc"]), float(r["yc"]), float(r["t1_w"]), float(r["t1_h"])
            pbox = box_poly(cx,cy,w,h)
            inter_area = pbox.intersection(union_all).area
            ratio = inter_area / (pbox.area + 1e-9)
            try:
                dist = pbox.centroid.distance(union_all.boundary)
            except Exception:
                dist = None
            flag = "OVERLAP" if ratio >= args.overlap_thr else "NO_OVERLAP"
            rows.append((int(r["row_id"]), r["image_name"], key, flag, ratio, dist))

    # per-box 지표 저장 (row_id 포함)
    per_box = pd.DataFrame(
        rows,
        columns=["row_id","image_name","template_key","overlap_flag","overlap_ratio","border_dist"]
    )
    per_box["near_border"] = per_box["border_dist"].apply(lambda d: 1 if (pd.notna(d) and d <= 5) else 0)
    per_box_path = os.path.join(args.out_dir, "proxy_per_box.csv")
    per_box.to_csv(per_box_path, index=False, encoding="utf-8-sig")

    # KPI (to_frame 사용 안 함)
    n = int(len(per_box))
    overlap_rate    = float((per_box["overlap_flag"] == "OVERLAP").mean()) if n>0 else 0.0
    near_border_rate= float(per_box["near_border"].mean())                 if n>0 else 0.0
    avg_overlap     = float(per_box["overlap_ratio"].mean())               if n>0 else 0.0
    kpi_df = pd.DataFrame([{
        "n": n,
        "overlap_rate": round(overlap_rate, 6),
        "near_border_rate": round(near_border_rate, 6),
        "avg_overlap": round(avg_overlap, 6),
        "overlap_thr": args.overlap_thr
    }])
    kpi_path = os.path.join(args.out_dir, "proxy_kpi.csv")
    kpi_df.to_csv(kpi_path, index=False, encoding="utf-8-sig")

    # 합의율(이미지별): row_id로 조인
    merged = df[["row_id","image_name","classification"]].merge(
        per_box[["row_id","overlap_flag"]],
        on="row_id", how="left"
    )
    agree_mask = (
        ((merged["classification"] != "OUTSIDE") & (merged["overlap_flag"] == "OVERLAP")) |
        ((merged["classification"] == "OUTSIDE") & (merged["overlap_flag"] == "NO_OVERLAP"))
    )
    merged["agreement"] = agree_mask.astype(int)
    by_img = merged.groupby("image_name", as_index=False)["agreement"].mean()
    agree_path = os.path.join(args.out_dir, "proxy_agreement_by_image.csv")
    by_img.to_csv(agree_path, index=False, encoding="utf-8-sig")

    print(f"[OK] 저장 완료:\n - {per_box_path}\n - {kpi_path}\n - {agree_path}")

if __name__ == "__main__":
    main()
