#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
후처리 통합 점검 스크립트 (확장자/대소문자 무시 매칭 지원, quicklook 타이틀 버그 픽스)
- stop_results_* 에서 T1 이미지 목록 추출(컬럼명 지정 가능)
- (선택) pairs.csv, name_alias/templates_map.csv 조인 확인
- (선택) Geo 레이어 요약(설치되어 있으면)
- 이미지 폴더를 인덱싱하여 확장자/대소문자 혼재 대응
- 이미지별 진단 리포트 및 퀵룩 생성
"""

import os
import sys
import argparse
import pandas as pd

_GEO_OK = True
try:
    import geopandas as gpd
except Exception:
    _GEO_OK = False

EXT_ORDER = {".jpg": 0, ".jpeg": 1, ".png": 2}

def read_csv_safe(path, msg):
    if not path:
        return None
    if not os.path.exists(path):
        print(f"[WARN] {msg} 경로 없음: {path}")
        return None
    try:
        return pd.read_csv(path)
    except Exception as e:
        print(f"[ERROR] {msg} 읽기 실패: {e}")
        return None

def index_images(root):
    table = {}
    for dirpath, _, files in os.walk(root):
        for fn in files:
            base, ext = os.path.splitext(fn)
            key = base.lower()
            ext_l = ext.lower()
            path = os.path.join(dirpath, fn)
            prev = table.get(key)
            if prev is None or EXT_ORDER.get(ext_l, 9) < EXT_ORDER.get(prev[1], 9):
                table[key] = (path, ext_l)
    return table

def load_geojson(path):
    if not path or not _GEO_OK:
        return None, "GEO_DISABLED" if not _GEO_OK else "NO_PATH"
    if not os.path.exists(path):
        return None, "NOT_FOUND"
    try:
        gdf = gpd.read_file(path)
        if gdf.empty:
            return gdf, "EMPTY"
        gdf["is_valid"] = gdf.geometry.is_valid
        if (~gdf["is_valid"]).any():
            gdf.geometry = gdf.buffer(0)
            gdf["is_valid"] = gdf.geometry.is_valid
        return gdf, "OK" if gdf["is_valid"].all() else "FIXED_SOME"
    except Exception as e:
        return None, f"LOAD_ERROR:{e}"

def coerce_crs(gdf, target_crs):
    if gdf is None or target_crs is None or not _GEO_OK:
        return gdf, False, "SKIP"
    try:
        if gdf.crs is None:
            return None, False, "NO_CRS"
        if str(gdf.crs) != str(target_crs):
            gdf = gdf.to_crs(target_crs)
            return gdf, True, "REPROJECTED"
        return gdf, True, "UNCHANGED"
    except Exception as e:
        return None, False, f"CRS_ERROR:{e}"

def summarize_geo_layer(name, gdf, status):
    if gdf is None:
        return {"name": name, "status": status, "crs": None, "count": 0, "valid_all": None, "bounds": None}
    return {
        "name": name, "status": status, "crs": str(gdf.crs),
        "count": len(gdf), "valid_all": bool(gdf.get("is_valid", pd.Series([True]*len(gdf))).all()),
        "bounds": list(gdf.total_bounds) if len(gdf) else None
    }

def detect_tpl_cols(df_tpl):
    cand_img = ["image_name", "img_name", "image", "alias", "img"]
    cand_key = ["template_key", "template", "poly_name", "polygon_name", "geo_name", "name"]
    cols = {c.lower(): c for c in df_tpl.columns}
    img_col = next((cols[c] for c in cand_img if c in cols), None)
    key_col = next((cols[c] for c in cand_key if c in cols), None)
    if img_col is None or key_col is None:
        if len(df_tpl.columns) >= 2:
            img_col = img_col or df_tpl.columns[0]
            key_col = key_col or df_tpl.columns[1]
    return img_col, key_col

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stop-results", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--t1-col", default="t1_image")
    ap.add_argument("--pairs-csv")
    ap.add_argument("--pairs-t1-col", default="t1_image")
    ap.add_argument("--templates-csv")
    ap.add_argument("--image-root")
    ap.add_argument("--boundary")
    ap.add_argument("--parking-area")
    ap.add_argument("--no-parking-area")
    ap.add_argument("--target-crs")
    ap.add_argument("--quicklook", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    df_stop = read_csv_safe(args.stop_results, "stop_results")
    if df_stop is None:
        print("[FATAL] stop_results 로드 실패"); sys.exit(1)
    if args.t1_col not in df_stop.columns:
        print(f"[FATAL] stop_results에 '{args.t1_col}' 컬럼 없음. 실제 컬럼: {list(df_stop.columns)[:20]}"); sys.exit(2)

    t1_list = (
        df_stop[args.t1_col].dropna().astype(str).str.strip().unique().tolist()
    )
    pd.DataFrame({"t1_image": t1_list}).to_csv(os.path.join(args.out_dir, "t1_list.csv"), index=False)
    print(f"[INFO] T1 이미지 수: {len(t1_list)}  → t1_list.csv")

    df_pairs = read_csv_safe(args.pairs_csv, "pairs.csv") if args.pairs_csv else None
    if df_pairs is not None and args.pairs_t1_col not in df_pairs.columns:
        print(f"[WARN] pairs CSV에 '{args.pairs_t1_col}' 컬럼 없음 → pairs 조인 생략"); df_pairs = None

    df_tpl = read_csv_safe(args.templates_csv, "templates/name_alias") if args.templates_csv else None
    if df_tpl is not None:
        tpl_img_col, tpl_key_col = detect_tpl_cols(df_tpl)
        if tpl_img_col and tpl_key_col:
            df_tpl = df_tpl.rename(columns={tpl_img_col: "__image_name__", tpl_key_col: "__template_key__"})
        else:
            print("[WARN] templates CSV에서 이미지/키 컬럼 추정 실패 → 템플릿 매핑 스킵"); df_tpl = None

    img_index = {}
    if args.image_root and os.path.isdir(args.image_root):
        img_index = index_images(args.image_root)
        print(f"[INFO] 이미지 인덱스 완료: {len(img_index)}개 베이스명")

    if _GEO_OK and any([args.boundary, args.parking_area, args.no_parking_area]):
        boundary_gdf, b_status = load_geojson(args.boundary)
        legal_gdf, pa_status = load_geojson(args.parking_area)
        illegal_gdf, np_status = load_geojson(args.no_parking_area)
        if args.target_crs:
            boundary_gdf, _, b_note = coerce_crs(boundary_gdf, args.target_crs)
            legal_gdf, _, l_note = coerce_crs(legal_gdf, args.target_crs)
            illegal_gdf, _, n_note = coerce_crs(illegal_gdf, args.target_crs)
        else:
            b_note = l_note = n_note = "NO_TARGET"
        pd.DataFrame([
            summarize_geo_layer("boundary", boundary_gdf, f"{b_status}|{b_note}"),
            summarize_geo_layer("parking_area", legal_gdf, f"{pa_status}|{l_note}"),
            summarize_geo_layer("no_parking_area", illegal_gdf, f"{np_status}|{n_note}"),
        ]).to_csv(os.path.join(args.out_dir, "geo_layers_summary.csv"), index=False)
        print("[INFO] Geo 레이어 요약 저장 → geo_layers_summary.csv")

    rows = []
    t1_set = set(t1_list)
    for img in t1_list:
        rec = {
            "image_name": img,
            "exists_on_disk": None,
            "resolved_path": None,
            "resolved_ext": None,
            "template_found": None,
            "pairs_found": None,
            "geo_ready": None,
            "reason": [],
        }

        if img_index:
            base = os.path.splitext(img)[0].lower()
            hit = img_index.get(base)
            if hit:
                rec["exists_on_disk"] = True
                rec["resolved_path"], rec["resolved_ext"] = hit[0], hit[1]
            else:
                rec["exists_on_disk"] = False
                rec["reason"].append("IMG_MISSING")

        if df_tpl is not None:
            hit = df_tpl.loc[df_tpl["__image_name__"].astype(str).str.strip() == img]
            rec["template_found"] = len(hit) > 0
            if not rec["template_found"]:
                rec["reason"].append("NO_TEMPLATE")

        if df_pairs is not None:
            hit = df_pairs.loc[df_pairs[args.pairs_t1_col].astype(str).str.strip() == img]
            rec["pairs_found"] = len(hit) > 0
            if not rec["pairs_found"]:
                rec["reason"].append("NO_PAIR")

        if any([args.boundary, args.parking_area, args.no_parking_area]):
            rec["geo_ready"] = True  # 요약 플래그 수준

        if not rec["reason"]:
            rec["reason"] = ["OK"]
        rows.append(rec)

    diag = pd.DataFrame(rows)
    diag["reason_str"] = diag["reason"].apply(lambda x: ";".join(x))
    diag.to_csv(os.path.join(args.out_dir, "diagnostics.csv"), index=False, encoding="utf-8-sig")
    print(f"[INFO] 진단 결과 저장 → diagnostics.csv (총 {len(diag)}행)")

    if df_tpl is not None:
        out_map = df_tpl[df_tpl["__image_name__"].astype(str).str.strip().isin(t1_set)]
        out_map.rename(columns={"__image_name__": "image_name", "__template_key__": "template_key"}).to_csv(
            os.path.join(args.out_dir, "template_map.csv"), index=False
        )
        print(f"[INFO] 템플릿 매핑 저장 → template_map.csv (행 {len(out_map)})")

    if args.quicklook and img_index:
        try:
            import matplotlib.pyplot as plt
            from PIL import Image
            qdir = os.path.join(args.out_dir, "quicklooks")
            os.makedirs(qdir, exist_ok=True)
            sample = [r for r in rows if r.get("exists_on_disk")] [: args.quicklook]
            for rec in sample:
                try:
                    im = Image.open(rec["resolved_path"])
                    plt.figure()
                    # 간단 타이틀(딕셔너리 단계에서 reason_str 없음 → 리스트로 구성)
                    title = f"{os.path.basename(rec['resolved_path'])} | {';'.join(rec['reason'])}"
                    plt.imshow(im); plt.title(title); plt.axis('off')
                    out_png = os.path.join(qdir, os.path.splitext(os.path.basename(rec["resolved_path"]))[0] + ".png")
                    plt.savefig(out_png, dpi=150, bbox_inches='tight'); plt.close()
                except Exception as e:
                    print(f"[WARN] quicklook 실패: {rec['resolved_path']} ({e})")
            print(f"[INFO] Quicklooks 생성 → {qdir}")
        except Exception as e:
            print(f"[WARN] matplotlib/PIL 오류로 Quicklook 스킵: {e}")

if __name__ == "__main__":
    main()
