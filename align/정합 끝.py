#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch alignment pipeline
- Produces full outputs under `align/` and logging/summary artifacts.
"""

import os
import sys
import json
import shutil
import math
import logging
from datetime import datetime
from typing import Tuple, Dict, Any, List

import numpy as np
import pandas as pd
import cv2
from tkinter import Tk, filedialog, messagebox, simpledialog

# ------------------ 0. 설정 (사용자 변경 가능) ------------------
# 입력 디렉토 / 파일 규칙 (스크립트 실행 시 확인)
TEMPLATE_CSV = "templates/zoneA_landmarks.csv"  # 템플릿 LM 목록
LANDMARKS_DIR = "landmarks"  # 각 이미지 LM: <image_id>.csv
MASKS_DIR = "masks"  # mask geojsons (no_parking, legal 등)

# 출력 루트 (프로젝트 루트하위 align/)
OUT_ROOT = "align"

# --- [요청사항 수정] ---
# 최소 인라이어 개수 기준을 20에서 4로 대폭 완화하여 RMSE가 좋으면 통과되도록 함
MIN_INLIERS = 4

# 상수
MIN_POINTS_FOR_HOMOGRAPHY = 4
RMSE_MAX = 15.0
Z_SCORE_THRESHOLD = 4.0

# RANSAC / 호모그래피 파라미터
RANSAC_REPROJ_THRESH = 5.0
RANSAC_MAX_ITERS = 2000

# preview 관련
PREVIEW_SIZE_MAX = 2048  # 생성될 preview 이미지의 한 변 최대 픽셀 (리샘플링 용)

# 로그 경로
LOGS_DIR = os.path.join(OUT_ROOT, "logs")
RUNTIME_LOG = os.path.join(LOGS_DIR, "runtime.log")
FAILURES_TXT = os.path.join(LOGS_DIR, "failures.txt")
SKIPPED_LOG = os.path.join(LOGS_DIR, "skipped.txt")

# summary paths
SUMMARY_DIR = os.path.join(OUT_ROOT, "summary")
METRICS_CSV = os.path.join(SUMMARY_DIR, "metrics.csv")
METRICS_SUMMARY_JSON = os.path.join(SUMMARY_DIR, "metrics_summary.json")
THRESHOLDS_USED_JSON = os.path.join(SUMMARY_DIR, "thresholds_used.json")

# manual backup dir
MANUAL_DIR = "manual"

# 기타
ENCODING = "utf-8"


# ------------------ 1. 디렉토리 생성 및 로깅 초기화 ------------------
def ensure_dirs():
    os.makedirs(OUT_ROOT, exist_ok=True)
    os.makedirs(os.path.join(OUT_ROOT, "H"), exist_ok=True)
    os.makedirs(os.path.join(OUT_ROOT, "quality"), exist_ok=True)
    os.makedirs(os.path.join(OUT_ROOT, "polygons"), exist_ok=True)
    os.makedirs(os.path.join(OUT_ROOT, "previews"), exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)
    os.makedirs(SUMMARY_DIR, exist_ok=True)
    os.makedirs(MANUAL_DIR, exist_ok=True)


def init_logging():
    ensure_dirs()
    logger = logging.getLogger("aligner")
    logger.setLevel(logging.INFO)
    # 파일 handler
    fh = logging.FileHandler(RUNTIME_LOG, mode="w", encoding=ENCODING)
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    fh.setFormatter(formatter)
    # console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.handlers = []
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


logger = init_logging()


# ------------------ 2. 유틸: CSV 로드 및 표준화 ------------------
def load_csv_any(path: str) -> pd.DataFrame:
    """CSV load with fallback encodings and basic validation."""
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    try:
        df = pd.read_csv(path, encoding="utf-8")
    except Exception:
        df = pd.read_csv(path, encoding="cp949", engine="python")
    return df


def standardize_landmark_df(df: pd.DataFrame, filename_for_log: str) -> pd.DataFrame:
    """
    Ensure columns named to x,y,LM (LM string).
    Accept common variants.
    """
    x_candidates = ['x', 'X', 'point_x', 'ux', 'lon', 'longitude']
    y_candidates = ['y', 'Y', 'point_y', 'uy', 'lat', 'latitude']
    lm_candidates = ['LM', 'lm', 'id', 'point_idx', 'landmark_id', 'name']

    cols = df.columns.tolist()
    x_col = next((c for c in x_candidates if c in cols), None)
    y_col = next((c for c in y_candidates if c in cols), None)
    lm_col = next((c for c in lm_candidates if c in cols), None)

    if not all([x_col, y_col, lm_col]):
        raise KeyError(f"{filename_for_log}: 필수 컬럼(x,y,LM) 없음. 현재: {cols}")

    out = df.rename(columns={x_col: "x", y_col: "y", lm_col: "LM"})[['LM', 'x', 'y']].copy()
    out['LM'] = out['LM'].astype(str).str.strip()
    out = out.drop_duplicates(subset='LM', keep='first').reset_index(drop=True)
    return out


# ------------------ 3. 정합 계산 (Similarity + Homography fallback) ------------------
def umeyama_similarity(src_pts: np.ndarray, dst_pts: np.ndarray) -> np.ndarray:
    """
    Return 3x3 transformation matrix mapping src_pts -> dst_pts (Umeyama similarity).
    src_pts/dst_pts: (N,2)
    """
    src = src_pts.astype(np.float64)
    dst = dst_pts.astype(np.float64)
    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)
    src_c = src - mu_src
    dst_c = dst - mu_dst
    cov = (dst_c.T @ src_c) / src.shape[0]
    U, S, Vt = np.linalg.svd(cov)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = U @ Vt
    var_src = (src_c ** 2).sum() / src.shape[0]
    scale = S.sum() / (var_src * src.shape[1]) if var_src > 1e-9 else 1.0
    t = mu_dst - scale * R @ mu_src
    H = np.eye(3, dtype=np.float64)
    H[:2, :2] = scale * R
    H[:2, 2] = t
    return H


def remove_outliers_zscore(mapped_pts: np.ndarray, template_pts: np.ndarray, z_thresh: float) -> np.ndarray:
    """
    return boolean mask of inliers based on z-score of euclidean residuals.
    mapped_pts and template_pts shape (N,2)
    """
    if mapped_pts.shape[0] == 0:
        return np.zeros(0, dtype=bool)
    errs = np.linalg.norm(mapped_pts - template_pts, axis=1)
    if errs.size < 2:
        return np.ones_like(errs, dtype=bool)
    mean = errs.mean()
    std = errs.std(ddof=0)
    if std == 0:
        return np.ones_like(errs, dtype=bool)
    z = (errs - mean) / std
    return np.abs(z) < z_thresh


def compute_alignment(merged_df: pd.DataFrame) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    merged_df: columns ['LM','x_template','y_template','x_image','y_image']
    returns H_final (3x3), quality_log dict
    """
    num_points = len(merged_df)
    if num_points < MIN_POINTS_FOR_HOMOGRAPHY:
        return np.eye(3, dtype=np.float64), {
            "transform_method": None,
            "matched_points": num_points,
            "inliers": 0,
            "rmse_px": None,
            "quality_gate_passed": False,
            "confidence_level": "low",
            "error": "Insufficient landmarks"
        }

    template_pts = merged_df[['x_template', 'y_template']].to_numpy(dtype=np.float64)
    image_pts = merged_df[['x_image', 'y_image']].to_numpy(dtype=np.float64)

    # 1) Similarity (Umeyama) as robust initial
    H_sim = umeyama_similarity(image_pts, template_pts)
    pts_sim = cv2.perspectiveTransform(image_pts.reshape(-1, 1, 2).astype(np.float32),
                                       H_sim.astype(np.float32)).reshape(-1, 2).astype(np.float64)

    mask_inliers = remove_outliers_zscore(pts_sim, template_pts, Z_SCORE_THRESHOLD)
    inliers_count = int(mask_inliers.sum())
    if inliers_count > 0:
        rmse_sim = float(np.sqrt(np.mean(np.sum((pts_sim[mask_inliers] - template_pts[mask_inliers]) ** 2, axis=1))))
    else:
        rmse_sim = float('inf')

    # 2) try homography on image_pts -> template_pts (original points)
    H_homo, homo_mask = None, None
    try:
        if num_points >= MIN_POINTS_FOR_HOMOGRAPHY:
            H_homo, homo_mask = cv2.findHomography(image_pts.astype(np.float64), template_pts.astype(np.float64),
                                                   cv2.RANSAC, RANSAC_REPROJ_THRESH, maxIters=RANSAC_MAX_ITERS)
    except Exception as e:
        H_homo = None

    H_final = H_homo if H_homo is not None else H_sim
    transform_method = "Homography" if H_homo is not None else "Similarity"

    # compute rmse using H_final
    mapped_pts = cv2.perspectiveTransform(image_pts.reshape(-1, 1, 2).astype(np.float32),
                                          H_final.astype(np.float32)).reshape(-1, 2).astype(np.float64)
    inlier_mask_from_z = remove_outliers_zscore(mapped_pts, template_pts, Z_SCORE_THRESHOLD)
    inliers_count_final = int(inlier_mask_from_z.sum())
    rmse_final = float(np.sqrt(np.mean(np.sum((mapped_pts[inlier_mask_from_z] - template_pts[inlier_mask_from_z]) ** 2,
                                              axis=1)))) if inliers_count_final > 0 else float('inf')

    # Quality gate 적용
    quality_gate_passed = (inliers_count_final >= MIN_INLIERS) and (rmse_final <= RMSE_MAX)

    # confidence
    if not quality_gate_passed:
        confidence = "low"
    elif inliers_count_final >= max(MIN_INLIERS, 50) and rmse_final <= (RMSE_MAX * 0.5):
        confidence = "high"
    else:
        confidence = "medium"

    error_msg = None
    if not quality_gate_passed:
        errors = []
        if inliers_count_final < MIN_INLIERS:
            errors.append(f"Not enough inliers ({inliers_count_final})")
        if rmse_final > RMSE_MAX:
            errors.append(f"High RMSE ({rmse_final:.2f})")
        error_msg = ", ".join(errors)

    q = {
        "transform_method": transform_method,
        "matched_points": num_points,
        "inliers": inliers_count_final,
        "rmse_px": None if math.isinf(rmse_final) else float(rmse_final),
        "quality_gate_passed": bool(quality_gate_passed),
        "confidence_level": confidence,
        "error": error_msg
    }
    return H_final.astype(np.float64), q


# ------------------ 4. GeoJSON 변환/정리 ------------------
def load_geojson_fix(path: str) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding=ENCODING) as f:
            data = json.load(f)
    except Exception:
        return {"type": "FeatureCollection", "features": []}
    if not isinstance(data, dict) or data.get("type") != "FeatureCollection":
        return {"type": "FeatureCollection", "features": []}
    fixed = []
    for feat in data.get("features", []):
        geom = feat.get("geometry")
        if not geom: continue
        if geom.get("type") == "Polygon":
            coords = geom.get("coordinates", [])
            # ensure polygon is [ [ [x,y], ... ] ] structure
            if len(coords) > 0 and isinstance(coords[0], list) and len(coords[0]) > 0 and isinstance(coords[0][0],
                                                                                                     (int, float)):
                geom["coordinates"] = [coords]
        fixed.append(feat)
    data["features"] = fixed
    return data


def transform_geojson(H: np.ndarray, geojson_data: Dict[str, Any]) -> Dict[str, Any]:
    out = {"type": "FeatureCollection", "features": []}
    if H is None:
        return out
    for feat in geojson_data.get("features", []):
        geom = feat.get("geometry")
        if not geom:
            continue
        gtype = geom.get("type")
        coords = geom.get("coordinates", [])
        new_feat = feat.copy()
        new_geom = new_feat.get("geometry", {}).copy()
        try:
            if gtype == "Polygon":
                new_coords = []
                for ring in coords:
                    if len(ring) < 1: continue
                    arr = np.array(ring, dtype=np.float32).reshape(-1, 1, 2)
                    mapped = cv2.perspectiveTransform(arr, H.astype(np.float32)).reshape(-1, 2).tolist()
                    new_coords.append(mapped)
                new_geom["coordinates"] = new_coords
            elif gtype == "MultiPolygon":
                new_mp = []
                for poly in coords:
                    new_poly = []
                    for ring in poly:
                        if len(ring) < 1: continue
                        arr = np.array(ring, dtype=np.float32).reshape(-1, 1, 2)
                        mapped = cv2.perspectiveTransform(arr, H.astype(np.float32)).reshape(-1, 2).tolist()
                        new_poly.append(mapped)
                    new_mp.append(new_poly)
                new_geom["coordinates"] = new_mp
            else:
                continue
            new_feat["geometry"] = new_geom
            out["features"].append(new_feat)
        except Exception as e:
            logger.warning(f"GeoJSON 변환 에러: {e}")
            continue
    return out


def shift_geojson_to_positive(polygons: Dict[str, Any]) -> Dict[str, Any]:
    all_pts = []
    for feat in polygons.get("features", []):
        geom = feat.get("geometry", {})
        if geom.get("type") == "Polygon":
            for ring in geom.get("coordinates", []):
                all_pts.extend(ring)
        elif geom.get("type") == "MultiPolygon":
            for poly in geom.get("coordinates", []):
                for ring in poly:
                    all_pts.extend(ring)
    if not all_pts:
        return polygons
    min_x = min(p[0] for p in all_pts)
    min_y = min(p[1] for p in all_pts)
    shift_x = -min_x if min_x < 0 else 0
    shift_y = -min_y if min_y < 0 else 0
    if shift_x == 0 and shift_y == 0:
        return polygons
    for feat in polygons.get("features", []):
        geom = feat.get("geometry", {})
        if geom.get("type") == "Polygon":
            for ring in geom.get("coordinates", []):
                for pt in ring:
                    pt[0] += shift_x
                    pt[1] += shift_y
        elif geom.get("type") == "MultiPolygon":
            for poly in geom.get("coordinates", []):
                for ring in poly:
                    for pt in ring:
                        pt[0] += shift_x
                        pt[1] += shift_y
    return polygons


# ------------------ 5. Preview 생성 (before / after overlay) ------------------
def build_canvas_from_template(polygons: List[Dict]) -> Tuple[np.ndarray, Tuple[int, int], float]:
    pts = []
    for feat in polygons:
        geom = feat.get("geometry", {})
        gtype = geom.get("type")
        if gtype == "Polygon":
            for ring in geom.get("coordinates", []):
                pts.extend(ring)
        elif gtype == "MultiPolygon":
            for poly in geom.get("coordinates", []):
                for ring in poly:
                    pts.extend(ring)
    if not pts:
        w, h = 512, 512
        return np.ones((h, w, 3), dtype=np.uint8) * 255, (w, h), 1.0
    arr = np.array(pts)
    min_x, min_y = arr.min(axis=0)
    max_x, max_y = arr.max(axis=0)
    width = max(1, int(math.ceil(max_x - min_x)))
    height = max(1, int(math.ceil(max_y - min_y)))
    scale = 1.0
    max_side = max(width, height)
    if max_side > PREVIEW_SIZE_MAX:
        scale = PREVIEW_SIZE_MAX / max_side
        width = int(width * scale)
        height = int(height * scale)
    canvas = np.ones((height + 20, width + 20, 3), dtype=np.uint8) * 255
    return canvas, (width, height), scale


def draw_geojson_on_canvas(canvas: np.ndarray, geojson_data: Dict[str, Any], color=(0, 0, 255), alpha=0.5, scale=1.0,
                           offset=(0.0, 0.0)):
    overlay = canvas.copy()
    offset_arr = np.array(offset)
    for feat in geojson_data.get("features", []):
        geom = feat.get("geometry", {})
        if geom.get("type") == "Polygon":
            for ring in geom.get("coordinates", []):
                if len(ring) < 1: continue
                pts = np.array(ring)
                pts = np.round((pts - offset_arr) * scale).astype(np.int32)
                if pts.size == 0:
                    continue
                cv2.fillPoly(overlay, [pts], color)
                cv2.polylines(overlay, [pts], isClosed=True, color=(0, 0, 0), thickness=1)
        elif geom.get("type") == "MultiPolygon":
            for poly in geom.get("coordinates", []):
                for ring in poly:
                    if len(ring) < 1: continue
                    pts = np.array(ring)
                    pts = np.round((pts - offset_arr) * scale).astype(np.int32)
                    if pts.size == 0:
                        continue
                    cv2.fillPoly(overlay, [pts], color)
                    cv2.polylines(overlay, [pts], isClosed=True, color=(0, 0, 0), thickness=1)
    cv2.addWeighted(overlay, alpha, canvas, 1 - alpha, 0, canvas)
    return canvas


def save_preview(image_id: str, original_geojson: Dict[str, Any], warped_geojson: Dict[str, Any], output_dir: str):
    all_features = warped_geojson.get("features", []) or original_geojson.get("features", [])
    if not all_features:
        logger.warning(f"{image_id}: 프리뷰 생성을 위한 GeoJSON 피처가 없습니다.")
        return None

    canvas, (w, h), scale = build_canvas_from_template(all_features)

    pts = []
    for feat in all_features:
        geom = feat.get("geometry", {})
        if geom.get("type") == "Polygon":
            for ring in geom.get("coordinates", []):
                pts.extend(ring)
    if pts:
        arr = np.array(pts)
        min_x, min_y = arr.min(axis=0)
    else:
        min_x, min_y = 0.0, 0.0

    # Template (warped) in Green
    after_canvas = canvas.copy()
    try:
        after_canvas = draw_geojson_on_canvas(after_canvas, warped_geojson, color=(0, 255, 0), alpha=0.5, scale=scale,
                                              offset=(min_x, min_y))
    except Exception as e:
        logger.warning(f"{image_id} 프리뷰 'after' 이미지 생성 실패: {e}")

    # Original (before warping) in Red, drawn on top of the 'after' canvas
    combined_canvas = after_canvas.copy()
    try:
        combined_canvas = draw_geojson_on_canvas(combined_canvas, original_geojson, color=(0, 0, 255), alpha=0.4,
                                                 scale=scale, offset=(min_x, min_y))
    except Exception as e:
        logger.warning(f"{image_id} 프리뷰 'before' 이미지 생성 실패: {e}")

    out_path = os.path.join(output_dir, f"{image_id}_overlay.jpg")
    cv2.imwrite(out_path, combined_canvas)
    return out_path


# ------------------ 6. Main batch runner ------------------
def list_landmark_files(landmarks_dir: str) -> List[str]:
    fs = [f for f in os.listdir(landmarks_dir) if f.lower().endswith(".csv")]
    return fs


def save_numpy_H(H: np.ndarray, out_dir: str, image_id: str, manual_backup=False):
    hpath = os.path.join(out_dir, f"{image_id}.npy")
    if manual_backup and os.path.exists(hpath):
        bdir = os.path.join(MANUAL_DIR, "H_backup")
        os.makedirs(bdir, exist_ok=True)
        shutil.copy2(hpath, os.path.join(bdir, f"{image_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.npy"))
    np.save(hpath, H)


def write_quality_json(out_dir: str, image_id: str, qlog: Dict[str, Any], extra_meta: Dict[str, Any] = None):
    os.makedirs(out_dir, exist_ok=True)
    p = os.path.join(out_dir, f"{image_id}.json")
    d = qlog.copy()
    if extra_meta:
        d.update(extra_meta)
    for k, v in d.items():
        if isinstance(v, (np.integer,)):
            d[k] = int(v)
        if isinstance(v, (np.floating,)):
            d[k] = float(v)
    with open(p, "w", encoding=ENCODING) as f:
        json.dump(d, f, ensure_ascii=False, indent=2)


def save_geojson(out_dir: str, image_id: str, source_fname: str, geojson_data: Dict[str, Any]):
    os.makedirs(out_dir, exist_ok=True)
    out_name = f"{image_id}_{os.path.basename(source_fname)}"
    p = os.path.join(out_dir, out_name)
    with open(p, "w", encoding=ENCODING) as f:
        json.dump(geojson_data, f, ensure_ascii=False, indent=2)
    return p


def append_failure(image_id: str, reason: str):
    with open(FAILURES_TXT, "a", encoding=ENCODING) as f:
        f.write(f"{image_id}\t{reason}\n")


def append_skipped(image_id: str, num_common: int):
    with open(SKIPPED_LOG, "a", encoding=ENCODING) as f:
        f.write(f"{image_id}\t{num_common}\n")


def run_full_batch(template_csv_path: str, landmarks_dir: str, masks_dir: str, out_root: str):
    logger.info("===== 배치 정합 시작 =====")
    out_H_dir = os.path.join(out_root, "H")
    out_quality_dir = os.path.join(out_root, "quality")
    out_polygons_dir = os.path.join(out_root, "polygons")
    out_previews_dir = os.path.join(out_root, "previews")

    template_df_raw = load_csv_any(template_csv_path)
    template_df = standardize_landmark_df(template_df_raw, template_csv_path)
    template_df = template_df.rename(columns={"x": "x_template", "y": "y_template"})

    files = list_landmark_files(landmarks_dir)
    metrics = []
    failures = []
    skipped = []
    total = len(files)
    logger.info(f"총 처리 대상: {total} CSV 파일")

    mask_files = [os.path.join(masks_dir, f) for f in os.listdir(masks_dir) if
                  f.lower().endswith(".geojson")] if os.path.isdir(masks_dir) else []
    mask_data_list = []
    for mf in mask_files:
        mask_data_list.append((mf, load_geojson_fix(mf)))

    for idx, fname in enumerate(files, start=1):
        image_id = os.path.splitext(fname)[0]
        logger.info(f"[{idx}/{total}] 처리 시작: {image_id}")
        image_csv_path = os.path.join(landmarks_dir, fname)
        try:
            image_df_raw = load_csv_any(image_csv_path)
            image_df = standardize_landmark_df(image_df_raw, image_csv_path)
            image_df = image_df.rename(columns={"x": "x_image", "y": "y_image"})

            merged = pd.merge(template_df[['LM', 'x_template', 'y_template']], image_df[['LM', 'x_image', 'y_image']],
                              on='LM', how='inner')
            num_common = len(merged)

            H, qlog = compute_alignment(merged)

            if qlog.get("error") == "Insufficient landmarks":
                logger.warning(f"{image_id}: 공통 LM < {MIN_POINTS_FOR_HOMOGRAPHY} ({num_common}) → 스킵 목록에 추가")
                append_skipped(image_id, num_common)
                skipped.append({"image_id": image_id, "num_common": num_common})
                continue

            qlog_save = qlog.copy()
            qlog_save["template_ref"] = os.path.basename(template_csv_path)
            qlog_save["image_id"] = image_id
            qlog_save["matched_points"] = int(qlog_save.get("matched_points", num_common))
            qlog_save["num_lm_common"] = int(num_common)
            qlog_save["generated_at"] = datetime.now().isoformat()

            save_numpy_H(H, out_H_dir, image_id, manual_backup=False)

            polygons_written = []
            specific_orig_mask_path = os.path.join(MASKS_DIR, f"{image_id}.geojson")
            orig_mask_data = load_geojson_fix(specific_orig_mask_path) if os.path.exists(
                specific_orig_mask_path) else None

            for mfpath, mfdata in mask_data_list:
                current_mask_data = orig_mask_data if orig_mask_data else mfdata

                transformed = transform_geojson(H, current_mask_data)
                if not transformed.get("features"):
                    continue
                transformed = shift_geojson_to_positive(transformed)
                poly_path = save_geojson(out_polygons_dir, image_id, os.path.basename(mfpath), transformed)
                polygons_written.append(poly_path)

                try:
                    with open(poly_path, "r", encoding=ENCODING) as pf:
                        poly_json = json.load(pf)
                    for feat in poly_json.get("features", []):
                        props = feat.get("properties", {}) or {}
                        props.update({
                            "image_id": image_id,
                            "method": qlog.get("transform_method"),
                            "rmse": qlog.get("rmse_px"),
                            "inliers": qlog.get("inliers"),
                            "template_ref": os.path.basename(template_csv_path),
                            "source_polygon_file": os.path.basename(mfpath),
                            "generated_at": datetime.now().isoformat()
                        })
                        feat["properties"] = props
                    with open(poly_path, "w", encoding=ENCODING) as pf:
                        json.dump(poly_json, pf, ensure_ascii=False, indent=2)
                except Exception as e:
                    logger.warning(f"폴리곤 metadata 삽입 실패: {e}")

            preview_path = None
            try:
                preview_orig_mask = orig_mask_data if orig_mask_data else (
                    mask_data_list[0][1] if mask_data_list else {"type": "FeatureCollection", "features": []})
                warped_mask = transform_geojson(H, preview_orig_mask)
                warped_mask = shift_geojson_to_positive(warped_mask)
                preview_path = save_preview(image_id, preview_orig_mask, warped_mask, out_previews_dir)
            except Exception as e:
                logger.warning(f"프리뷰 생성 실패: {e}")

            q_extra = {"polygons_written": [os.path.basename(p) for p in polygons_written],
                       "preview": os.path.basename(preview_path) if preview_path else None}
            write_quality_json(out_quality_dir, image_id, qlog_save, extra_meta=q_extra)

            metrics.append({
                "image_id": image_id,
                "method": qlog_save.get("transform_method"),
                "rmse": qlog_save.get("rmse_px"),
                "inliers": qlog_save.get("inliers"),
                "num_lm_common": num_common,
                "passed": bool(qlog_save.get("quality_gate_passed")),
                "confidence": qlog_save.get("confidence_level"),
            })

            logger.info(
                f"{image_id} 완료: passed={qlog_save.get('quality_gate_passed')} rmse={qlog_save.get('rmse_px')} inliers={qlog_save.get('inliers')}")

            if not qlog_save.get("quality_gate_passed"):
                append_failure(image_id, qlog_save.get("error") or "failed_gate")
                failures.append({"image_id": image_id, "reason": qlog_save.get("error")})

        except Exception as e:
            logger.exception(f"{image_id} 처리 중 예외 발생: {e}")
            append_failure(image_id, f"exception: {str(e)}")
            failures.append({"image_id": image_id, "reason": str(e)})
            continue

    if metrics:
        dfm = pd.DataFrame(metrics)
        dfm.to_csv(METRICS_CSV, index=False, encoding="utf-8-sig")
    else:
        pd.DataFrame([],
                     columns=["image_id", "method", "rmse", "inliers", "num_lm_common", "passed", "confidence"]).to_csv(
            METRICS_CSV, index=False, encoding="utf-8-sig")

    total_processed = len(metrics)
    passed_count = sum(1 for m in metrics if m.get("passed"))
    rmse_vals = [m["rmse"] for m in metrics if m.get("rmse") is not None]
    inliers_vals = [m["inliers"] for m in metrics if m.get("inliers") is not None]
    summary = {
        "total_images_in_folder": total,
        "total_processed_for_metrics": total_processed,
        "passed_auto_alignment": passed_count,
        "passed_rate": passed_count / total_processed if total_processed > 0 else 0,
        "mean_rmse_of_processed": float(np.nanmean(rmse_vals)) if rmse_vals else None,
        "median_rmse_of_processed": float(np.nanmedian(rmse_vals)) if rmse_vals else None,
        "mean_inliers_of_processed": float(np.nanmean(inliers_vals)) if inliers_vals else None,
        "skipped_due_to_min_lm": len(skipped),
        "failed_quality_gate": len(failures),
        "generated_at": datetime.now().isoformat()
    }
    with open(METRICS_SUMMARY_JSON, "w", encoding=ENCODING) as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    thresholds = {
        "rmse_max": RMSE_MAX,
        "inliers_min": MIN_INLIERS,
        "min_points_for_homography": MIN_POINTS_FOR_HOMOGRAPHY,
        "ransac_reproj_thresh": RANSAC_REPROJ_THRESH,
        "z_score_threshold": Z_SCORE_THRESHOLD
    }
    with open(THRESHOLDS_USED_JSON, "w", encoding=ENCODING) as f:
        json.dump(thresholds, f, ensure_ascii=False, indent=2)

    generate_readme_and_configs(template_csv_path, landmarks_dir, masks_dir)

    logger.info("===== 배치 정합 완료 =====")
    logger.info(
        f"총 처리: {total}, metrics 기록: {len(metrics)}, skipped(<{MIN_POINTS_FOR_HOMOGRAPHY} LM): {len(skipped)}, failed: {len(failures)}")
    return {
        "metrics": metrics,
        "summary": summary,
        "failures": failures,
        "skipped": skipped
    }


# ------------------ 7. Manual fallback GUI helper ------------------
def manual_fallback_gui(low_list: List[Dict[str, Any]], template_df: pd.DataFrame, landmarks_dir: str, masks_dir: str,
                        out_root: str):
    root = Tk()
    root.withdraw()
    if not low_list:
        messagebox.showinfo("수동 폴백", "수동 폴백 대상이 없습니다.")
        return

    for item in low_list:
        image_id = item.get("image_id")
        lm_path = os.path.join(landmarks_dir, f"{image_id}.csv")
        try:
            template_df_local = template_df.copy()
            image_df_raw = load_csv_any(lm_path)
            image_df = standardize_landmark_df(image_df_raw, lm_path).rename(columns={"x": "x_image", "y": "y_image"})

            merged = pd.merge(template_df_local[['LM', 'x_template', 'y_template']],
                              image_df[['LM', 'x_image', 'y_image']], on='LM', how='inner')
            if len(merged) < MIN_POINTS_FOR_HOMOGRAPHY:
                messagebox.showwarning("수동 불가", f"{image_id} 공통 LM < {MIN_POINTS_FOR_HOMOGRAPHY}, 수동 처리 불가.")
                continue

            manual_points = []
            canceled = False
            for _, row in merged.iterrows():
                qx = simpledialog.askfloat("수동 좌표 입력",
                                           f"{image_id} - LM:{row['LM']}\n기존 x_image: {row['x_image']}\n새 x 입력 (취소 시 전체 중단)")
                if qx is None:
                    canceled = True
                    break
                qy = simpledialog.askfloat("수동 좌표 입력",
                                           f"{image_id} - LM:{row['LM']}\n기존 y_image: {row['y_image']}\n새 y 입력 (취소 시 전체 중단)")
                if qy is None:
                    canceled = True
                    break
                manual_points.append({"LM": row['LM'], "x_image_manual": float(qx), "y_image_manual": float(qy),
                                      "x_template": float(row['x_template']), "y_template": float(row['y_template'])})

            if canceled:
                response = messagebox.askyesno("취소 확인", "현재 파일의 수동 입력을 취소합니다. 전체 수동 처리를 중단하시겠습니까?")
                if response:
                    logger.info("사용자에 의해 전체 수동 처리 중단됨")
                    break
                else:
                    continue

            if len(manual_points) < MIN_POINTS_FOR_HOMOGRAPHY:
                logger.warning(f"{image_id}: 수동 입력 포인트 부족 ({len(manual_points)}), 건너뜁니다.")
                continue

            manual_points_df = pd.DataFrame(manual_points)
            manual_points_path = os.path.join(MANUAL_DIR, f"{image_id}_manual_points.csv")
            manual_points_df.to_csv(manual_points_path, index=False, encoding="utf-8-sig")

            image_pts = manual_points_df[['x_image_manual', 'y_image_manual']].to_numpy(dtype=np.float64)
            template_pts = manual_points_df[['x_template', 'y_template']].to_numpy(dtype=np.float64)

            H_manual = None
            if len(image_pts) >= 4:
                try:
                    H_manual, _ = cv2.findHomography(image_pts, template_pts, cv2.RANSAC, RANSAC_REPROJ_THRESH,
                                                     maxIters=RANSAC_MAX_ITERS)
                except Exception:
                    H_manual = None
            if H_manual is None:
                H_manual = umeyama_similarity(image_pts, template_pts)

            os.makedirs(os.path.join(MANUAL_DIR, "H"), exist_ok=True)
            manual_H_path = os.path.join(MANUAL_DIR, "H", f"{image_id}_H_manual.npy")
            np.save(manual_H_path, H_manual)

            save_numpy_H(H_manual, os.path.join(out_root, "H"), image_id, manual_backup=True)

            mask_files = [os.path.join(MASKS_DIR, f) for f in os.listdir(MASKS_DIR) if
                          f.lower().endswith(".geojson")] if os.path.isdir(MASKS_DIR) else []
            polygons_written = []
            for mf in mask_files:
                mdata = load_geojson_fix(mf)
                transformed = transform_geojson(H_manual, mdata)
                if not transformed.get("features"):
                    continue
                transformed = shift_geojson_to_positive(transformed)
                ppath = save_geojson(os.path.join(out_root, "polygons"), image_id, os.path.basename(mf), transformed)
                polygons_written.append(ppath)

            qlog_save = {
                "image_id": image_id,
                "manual_override": True,
                "manual_points_path": manual_points_path,
                "generated_at": datetime.now().isoformat(),
                "quality_gate_passed": True,
                "confidence_level": "manual",
                "error": None
            }
            q_extra = {"polygons_written": [os.path.basename(p) for p in polygons_written]}
            write_quality_json(os.path.join(out_root, "quality"), image_id, qlog_save, extra_meta=q_extra)

            try:
                specific_orig_mask_path = os.path.join(MASKS_DIR, f"{image_id}.geojson")
                orig_mask = load_geojson_fix(specific_orig_mask_path) if os.path.exists(specific_orig_mask_path) else (
                    load_geojson_fix(mask_files[0]) if mask_files else {"type": "FeatureCollection", "features": []})
                warped_mask = transform_geojson(H_manual, orig_mask)
                warped_mask = shift_geojson_to_positive(warped_mask)
                save_preview(image_id, orig_mask, warped_mask, os.path.join(out_root, "previews"))
            except Exception as e:
                logger.warning(f"{image_id} 수동 프리뷰 생성 실패: {e}")

            logger.info(f"{image_id} 수동 처리가 완료되어 align/H에 덮어쓰기 되었습니다.")

        except Exception as e:
            logger.exception(f"{image_id} 수동 처리 중 예외: {e}")
            messagebox.showerror("오류", f"{image_id} 수동 처리 중 오류 발생: {e}")


# ------------------ 8. README / requirements / run_config 작성 ------------------
def generate_readme_and_configs(template_csv_path, landmarks_dir, masks_dir):
    readme = f"""# README_align

## 입력 파일 (반드시 확인)
- 템플릿 랜드마크: {template_csv_path}
  - CSV with columns including: LM (id/name), x, y
- 랜드마크 폴더: {landmarks_dir}
  - 각 이미지별 `<image_id>.csv` 파일 (columns include LM,x,y)
- 마스크 폴더: {masks_dir}
  - GeoJSON files (e.g. zoneA_no_parking.geojson, legal_areas.geojson)

## 출력(align/ 루트)
- align/H/<image_id>.npy  (3x3 float64)
- align/quality/<image_id>.json
- align/polygons/<image_id>_<source_mask>.geojson (좌표계: 템플릿 픽셀)
- align/previews/<image_id>_overlay.jpg
- logs/runtime.log
- logs/failures.txt
- logs/skipped.txt
- summary/metrics.csv
- summary/metrics_summary.json
- summary/thresholds_used.json

## 실행 예시
$ python your_script_name.py
(또는 필요시 경로를 스크립트 상단에서 수정)

## 사용 임계값 (현재 파일 thresholds_used.json 참고)
- RMSE_MAX: {RMSE_MAX}
- MIN_INLIERS: {MIN_INLIERS}
- Z_SCORE_THRESHOLD: {Z_SCORE_THRESHOLD}
- RANSAC_REPROJ_THRESH: {RANSAC_REPROJ_THRESH}

## 좌표계
- 입력 랜드마크: 이미지 좌표 (각 이미지의 CSV에 저장된 좌표)
- 템플릿 랜드마크: 템플릿 픽셀 좌표 (templates/...)
- 출력 폴리곤: 템플릿 픽셀 좌표계로 정합되어 저장

"""
    with open(os.path.join(OUT_ROOT, "README_align.md"), "w", encoding=ENCODING) as f:
        f.write(readme)

    # requirements.txt
    reqs = """numpy
pandas
opencv-python
PyYAML
"""
    with open(os.path.join(OUT_ROOT, "requirements.txt"), "w", encoding=ENCODING) as f:
        f.write(reqs)

    # run_config.yaml
    run_cfg = {
        "rmse_max": RMSE_MAX,
        "inliers_min": MIN_INLIERS,
        "z_score_threshold": Z_SCORE_THRESHOLD,
        "ransac_reproj_thresh": RANSAC_REPROJ_THRESH,
        "min_points_for_homography": MIN_POINTS_FOR_HOMOGRAPHY,
        "preview_size_max": PREVIEW_SIZE_MAX
    }
    try:
        import yaml
        with open(os.path.join(OUT_ROOT, "run_config.yaml"), "w", encoding=ENCODING) as f:
            yaml.dump(run_cfg, f, allow_unicode=True)
    except ImportError:
        with open(os.path.join(OUT_ROOT, "run_config.json"), "w", encoding=ENCODING) as f:
            json.dump(run_cfg, f, ensure_ascii=False, indent=2)


# ------------------ 9. CLI helper ------------------
def pick_or_use_default(path_hint: str, prompt: str) -> str:
    if os.path.exists(path_hint):
        return path_hint
    root = Tk()
    root.withdraw()
    if os.path.isdir(path_hint) or prompt.lower().find("폴더") >= 0:
        selected = filedialog.askdirectory(title=f"{prompt} (선택, 취소 시 빈값)")
    else:
        selected = filedialog.askopenfilename(title=f"{prompt} (선택, 취소 시 빈값)")
    root.destroy()
    if selected:
        return selected
    return path_hint


# ------------------ 10. main ------------------
def main():
    ensure_dirs()
    logger.info("Align pipeline 시작")
    template_csv_path = pick_or_use_default(TEMPLATE_CSV, "템플릿 CSV 선택 (templates/zoneA_landmarks.csv)")
    if not os.path.exists(template_csv_path):
        logger.error("템플릿 CSV가 존재하지 않습니다. 경로를 확인하세요.")
        print("템플릿 CSV가 존재하지 않습니다. 상단 스크립트 내 경로를 확인하거나, GUI에서 선택하세요.")
        return
    landmarks_dir = pick_or_use_default(LANDMARKS_DIR, "이미지 랜드마크 폴더 선택 (landmarks/)")
    masks_dir = pick_or_use_default(MASKS_DIR, "마스크(GeoJSON) 폴더 선택 (masks/)")

    result = run_full_batch(template_csv_path, landmarks_dir, masks_dir, OUT_ROOT)

    low_list = [
        {"image_id": m["image_id"], "reason": "quality_low"}
        for m in result.get("metrics", [])
        if not m.get("passed", False)
    ]
    metric_ids = {m["image_id"] for m in low_list}
    for fail in result.get("failures", []):
        if fail["image_id"] not in metric_ids:
            low_list.append({"image_id": fail["image_id"], "reason": fail.get("reason")})

    if low_list:
        logger.info(f"수동 검토 대상: {len(low_list)} 건. 수동 폴백 GUI를 실행할까요?")
        root = Tk()
        root.withdraw()
        do_manual = messagebox.askyesno("수동 폴백", f"자동 처리에 실패한 {len(low_list)}건에 대해 수동 처리를 진행하시겠습니까?")
        root.destroy()

        if do_manual:
            template_df_raw = load_csv_any(template_csv_path)
            template_df = standardize_landmark_df(template_df_raw, template_csv_path)
            template_df = template_df.rename(columns={"x": "x_template", "y": "y_template"})
            manual_fallback_gui(low_list, template_df, landmarks_dir, masks_dir, OUT_ROOT)

    logger.info("모든 처리 완료. 출력 폴더: %s" % os.path.abspath(OUT_ROOT))


if __name__ == "__main__":
    main()