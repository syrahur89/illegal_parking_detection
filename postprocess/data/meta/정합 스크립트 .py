import os
import json
import pandas as pd
import numpy as np
import cv2
import logging
from tkinter import Tk, filedialog, messagebox

# ------------------ 1. 상수 설정 ------------------
MIN_POINTS_FOR_HOMOGRAPHY = 8
MIN_POINTS_FOR_AFFINE = 3
RMSE_THRESHOLD = 8.0
MIN_INLIERS = 50

# ------------------ 2. 로깅 설정 ------------------
log_file = "alignment_process.log"
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler(log_file, mode='w', encoding='utf-8'),
                              logging.StreamHandler()])
logger = logging.getLogger()

# ------------------ 3. CSV 로드 및 표준화 ------------------
def load_csv(path):
    try:
        return pd.read_csv(path, encoding='utf-8')
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(path, encoding='cp949')
            logger.info(f"CSV 로드 완료 (cp949): {os.path.basename(path)}, {len(df)} 행")
            return df
        except Exception as e:
            logger.error(f"CSV 로드 실패: {os.path.basename(path)}, {e}")
            return None
    except Exception as e:
        logger.error(f"CSV 로드 실패: {os.path.basename(path)}, {e}")
        return None

def standardize_columns(df, filename_for_log):
    x_candidates = ['x', 'X', 'lon', 'longitude', 'point_x', 'ux']
    y_candidates = ['y', 'Y', 'lat', 'latitude', 'point_y', 'uy']
    lm_candidates = ['LM', 'lm', 'id', 'point_idx', 'landmark_id']

    x_col = next((c for c in x_candidates if c in df.columns), None)
    y_col = next((c for c in y_candidates if c in df.columns), None)
    lm_col = next((c for c in lm_candidates if c in df.columns), None)

    if not all([x_col, y_col, lm_col]):
        raise KeyError(f"'{filename_for_log}' 파일에서 필수 컬럼을 찾을 수 없습니다. 현재 컬럼: {list(df.columns)}")

    df = df.rename(columns={x_col: 'x', y_col: 'y', lm_col: 'LM'})
    df['LM'] = df['LM'].astype(str).str.strip()
    logger.info(f"[{filename_for_log}] 컬럼 표준화 완료")
    return df

def merge_common_landmarks(df_template, df_image):
    merged_df = pd.merge(df_template[['LM', 'x', 'y']],
                         df_image[['LM', 'x', 'y']],
                         on='LM', suffixes=('_template', '_image'))
    logger.info(f"총 {len(merged_df)}개의 공통 랜드마크 확인")
    return merged_df

# ------------------ 4. Similarity ------------------
def umeyama_similarity(src_pts, dst_pts):
    src_mean = np.mean(src_pts, axis=0)
    dst_mean = np.mean(dst_pts, axis=0)
    src_centered = src_pts - src_mean
    dst_centered = dst_pts - dst_mean
    cov = dst_centered.T @ src_centered / len(src_pts)
    U, S, Vt = np.linalg.svd(cov)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = U @ Vt
    var_src = np.var(src_centered, axis=0).sum()
    scale = np.sum(S) / var_src
    t = dst_mean - scale * R @ src_mean
    H = np.eye(3)
    H[:2, :2] = scale * R
    H[:2, 2] = t
    return H

# ------------------ 5. 정합 계산 (신뢰도 포함) ------------------
def compute_alignment(merged_df):
    num_points = len(merged_df)
    template_pts = merged_df[['x_template', 'y_template']].to_numpy(dtype=np.float32)
    image_pts = merged_df[['x_image', 'y_image']].to_numpy(dtype=np.float32)

    def determine_confidence(rmse, inliers):
        if rmse <= RMSE_THRESHOLD and inliers >= MIN_INLIERS:
            return "high"
        elif rmse <= 2*RMSE_THRESHOLD or inliers >= MIN_INLIERS:
            return "medium"
        else:
            return "low"

    # Case 1: 8개 이상 → Similarity → Homography
    if num_points >= MIN_POINTS_FOR_HOMOGRAPHY:
        logger.info(f"{num_points}개 랜드마크로 Similarity/Homography 정합 시도...")
        H_sim = umeyama_similarity(image_pts, template_pts)
        pts_hom = cv2.perspectiveTransform(image_pts.reshape(-1, 1, 2), H_sim)
        rmse_sim = np.sqrt(np.mean((pts_hom.reshape(-1, 2) - template_pts) ** 2))

        confidence = determine_confidence(rmse_sim, num_points)
        if rmse_sim <= RMSE_THRESHOLD:
            quality_log = {"transform_method": "Similarity", "matched_points": num_points,
                           "rmse_px": round(rmse_sim, 4), "inliers": num_points,
                           "quality_gate_passed": True, "confidence_level": confidence}
            logger.info(f"Similarity 성공 - RMSE: {rmse_sim:.2f}, 신뢰도: {confidence}")
            return H_sim, quality_log

        logger.warning("Similarity 품질 미달. Homography 재시도...")
        H_homo, mask = cv2.findHomography(image_pts, template_pts, cv2.RANSAC, 5.0)
        if H_homo is None:
            logger.error("Homography 계산 실패")
            return None, {"quality_gate_passed": False, "error": "Homography calculation failed"}

        inliers = int(mask.sum())
        pts_trans = cv2.perspectiveTransform(image_pts.reshape(-1, 1, 2), H_homo)
        errors = np.sqrt(np.sum((pts_trans - template_pts.reshape(-1, 1, 2)) ** 2, axis=2))
        rmse_homo = np.mean(errors[mask.flatten() == 1]) if inliers > 0 else float('inf')

        confidence = determine_confidence(rmse_homo, inliers)
        quality_log = {"transform_method": "Homography", "matched_points": num_points,
                       "rmse_px": round(rmse_homo, 4), "inliers": inliers,
                       "quality_gate_passed": confidence != "low", "confidence_level": confidence}
        logger.info(f"Homography 결과 - RMSE: {rmse_homo:.2f}, Inliers: {inliers}, 신뢰도: {confidence}")
        return H_homo if confidence != "low" else None, quality_log

    # Case 2: 3~7개 → Affine
    elif MIN_POINTS_FOR_AFFINE <= num_points < MIN_POINTS_FOR_HOMOGRAPHY:
        logger.info(f"{num_points}개 랜드마크로 Affine 정합 시도...")
        H_affine_2x3, _ = cv2.estimateAffinePartial2D(image_pts, template_pts)
        if H_affine_2x3 is None:
            logger.error("Affine 계산 실패")
            return None, {"quality_gate_passed": False, "error": "Affine calculation failed"}

        H_affine_3x3 = np.vstack([H_affine_2x3, [0, 0, 1]])
        pts_trans = cv2.transform(image_pts.reshape(-1, 1, 2), H_affine_2x3)
        rmse_affine = np.sqrt(np.mean((pts_trans.reshape(-1, 2) - template_pts) ** 2))
        confidence = determine_confidence(rmse_affine, num_points)
        quality_log = {"transform_method": "Affine", "matched_points": num_points,
                       "rmse_px": round(rmse_affine, 4), "inliers": num_points,
                       "quality_gate_passed": confidence != "low", "confidence_level": confidence}
        logger.info(f"Affine 결과 - RMSE: {rmse_affine:.2f}, 신뢰도: {confidence}")
        return H_affine_3x3 if confidence != "low" else None, quality_log

    else:
        logger.warning(f"공통 랜드마크 부족 ({num_points}개)")
        return None, {"quality_gate_passed": False, "matched_points": num_points,
                      "error": "Insufficient landmarks", "confidence_level": "low"}

# ------------------ 6. GeoJSON 정합 ------------------
def load_geojson_with_fix(path):
    try:
        with open(path, "r", encoding="utf-8-sig") as f:
            data = json.load(f)
    except:
        return {"type": "FeatureCollection", "features": []}
    if "type" not in data or data["type"] != "FeatureCollection":
        data = {"type": "FeatureCollection", "features": []}
    fixed_features = []
    for feat in data.get("features", []):
        geom = feat.get("geometry")
        if not geom or "type" not in geom or "coordinates" not in geom:
            continue
        coords = geom["coordinates"]
        if geom["type"] == "Polygon" and isinstance(coords, list) and len(coords) > 0 and not isinstance(coords[0][0], list):
            geom["coordinates"] = [coords]
        fixed_features.append(feat)
    data["features"] = fixed_features
    return data

def align_geojson(H, geojson_data):
    aligned_features = []
    for feat in geojson_data.get("features", []):
        geom = feat.get("geometry")
        if not geom: continue
        geom_type, coords = geom.get("type"), geom.get("coordinates")
        try:
            new_feat = feat.copy()
            new_geom = new_feat['geometry'] = new_feat['geometry'].copy()
            if geom_type == "Polygon":
                rings = [cv2.perspectiveTransform(np.array(r, dtype=np.float32).reshape(-1, 1, 2), H).reshape(-1, 2).tolist() for r in coords]
                new_geom['coordinates'] = rings
                aligned_features.append(new_feat)
            elif geom_type == "MultiPolygon":
                polys = [[cv2.perspectiveTransform(np.array(r, dtype=np.float32).reshape(-1, 1, 2), H).reshape(-1,2).tolist() for r in p] for p in coords]
                new_geom['coordinates'] = polys
                aligned_features.append(new_feat)
        except Exception as e:
            logger.error(f"Feature 변환 실패: {e}")
    return {"type": "FeatureCollection", "features": aligned_features}

# ------------------ 7. 실행 ------------------
def run_alignment():
    root = Tk(); root.withdraw()

    template_csv_path = filedialog.askopenfilename(title="템플릿 CSV 선택", filetypes=[("CSV", "*.csv")])
    if not template_csv_path: return
    image_csv_path = filedialog.askopenfilename(title="이미지 CSV 선택", filetypes=[("CSV", "*.csv")])
    if not image_csv_path: return
    geojson_dir = filedialog.askdirectory(title="GeoJSON 폴더 선택")
    if not geojson_dir: return
    save_dir = filedialog.askdirectory(title="결과 저장 폴더 선택")
    if not save_dir: return

    template_df = standardize_columns(load_csv(template_csv_path), os.path.basename(template_csv_path))
    image_df = standardize_columns(load_csv(image_csv_path), os.path.basename(image_csv_path))
    merged = merge_common_landmarks(template_df, image_df)

    H, quality_log = compute_alignment(merged)
    image_id = os.path.splitext(os.path.basename(image_csv_path))[0]

    os.makedirs(os.path.join(save_dir, "H"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "quality"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "polygons"), exist_ok=True)

    with open(os.path.join(save_dir, "quality", f"{image_id}.json"), "w", encoding="utf-8") as f:
        log_to_save = {k: (int(v) if isinstance(v, np.integer) else float(v) if isinstance(v, np.floating) else v) for k, v in quality_log.items()}
        json.dump(log_to_save, f, ensure_ascii=False, indent=2)

    if H is None:
        messagebox.showerror("정합 실패",
                             f"정합 실패. 자세한 내용은 quality 폴더 확인\n(RMSE={quality_log.get('rmse_px','N/A')}, Inliers={quality_log.get('inliers','N/A')}, 신뢰도={quality_log.get('confidence_level','low')})")
        return

    np.save(os.path.join(save_dir, "H", f"{image_id}.npy"), H)

    processed_count = 0
    for fname in os.listdir(geojson_dir):
        if fname.lower().endswith(".geojson"):
            fpath = os.path.join(geojson_dir, fname)
            geojson_data = load_geojson_with_fix(fpath)
            aligned = align_geojson(H, geojson_data)
            if not aligned["features"]: continue
            save_path = os.path.join(save_dir, "polygons", f"{image_id}_{fname}")
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(aligned, f, ensure_ascii=False, indent=2)
            processed_count += 1
            logger.info(f"{fname} -> {save_path} 완료")

    messagebox.showinfo("완료", f"{processed_count}개의 GeoJSON 처리 완료\n정합 신뢰도: {quality_log.get('confidence_level','low')}")

# ------------------ 8. 실행 ------------------
if __name__ == "__main__":
    run_alignment()
