# -*- coding: utf-8 -*-

import csv
import json
import math
from pathlib import Path
import numpy as np
import argparse

try:
    from shapely.geometry import Point, Polygon, MultiPolygon
    from shapely.ops import nearest_points
    HAS_SHAPELY = True
except Exception:
    HAS_SHAPELY = False

def ffloat(v):
    try:
        return float(v)
    except:
        return None

def parse_bbox(row):
    x1 = ffloat(row.get("t1_orig_x1"))
    y1 = ffloat(row.get("t1_orig_y1"))
    x2 = ffloat(row.get("t1_orig_x2"))
    y2 = ffloat(row.get("t1_orig_y2"))
    if None not in (x1, y1, x2, y2):
        cx = (x1 + x2) * 0.5
        cy = (y1 + y2) * 0.5
        return cx, cy, x1, y1, x2, y2

    bx = ffloat(row.get("bbox_x"))
    by = ffloat(row.get("bbox_y"))
    bw = ffloat(row.get("bbox_w"))
    bh = ffloat(row.get("bbox_h"))
    if None not in (bx, by, bw, bh):
        cx = bx + bw * 0.5
        cy = by + bh * 0.5
        return cx, cy, bx, by, bx + bw, by + bh

    return None, None, None, None, None, None


def load_prefix_map(p: Path):
    mp = {}
    if not p.exists():
        return mp
    with p.open("r", encoding="utf-8") as fi:
        r = csv.DictReader(fi)
        for row in r:
            pref = (row.get("prefix") or "").strip()
            tid = (row.get("template_id") or row.get("template") or row.get("best_template") or "").strip()
            if pref and tid:
                mp[pref] = tid
    return mp

def stem_to_prefix(stem: str):
    # 예: 0001_top_20250704_t1 -> 0001_top
    parts = stem.split("_")
    if len(parts) >= 2:
        return "_".join(parts[0:2])
    return stem

def load_catalog(p: Path):
    tid2poly = {}
    pref2poly = {}
    if not p.exists():
        return tid2poly, pref2poly

    with p.open("r", encoding="utf-8") as fi:
        r = csv.DictReader(fi)
        for row in r:
            tid = (row.get("template_id") or "").strip()
            poly = (row.get("polygon_path") or row.get("path") or "").strip()
            pref = (row.get("prefix") or "").strip()
            if tid and poly:
                tid2poly[tid] = poly
            elif pref and poly:
                pref2poly[pref] = poly
    return tid2poly, pref2poly

def load_geojson_polygons(gj_path: Path):
    if not gj_path.exists():
        return []

    with gj_path.open("r", encoding="utf-8") as f:
        gj = json.load(f)

    polys = []
    feats = gj.get("features") or []
    for ft in feats:
        prop = ft.get("properties") or {}
        tag = (prop.get("zone_type") or prop.get("type") or prop.get("label") or "").strip().lower()
        geom = ft.get("geometry") or {}
        gtype = (geom.get("type") or "").lower()
        coords = geom.get("coordinates")

        if not coords:
            continue

        if gtype == "polygon":
            rings = coords[0]
            polys.append((rings, tag))
        elif gtype == "multipolygon":
            for poly in coords:
                rings = poly[0]
                polys.append((rings, tag))
        else:
            continue
    return polys

def to_homog_mat(np_path: Path):
    import numpy as np
    M = np.load(str(np_path))
    M = np.array(M)
    if M.shape == (4, 3):
        M = M[:3, :]
    if M.shape != (3, 3):
        raise RuntimeError(f"Invalid H shape: {M.shape}")
    return M

def apply_H_img2tpl(H, x, y, xy_swap=False):
    import numpy as np
    if xy_swap:
        x, y = y, x
    v = np.array([x, y, 1.0], dtype=float)
    w = H @ v
    if abs(w[2]) < 1e-9:
        return None, None
    return float(w[0] / w[2]), float(w[1] / w[2])

def point_inout_and_dist(polys, tx, ty, meaning="legal"):
    if tx is None or ty is None or not polys:
        return "out", None, None

    if HAS_SHAPELY:
        pt = Point(tx, ty)
        wanted = []
        for coords, tag in polys:
            wanted.append(Polygon(coords))
        if not wanted:
            return "out", None, None

        union = wanted[0]
        for p in wanted[1:]:
            union = union.union(p)

        inside = union.contains(pt) or union.touches(pt)
        d = union.exterior.distance(pt)
        edge_dist = -d if inside else d
        return ("in" if inside else "out"), ("legal" if meaning == "legal" else "illegal"), edge_dist
    else:
        inside = False
        for coords, tag in polys:
            xs = [c[0] for c in coords]
            ys = [c[1] for c in coords]
            if min(xs) <= tx <= max(xs) and min(ys) <= ty <= max(ys):
                inside = True
                break
        return ("in" if inside else "out"), ("legal" if meaning == "legal" else "illegal"), None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--detections", required=True)
    ap.add_argument("--prefix-map", required=True)
    ap.add_argument("--catalog", required=True)
    ap.add_argument("--polygons-root", required=True)
    ap.add_argument("--H-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--invert-h", action="store_true", help="(드물게) H를 역행렬로 사용")
    ap.add_argument("--xy-swap", action="store_true", help="x/y 좌표 뒤집기(정합 오해결일 때 임시)")
    ap.add_argument("--hit-buffer", type=float, default=0.0, help="경계 여유(px)")
    ap.add_argument("--post-shift-x", type=float, default=0.0)
    ap.add_argument("--post-shift-y", type=float, default=0.0)
    ap.add_argument("--polygons-meaning", choices=["legal", "illegal", "auto"], default="legal")
    ap.add_argument("--outside-legal-as", choices=["unknown", "illegal"], default="unknown",
                    help="meaning=legal일 때, 폴리곤 밖을 unknown/illegal 중 무엇으로 볼지")
    args = ap.parse_args()

    det_path = Path(args.detections)
    prefmap_path = Path(args.prefix_map)
    catalog_path = Path(args.catalog)
    poly_root = Path(args.polygons_root)
    H_root = Path(args.H_root)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    prefix_map = load_prefix_map(prefmap_path)
    tid2poly, pref2poly = load_catalog(catalog_path)

    rows_in = list(csv.DictReader(det_path.open("r", encoding="utf-8")))
    out_cols = [
        "t1_image", "vehicle_id", "zone_id",
        "status", "decision", "illegal",
        "poly_hit", "edge_dist",
        "tx", "ty",
        "template_id", "H_path", "polygon_path",
        "t1_orig_x1", "t1_orig_y1", "t1_orig_x2", "t1_orig_y2",
        "bbox_x", "bbox_y", "bbox_w", "bbox_h",
        "delta_min", "diag0", "diag1", "diag_ratio",
        "move_dist", "move_ratio", "stop", "stop_bool",
        "t1_orig_w", "t1_orig_h"
    ]
    rows_out = []

    for r in rows_in:
        t1 = (r.get("t1_image") or r.get("image_path") or "").strip()
        stem = Path(t1).stem
        pref = stem_to_prefix(stem)
        tpl = prefix_map.get(pref) or stem
        template_id = tpl
        H_path = H_root / f"{template_id}.npy"
        if not H_path.exists():
            H_path = H_root / f"{stem}.npy"
        polygon_path = None
        if template_id in tid2poly:
            polygon_path = tid2poly[template_id]
        elif pref in pref2poly:
            polygon_path = pref2poly[pref]
        else:
            guessed = poly_root / f"{template_id}.geojson"
            if guessed.exists():
                polygon_path = str(guessed)

        cx, cy, x1, y1, x2, y2 = parse_bbox(r)

        status = "OK"
        decision = ""
        illegal = ""
        poly_hit = ""
        edge_dist = ""
        tx = ""
        ty = ""

        try:
            if not H_path.exists():
                status = "NO_H"
                rows_out.append({
                    **{k: r.get(k, "") for k in out_cols if k not in ["status","decision","illegal","poly_hit","edge_dist","tx","ty","template_id","H_path","polygon_path"]},
                    "t1_image": t1, "status": status, "decision": decision, "illegal": illegal,
                    "poly_hit": poly_hit, "edge_dist": edge_dist, "tx": tx, "ty": ty,
                    "template_id": template_id, "H_path": str(H_path), "polygon_path": (polygon_path or "")
                })
                continue

            H = to_homog_mat(H_path)
            if args.invert_h:
                H = np.linalg.inv(H)

            tx_v, ty_v = None, None
            if cx is not None and cy is not None:
                tx_v, ty_v = apply_H_img2tpl(H, cx + args.post_shift_x, cy + args.post_shift_y, xy_swap=args.xy_swap)

            tx = "" if tx_v is None else f"{tx_v:.3f}"
            ty = "" if ty_v is None else f"{ty_v:.3f}"

            if not polygon_path:
                status = "NO_POLYGON"
            else:
                polys = load_geojson_polygons(Path(polygon_path))
                meaning = args.polygons_meaning
                if meaning == "auto":
                    meaning = "legal"

                hit, tag, ed = point_inout_and_dist(polys, tx_v, ty_v, meaning=meaning)

                poly_hit = hit
                edge_out = ""
                if ed is not None:
                    edge_out = f"{ed:.2f}"

                if meaning == "legal":
                    if hit == "in":
                        decision = "LEGAL"
                        illegal = "0"
                    else:
                        if args.outside_legal_as == "illegal":
                            decision = "ILLEGAL"
                            illegal = "1"
                        else:
                            decision = "UNKNOWN"
                            illegal = "0"
                else:
                    if hit == "in":
                        decision = "ILLEGAL"
                        illegal = "1"
                    else:
                        decision = "UNKNOWN"
                        illegal = "0"

                edge_dist = edge_out

            rows_out.append({
                **{k: r.get(k, "") for k in out_cols if k not in ["status","decision","illegal","poly_hit","edge_dist","tx","ty","template_id","H_path","polygon_path"]},
                "t1_image": t1,
                "status": status,
                "decision": decision,
                "illegal": illegal,
                "poly_hit": poly_hit,
                "edge_dist": edge_dist,
                "tx": tx,
                "ty": ty,
                "template_id": template_id,
                "H_path": str(H_path),
                "polygon_path": (polygon_path or "")
            })
        except Exception as e:
            rows_out.append({
                **{k: r.get(k, "") for k in out_cols if k not in ["status","decision","illegal","poly_hit","edge_dist","tx","ty","template_id","H_path","polygon_path"]},
                "t1_image": t1, "status": f"ERR:{e}", "decision": "", "illegal": "",
                "poly_hit": "", "edge_dist": "", "tx": "", "ty": "",
                "template_id": template_id, "H_path": str(H_path), "polygon_path": (polygon_path or "")
            })

    with out_path.open("w", newline="", encoding="utf-8") as fo:
        w = csv.DictWriter(fo, fieldnames=out_cols)
        w.writeheader()
        for row in rows_out:
            w.writerow({k: row.get(k, "") for k in out_cols})

if __name__ == "__main__":
    main()


# python D:\projects\illegal_parking_project\postprocess\scripts\classify_stops.py `
#   --detections "D:\projects\illegal_parking_project\post_process_out\stop_results_for_classify.csv" `
#   --prefix-map "D:\projects\illegal_parking_project\post_process_out\prefix2template.csv" `
#   --catalog "D:\projects\illegal_parking_project\post_process_out\template_catalog.csv" `
#   --polygons-root "D:\projects\illegal_parking_project\align\polygons" `
#   --H-root "D:\projects\illegal_parking_project\align\H" `
#   --xy-swap `
#   --polygons-meaning legal `
#   --outside-legal-as illegal `
#   --hit-buffer 24 `
#   --out "D:\projects\illegal_parking_project\post_process_out\classified_nounk24.csv"