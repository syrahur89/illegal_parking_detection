# -*- coding: utf-8 -*-

import csv
import json
from pathlib import Path
import argparse
from PIL import Image, ImageDraw
import numpy as np

def ffloat(v):
    try:
        return float(v)
    except:
        return None

def parse_box(row):
    # 1) t1_orig_*
    x1 = ffloat(row.get("t1_orig_x1")); y1 = ffloat(row.get("t1_orig_y1"))
    x2 = ffloat(row.get("t1_orig_x2")); y2 = ffloat(row.get("t1_orig_y2"))
    if None not in (x1, y1, x2, y2):
        return int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))
    # 2) bbox_x,y,w,h
    bx = ffloat(row.get("bbox_x")); by = ffloat(row.get("bbox_y"))
    bw = ffloat(row.get("bbox_w")); bh = ffloat(row.get("bbox_h"))
    if None not in (bx, by, bw, bh):
        x1 = int(round(bx)); y1 = int(round(by))
        x2 = int(round(bx + bw)); y2 = int(round(by + bh))
        return x1, y1, x2, y2
    return None

def load_catalog_map(path: Path):
    tid2poly, pref2poly = {}, {}
    if not path.exists():
        return tid2poly, pref2poly
    with path.open("r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            tid = (row.get("template_id") or "").strip()
            pth = (row.get("polygon_path") or row.get("path") or "").strip()
            pref = (row.get("prefix") or "").strip()
            if tid and pth:
                tid2poly[tid] = pth
            elif pref and pth:
                pref2poly[pref] = pth
    return tid2poly, pref2poly


def stem_to_prefix(stem: str):
    parts = stem.split("_")
    if len(parts) >= 2:
        return "_".join(parts[:2])
    return stem


def to_homog_mat(np_path: Path):
    M = np.load(str(np_path))
    M = np.array(M)
    if M.shape == (4, 3):
        M = M[:3, :]
    if M.shape != (3, 3):
        raise RuntimeError(f"Invalid H shape: {M.shape}")
    return M


def apply_H_xy(H, x, y):
    v = np.array([x, y, 1.0], dtype=float)
    w = H @ v
    if abs(w[2]) < 1e-9:
        return None, None
    return float(w[0] / w[2]), float(w[1] / w[2])


def draw_thick_poly(draw: ImageDraw.ImageDraw, pts, color, width=2):
    if len(pts) < 2:
        return
    for i in range(len(pts) - 1):
        draw.line([pts[i], pts[i + 1]], fill=color, width=width)
    draw.line([pts[-1], pts[0]], fill=color, width=width)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--classified", required=True)
    ap.add_argument("--images-root", required=True)
    ap.add_argument("--H-root", required=True)
    ap.add_argument("--catalog", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--thick", type=int, default=3)
    ap.add_argument("--draw-polys", type=int, default=1, help="1이면 폴리곤 윤곽선 그림")
    ap.add_argument("--assume_direction", choices=["tpl2img", "img2tpl"], default="tpl2img",
                    help="H가 템플릿→이미지인지(img2tpl이면 역행렬 사용)")
    ap.add_argument("--polygons-root", default="", help="(옵션) 카탈로그에 없을 때 파일명 규칙으로 찾는 루트")
    ap.add_argument("--respect-illegal-from-csv", action="store_true",
                    help="CSV illegal(1/0) 그대로 색상에 반영")

    args = ap.parse_args()

    classified = Path(args.classified)
    img_root = Path(args.images_root)
    H_root = Path(args.H_root)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    tid2poly, pref2poly = load_catalog_map(Path(args.catalog))
    poly_root = Path(args.polygons_root) if args.polygons_root else None

    RED = (255, 60, 60)
    GREEN = (80, 220, 100)
    GRAY = (170, 170, 170)
    POLY = (255, 200, 0)

    rows = list(csv.DictReader(classified.open("r", encoding="utf-8")))
    for row in rows:
        t1 = (row.get("t1_image") or row.get("image") or "").strip()
        if not t1:
            continue
        stem = Path(t1).stem

        imgp = None
        for ext in [".jpg",".jpeg",".png",".JPG",".JPEG",".PNG"]:
            p = img_root / f"{stem}{ext}"
            if p.exists():
                imgp = p; break
        if imgp is None:
            hits = list(img_root.rglob(f"{stem}.jpg")) + list(img_root.rglob(f"{stem}.png"))
            if hits: imgp = hits[0]
        if imgp is None:
            continue

        try:
            im = Image.open(str(imgp)).convert("RGB")
        except Exception:
            continue
        W, Himg = im.size
        d = ImageDraw.Draw(im)

        color = GREEN
        if args.respect_illegal_from_csv:
            illeg = (row.get("illegal") or "").strip().lower()
            if illeg in ("1", "true", "t", "yes", "y"):
                color = RED
            else:
                dec = (row.get("decision") or "").strip().upper()
                if dec == "UNKNOWN":
                    color = GRAY
                else:
                    color = GREEN
        else:
            dec = (row.get("decision") or "").strip().upper()
            if dec == "ILLEGAL":
                color = RED
            elif dec == "LEGAL":
                color = GREEN
            else:
                color = GRAY

        box = parse_box(row)
        if box:
            x1,y1,x2,y2 = box
            x1 = max(0, min(W - 1, x1)); x2 = max(0, min(W - 1, x2))
            y1 = max(0, min(Himg - 1, y1)); y2 = max(0, min(Himg - 1, y2))
            if x2 > x1 and y2 > y1:
                for k in range(max(1, args.thick)):
                    d.rectangle([x1 - k, y1 - k, x2 + k, y2 + k], outline=color)

        if args.draw_polys:
            polygon_path = (row.get("polygon_path") or "").strip()
            template_id = (row.get("template_id") or "").strip()
            if not polygon_path:
                if template_id and template_id in tid2poly:
                    polygon_path = tid2poly[template_id]
                elif template_id:
                    pref = stem_to_prefix(template_id)
                    if pref in pref2poly:
                        polygon_path = pref2poly[pref]
                    elif poly_root:
                        guess = poly_root / f"{template_id}.geojson"
                        if guess.exists():
                            polygon_path = str(guess)

            H_path = (row.get("H_path") or "").strip()
            if not H_path:
                if template_id:
                    H_guess = H_root / f"{template_id}.npy"
                else:
                    H_guess = H_root / f"{stem}.npy"
                H_path = str(H_guess)

            try:
                if polygon_path and Path(polygon_path).exists() and H_path and Path(H_path).exists():
                    # load geojson
                    gj = json.load(open(polygon_path, "r", encoding="utf-8"))
                    polys = []
                    for ft in (gj.get("features") or []):
                        geom = ft.get("geometry") or {}
                        gtype = (geom.get("type") or "").lower()
                        coords = geom.get("coordinates")
                        if not coords: continue
                        if gtype == "polygon":
                            polys.append(coords[0])
                        elif gtype == "multipolygon":
                            for poly in coords:
                                polys.append(poly[0])

                    Hm = to_homog_mat(Path(H_path))
                    if args.assume_direction == "tpl2img":
                        H_draw = Hm
                    else:
                        H_draw = np.linalg.inv(Hm)

                    for ring in polys:
                        pts = []
                        for x, y in ring:
                            u, v = apply_H_xy(H_draw, x, y)
                            if u is not None and v is not None:
                                pts.append((int(round(u)), int(round(v))))
                        if len(pts) >= 2:
                            draw_thick_poly(d, pts, POLY, width=2)
            except Exception:
                pass

        vid = (row.get("vehicle_id") or "").strip()
        sfx = f"_{vid}" if vid else ""
        outp = out_dir / f"{stem}{sfx}_overlay.jpg"
        try:
            im.save(str(outp), quality=92)
        except Exception:
            outp = out_dir / f"{stem}{sfx}_overlay_fallback.jpg"
            try:
                im.save(str(outp), quality=92)
            except Exception:
                pass

if __name__ == "__main__":
    main()


# 폴리곤 포함한 오버레이
# python D:\projects\illegal_parking_project\postprocess\scripts\make_overlays_from_classified.py `
#   --classified "D:\projects\illegal_parking_project\post_process_out\classified_nounk24.csv" `
#   --images-root "D:\projects\illegal_parking_project\data\images" `
#   --H-root "D:\projects\illegal_parking_project\align\H" `
#   --catalog "D:\projects\illegal_parking_project\post_process_out\overlay_catalog.csv" `
#   --out-dir "D:\projects\illegal_parking_project\post_process_out\overlays_nounk24_polys" `
#   --thick 3 `
#   --draw-polys 1 `
#   --assume_direction tpl2img `
#   --respect-illegal-from-csv