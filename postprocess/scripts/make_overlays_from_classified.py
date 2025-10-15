#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
classified_stops.csv 기반 시각 오버레이 생성
- 불법(ILLEGAL)=빨강, 합법(LEGAL)=초록, 나무(TREE)=파랑, 기타/외부 회색(OUTSIDE는 점선)
- 이미지 확장자/대소문자 자동 매칭
출력: out_dir/overlays/*.overlay.png
"""
import os, argparse, pandas as pd
from PIL import Image, ImageDraw, ImageFont

COLORS = {
    "ILLEGAL": (255, 0, 0),
    "LEGAL":   (0, 200, 0),
    "TREE":    (0, 160, 255),
    "OTHER":   (180, 180, 180),
    "OUTSIDE": (140, 140, 140),
    "NO_POLYGON": (255, 128, 0),
    "NO_ALIAS":   (180, 0, 180),
}

def build_index(image_root):
    idx = {}
    for root, _, files in os.walk(image_root):
        for fn in files:
            base = os.path.splitext(fn)[0].lower()
            idx.setdefault(base, os.path.join(root, fn))
    return idx

def draw_dashed_rect(draw, box, color, width=3, dash=10, gap=6):
    x1, y1, x2, y2 = box
    x = x1
    while x < x2:
        draw.line([(x, y1), (min(x+dash, x2), y1)], fill=color, width=width)
        draw.line([(x, y2), (min(x+dash, x2), y2)], fill=color, width=width)
        x += dash + gap
    y = y1
    while y < y2:
        draw.line([(x1, y), (x1, min(y+dash, y2))], fill=color, width=width)
        draw.line([(x2, y), (x2, min(y+dash, y2))], fill=color, width=width)
        y += dash + gap

def text_size(draw, text, font):
    """Pillow 10+ 호환: textbbox 우선, 폴백 getlength/height"""
    if hasattr(draw, "textbbox"):
        l, t, r, b = draw.textbbox((0, 0), text, font=font)
        return r - l, b - t
    # 폴백
    try:
        return font.getsize(text)
    except Exception:
        # 마지막 폴백: 대략값
        return (8 * len(text), 16)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--classified", required=True)
    ap.add_argument("--image-root", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-per-image", type=int, default=999999)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    ov_dir = os.path.join(args.out_dir, "overlays")
    os.makedirs(ov_dir, exist_ok=True)

    df = pd.read_csv(args.classified)
    need = {"image_name","xc","yc","t1_w","t1_h","classification"}
    miss = list(need - set(df.columns))
    if miss:
        raise SystemExit(f"[FATAL] 필요한 컬럼 없음: {miss}")

    index = build_index(args.image_root)

    # 폰트
    try:
        font = ImageFont.truetype("malgun.ttf", 18)
    except Exception:
        try:
            font = ImageFont.truetype("arial.ttf", 18)
        except Exception:
            font = ImageFont.load_default()

    for img_name, sub in df.groupby("image_name"):
        base = os.path.splitext(str(img_name))[0].lower()
        path = index.get(base)
        if not path:
            continue
        try:
            im = Image.open(path).convert("RGB")
        except Exception:
            continue

        draw = ImageDraw.Draw(im, "RGBA")
        count_by_cls = {}

        for i, r in enumerate(sub.itertuples(index=False), 1):
            if i > args.max_per_image:
                break
            cls = str(r.classification)
            x, y, w, h = float(r.xc), float(r.yc), float(r.t1_w), float(r.t1_h)
            x1, y1, x2, y2 = x - w/2, y - h/2, x + w/2, y + h/2
            color = COLORS.get(cls, (255,255,0))

            if cls == "OUTSIDE":
                draw_dashed_rect(draw, (x1, y1, x2, y2), color, width=3)
            else:
                draw.rectangle([x1, y1, x2, y2], outline=color + (255,), width=3)

            # 라벨 박스
            label = "ILLEGAL" if cls == "ILLEGAL" else ("LEGAL" if cls == "LEGAL" else cls)
            tw, th = text_size(draw, label, font)
            draw.rectangle([x1, y1 - th - 6, x1 + tw + 6, y1], fill=(0,0,0,120))
            draw.text((x1 + 3, y1 - th - 3), label, fill=(255,255,255,230), font=font)

            count_by_cls[cls] = count_by_cls.get(cls, 0) + 1

        # 범례/카운트
        legend_lines = [f"{k}: {count_by_cls[k]}" for k in ["ILLEGAL","LEGAL","TREE","OTHER","OUTSIDE"] if count_by_cls.get(k,0)>0]
        if legend_lines:
            widths = [text_size(draw, s, font)[0] for s in legend_lines]
            box_w = max(widths) + 16
            line_h = text_size(draw, legend_lines[0], font)[1] + 6
            box_h = line_h * len(legend_lines) + 12
            draw.rectangle([8, 8, 8 + box_w, 8 + box_h], fill=(0,0,0,120))
            ycur = 12
            for s in legend_lines:
                key = s.split(":")[0]
                col = COLORS.get(key, (255,255,255))
                draw.text((16, ycur), s, fill=col + (255,), font=font)
                ycur += line_h

        out_path = os.path.join(ov_dir, f"{base}.overlay.png")
        im.save(out_path)

    print(f"[OK] overlays 생성 → {ov_dir}")

if __name__ == "__main__":
    main()

