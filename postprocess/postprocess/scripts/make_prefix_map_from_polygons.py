# -*- coding: utf-8 -*-
import os, csv, glob

POLY_DIR = r"D:\projects\illegal_parking_project\align\polygons"
OUT = r"D:\projects\illegal_parking_project\post_process_out\prefix2template.csv"

rows = []
for path in glob.glob(os.path.join(POLY_DIR, "*.geojson")):
    base = os.path.basename(path)
    stem = os.path.splitext(base)[0]  # template_id (예: 0001_top_20250619_t1)
    parts = stem.split('_')
    # prefix: 앞 2토큰 (예: 0001_top)
    if len(parts) >= 2 and parts[0].isdigit() and len(parts[0]) == 4:
        prefix = f"{parts[0]}_{parts[1]}"
    else:
        prefix = '_'.join(parts[:2]) if len(parts) >= 2 else parts[0]
    rows.append({'prefix': prefix, 'template_id': stem})

# prefix 중복시 첫 항목만 사용(필요하면 규칙 바꿔도 됨)
seen = set(); uniq = []
for r in rows:
    p = r['prefix']
    if p in seen:
        continue
    seen.add(p); uniq.append(r)

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, 'w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=['prefix','template_id'])
    w.writeheader()
    w.writerows(uniq)

print(f"OK -> {OUT}, rows={len(uniq)}")
