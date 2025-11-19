# -*- coding: utf-8 -*-
# 폴리곤 기반 카탈로그 생성
import csv, re
from pathlib import Path

POLY_DIR = Path(r"D:\projects\illegal_parking_project\align\polygons")
OUT = Path(r"D:\projects\illegal_parking_project\align\catalog\template_catalog.csv")

PAT = re.compile(r"^([0-9]{4}_[a-z0-9]+)_(\d{8})_(t0|t1)$", re.I)

def pick_best(per_prefix_files):
    def keyfn(p):
        m = PAT.match(p.stem.lower())
        if m:
            pre, date8, tt = m.groups()
            t1_score = 1 if tt.lower() == "t1" else 0
            return (t1_score, int(date8))
        return (0, -1)
    return max(per_prefix_files, key=keyfn)

def get_prefix(stem: str):
    m = PAT.match(stem.lower())
    if m:
        return m.group(1)
    m2 = re.match(r"^([0-9]{4}_[a-z0-9]+)", stem.lower())
    return m2.group(1) if m2 else stem.lower()

bucket = {}
for p in POLY_DIR.glob("*.geojson"):
    pre = get_prefix(p.stem)
    bucket.setdefault(pre, []).append(p)

rows = []
for pre, files in bucket.items():
    best = pick_best(files) if files else None
    if best:
        rows.append({"prefix": pre, "path": str(best)})

OUT.parent.mkdir(parents=True, exist_ok=True)
with OUT.open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["prefix","path"])
    w.writeheader(); w.writerows(rows)

print(f"[OK] catalog written: {OUT} rows: {len(rows)}")
