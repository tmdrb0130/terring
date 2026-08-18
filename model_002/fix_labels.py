# -*- coding: utf-8 -*-
"""사용자 판정(14건)에 따라 terring_1_85.csv 라벨 수정.
같은 장면의 연속 사진에도 동일 적용. 수정 후 정합성(score=max, total=sum) 재검증."""
import csv
import os
import shutil

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(MODEL_DIR)
CSV = os.path.join(ROOT, "data", "001~085 (08.16)", "terring_1_85.csv")

# (이미지, 스톤색, 기존 cells) -> 새 cells
FIXES = {
    # case01: 빨간 스톤 2번 칸에도 닿음 (345 -> 2345)
    ("001.jpg", "red", "4535"): "2345",
    ("002.jpg", "red", "4535"): "2345",
    ("003.jpg", "red", "4535"): "2345",
    # case02: 파란 스톤은 3,5만 닿음 (1235 -> 35)
    ("006.jpg", "blue", "3521"): "35",
    ("008.jpg", "blue", "3521"): "35",
    ("009.jpg", "blue", "3521"): "35",
    ("010.jpg", "blue", "3521"): "35",
    # case09: 빨간 스톤은 보드 밖 (12 -> 0)
    ("018.jpg", "red", "12"): "0",
    ("020.jpg", "red", "12"): "0",
    ("021.jpg", "red", "12"): "0",
    # case10: 라벨 오타 (4 -> 1)
    ("038.jpg", "blue", "4"): "1",
}

shutil.copy(CSV, CSV + ".bak")
rows = []
with open(CSV, encoding="utf-8-sig") as f:
    reader = csv.reader(f)
    header = next(reader)
    rows = list(reader)

col = {name: i for i, name in enumerate(header)}
applied = []
for r in rows:
    img = r[col["image"]]
    total = 0
    n_stones = 0
    for i in range(1, 7):
        c_col, cells_col, score_col = col[f"s{i}_color"], col[f"s{i}_cells"], col[f"s{i}_score"]
        color = r[c_col].strip()
        if not color:
            continue
        n_stones += 1
        key = (img, color, r[cells_col].strip())
        if key in FIXES:
            r[cells_col] = FIXES[key]
            applied.append(f"{img} s{i} {color}: {key[2]} -> {FIXES[key]}")
        # score/total 재계산
        new_score = max(int(ch) for ch in r[cells_col].strip())
        r[score_col] = str(new_score)
        total += new_score
    old_total = r[col["total_score"]]
    r[col["total_score"]] = str(total)
    r[col["num_stones"]] = str(n_stones)
    if old_total != str(total):
        applied.append(f"  {img} total {old_total} -> {total}")

with open(CSV, "w", newline="", encoding="utf-8-sig") as f:
    w = csv.writer(f)
    w.writerow(header)
    w.writerows(rows)

print(f"적용된 수정 ({len([a for a in applied if not a.startswith(' ')])}건 셀 + 총점 변경):")
for a in applied:
    print(" ", a)
print(f"\n백업: {CSV}.bak")
