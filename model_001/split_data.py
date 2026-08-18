# -*- coding: utf-8 -*-
"""terring_1_85.csv — 난이도(보드 위 스톤 수 + 칸 겹침 정도) 기준 층화 7:3 분할"""
import csv
import os
import random
from collections import Counter

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))          # .../model_001
ROOT = os.path.dirname(MODEL_DIR)                               # .../terring
SRC = os.path.join(ROOT, "data", "001~085 (08.16)", "terring_1_85.csv")
SEED = 42

rows = []
with open(SRC, encoding="utf-8-sig") as f:
    for row in csv.DictReader(f):
        rows.append(row)

# ---------- 난이도 피처 계산 ----------
for r in rows:
    on_board = 0      # 보드 위 스톤 수 (0점 아웃 제외)
    overlap_sum = 0   # 겹침 정도: 스톤마다 (닿은 칸 수 - 1)의 합
    max_span = 0      # 가장 여러 칸에 걸친 스톤의 칸 수
    for i in range(1, 7):
        color = (r[f"s{i}_color"] or "").strip()
        cells = (r[f"s{i}_cells"] or "").strip()
        if not color:
            continue
        if max(int(c) for c in cells) == 0:
            continue
        on_board += 1
        overlap_sum += len(cells) - 1
        max_span = max(max_span, len(cells))
    r["_on_board"] = on_board
    r["_overlap"] = overlap_sum
    r["_max_span"] = max_span

# ---------- 난이도 구간 정의 ----------
def stone_bin(n):
    if n <= 1: return "S1"      # 스톤 1개
    if n <= 3: return "S2-3"    # 2~3개
    return "S4-6"               # 4개 이상

def overlap_bin(o):
    if o == 0: return "O0"      # 전부 단일 칸
    if o <= 2: return "O1-2"    # 가벼운 겹침
    return "O3+"                # 겹침 많음 (모서리 4칸 걸침 등)

for r in rows:
    r["_stratum"] = stone_bin(r["_on_board"]) + "/" + overlap_bin(r["_overlap"])

# ---------- 분포 출력 ----------
print("--- 보드 위 스톤 수 분포 ---")
c = Counter(r["_on_board"] for r in rows)
for k in sorted(c): print(f"  {k}개: {c[k]:>2}장 {'#'*c[k]}")

print("\n--- 사진별 겹침 정도(overlap_sum) 분포 ---")
c = Counter(r["_overlap"] for r in rows)
for k in sorted(c): print(f"  {k}: {c[k]:>2}장 {'#'*c[k]}")

print("\n--- 난이도 층(stratum) 분포 ---")
strata = {}
for r in rows:
    strata.setdefault(r["_stratum"], []).append(r)
for k in sorted(strata):
    print(f"  {k:<10}: {len(strata[k]):>2}장")

# 3장 미만인 층은 겹침 축을 무시하고 스톤 수 층으로 합침
merged = {}
for k, grp in strata.items():
    key = k if len(grp) >= 3 else k.split("/")[0] + "/merged"
    merged.setdefault(key, []).extend(grp)

# ---------- 층화 7:3 분할 ----------
rng = random.Random(SEED)
train, test = [], []
print("\n--- 층화 분할 (seed=42) ---")
for k in sorted(merged):
    grp = merged[k][:]
    rng.shuffle(grp)
    n_train = round(len(grp) * 0.7)
    if len(grp) >= 2:
        n_train = min(n_train, len(grp) - 1)  # 2장 이상이면 test에 최소 1장
    train += grp[:n_train]
    test += grp[n_train:]
    print(f"  {k:<12}: 전체 {len(grp):>2} -> train {n_train}, test {len(grp)-n_train}")

rng.shuffle(train)
rng.shuffle(test)

fieldnames = [k for k in rows[0].keys() if not k.startswith("_")]
for name, data in [("train.csv", train), ("test.csv", test)]:
    path = os.path.join(MODEL_DIR, name)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(data)
    ts = [int(r["total_score"]) for r in data]
    ob = [r["_on_board"] for r in data]
    ov = [r["_overlap"] for r in data]
    print(f"\n{name}: {len(data)}장")
    print(f"  총점    평균 {sum(ts)/len(ts):.1f} (범위 {min(ts)}~{max(ts)})")
    print(f"  보드 위 스톤 평균 {sum(ob)/len(ob):.2f}개, 겹침 평균 {sum(ov)/len(ov):.2f}")
    print("  " + ", ".join(sorted(r["image"] for r in data)))
