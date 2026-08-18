# -*- coding: utf-8 -*-
"""스톤(이미지 좌표) ↔ 사람 라벨(CSV의 s1~s6) 매핑표 생성/갱신 → data/stone_map.csv(.json)

동작:
  · 입력: model_004/labels.json (검수된 CV 좌표), data/terring_1_583.csv (사람 라벨)
  · 같은 색끼리 기하 예측 칸과 라벨 칸의 최적 할당으로 짝을 맞춤
  · sure=0 (모호): 같은 색 스톤 2개 이상 + 라벨이 서로 다름 + 1/2등 할당 점수차 < 0.15
  · 이미 사람이 확정한 항목은 보존: 기존 data/stone_map.csv 에서 human=1 인 행은
    다시 계산하지 않고 그대로 유지 (모호 판정을 사람이 고쳐준 뒤 human=1 로 표시)

데이터 추가 시: 새 이미지를 labels.json 과 라벨 CSV 에 넣고 이 스크립트를 다시 실행하면
기존 매핑(사람 확정 포함)은 유지되고 새 이미지 것만 추가된다.
"""
import csv
import io
import json
import os
import sys
from itertools import permutations

import numpy as np
import cv2

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(MODEL_DIR)
sys.path.insert(0, os.path.join(ROOT, "model_002"))
import terring_cv as tc  # noqa: E402

LABELS_JSON = os.path.join(MODEL_DIR, "labels.json")
LABEL_CSV = os.path.join(ROOT, "data", "terring_1_583.csv")
OUT_CSV = os.path.join(ROOT, "data", "stone_map.csv")
OUT_JSON = os.path.join(ROOT, "data", "stone_map.json")
MARGIN = 0.15
FIELDS = ["image", "s_idx", "color", "cells", "x", "y", "sure", "human", "geo_pred"]


def entries_of(r):
    out = []
    for i in range(1, 7):
        c = (r[f"s{i}_color"] or "").strip()
        cells = (r[f"s{i}_cells"] or "").strip()
        if not c:
            continue
        cs = frozenset(int(x) for x in cells)
        out.append({"idx": i, "color": c, "cells": cs if cs == {0} else frozenset(cs - {0})})
    return out


def jac(a, b):
    return 1 - len(a & b) / len(a | b)


def map_image(name, info, ent):
    """한 이미지 매핑 → (stones 행 목록, 모호 색 목록, 개수불일치 색 목록)"""
    st = info["stones"]
    Hm = tc.get_homography(np.array(info["quad"], dtype=np.float64))
    r_stone = tc.GEO["stone_r_ratio"] * tc.BW
    pred = []
    for (x, y, c) in st:
        p = cv2.perspectiveTransform(np.array([[[x, y]]], np.float64), Hm)[0, 0]
        pred.append(frozenset(tc.touched_cells(float(p[0]), float(p[1]), r_stone)))
    rows, amb_colors, bad_colors = [], [], []
    for color in ("red", "blue"):
        E = [e for e in ent if e["color"] == color]
        S = [(k, st[k], pred[k]) for k in range(len(st)) if st[k][2] == color]
        if len(E) != len(S):
            bad_colors.append(color)
            continue
        if not E:
            continue
        costs = sorted(
            (sum(jac(S[i][2], E[perm[i]]["cells"]) for i in range(len(E))), perm)
            for perm in permutations(range(len(E))))
        best_cost, best_perm = costs[0]
        distinct = len({e["cells"] for e in E}) > 1
        amb = False
        if distinct and len(costs) > 1:
            for c2, p2 in costs[1:]:
                if any(E[best_perm[i]]["cells"] != E[p2[i]]["cells"] for i in range(len(E))):
                    amb = (c2 - best_cost) < MARGIN
                    break
        if amb:
            amb_colors.append(color)
        for i, pi in enumerate(best_perm):
            _k, (x, y, c), pcells = S[i]
            e = E[pi]
            rows.append({
                "image": name, "s_idx": f"s{e['idx']}", "color": c,
                "cells": "".join(map(str, sorted(e["cells"]))),
                "x": round(x), "y": round(y),
                "sure": 0 if amb else 1, "human": 0,
                "geo_pred": "".join(map(str, sorted(pcells)))})
    return rows, amb_colors, bad_colors


def main():
    coord = json.load(open(LABELS_JSON, encoding="utf-8"))
    label_rows = list(csv.DictReader(io.open(LABEL_CSV, encoding="utf-8-sig", newline="")))

    # 사람이 확정(human=1)한 이미지는 통째로 보존
    keep = {}
    if os.path.exists(OUT_CSV):
        for r in csv.DictReader(io.open(OUT_CSV, encoding="utf-8-sig", newline="")):
            if r["image"] not in keep and any(
                    x["human"] == "1" for x in keep.get(r["image"], [])) is False:
                pass
            keep.setdefault(r["image"], []).append(r)
        keep = {img: rows for img, rows in keep.items()
                if any(r.get("human") == "1" for r in rows)}
        print(f"기존 매핑에서 사람 확정 이미지 {len(keep)}장 보존")

    out, amb_imgs, bad_imgs = [], [], []
    for r in label_rows:
        name = r["image"]
        if name in keep:
            out += [{k: v for k, v in row.items()} for row in keep[name]]
            continue
        info = coord.get(name)
        if info is None or not info["stones"]:
            if entries_of(r):
                bad_imgs.append(name)
            continue
        rows, amb, bad = map_image(name, info, entries_of(r))
        out += rows
        if amb:
            amb_imgs.append(name)
        if bad:
            bad_imgs.append(name)

    with io.open(OUT_CSV, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(out)
    by_img = {}
    for row in out:
        by_img.setdefault(row["image"], []).append(row)
    json.dump(by_img, open(OUT_JSON, "w", encoding="utf-8"), indent=1)

    n_sure = sum(1 for r in out if str(r["sure"]) == "1" or str(r.get("human")) == "1")
    print(f"스톤 {len(out)}개 매핑 (확정 {n_sure} / 모호 {len(out)-n_sure})")
    print(f"모호 이미지 {len(amb_imgs)}장:", amb_imgs)
    print(f"개수불일치(매핑 불가 색 존재) {len(bad_imgs)}장:", bad_imgs)
    print("→", OUT_CSV)


if __name__ == "__main__":
    main()
