# -*- coding: utf-8 -*-
"""583장을 난이도 층화로 학습/테스트 8:2 분할 → train.csv / test.csv

난이도 특성 3축 (유형 = 세 버킷의 조합):
  ① 보드 위 스톤 수      S0-1 / S2-3 / S4+     (사람 라벨 기준)
  ② 스톤당 걸친 칸 수 합  O0 / O1-2 / O3+       (on-board 스톤의 len(cells)-1 합)
  ③ 경계 밀착도          T0 / T1               (CV 좌표 기준, 어느 스톤이든 칸 경계·중앙원까지
                                                거리가 |d - r| < 0.15r 이면 T1 — 판정이 아슬아슬한 장면)
각 유형 안에서 셔플 후 5장마다 1장을 테스트로 → 유형별로 정확히 20%씩 균등 배분.
출력 CSV는 두 세트 모두 행 순서를 다시 셔플(장면 연속성 제거)."""
import csv
import io
import json
import os
import random
import sys

import numpy as np

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(MODEL_DIR)
sys.path.insert(0, os.path.join(ROOT, "model_002"))
import terring_cv as tc  # noqa: E402

SEED = 42
CSV = os.path.join(ROOT, "data", "terring_1_583.csv")
LABELS_JSON = os.path.join(MODEL_DIR, "labels.json")


def stone_margin(wx, wy):
    """워프 좌표의 스톤이 칸 경계/중앙원에 얼마나 밀착했는지: min |d - r| / r"""
    r = tc.GEO["stone_r_ratio"] * tc.BW
    best = 1e9
    for (x0, y0, x1, y1, _v) in tc._cell_rects():
        nx, ny = tc.circle_rect_nearest(wx, wy, x0, y0, x1, y1)
        best = min(best, abs(((wx - nx) ** 2 + (wy - ny) ** 2) ** 0.5 - r))
    ccx = tc.MARGIN + tc.GEO["circle_cx"] * tc.BW
    ccy = tc.MARGIN + tc.GEO["circle_cy"] * tc.BH
    cr = tc.GEO["circle_r"] * min(tc.BW, tc.BH)
    d = ((wx - ccx) ** 2 + (wy - ccy) ** 2) ** 0.5
    best = min(best, abs(d - (r + cr)))
    return best / r


def tightness(coord_entry):
    """이미지의 최소 밀착도 (CV 검출 좌표 기준). 스톤 없으면 큰 값."""
    quad = np.array(coord_entry["quad"], dtype=np.float64)
    Hm = tc.get_homography(quad)
    m = 1e9
    for (x, y, _c) in coord_entry["stones"]:
        import cv2
        p = cv2.perspectiveTransform(np.array([[[x, y]]], dtype=np.float64), Hm)[0, 0]
        m = min(m, stone_margin(float(p[0]), float(p[1])))
    return m


def main():
    coord = json.load(open(LABELS_JSON, encoding="utf-8"))
    rows = list(csv.DictReader(io.open(CSV, encoding="utf-8-sig", newline="")))
    hdr = list(rows[0].keys())

    strata = {}
    for r in rows:
        on_board = 0
        overlap = 0
        for i in range(1, 7):
            color = (r[f"s{i}_color"] or "").strip()
            cells = (r[f"s{i}_cells"] or "").strip()
            if not color:
                continue
            s = max(int(c) for c in cells)
            if s > 0:
                on_board += 1
                overlap += len(cells) - 1
        sb = "S0-1" if on_board <= 1 else ("S2-3" if on_board <= 3 else "S4+")
        ob = "O0" if overlap == 0 else ("O1-2" if overlap <= 2 else "O3+")
        ce = coord.get(r["image"])
        tb = "T?" if ce is None else ("T1" if tightness(ce) < 0.15 else "T0")
        strata.setdefault(sb + "/" + ob + "/" + tb, []).append(r)

    rng = random.Random(SEED)
    train, test = [], []
    print(f"{'유형':<16}{'전체':>5}{'test':>6}")
    for key in sorted(strata):
        g = strata[key]
        rng.shuffle(g)
        t = [r for i, r in enumerate(g) if i % 5 == 4]   # 5장마다 1장 → 20%
        tr = [r for i, r in enumerate(g) if i % 5 != 4]
        train += tr
        test += t
        print(f"{key:<16}{len(g):>5}{len(t):>6}")
    rng.shuffle(train)
    rng.shuffle(test)

    for name, data in [("train.csv", train), ("test.csv", test)]:
        with io.open(os.path.join(MODEL_DIR, name), "w",
                     encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=hdr)
            w.writeheader()
            w.writerows(data)
    print(f"\ntrain {len(train)} / test {len(test)}  (seed {SEED})")


if __name__ == "__main__":
    main()
