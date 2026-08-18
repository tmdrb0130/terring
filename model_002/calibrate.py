# -*- coding: utf-8 -*-
"""보드 기하 캘리브레이션 v3: 85장 전체 프로파일 합산 후 경계 실측.
- 칸 경계 검은 선: 합산 V(밝기) 프로파일의 국소 최저점 (임계값 불필요)
- red/blue 경계: 합산 색 마스크 프로파일의 우세 교차점
- 스톤/장애물 영향은 합산으로 상쇄"""
import glob
import json
import os
import cv2
import numpy as np
import terring_cv as tc

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
BW, BH, M = tc.BW, tc.BH, tc.MARGIN


def masks_warp(img, Hm):
    warp = cv2.warpPerspective(img, Hm, (BW + 2 * M, BH + 2 * M))
    hsv = cv2.cvtColor(warp, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    red = (((H <= 10) | (H >= 168)) & (S > 110) & (V > 90)).astype(np.float64)
    blue = ((H >= 95) & (H <= 130) & (S > 110) & (V > 60)).astype(np.float64)
    yellow = ((H >= 18) & (H <= 40) & (S > 110) & (V > 140)).astype(np.float64)
    return red, blue, yellow, V.astype(np.float64)


def crossing(pr, pb, lo, hi):
    diff = pr - pb
    for y in range(lo, hi):
        if diff[y] >= 0 > diff[y + 1]:
            before = diff[max(0, y - 60):max(1, y - 15)].mean()
            after = diff[y + 15:y + 60].mean()
            if before > 0.2 and after < -0.2:
                return y + 1
    return None


def vmin(profile, lo, hi):
    s = np.convolve(profile, np.ones(7) / 7, "same")
    return lo + int(np.argmin(s[lo:hi]))


def main():
    files = sorted(glob.glob(os.path.join(tc.IMG_DIR, "*.jpg")))
    accs = None
    n_ok = 0
    circles = []
    for f in files:
        img = cv2.imread(f)
        quad = tc.find_board_quad(img)
        if quad is None:
            continue
        Hm = tc.get_homography(quad)
        red, blue, yellow, V = masks_warp(img, Hm)
        board = lambda m: m[M:M + BH, M:M + BW]
        if accs is None:
            accs = {"red": board(red).copy(), "blue": board(blue).copy(), "V": board(V).copy()}
        else:
            accs["red"] += board(red)
            accs["blue"] += board(blue)
            accs["V"] += board(V)
        ys, xs = np.nonzero(board(yellow) > 0)
        if len(xs) > 3000:
            circles.append((xs.mean() / BW, ys.mean() / BH,
                            np.sqrt(len(xs) / np.pi) / min(BW, BH)))
        n_ok += 1
    for k in accs:
        accs[k] /= n_ok
    print(f"합산 이미지 {n_ok}/{len(files)}장")

    geo = {}
    # 세로선 c1, c2: 위+아래 행 V 프로파일 최저점
    strip = np.vstack([accs["V"][int(0.05 * BH):int(0.28 * BH)],
                       accs["V"][int(0.74 * BH):int(0.96 * BH)]]).mean(axis=0)
    geo["c1"] = vmin(strip, int(0.20 * BW), int(0.45 * BW)) / BW
    geo["c2"] = vmin(strip, int(0.55 * BW), int(0.85 * BW)) / BW

    # 좌열 red→blue = h1L / 우열 red→blue = h2R
    def col(m, x0, x1):
        return accs[m][:, int(x0 * BW):int(x1 * BW)].mean(axis=1)

    geo["h1L"] = crossing(col("red", 0.04, geo["c1"] - 0.03),
                          col("blue", 0.04, geo["c1"] - 0.03),
                          int(0.10 * BH), int(0.60 * BH)) / BH
    geo["h2R"] = crossing(col("red", geo["c2"] + 0.03, 0.96),
                          col("blue", geo["c2"] + 0.03, 0.96),
                          int(0.40 * BH), int(0.90 * BH)) / BH

    # 좌열 3/2 선(h2L), 우열 4/3 선(h1R): V 최저점
    geo["h2L"] = vmin(col("V", 0.04, geo["c1"] - 0.03),
                      int(geo["h1L"] * BH) + 80, int(0.92 * BH)) / BH
    geo["h1R"] = vmin(col("V", geo["c2"] + 0.03, 0.96),
                      int(0.08 * BH), int(geo["h2R"] * BH) - 80) / BH

    carr = np.array(circles)
    geo["circle_cx"], geo["circle_cy"], geo["circle_r"] = \
        [float(np.median(carr[:, i])) for i in range(3)]
    geo["hm"] = geo["circle_cy"]  # 중열 2/1 경계는 원이 가림 → 원 중심 높이로 근사

    for k, v in geo.items():
        print(f"{k:>10}: {v:.4f}")
    path = os.path.join(MODEL_DIR, "geo.json")
    old = json.load(open(path, encoding="utf-8")) if os.path.exists(path) else {}
    old.update(geo)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(old, f, indent=2)
    print("saved geo.json")


if __name__ == "__main__":
    main()
