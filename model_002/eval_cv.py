# -*- coding: utf-8 -*-
"""
85장 전수 평가 + alpha(접촉점 보정) / stone_r_ratio / circle_r 그리드서치.

평가 기준(엄격한 순):
  [완전 일치] 스톤 수 + 각 스톤의 (색, 닿은 칸 집합)이 라벨과 전부 일치
  [총점 일치] 이미지 총점 일치
  [스톤 단위] 라벨 스톤 중 (색, 닿은 칸 집합)까지 일치한 비율
보드/스톤 검출은 이미지당 1회만 수행하고, 파라미터 탐색은 캐시로 빠르게.
"""
import csv
import json
import os
import cv2
import numpy as np
import terring_cv as tc

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(MODEL_DIR)
LABELS = os.path.join(ROOT, "data", "001~085 (08.16)", "terring_1_85.csv")
BW = tc.BW


def load_labels():
    out = {}
    with open(LABELS, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            stones = []
            for i in range(1, 7):
                color = (r[f"s{i}_color"] or "").strip()
                cells = (r[f"s{i}_cells"] or "").strip()
                if not color:
                    continue
                cs = frozenset(int(c) for c in cells)
                if cs != {0}:
                    cs = frozenset(cs - {0})
                stones.append((color, cs))
            out[r["image"]] = {"stones": stones, "total": int(r["total_score"])}
    return out


def detect_all():
    """이미지별 검출 캐시: H행렬, 스톤 (x, y, color, 실루엣 최하단 y)"""
    cache = {}
    labels = load_labels()
    for name in sorted(labels):
        img = cv2.imread(os.path.join(tc.IMG_DIR, name))
        quad = tc.find_board_quad(img)
        if quad is None:
            cache[name] = None
            continue
        Hm = tc.get_homography(quad)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        stones = []
        for (x, y, color) in tc.find_stones(img):
            sb = tc.silhouette_bottom(hsv, x, y, color)
            stones.append((x, y, color, sb))
        cache[name] = {"H": Hm, "stones": stones}
    return cache, labels


def predict(cache_entry, alpha, ratio):
    Hm = cache_entry["H"]
    stone_r = ratio * BW
    preds = []
    for (x, y, color, sb) in cache_entry["stones"]:
        p = Hm @ np.array([x, y + alpha * (sb - y), 1.0])
        wx, wy = p[0] / p[2], p[1] / p[2]
        vals = tc.touched_cells(wx, wy, stone_r)
        preds.append((color, frozenset(vals)))
    return preds


def score_of(cells):
    return max(cells)


def compare(preds, label_stones):
    """반환: (완전일치, 총점일치, 스톤단위 일치 수, 라벨 스톤 수)"""
    total_p = sum(score_of(c) for _, c in preds)
    total_l = sum(score_of(c) for _, c in label_stones)
    rest = list(preds)
    matched = 0
    for ls in label_stones:
        if ls in rest:
            rest.remove(ls)
            matched += 1
    full = (len(preds) == len(label_stones)) and (matched == len(label_stones))
    return full, total_p == total_l, matched, len(label_stones)


def evaluate(cache, labels, alpha, ratio):
    full = tot = ms = ls = 0
    n = 0
    for name, lab in labels.items():
        ce = cache[name]
        if ce is None:
            ls += len(lab["stones"])
            n += 1
            continue
        f_, t_, m_, l_ = compare(predict(ce, alpha, ratio), lab["stones"])
        full += f_
        tot += t_
        ms += m_
        ls += l_
        n += 1
    return full, tot, ms, ls, n


def main():
    print("검출 캐시 생성 중...")
    cache, labels = detect_all()
    n_no_board = sum(1 for v in cache.values() if v is None)
    n_det = {name: len(v["stones"]) if v else 0 for name, v in cache.items()}
    n_lab = {name: len(v["stones"]) for name, v in labels.items()}
    cnt_match = sum(1 for k in labels if n_det[k] == n_lab[k])
    print(f"보드 검출 실패 {n_no_board}장 / 스톤 개수 일치 {cnt_match}/{len(labels)}장")
    for k in sorted(labels):
        if n_det[k] != n_lab[k]:
            print(f"  개수 불일치 {k}: 검출 {n_det[k]} vs 라벨 {n_lab[k]}")

    print("\n그리드서치 (alpha, ratio, circle_r 보정) ...")
    # 캘리브레이션 실측값을 기준으로 보정 (튜닝값 재보정 누적 방지)
    path0 = os.path.join(MODEL_DIR, "geo.json")
    geo0 = json.load(open(path0, encoding="utf-8"))
    base_circle_r = geo0.get("circle_r_cal", 0.1751)
    best = None
    for cr_d in [-0.005, 0.0, 0.005, 0.01, 0.015]:
        tc.GEO["circle_r"] = base_circle_r + cr_d
        for alpha in np.arange(0.0, 0.75, 0.05):
            for ratio in np.arange(0.055, 0.108, 0.0025):
                full, tot, ms, ls, n = evaluate(cache, labels, alpha, ratio)
                key = (full, tot, ms)
                if best is None or key > best[0]:
                    best = (key, alpha, ratio, cr_d)
    (full, tot, ms), alpha, ratio, cr_d = best
    tc.GEO["circle_r"] = base_circle_r + cr_d
    print(f"circle_r 보정: {cr_d:+.3f} → {tc.GEO['circle_r']:.4f}")
    n = len(labels)
    ls = sum(len(v["stones"]) for v in labels.values())
    print(f"\n최적: alpha={alpha:.2f}, stone_r_ratio={ratio:.4f}")
    print(f"[완전 일치] {full}/{n} ({full/n*100:.0f}%)  — 모든 스톤의 색+닿은 칸까지 일치")
    print(f"[총점 일치] {tot}/{n} ({tot/n*100:.0f}%)")
    print(f"[스톤 단위] {ms}/{ls} ({ms/ls*100:.0f}%)  — (색, 닿은 칸 집합) 일치 비율")

    path = os.path.join(MODEL_DIR, "geo.json")
    geo = json.load(open(path, encoding="utf-8"))
    geo["alpha"] = round(float(alpha), 3)
    geo["stone_r_ratio"] = round(float(ratio), 4)
    geo["circle_r_cal"] = round(float(base_circle_r), 4)
    geo["circle_r"] = round(float(tc.GEO["circle_r"]), 4)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(geo, f, indent=2)
    tc.GEO.update(geo)

    print("\n--- 불일치 이미지 상세 ---")
    for name in sorted(labels):
        ce = cache[name]
        lab = labels[name]
        if ce is None:
            print(f"{name}: 보드 검출 실패")
            continue
        preds = predict(ce, alpha, ratio)
        f_, t_, m_, l_ = compare(preds, lab["stones"])
        if not f_:
            pd = sorted("".join(map(str, sorted(c))) + col[0] for col, c in preds)
            lb = sorted("".join(map(str, sorted(c))) + col[0] for col, c in lab["stones"])
            mark = "" if t_ else "  [총점도 불일치]"
            print(f"{name}: 예측 {pd} vs 라벨 {lb}{mark}")


if __name__ == "__main__":
    main()
