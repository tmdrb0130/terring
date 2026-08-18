# -*- coding: utf-8 -*-
"""오답의 칸별 분포 분석 — 칸별 임계값 분리가 이득인지 판단하기 위한 사전 조사.

하는 일:
  1) test 스톤을 라벨과 1:1 매칭(가장 가까운 예측 스톤)하고, 칸 5개 각각에 대해
     놓침(정답=닿음인데 예측=안닿음) / 헛detect(반대) 를 집계
  2) 그 칸에서 틀린 스톤들의 cells 확률이 실제로 어디 분포하는지(평균·중앙값) 출력
     → 확률이 임계값 바로 아래에 몰려 있으면 임계값 낮추기가 먹힌다는 신호
  3) 칸별 임계값을 각각 독립 스윕(train)한 뒤 test 에 적용해 공통 임계값과 비교

사용법: python analyze_cells.py [--tta flip|full]
"""
import argparse
from collections import defaultdict

import numpy as np
import torch

from train_cnn import (CFG, HH, HW, MERGE_DIST_HM, NMS_K, compare, load_human_labels,
                       load_split, compute_maps, stones_to_preds)
from infer import THR, CELLS_THR, load_ensemble
import torch.nn.functional as F

CELL_NAME = {1: "1점칸", 2: "2점칸", 3: "3점칸", 4: "4점칸", 5: "중앙5원"}


def decode_with_probs(hm_s, hm_c, off, cells_map, thr):
    """decode_outputs 와 동일하되 cells 확률 벡터를 함께 반환"""
    hm_t = hm_s[None]
    pool = F.max_pool2d(hm_t, NMS_K, stride=1, padding=NMS_K // 2)
    peaks = (hm_t == pool) & (hm_t > thr)
    cands = []
    for ch, color in [(0, "red"), (1, "blue")]:
        ys, xs = torch.nonzero(peaks[0, ch], as_tuple=True)
        scored = sorted(zip(hm_s[ch, ys, xs].tolist(), ys.tolist(), xs.tolist()),
                        reverse=True)
        for score, y, x in scored[:CFG["max_per_color"]]:
            ox = float(off[0, y, x]) - 0.5
            oy = float(off[1, y, x]) - 0.5
            y0, y1 = max(0, y - 1), min(HH, y + 2)
            x0, x1 = max(0, x - 1), min(HW, x + 2)
            prob = cells_map[:, y0:y1, x0:x1].mean(dim=(1, 2)).numpy()
            cands.append((score, (x + 0.5 + ox) * CFG["stride"],
                          (y + 0.5 + oy) * CFG["stride"], color, prob))
    cands.sort(key=lambda t: t[0], reverse=True)
    out = []
    for (score, x, y, color, prob) in cands:
        if all((x - sx) ** 2 + (y - sy) ** 2 > (MERGE_DIST_HM * CFG["stride"]) ** 2
               for sx, sy, *_ in out):
            out.append((x, y, color, prob))
    return out


def match_to_labels(dets, label_stones):
    """예측 스톤 ↔ 라벨 스톤 매칭: 색이 같은 것끼리 그리디로 짝지음.
    라벨에는 좌표가 없으므로 색+cells 가 같은 것을 우선 매칭하고,
    남은 것은 같은 색끼리 임의 대응 (칸별 통계 목적이라 이 정도로 충분)."""
    pairs = []
    rest_det = list(dets)
    rest_lab = list(label_stones)
    for thr_pass in (0, 1):
        for lab in list(rest_lab):
            color, cs = lab
            for d in list(rest_det):
                if d[2] != color:
                    continue
                pred = frozenset(v + 1 for v in range(5) if d[3][v] >= CELLS_THR)
                lab_cs = frozenset() if cs == frozenset({0}) else cs
                if thr_pass == 0 and pred != lab_cs:
                    continue
                pairs.append((d, lab_cs))
                rest_det.remove(d)
                rest_lab.remove(lab)
                break
    return pairs


def sweep_per_cell(probs, targets, grid):
    """칸별 독립 스윕 — 각 칸에서 비트 정확도를 최대화하는 임계값"""
    best = []
    for v in range(5):
        p, t = probs[:, v], targets[:, v]
        scores = [(int(((p >= g) == (t > 0.5)).sum()), g) for g in grid]
        best.append(max(scores)[1])
    return best


def collect(models_list, names, human, device, tta):
    maps = compute_maps(models_list, names, device, use_tta=True, tta=tta)
    probs, targets, per_img = [], [], []
    for name in names:
        hm_s, hm_c, off, cells_map = maps[name]
        dets = decode_with_probs(hm_s, hm_c, off, cells_map, THR)
        pairs = match_to_labels(dets, human[name]["stones"])
        for d, lab_cs in pairs:
            probs.append(d[3])
            t = np.zeros(5, dtype=np.float32)
            for v in lab_cs:
                t[v - 1] = 1.0
            targets.append(t)
        per_img.append((name, dets, human[name]["stones"]))
    return np.array(probs), np.array(targets), per_img


def report_distribution(probs, targets, thr_vec):
    miss = defaultdict(list)    # 놓침: 정답 1인데 확률 < thr
    false = defaultdict(list)   # 헛detect: 정답 0인데 확률 >= thr
    for p, t in zip(probs, targets):
        for v in range(5):
            if t[v] > 0.5 and p[v] < thr_vec[v]:
                miss[v + 1].append(float(p[v]))
            elif t[v] < 0.5 and p[v] >= thr_vec[v]:
                false[v + 1].append(float(p[v]))
    print(f"\n{'칸':<8}{'정답 수':>7}{'놓침':>6}{'헛detect':>9}   놓친 확률(평균/중앙/최대)")
    for v in range(1, 6):
        n_pos = int(targets[:, v - 1].sum())
        m, f = miss[v], false[v]
        if m:
            s = f"{np.mean(m):.2f} / {np.median(m):.2f} / {np.max(m):.2f}"
        else:
            s = "-"
        print(f"{CELL_NAME[v]:<8}{n_pos:>7}{len(m):>6}{len(f):>9}   {s}")
    total_miss = sum(len(v) for v in miss.values())
    total_false = sum(len(v) for v in false.values())
    print(f"합계: 놓침 {total_miss} / 헛detect {total_false} "
          f"(스톤 {len(probs)}개 x 5칸 = {len(probs) * 5} 비트 중)")
    return miss, false


def stone_exact(probs, targets, thr_vec):
    ok = 0
    for p, t in zip(probs, targets):
        pred = tuple(p[v] >= thr_vec[v] for v in range(5))
        lab = tuple(t[v] > 0.5 for v in range(5))
        ok += (pred == lab)
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tta", choices=["flip", "full"], default="flip")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    human = load_human_labels()
    test_names = [r["image"] for r in load_split("test.csv")]
    train_names = [r["image"] for r in load_split("train.csv")]
    print(f"device: {device} | TTA({args.tta}) | 공통 임계값 {CELLS_THR}", flush=True)
    models_list = load_ensemble(device)

    print("\n[test 110장] 칸별 오답 분포 (공통 임계값 %.2f)" % CELLS_THR)
    p_te, t_te, _ = collect(models_list, test_names, human, device, args.tta)
    common = [CELLS_THR] * 5
    report_distribution(p_te, t_te, common)
    base = stone_exact(p_te, t_te, common)
    print(f"스톤 단위 cells 완전일치(공통): {base}/{len(p_te)} "
          f"({base / len(p_te) * 100:.1f}%)")

    print("\n[train 473장] 칸별 임계값 독립 스윕 중...", flush=True)
    p_tr, t_tr, _ = collect(models_list, train_names, human, device, args.tta)
    grid = [round(g, 2) for g in np.arange(0.20, 0.86, 0.05)]
    thr_vec = sweep_per_cell(p_tr, t_tr, grid)
    print("선택된 칸별 임계값: " +
          ", ".join(f"{CELL_NAME[v + 1]} {thr_vec[v]:.2f}" for v in range(5)))
    tr_c = stone_exact(p_tr, t_tr, common)
    tr_v = stone_exact(p_tr, t_tr, thr_vec)
    print(f"train 스톤 완전일치: 공통 {tr_c}/{len(p_tr)} -> 칸별 {tr_v}/{len(p_tr)}")

    print("\n[test] 칸별 임계값 적용 결과")
    report_distribution(p_te, t_te, thr_vec)
    new = stone_exact(p_te, t_te, thr_vec)
    print(f"스톤 단위 cells 완전일치: 공통 {base}/{len(p_te)} -> "
          f"칸별 {new}/{len(p_te)} ({new / len(p_te) * 100:.1f}%)")
    print("\n판단 기준: 놓침이 특정 칸에 몰려 있고 그 확률 평균이 임계값 바로 아래면 "
          "칸별 분리가 이득. 고르게 흩어져 있으면 효과 없음.")


if __name__ == "__main__":
    main()
