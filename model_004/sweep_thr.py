# -*- coding: utf-8 -*-
"""피크 임계값을 낮은 쪽까지 확장해 재스윕.

배경: 기존 스윕 그리드가 [0.20, 0.30, 0.40, 0.50] 이었고 최저값 0.20 이 선택됐다.
경계에서 잘린 스톤은 피크가 약해(080 의 경우 0.106) 0.20 에 걸려 버려진다.
그리드를 0.05 까지 내려 train 에서 최적값을 다시 고른다 (test 미접촉).

사용법: python sweep_thr.py
"""
import torch

from train_cnn import compute_maps, eval_maps, load_human_labels, load_split
from infer import CELLS_THR, load_ensemble

GRID = [0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30, 0.40]


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    human = load_human_labels()
    train_names = [r["image"] for r in load_split("train.csv")]
    test_names = [r["image"] for r in load_split("test.csv")]
    models_list = load_ensemble(device)
    print(f"device: {device} | cells_thr={CELLS_THR} 고정, 피크 임계값만 스윕", flush=True)

    print(f"\n[train {len(train_names)}장] 맵 계산 중...", flush=True)
    maps_tr = compute_maps(models_list, train_names, device)
    print(f"{'thr':>6}  {'완전일치':>12}  {'총점':>12}  {'스톤':>12}")
    best = None
    for thr in GRID:
        ev = eval_maps(maps_tr, train_names, human, thr, CELLS_THR)
        n, ls = ev["n"], ev["ls"]
        print(f"{thr:>6.2f}  {ev['full']:>4}/{n} ({ev['full']/n*100:>4.1f}%)  "
              f"{ev['tot']:>4}/{n} ({ev['tot']/n*100:>4.1f}%)  {ev['ms']:>4}/{ls}")
        key = (ev["full"], ev["ms"], ev["tot"])
        if best is None or key > best[0]:
            best = (key, thr)
    thr_best = best[1]
    print(f"train 최적 peak 임계값: {thr_best}")
    del maps_tr

    print(f"\n[test {len(test_names)}장] 확인", flush=True)
    maps_te = compute_maps(models_list, test_names, device)
    print(f"{'thr':>6}  {'완전일치':>12}  {'총점':>12}  {'스톤':>12}")
    for thr in GRID:
        ev = eval_maps(maps_te, test_names, human, thr, CELLS_THR)
        n, ls = ev["n"], ev["ls"]
        mark = "  <- train 선택" if thr == thr_best else ""
        print(f"{thr:>6.2f}  {ev['full']:>4}/{n} ({ev['full']/n*100:>4.1f}%)  "
              f"{ev['tot']:>4}/{n} ({ev['tot']/n*100:>4.1f}%)  {ev['ms']:>4}/{ls}{mark}")

    ev = eval_maps(maps_te, test_names, human, thr_best, CELLS_THR, detail=True)
    print(f"\n선택값 {thr_best} 에서 남은 오답:")
    pure = [x for x in test_names if int(x.split(".")[0]) >= 86]
    ev_p = eval_maps(maps_te, pure, human, thr_best, CELLS_THR)
    print(f"[완전일치·순수 test {ev_p['n']}장] {ev_p['full']}/{ev_p['n']} "
          f"({ev_p['full']/ev_p['n']*100:.1f}%)")
    for name, tl, tp, cells in ev["rows"]:
        if tp != tl:
            print(f"  {name:<10} 라벨 {tl:>3} 예측 {tp:>3}  {cells}")


if __name__ == "__main__":
    main()
