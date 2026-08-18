# -*- coding: utf-8 -*-
"""model_004 (구 model_005) test 110장 평가 (재학습 없음 — 저장된 체크포인트로 앙상블+TTA만 실행).

사용법: python eval_test.py [--tta flip|full]
  --tta full: 좌우반전에 회전 ±10도 / 스케일 0.92·1.08 뷰를 추가한 확장 TTA
지표: 스톤 단위 / 총점 일치 / 완전 일치(스톤 수+색+닿은 칸 전부) / MAE + 오답 목록
"""
import argparse

import torch

from train_cnn import CFG, evaluate_names, load_human_labels, load_split
from infer import THR, CELLS_THR, load_ensemble


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tta", choices=["flip", "full"], default="flip")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    test_names = [r["image"] for r in load_split("test.csv")]
    human = load_human_labels()
    print(f"device: {device} | 앙상블 {CFG['n_folds']}개 fold + TTA({args.tta}) | "
          f"thr={THR} cells={CELLS_THR} | test {len(test_names)}장", flush=True)
    models_list = load_ensemble(device)

    ev = evaluate_names(models_list, test_names, human, device, THR, CELLS_THR,
                        detail=True, tta=args.tta)
    n, ls = ev["n"], ev["ls"]
    print(f"\n=== test {n}장 결과 ===")
    print(f"[스톤 단위] {ev['ms']}/{ls} ({ev['ms']/ls*100:.0f}%)")
    print(f"[총점 일치] {ev['tot']}/{n} ({ev['tot']/n*100:.0f}%) | MAE {ev['mae']:.2f}")
    print(f"[완전 일치] {ev['full']}/{n} ({ev['full']/n*100:.0f}%)")

    pure = [x for x in test_names if int(x.split(".")[0]) >= 86]
    ev_p = evaluate_names(models_list, pure, human, device, THR, CELLS_THR, tta=args.tta)
    print(f"[완전 일치·순수 test {ev_p['n']}장] {ev_p['full']}/{ev_p['n']} "
          f"({ev_p['full']/ev_p['n']*100:.1f}%)")

    print(f"\n{'image':<10}{'정답':>5}{'예측':>5}  스톤별 판정 (총점 오답만)")
    for name, tl, tp, cells in ev["rows"]:
        if tp != tl:
            mark = "  <<<" if abs(tp - tl) >= 3 else "  <"
            print(f"{name:<10}{tl:>5}{tp:>5}  {cells}{mark}")


if __name__ == "__main__":
    main()
