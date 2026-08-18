# -*- coding: utf-8 -*-
"""model_005 (구 model_006) 2단계 파이프라인 test 110장 평가.

파이프라인: 1단계 model_004 앙상블(위치+색 탐지) → 2단계 model_005 crop 앙상블(칸 판정).
비교를 위해 model_004 단독(칸 판정까지 1단계 헤드) 결과도 같은 위치에서 함께 출력.

사용법: python eval_test.py
"""
import json
import os
import sys

import numpy as np
import torch

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(MODEL_DIR)
sys.path.insert(0, os.path.join(ROOT, "model_004"))

from train_cnn import (SX, SY, compare, decode_outputs, load_human_labels,  # noqa: E402
                       load_split, predict_maps, stones_to_preds)
from infer import THR, CELLS_THR, load_ensemble  # noqa: E402
from train_crop import (CROP, CropNet, OUT_DIR as M6_OUT, center_crop,  # noqa: E402
                        crop_region, load_original, normalize)

REGION_EVAL = CROP  # 평가 시엔 지터 여유 불필요 — 바로 CROP 크기로 자름


def load_crop_ensemble(device):
    cfg = json.load(open(os.path.join(M6_OUT, "config.json"), encoding="utf-8"))
    models_list = []
    for i in range(cfg["n_folds"]):
        ck = torch.load(os.path.join(M6_OUT, f"fold{i}_best.pt"),
                        map_location=device, weights_only=False)
        m = CropNet().to(device)
        m.load_state_dict(ck["state_dict"])
        m.eval()
        models_list.append(m)
    return models_list, cfg["cells_thr_final"]


@torch.no_grad()
def crop_cells(models_list, img_orig, stones_in, thr, device):
    """1단계 스톤 위치(입력 좌표계) → 원본 crop → cells 집합 목록 (앙상블+반전 TTA)"""
    if not stones_in:
        return []
    batch = []
    for (x, y, _c, _cs) in stones_in:
        c = crop_region(img_orig, x / SX, y / SY, CROP)
        batch.append(normalize(np.ascontiguousarray(c)))
        batch.append(normalize(np.ascontiguousarray(c[:, ::-1])))  # 좌우반전 TTA
    xb = torch.stack(batch).to(device)
    probs = torch.stack([torch.sigmoid(m(xb)) for m in models_list]).mean(0)
    probs = probs.view(len(stones_in), 2, 5).mean(1).cpu().numpy()
    return [frozenset(v + 1 for v in range(5) if p[v] >= thr) for p in probs]


def metrics_init():
    return {"full": 0, "tot": 0, "ms": 0, "ls": 0, "n": 0, "abs": [], "rows": []}


def metrics_add(mt, name, preds, label):
    f_, t_, m_, l_, tp, tl = compare(preds, label["stones"])
    mt["full"] += f_
    mt["tot"] += t_
    mt["ms"] += m_
    mt["ls"] += l_
    mt["n"] += 1
    mt["abs"].append(abs(tp - tl))
    mt["rows"].append((name, f_, tl, tp,
                       sorted("".join(map(str, sorted(c))) + col[0] for col, c in preds)))


def metrics_print(tag, mt):
    n, ls = mt["n"], mt["ls"]
    mae = sum(mt["abs"]) / max(1, n)
    print(f"[{tag}] 스톤 {mt['ms']}/{ls} ({mt['ms'] / ls * 100:.0f}%) | "
          f"총점 {mt['tot']}/{n} ({mt['tot'] / n * 100:.0f}%) MAE {mae:.2f} | "
          f"완전 일치 {mt['full']}/{n} ({mt['full'] / n * 100:.1f}%)")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    test_names = [r["image"] for r in load_split("test.csv")]
    human = load_human_labels()
    print(f"device: {device} | 1단계 model_004 앙상블+TTA (thr={THR}) → "
          f"2단계 model_005 crop 앙상블+TTA | test {len(test_names)}장", flush=True)
    m5 = load_ensemble(device)
    m6, cthr = load_crop_ensemble(device)
    print(f"2단계 cells_thr = {cthr} (train OOF 스윕 선택)", flush=True)

    mt5, mt6 = metrics_init(), metrics_init()
    for i, name in enumerate(test_names):
        hm_s, hm_c, off, cells_map = predict_maps(m5, name, device, use_tta=True)
        stones = decode_outputs(hm_s, hm_c, off, cells_map, THR, CELLS_THR)
        img = load_original(name)
        cs6 = crop_cells(m6, img, stones, cthr, device)
        preds5 = stones_to_preds(stones)
        preds6 = [(c, cs if cs else frozenset({0}))
                  for (_x, _y, c, _), cs in zip(stones, cs6)]
        metrics_add(mt5, name, preds5, human[name])
        metrics_add(mt6, name, preds6, human[name])
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{len(test_names)}장...", flush=True)

    print(f"\n=== test {mt6['n']}장 결과 (같은 1단계 위치 기준) ===")
    metrics_print("m4 단독 ", mt5)
    metrics_print("m5 2단계", mt6)

    pure = {x for x in test_names if int(x.split(".")[0]) >= 86}
    for tag, mt in [("m4", mt5), ("m5", mt6)]:  # 변수명 mt5/mt6 는 구 번호 흔적
        f_ = sum(r[1] for r in mt["rows"] if r[0] in pure)
        print(f"[완전 일치·순수 test {len(pure)}장·{tag}] {f_}/{len(pure)} "
              f"({f_ / len(pure) * 100:.1f}%)")

    wrong5 = {r[0] for r in mt5["rows"] if not r[1]}
    wrong6 = {r[0] for r in mt6["rows"] if not r[1]}
    print(f"\nm4 오답 {len(wrong5)}장 -> m5 오답 {len(wrong6)}장 | "
          f"고쳐진 {sorted(wrong5 - wrong6)} | 새로 틀린 {sorted(wrong6 - wrong5)}")

    print(f"\n{'image':<10}{'정답':>5}{'예측':>5}  m5 스톤별 판정 (완전 일치 실패만)")
    for name, f_, tl, tp, cells in mt6["rows"]:
        if not f_:
            print(f"{name:<10}{tl:>5}{tp:>5}  {cells}")


if __name__ == "__main__":
    main()
