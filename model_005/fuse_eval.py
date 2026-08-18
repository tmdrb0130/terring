# -*- coding: utf-8 -*-
"""model_004(전역 저해상도) + model_005(국소 고해상도) cells 확률 융합 실험.
(코드 안 변수명 m5/m6 는 구 번호 흔적 — 구 005=현 004, 구 006=현 005)

두 모델은 서로 다른 이미지에서 틀렸다(현 005는 475/578을 고쳤지만 008/207/394를 깨뜨림).
오류가 상보적이면 확률 평균이 양쪽보다 낫다 — 가중치 w 를 스윕해 확인한다.
w=0 이면 model_004 단독, w=1 이면 model_005 단독.

사용법: python fuse_eval.py
"""
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(MODEL_DIR)
sys.path.insert(0, os.path.join(ROOT, "model_004"))

from train_cnn import (CFG, HH, HW, MERGE_DIST_HM, NMS_K, SX, SY, compare,  # noqa: E402
                       load_human_labels, load_split, predict_maps, stones_to_preds)
from infer import THR, CELLS_THR, load_ensemble  # noqa: E402
from train_crop import (CROP, CropNet, OUT_DIR as M6_OUT, crop_region,  # noqa: E402
                        load_original, normalize)


def load_crop_ensemble(device):
    """model_005 crop 앙상블 (eval_test 와 동일 — model_004 의 동명 모듈과 충돌해 여기 둠)"""
    import json
    cfg = json.load(open(os.path.join(M6_OUT, "config.json"), encoding="utf-8"))
    out = []
    for i in range(cfg["n_folds"]):
        ck = torch.load(os.path.join(M6_OUT, f"fold{i}_best.pt"),
                        map_location=device, weights_only=False)
        m = CropNet().to(device)
        m.load_state_dict(ck["state_dict"])
        m.eval()
        out.append(m)
    return out, cfg["cells_thr_final"]


def decode_with_probs(hm_s, off, cells_map, thr):
    """스톤 피크 검출 + m5 cells 확률 벡터"""
    hm_t = hm_s[None]
    pool = F.max_pool2d(hm_t, NMS_K, stride=1, padding=NMS_K // 2)
    peaks = (hm_t == pool) & (hm_t > thr)
    cands = []
    for ch, color in [(0, "red"), (1, "blue")]:
        ys, xs = torch.nonzero(peaks[0, ch], as_tuple=True)
        scored = sorted(zip(hm_s[ch, ys, xs].tolist(), ys.tolist(), xs.tolist()),
                        reverse=True)
        for score, y, x in scored[:CFG["max_per_color"]]:
            ox, oy = float(off[0, y, x]) - 0.5, float(off[1, y, x]) - 0.5
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


@torch.no_grad()
def crop_probs(models_list, img_orig, dets, device):
    if not dets:
        return np.zeros((0, 5), dtype=np.float32)
    batch = []
    for (x, y, _c, _p) in dets:
        c = crop_region(img_orig, x / SX, y / SY, CROP)
        batch.append(normalize(np.ascontiguousarray(c)))
        batch.append(normalize(np.ascontiguousarray(c[:, ::-1])))
    xb = torch.stack(batch).to(device)
    p = torch.stack([m(xb) for m in models_list])
    p = torch.sigmoid(p).mean(0).view(len(dets), 2, 5).mean(1)
    return p.cpu().numpy()


def score_at(cache, names, human, w, thr5, thr6):
    """융합 확률로 완전일치/총점 집계. 임계값은 두 모델 임계값의 가중 평균."""
    thr = (1 - w) * thr5 + w * thr6
    full = tot = ms = ls = 0
    wrong = []
    for name in names:
        dets, p5, p6 = cache[name]
        preds = []
        for i, (_x, _y, color, _) in enumerate(dets):
            p = (1 - w) * p5[i] + w * p6[i]
            cs = frozenset(v + 1 for v in range(5) if p[v] >= thr)
            preds.append((color, cs if cs else frozenset({0})))
        f_, t_, m_, l_, _tp, _tl = compare(preds, human[name]["stones"])
        full += f_
        tot += t_
        ms += m_
        ls += l_
        if not f_:
            wrong.append(name)
    return full, tot, ms, ls, wrong


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    test_names = [r["image"] for r in load_split("test.csv")]
    human = load_human_labels()
    m5 = load_ensemble(device)
    m6, thr6 = load_crop_ensemble(device)
    print(f"device: {device} | test {len(test_names)}장 | m5 cells_thr={CELLS_THR} "
          f"m6 cells_thr={thr6}", flush=True)

    cache = {}
    for i, name in enumerate(test_names):
        hm_s, _hm_c, off, cells_map = predict_maps(m5, name, device, use_tta=True)
        dets = decode_with_probs(hm_s, off, cells_map, THR)
        p5 = np.array([d[3] for d in dets], dtype=np.float32) if dets else np.zeros((0, 5))
        p6 = crop_probs(m6, load_original(name), dets, device)
        cache[name] = (dets, p5, p6)
        if (i + 1) % 30 == 0:
            print(f"  {i + 1}/{len(test_names)}장...", flush=True)

    pure = [n for n in test_names if int(n.split(".")[0]) >= 86]
    print(f"\n{'w':>5}  {'전체 완전일치':>14}  {'순수 91장':>12}  {'스톤':>10}  총점")
    results = {}
    for w in [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]:
        full, tot, ms, ls, wrong = score_at(cache, test_names, human, w, CELLS_THR, thr6)
        pf = sum(1 for n in pure if n not in wrong)
        results[w] = wrong
        tag = " (m5 단독)" if w == 0 else (" (m6 단독)" if w == 1 else "")
        print(f"{w:>5.1f}  {full:>4}/110 ({full/110*100:>5.1f}%)  "
              f"{pf:>3}/91 ({pf/91*100:>5.1f}%)  {ms:>4}/{ls}  {tot}/110{tag}")

    best_w = max(results, key=lambda w: 110 - len(results[w]))
    print(f"\n최적 w={best_w} 오답: {sorted(results[best_w])}")
    print(f"m5 단독 오답: {sorted(results[0.0])}")
    print(f"m6 단독 오답: {sorted(results[1.0])}")


if __name__ == "__main__":
    main()
