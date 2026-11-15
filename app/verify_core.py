# -*- coding: utf-8 -*-
"""terring_core 가 원본 추론 코드(model_004/infer.py + model_005/eval_test.py)와
같은 결과를 내는지 검사한다.

사용법: python verify_core.py 사진1.jpg [사진2.jpg ...]
"""
import os
import sys

import cv2
import numpy as np
import torch
import torchvision.models as tvm

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "model_005"))
sys.path.insert(0, os.path.join(ROOT, "model_004"))
sys.path.insert(0, HERE)

# 원본 코드는 모델 생성 때 ImageNet 가중치를 내려받는다. 어차피 체크포인트가 덮어쓰므로 생략.
_orig_resnet18 = tvm.resnet18
tvm.resnet18 = lambda *a, **k: _orig_resnet18(weights=None)

import train_cnn as t4                      # noqa: E402
import infer as inf4                        # noqa: E402
import importlib.util                       # noqa: E402
# model_004 에도 eval_test.py 가 있어 이름이 겹친다 — model_005 것을 경로로 직접 로드
_spec = importlib.util.spec_from_file_location(
    "eval_test_m5", os.path.join(ROOT, "model_005", "eval_test.py"))
ev5 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ev5)
import terring_core as core                 # noqa: E402


def main():
    paths = sys.argv[1:]
    device = "cpu"
    m4 = inf4.load_ensemble(device)
    m5, cthr = ev5.load_crop_ensemble(device)
    scorer = core.Scorer(device=device)
    assert scorer.crop_thr == cthr
    bad = 0
    for p in paths:
        # --- 원본 경로 ---
        key, path = inf4.resolve_image(p)
        with torch.no_grad():
            hm_s, hm_c, off, cells_map = t4.predict_maps(m4, key, device, use_tta=True)
        stones = t4.decode_outputs(hm_s, hm_c, off, cells_map, inf4.THR, inf4.CELLS_THR)
        raw = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        cs5 = ev5.crop_cells(m5, cv2.cvtColor(raw, cv2.COLOR_BGR2RGB), stones, cthr, device)
        ref = [(round(x, 3), round(y, 3), c, sorted(cs1), sorted(cs2))
               for (x, y, c, cs1), cs2 in zip(stones, cs5)]
        # --- 새 코어 ---
        res = scorer.score_image(core.read_image(p))
        got = [(round(s["x"] * core.ORIG_W * core.SX, 3), round(s["y"] * core.ORIG_H * core.SY, 3),
                s["color"], s["stage1_cells"], s["cells"]) for s in res["stones"]]
        ok = ref == got
        bad += not ok
        print(f"{os.path.basename(p)}: {'OK' if ok else 'MISMATCH'} | stones {len(ref)} | "
              f"total {res['total']} | size {raw.shape[1]}x{raw.shape[0]}")
        if not ok:
            print("  ref:", ref)
            print("  got:", got)
        else:
            for s in res["stones"]:
                print(f"   S{s['index']} {s['color']:<4} cells {s['cells']} score {s['score']}"
                      + ("  (stage1 differs)" if s["uncertain"] else ""))
    print("RESULT:", "ALL MATCH" if bad == 0 else f"{bad} MISMATCH")
    return bad


if __name__ == "__main__":
    sys.exit(main())
