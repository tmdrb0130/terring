# -*- coding: utf-8 -*-
"""model_002 CV 파이프라인으로 583장 전체의 좌표 라벨 자동 생성 → labels.json
(보드 코너 4점 + 스톤 중심 좌표/색). CNN 학습의 지도 신호로 사용.
구 model_003(85장판, 폴더 제거됨)의 gen_labels.py와 동일하되 두 데이터 폴더(001~085, 086~583)를 모두 처리."""
import glob
import json
import os
import sys

import cv2
import numpy as np

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(MODEL_DIR)
sys.path.insert(0, os.path.join(ROOT, "model_002"))
import terring_cv as tc  # noqa: E402

IMG_DIRS = [os.path.join(ROOT, "data", "001~085 (08.16)"),
            os.path.join(ROOT, "data", "086~583 (08.19)")]

out = {}
files = sorted(f for d in IMG_DIRS for f in glob.glob(os.path.join(d, "*.jpg")))
for f in files:
    name = os.path.basename(f)
    img = cv2.imdecode(np.fromfile(f, dtype=np.uint8), cv2.IMREAD_COLOR)
    quad = tc.find_board_quad(img)
    if quad is None:
        print("board fail:", name)
        continue
    stones = tc.find_stones(img, tc.get_homography(quad))
    out[name] = {
        "quad": [[round(float(x), 1), round(float(y), 1)] for x, y in quad],
        "stones": [[round(float(x), 1), round(float(y), 1), c] for x, y, c in stones],
    }

with open(os.path.join(MODEL_DIR, "labels.json"), "w", encoding="utf-8") as fp:
    json.dump(out, fp, indent=1)
print(f"saved labels.json ({len(out)}/{len(files)}장)")
