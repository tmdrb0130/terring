# -*- coding: utf-8 -*-
"""model_002를 구 model_003(85장판, 폴더 제거됨)과 같은 조건(구 test 27장)으로만 평가 — 공정 비교용 legacy"""
import csv
import os

import cv2
import terring_cv as tc
from eval_cv import load_labels, compare, predict

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(MODEL_DIR)

test_names = []
with open(os.path.join(ROOT, "model_001", "test.csv"), encoding="utf-8-sig") as f:
    for r in csv.DictReader(f):
        test_names.append(r["image"])

labels = load_labels()
alpha = tc.GEO.get("alpha", 0.0)
ratio = tc.GEO["stone_r_ratio"]

full = tot = ms = ls = 0
for name in test_names:
    img = cv2.imread(os.path.join(tc.IMG_DIR, name))
    quad = tc.find_board_quad(img)
    Hm = tc.get_homography(quad)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    stones = []
    for (x, y, color) in tc.find_stones(img):
        sb = tc.silhouette_bottom(hsv, x, y, color)
        stones.append((x, y, color, sb))
    ce = {"H": Hm, "stones": stones}
    f_, t_, m_, l_ = compare(predict(ce, alpha, ratio), labels[name]["stones"])
    full += f_
    tot += t_
    ms += m_
    ls += l_

n = len(test_names)
print(f"model_002 / test {n}장 기준")
print(f"[완전 일치] {full}/{n} ({full/n*100:.0f}%)")
print(f"[총점 일치] {tot}/{n} ({tot/n*100:.0f}%)")
print(f"[스톤 단위] {ms}/{ls} ({ms/ls*100:.0f}%)")
