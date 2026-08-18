# -*- coding: utf-8 -*-
"""[구 번호 체계 legacy 스크립트 — 2026-08-25 폴더 재정리 이전 기준]
구 model_001(총점 회귀) ~ 구 model_004(현 model_003) 를 test 110장에서 동일 지표로 비교 평가.
※ 본문·코드의 번호는 전부 구 체계: 구 003→폴더 제거(85장판), 구 004→현 003.
   구 001(총점 회귀)은 현재 model_001/legacy_total_regression.py 로 보존 — 이 스크립트가
   로드하는 model_001/train_cnn.py 는 지금은 다른 모델(멀티셋 분류)이라 그대로 실행 불가.
   현행 전 모델 비교는 루트의 eval_all_models.py 를 사용할 것.

지표 (기존과 동일):
  [완전 일치] 스톤 수 + 각 스톤의 (색, 닿은 칸 집합) 전부 일치
  [총점 일치] 이미지 총점 일치            [MAE] |예측총점 - 정답총점| 평균
  [스톤 단위] 라벨 스톤 중 (색, 닿은 칸)까지 맞춘 비율
  ※ model_001 은 총점 회귀 모델이라 총점/MAE 만 산출 가능

기준 통일:
  · 재학습 없음 — 각 모델의 저장된 체크포인트/파이프라인 그대로
  · CNN(003/004) 임계값: model_004 train 473장에서 스윕 (test 110장 미접촉)
  · 부분 집계: model_001/003 은 test 중 옛 train 58장 출신을 학습 때 봤으므로
    '신규 086~ 데이터만' 집계를 함께 출력
"""
import csv
import importlib.util
import io
import os
import sys

import numpy as np
import torch
from PIL import Image

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(MODEL_DIR)
sys.path.insert(0, os.path.join(ROOT, "model_002"))
import terring_cv as tc  # noqa: E402
import cv2  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
THRS = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]


def load_module(name, path):
    """모듈명이 전부 train_cnn 이라 파일 경로로 각각 다른 이름으로 로드"""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_rows(path):
    return list(csv.DictReader(io.open(path, encoding="utf-8-sig", newline="")))


def human_stones(r):
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
    return stones


def img_path(name):
    for d in (os.path.join(ROOT, "data", "001~085 (08.16)"),
              os.path.join(ROOT, "data", "086~583 (08.19)")):
        p = os.path.join(d, name)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(name)


def agg(rows):
    """rows: (name, preds or None, total_pred, label_stones) → 지표 dict"""
    full = tot = ms = ls = 0
    abs_err = []
    for name, preds, tp, lab in rows:
        tl = sum(max(c) for _, c in lab)
        abs_err.append(abs(tp - tl))
        tot += (tp == tl)
        if preds is None:            # 총점만 내는 모델
            ls += len(lab)
            continue
        rest = list(preds)
        m = 0
        for x in lab:
            if x in rest:
                rest.remove(x)
                m += 1
        full += (len(preds) == len(lab) and m == len(lab))
        ms += m
        ls += len(lab)
    n = len(rows)
    return {"n": n, "full": full, "tot": tot, "ms": ms, "ls": ls,
            "mae": sum(abs_err) / n}


def fmt(name, a, stone_ok=True):
    s_full = f"{a['full']/a['n']*100:5.1f}%" if stone_ok else "    - "
    s_ms = f"{a['ms']/a['ls']*100:5.1f}%" if stone_ok else "    - "
    return (f"{name:<10} {s_full}   {a['tot']/a['n']*100:5.1f}%   {s_ms}"
            f"   {a['mae']:5.2f}")


# ---------- 각 모델 예측 ----------
def eval_cv(names, labels):
    rows = []
    for name in names:
        img = cv2.imdecode(np.fromfile(img_path(name), np.uint8), cv2.IMREAD_COLOR)
        res = tc.process_image(img)
        preds = [] if res is None else \
            [(s["color"], frozenset(s["cells"])) for s in res["stones"]]
        rows.append((name, preds, sum(max(c) for _, c in preds), labels[name]))
    return rows


def eval_m001(names, labels):
    m1 = load_module("m1_train", os.path.join(ROOT, "model_001", "train_cnn.py"))
    sv = m1.SCORE_VALUES.to(DEVICE)
    models_list = []
    for i in range(m1.CFG["n_folds"]):
        ck = torch.load(os.path.join(ROOT, "model_001", "output", f"fold{i}_best.pt"),
                        weights_only=False)
        m = m1.build_model().to(DEVICE)
        m.load_state_dict(ck["state_dict"])
        m.eval()
        models_list.append(m)
    eval_tf = m1.make_eval_tf(m1.CFG["img_size"])
    tta_tf = m1.make_tta_tf(m1.CFG["img_size"])
    rows = []
    with torch.no_grad():
        for name in names:
            img = Image.open(img_path(name)).convert("RGB")
            preds = []
            for m in models_list:
                preds.append(m1.predict_total(m, img, eval_tf, DEVICE, sv))
                for _ in range(m1.CFG["tta_rounds"]):
                    preds.append(m1.predict_total(m, img, tta_tf, DEVICE, sv))
            tp = int(np.clip(round(float(np.mean(preds))), 0, 30))
            rows.append((name, None, tp, labels[name]))
    del models_list
    torch.cuda.empty_cache()
    return rows


def eval_det(mod, out_dir, tag, test_names, train_names, labels):
    """model_003/004 공용: 앙상블 로드 → train 스윕으로 thr 선정 → test 평가"""
    models_list = []
    for i in range(mod.CFG["n_folds"]):
        ck = torch.load(os.path.join(out_dir, f"fold{i}_best.pt"), weights_only=False)
        m = mod.DetNet().to(DEVICE)
        m.load_state_dict(ck["state_dict"])
        m.eval()
        models_list.append(m)

    def decode_rows(maps, names, thr):
        rows = []
        for name in names:
            hm_s, hm_c, off, reg = maps[name]
            stones, corners = mod.decode_outputs(hm_s, hm_c, off, reg, thr)
            preds = mod.to_cells(stones, corners)
            rows.append((name, preds, sum(max(c) for _, c in preds), labels[name]))
        return rows

    with torch.no_grad():
        maps_tr = {n: mod.predict_maps(models_list, n, DEVICE) for n in train_names}
    best = None
    for thr in THRS:
        a = agg(decode_rows(maps_tr, train_names, thr))
        key = (a["ms"], a["tot"], a["full"])
        if best is None or key > best[0]:
            best = (key, thr)
    thr = best[1]
    del maps_tr
    print(f"  {tag}: 선택 임계값 {thr}", flush=True)

    with torch.no_grad():
        maps_te = {n: mod.predict_maps(models_list, n, DEVICE) for n in test_names}
    rows = decode_rows(maps_te, test_names, thr)
    del models_list, maps_te
    torch.cuda.empty_cache()
    return rows


def main():
    test_rows = load_rows(os.path.join(MODEL_DIR, "test.csv"))
    train_rows = load_rows(os.path.join(MODEL_DIR, "train.csv"))
    labels = {r["image"]: human_stones(r)
              for r in load_rows(os.path.join(ROOT, "data", "terring_1_583.csv"))}
    train_names = [r["image"] for r in train_rows]

    # 순수 테스트: 어느 모델의 학습(001/003: 구 train 58)·보정(002: 구 85)에도
    # 노출된 적 없는 신규 데이터(086~) 출신만
    pure = [r["image"] for r in test_rows if int(r["image"].split(".")[0]) >= 86]
    print(f"순수 test {len(pure)}장 (086~ 출신, 전체 test 110장 중) | device {DEVICE}", flush=True)
    print("※ model_001 은 총점 회귀 모델이라 완전 일치 산출 불가 → 제외\n", flush=True)

    results = {}
    results["model_002"] = eval_cv(pure, labels)

    m3 = load_module("m3_train", os.path.join(ROOT, "model_003", "train_cnn.py"))
    m3._IMG_CACHE = {}
    m3.load_input_image = lambda name, _m=m3: _load_resized(name, _m)
    results["model_003"] = eval_det(m3, os.path.join(ROOT, "model_003", "output"),
                                    "model_003", pure, train_names, labels)

    m4 = load_module("m4_train", os.path.join(MODEL_DIR, "train_cnn.py"))
    results["model_004"] = eval_det(m4, os.path.join(MODEL_DIR, "output"),
                                    "model_004", pure, train_names, labels)

    print(f"\n=== 완전 일치 (순수 test {len(pure)}장) ===")
    for k in ("model_002", "model_003", "model_004"):
        a = agg(results[k])
        print(f"{k:<10}  {a['full']}/{a['n']}  ({a['full']/a['n']*100:.1f}%)")


def _load_resized(name, mod):
    if name not in mod._IMG_CACHE:
        img = cv2.imdecode(np.fromfile(img_path(name), np.uint8), cv2.IMREAD_COLOR)
        img = cv2.resize(img, (mod.IW, mod.IH), interpolation=cv2.INTER_AREA)
        mod._IMG_CACHE[name] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return mod._IMG_CACHE[name]


if __name__ == "__main__":
    main()
