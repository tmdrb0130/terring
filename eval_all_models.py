# -*- coding: utf-8 -*-
"""model_001 ~ model_005 를 동일 조건(test 110장)에서 완전일치 기준으로 비교.

완전 일치 = 스톤 수 + 각 스톤의 (색, 닿은 칸 집합) 이 전부 일치해야 그 사진이 정답.
부수 지표로 총점 일치 / 스톤 단위 / MAE 도 함께 낸다.

라인업 (2026-08-24 폴더 정리 후):
  model_001  전역 CNN 멀티셋 분류 (위치 탐지 없음, ResNet34)
  model_002  OpenCV + 기하 계산 (학습 없음)
  model_003  탐지 CNN + 기하 공식 (CV pseudo-label 로 학습)
  model_004  탐지 CNN + cells 를 사람 라벨로 직접 학습
  model_005  model_004 위에 2단계 고해상도 crop 판정

전부 저장된 체크포인트 그대로 사용(재학습 없음), 임계값은 각 모델이 train 에서
고른 값. test 110장은 어느 모델의 임계값 선정에도 쓰이지 않는다.

사용법: python eval_all_models.py
"""
import csv
import importlib.util
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SPLIT_DIR = os.path.join(ROOT, "model_003")     # 분할 CSV 보유 폴더
IMG_DIRS = [os.path.join(ROOT, "data", "001~085 (08.16)"),
            os.path.join(ROOT, "data", "086~583 (08.19)")]


def img_path(name):
    for d in IMG_DIRS:
        p = os.path.join(d, name)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(name)


def load_module(alias, path):
    """폴더별 동명 모듈(train_cnn 등) 충돌을 피해 개별 로드"""
    spec = importlib.util.spec_from_file_location(alias, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[alias] = mod
    old = sys.path[:]
    sys.path.insert(0, os.path.dirname(path))
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.path[:] = old
    return mod


# ---------- 공통 라벨/지표 ----------
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


def load_labels():
    p = os.path.join(ROOT, "data", "terring_1_583.csv")
    return {r["image"]: human_stones(r)
            for r in csv.DictReader(open(p, encoding="utf-8-sig"))}


def score_of(stones):
    return sum(max(c) for _c, c in stones)


def agg(rows):
    """rows: [(name, preds 또는 None, 예측총점, 라벨스톤)]"""
    full = tot = ms = ls = 0
    err = []
    wrong = []
    for name, preds, tp, lab in rows:
        tl = score_of(lab)
        tot += (tp == tl)
        err.append(abs(tp - tl))
        ls += len(lab)
        if preds is None:                 # 총점만 내는 모델
            continue
        rest = list(preds)
        matched = 0
        for s in lab:
            if s in rest:
                rest.remove(s)
                matched += 1
        ms += matched
        f_ = (len(preds) == len(lab)) and (matched == len(lab))
        full += f_
        if not f_:
            wrong.append(name)
    return {"full": full, "tot": tot, "ms": ms, "ls": ls, "n": len(rows),
            "mae": sum(err) / max(1, len(err)), "wrong": wrong,
            "has_full": rows[0][1] is not None}


# ---------- 모델별 예측 ----------
def eval_m001(names, labels):
    m1 = load_module("m1", os.path.join(ROOT, "model_001", "train_cnn.py"))
    models_list, vocab = m1.load_ensemble(DEVICE)
    probs = m1.predict_probs(models_list, names, DEVICE)
    rows = []
    for n in names:
        c = m1.decode(probs[n], vocab)
        preds = []
        for (color, cells), cnt in c.items():
            cs = frozenset({0}) if cells == "0" else frozenset(int(ch) for ch in cells)
            preds.extend([(color, cs)] * cnt)
        rows.append((n, preds, score_of(preds), labels[n]))
    del models_list
    torch.cuda.empty_cache()
    return rows


def eval_m002(names, labels):
    import cv2
    sys.path.insert(0, os.path.join(ROOT, "model_002"))
    import terring_cv as tc
    rows = []
    for n in names:
        img = cv2.imdecode(np.fromfile(img_path(n), np.uint8), cv2.IMREAD_COLOR)
        res = tc.process_image(img)
        preds = [] if res is None else \
            [(s["color"], frozenset(s["cells"])) for s in res["stones"]]
        rows.append((n, preds, score_of(preds), labels[n]))
    return rows


def eval_m003(names, train_names, labels):
    """탐지 CNN + 기하 공식 — train 에서 임계값 스윕"""
    m3 = load_module("m3", os.path.join(ROOT, "model_003", "train_cnn.py"))
    models_list = []
    for i in range(m3.CFG["n_folds"]):
        ck = torch.load(os.path.join(ROOT, "model_003", "output", f"fold{i}_best.pt"),
                        map_location=DEVICE, weights_only=False)
        m = m3.DetNet().to(DEVICE)
        m.load_state_dict(ck["state_dict"])
        m.eval()
        models_list.append(m)

    def rows_at(maps, ns, thr):
        out = []
        for n in ns:
            hm_s, hm_c, off, reg = maps[n]
            stones, corners = m3.decode_outputs(hm_s, hm_c, off, reg, thr)
            preds = m3.to_cells(stones, corners)
            out.append((n, preds, score_of(preds), labels[n]))
        return out

    with torch.no_grad():
        maps_tr = {n: m3.predict_maps(models_list, n, DEVICE) for n in train_names}
    best = None
    for thr in [0.05, 0.10, 0.15, 0.20, 0.30, 0.40]:
        a = agg(rows_at(maps_tr, train_names, thr))
        key = (a["full"], a["ms"], a["tot"])
        if best is None or key > best[0]:
            best = (key, thr)
    thr = best[1]
    del maps_tr
    print(f"  model_003 선택 임계값 {thr}", flush=True)
    with torch.no_grad():
        maps_te = {n: m3.predict_maps(models_list, n, DEVICE) for n in names}
    rows = rows_at(maps_te, names, thr)
    del models_list, maps_te
    torch.cuda.empty_cache()
    return rows


def eval_m004(names, labels):
    m4 = load_module("m4", os.path.join(ROOT, "model_004", "train_cnn.py"))
    inf = load_module("m4i", os.path.join(ROOT, "model_004", "infer.py"))
    models_list = inf.load_ensemble(DEVICE)
    print(f"  model_004 임계값 peak {inf.THR} / cells {inf.CELLS_THR}", flush=True)
    rows = []
    with torch.no_grad():
        for n in names:
            maps = m4.predict_maps(models_list, n, DEVICE)
            stones = m4.decode_outputs(*maps, inf.THR, inf.CELLS_THR)
            preds = m4.stones_to_preds(stones)
            rows.append((n, preds, score_of(preds), labels[n]))
    del models_list
    torch.cuda.empty_cache()
    return rows


def eval_m005(names, labels):
    """1단계 model_004 위치 → 2단계 crop 칸 판정"""
    m4 = load_module("m4b", os.path.join(ROOT, "model_004", "train_cnn.py"))
    inf = load_module("m4ib", os.path.join(ROOT, "model_004", "infer.py"))
    m5 = load_module("m5", os.path.join(ROOT, "model_005", "train_crop.py"))
    ev5 = load_module("m5e", os.path.join(ROOT, "model_005", "eval_test.py"))
    stage1 = inf.load_ensemble(DEVICE)
    crop_models, cthr = ev5.load_crop_ensemble(DEVICE)
    print(f"  model_005 2단계 cells_thr {cthr}", flush=True)
    rows = []
    with torch.no_grad():
        for n in names:
            maps = m4.predict_maps(stage1, n, DEVICE)
            stones = m4.decode_outputs(*maps, inf.THR, inf.CELLS_THR)
            cs = ev5.crop_cells(crop_models, m5.load_original(n), stones, cthr, DEVICE)
            preds = [(c, s if s else frozenset({0}))
                     for (_x, _y, c, _), s in zip(stones, cs)]
            rows.append((n, preds, score_of(preds), labels[n]))
    del stage1, crop_models
    torch.cuda.empty_cache()
    return rows


DESC = {
    "model_001": "전역 CNN 멀티셋 분류 (위치 탐지 없음)",
    "model_002": "OpenCV + 기하 계산 (학습 없음)",
    "model_003": "탐지 CNN + 기하 공식",
    "model_004": "탐지 CNN + cells 사람 라벨 직접 학습",
    "model_005": "model_004 + 2단계 고해상도 crop 판정",
}


def main():
    labels = load_labels()
    test_names = [r["image"] for r in
                  csv.DictReader(open(os.path.join(SPLIT_DIR, "test.csv"),
                                      encoding="utf-8-sig"))]
    train_names = [r["image"] for r in
                   csv.DictReader(open(os.path.join(SPLIT_DIR, "train.csv"),
                                       encoding="utf-8-sig"))]
    # 2026-08-24 make_split.py 로 583장 전체를 8개 축 균형 8:2 재분할 (최대 격차 0.9%p).
    # 전 모델을 이 분할로 처음부터 재학습했으므로 노출 구분이 필요 없다.
    print(f"device {DEVICE} | test {len(test_names)}장 | train {len(train_names)}장 "
          f"(583장 다기준 균형 분할)", flush=True)

    results = {}
    for tag, fn in [("model_001", lambda: eval_m001(test_names, labels)),
                    ("model_002", lambda: eval_m002(test_names, labels)),
                    ("model_003", lambda: eval_m003(test_names, train_names, labels)),
                    ("model_004", lambda: eval_m004(test_names, labels)),
                    ("model_005", lambda: eval_m005(test_names, labels))]:
        try:
            print(f"\n[{tag}] 평가 중...", flush=True)
            results[tag] = agg(fn())
        except Exception as e:
            print(f"  {tag} 실패: {type(e).__name__}: {e}", flush=True)

    print("\n" + "=" * 72)
    print("완전 일치 = 스톤 수 + 각 스톤의 (색, 닿은 칸 집합) 전부 일치")
    print("=" * 72)
    print(f"{'모델':<11}{'완전일치':>16}{'총점일치':>12}{'스톤단위':>14}{'MAE':>7}")
    for tag in ["model_001", "model_002", "model_003", "model_004", "model_005"]:
        if tag not in results:
            continue
        a = results[tag]
        n = a["n"]
        full_s = (f"{a['full']:>3}/{n} ({a['full']/n*100:4.1f}%)"
                  if a["has_full"] else "산출 불가")
        stone_s = f"{a['ms']}/{a['ls']}" if a["has_full"] else "-"
        print(f"{tag:<11}{full_s:>16}{a['tot']/n*100:>11.1f}%{stone_s:>14}"
              f"{a['mae']:>7.2f}")
    print("\n" + "\n".join(f"  {t}: {DESC[t]}" for t in DESC if t in results))
    print("\n※ fold 단위 학습 편차가 test 기준 ±8장 수준이므로, 5장 이내 차이는 "
          "유의미한 우열로 보기 어렵다.")


if __name__ == "__main__":
    main()
