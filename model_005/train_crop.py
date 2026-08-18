# -*- coding: utf-8 -*-
"""
model_005 (구 model_006): 2단계 고해상도 crop 칸 판정 — 경계 분쟁의 구조적 해법

배경: model_004(구 model_005)는 원본 1050x1400을 576x768로 축소해 넣으므로 경계에 걸친 스톤의
"닿았나/안 닿았나" 정보가 다운샘플에서 뭉개진다 (남은 오답 대부분이 한 칸 경계 분쟁).

구조:
  · 1단계(model_004 앙상블): 스톤 위치+색 탐지만 담당
  · 2단계(이 파일): 스톤 중심 주변을 원본 해상도로 384x384 crop → ResNet18이
    cells 5비트(multi-label) 직접 분류. 칸 숫자·색이 crop 안에 그대로 보이므로
    국소 판정이 성립한다. 빈 집합 = 보드 밖 0.

학습 데이터: data/stone_map.json (사람 검수 좌표+cells) 중 train.csv 473장의 스톤.
분할: model_003/004와 동일한 5-fold (make_folds 재사용, seed 42).
증강: 중심 지터 ±16px(1단계 위치 오차 모사), 회전 ±20°, 스케일 0.8~1.2,
      좌우반전(보드 값 배치 좌우 대칭이라 cells 불변), 색/밝기 지터.
임계값: 학습 후 out-of-fold 예측으로 스윕 → output/config.json 에 저장.
평가: eval_test.py 가 1단계 위치 → 2단계 crop 파이프라인으로 test 110장 측정.
"""
import csv
import json
import os
import random
import sys

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(MODEL_DIR)
sys.path.insert(0, os.path.join(ROOT, "model_004"))
from train_cnn import load_split, make_folds  # noqa: E402  (동일 fold 분할 재사용)

OUT_DIR = os.path.join(MODEL_DIR, "output")
IMG_DIRS = [os.path.join(ROOT, "data", "001~085 (08.16)"),
            os.path.join(ROOT, "data", "086~583 (08.19)")]

CFG = {
    "seed": 42,
    "crop": 384,           # 모델 입력 (원본 해상도 그대로, 리사이즈 없음)
    "region": 512,         # 증강 여유를 포함해 미리 잘라두는 영역
    "jitter": 16,          # 중심 지터 (1단계 위치 오차 모사)
    "rot": 20.0,
    "scale_lo": 0.8, "scale_hi": 1.2,
    "max_epochs": 60,
    "patience": 10,
    "batch_size": 32,
    "lr_backbone": 1e-4,
    "lr_head": 1e-3,
    "weight_decay": 1e-4,
    "repeat_per_epoch": 2,
    "cells_thr": 0.5,      # 학습 중 val 측정용 (최종은 OOF 스윕)
    "n_folds": 5,
}
random.seed(CFG["seed"])
np.random.seed(CFG["seed"])
torch.manual_seed(CFG["seed"])
torch.cuda.manual_seed_all(CFG["seed"])

CROP, REGION = CFG["crop"], CFG["region"]
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def img_path(name):
    for d in IMG_DIRS:
        p = os.path.join(d, name)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(name)


def load_original(name):
    """원본 해상도(1050x1400) RGB"""
    img = cv2.imdecode(np.fromfile(img_path(name), dtype=np.uint8), cv2.IMREAD_COLOR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def crop_region(img, cx, cy, size):
    """(cx,cy) 중심 size x size crop — 경계 밖은 회색(127) 패딩"""
    h, w = img.shape[:2]
    half = size // 2
    x0, y0 = int(round(cx)) - half, int(round(cy)) - half
    x1, y1 = x0 + size, y0 + size
    pl, pt = max(0, -x0), max(0, -y0)
    pr, pb = max(0, x1 - w), max(0, y1 - h)
    c = img[max(0, y0):min(h, y1), max(0, x0):min(w, x1)]
    if pl or pt or pr or pb:
        c = cv2.copyMakeBorder(c, pt, pb, pl, pr, cv2.BORDER_CONSTANT,
                               value=(127, 127, 127))
    return c


def load_stone_rows(names):
    """stone_map.json → [(image, s_idx, x, y, cells 집합)] (원본 좌표계, names 소속만)"""
    raw = json.load(open(os.path.join(ROOT, "data", "stone_map.json"), encoding="utf-8"))
    nameset = set(names)
    rows = []
    for name, stones in raw.items():
        if name not in nameset:
            continue
        for s in stones:
            cs = frozenset() if s["cells"] == "0" else \
                frozenset(int(c) for c in s["cells"])
            rows.append((name, s["s_idx"], float(s["x"]), float(s["y"]), cs))
    return rows


def cells_target(cs):
    t = np.zeros(5, dtype=np.float32)
    for v in cs:
        t[v - 1] = 1.0
    return t


def build_region_cache(rows):
    """이미지당 1회 로드 → 스톤별 REGION crop 캐시 (약 1.5GB)"""
    by_img = {}
    for r in rows:
        by_img.setdefault(r[0], []).append(r)
    cache = {}
    for i, (name, rs) in enumerate(sorted(by_img.items())):
        img = load_original(name)
        for (nm, sidx, x, y, cs) in rs:
            cache[(nm, sidx)] = crop_region(img, x, y, REGION)
        if (i + 1) % 100 == 0:
            print(f"  crop 캐시 {i + 1}/{len(by_img)}장", flush=True)
    return cache


def center_crop(region):
    o = (REGION - CROP) // 2
    return region[o:o + CROP, o:o + CROP]


def normalize(img):
    x = (img.astype(np.float32) / 255.0 - MEAN) / STD
    return torch.from_numpy(x.transpose(2, 0, 1))


class CropDS(Dataset):
    def __init__(self, rows, cache, train, repeat=1):
        self.rows = rows
        self.cache = cache
        self.train = train
        self.repeat = repeat

    def __len__(self):
        return len(self.rows) * self.repeat

    def __getitem__(self, idx):
        name, sidx, _x, _y, cs = self.rows[idx % len(self.rows)]
        region = self.cache[(name, sidx)]
        if self.train:
            ang = random.uniform(-CFG["rot"], CFG["rot"])
            sc = random.uniform(CFG["scale_lo"], CFG["scale_hi"])
            jx = random.uniform(-CFG["jitter"], CFG["jitter"])
            jy = random.uniform(-CFG["jitter"], CFG["jitter"])
            Mm = cv2.getRotationMatrix2D((REGION / 2, REGION / 2), ang, sc)
            Mm[0, 2] += jx
            Mm[1, 2] += jy
            img = cv2.warpAffine(region, Mm, (REGION, REGION),
                                 borderValue=(127, 127, 127))
            img = center_crop(img)
            if random.random() < 0.5:
                # 좌우반전: 보드 값 배치가 좌우 대칭(4-2-4/3-5-3/2-1-2)이라 cells 불변
                img = img[:, ::-1]
            f = np.random.uniform(0.7, 1.3, size=3).astype(np.float32)
            img = np.clip(img.astype(np.float32) * f, 0, 255)
            img = np.clip(img + np.random.uniform(-25, 25), 0, 255).astype(np.uint8)
        else:
            img = center_crop(region)
        return normalize(np.ascontiguousarray(img)), torch.from_numpy(cells_target(cs))


class CropNet(nn.Module):
    def __init__(self):
        super().__init__()
        r = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        r.fc = nn.Linear(512, 5)
        self.net = r

    def forward(self, x):
        return self.net(x)   # logits


@torch.no_grad()
def predict_probs(model, ds, device, batch=64):
    """deterministic crop 확률 (N,5)"""
    model.eval()
    dl = DataLoader(ds, batch_size=batch, shuffle=False, num_workers=0)
    out = []
    for x, _t in dl:
        out.append(torch.sigmoid(model(x.to(device))).cpu())
    return torch.cat(out).numpy()


def exact_match(probs, rows, thr):
    ok = 0
    for p, (_n, _s, _x, _y, cs) in zip(probs, rows):
        pred = frozenset(v + 1 for v in range(5) if p[v] >= thr)
        ok += (pred == cs)
    return ok


def train_fold(fold_i, tr_rows, va_rows, cache, device, log_writer):
    ds_tr = CropDS(tr_rows, cache, train=True, repeat=CFG["repeat_per_epoch"])
    ds_va = CropDS(va_rows, cache, train=False)
    dl = DataLoader(ds_tr, batch_size=CFG["batch_size"], shuffle=True, num_workers=0)
    model = CropNet().to(device)
    bb = [p for n, p in model.named_parameters() if not n.startswith("net.fc")]
    hd = [p for n, p in model.named_parameters() if n.startswith("net.fc")]
    opt = torch.optim.AdamW([{"params": bb, "lr": CFG["lr_backbone"]},
                             {"params": hd, "lr": CFG["lr_head"]}],
                            weight_decay=CFG["weight_decay"])
    crit = nn.BCEWithLogitsLoss()
    ckpt = os.path.join(OUT_DIR, f"fold{fold_i}_best.pt")
    best, best_epoch, epoch = -1, 0, 0
    while True:
        epoch += 1
        model.train()
        tr_loss = 0.0
        for x, t in dl:
            x, t = x.to(device), t.to(device)
            loss = crit(model(x), t)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tr_loss += loss.item() * x.size(0)
        tr_loss /= len(ds_tr)

        probs = predict_probs(model, ds_va, device)
        em = exact_match(probs, va_rows, CFG["cells_thr"])
        is_best = em > best
        if is_best:
            best, best_epoch = em, epoch
            torch.save({"epoch": epoch, "config": CFG,
                        "state_dict": model.state_dict()}, ckpt)
        log_writer.writerow([fold_i, epoch, f"{tr_loss:.4f}",
                             f"{em}/{len(va_rows)}", int(is_best)])
        if epoch % 5 == 0 or is_best:
            print(f"[fold {fold_i}] ep {epoch:>3} | loss {tr_loss:.4f} | "
                  f"val 스톤 cells 일치 {em}/{len(va_rows)}"
                  + ("  * best" if is_best else ""), flush=True)
        if epoch - best_epoch >= CFG["patience"] or epoch >= CFG["max_epochs"]:
            print(f"[fold {fold_i}] 종료 (best ep {best_epoch}: {best}/{len(va_rows)})",
                  flush=True)
            break
    return ckpt


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device, flush=True)

    train_all = load_split("train.csv")
    folds = make_folds(train_all, CFG["n_folds"], CFG["seed"])
    all_names = [n for f_ in folds for n in f_]
    rows = load_stone_rows(all_names)
    print(f"train {len(all_names)}장 / 스톤 {len(rows)}개 → {CFG['n_folds']}-fold "
          f"(crop {CROP} 원본 해상도)", flush=True)
    print("crop 캐시 구축 중...", flush=True)
    cache = build_region_cache(rows)

    fold_of = {n: i for i, f_ in enumerate(folds) for n in f_}
    log_f = open(os.path.join(OUT_DIR, "train_log.csv"), "w",
                 newline="", encoding="utf-8-sig")
    log_writer = csv.writer(log_f)
    log_writer.writerow(["fold", "epoch", "loss", "val_exact", "best"])

    oof_probs = np.zeros((len(rows), 5), dtype=np.float32)
    for i in range(CFG["n_folds"]):
        tr_rows = [r for r in rows if fold_of[r[0]] != i]
        va_rows = [r for r in rows if fold_of[r[0]] == i]
        print(f"\n===== fold {i}: 스톤 train {len(tr_rows)} / val {len(va_rows)} =====",
              flush=True)
        ckpt = train_fold(i, tr_rows, va_rows, cache, device, log_writer)
        log_f.flush()
        # best 체크포인트로 out-of-fold 확률 수집 (임계값 스윕용)
        model = CropNet().to(device)
        model.load_state_dict(torch.load(ckpt, weights_only=False)["state_dict"])
        probs = predict_probs(model, CropDS(va_rows, cache, train=False), device)
        idx = [j for j, r in enumerate(rows) if fold_of[r[0]] == i]
        oof_probs[idx] = probs
    log_f.close()

    print("\n임계값 스윕 (out-of-fold, 스톤 단위 cells 완전 일치):", flush=True)
    best = None
    for thr in [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]:
        em = exact_match(oof_probs, rows, thr)
        print(f"  cells_thr {thr:.2f}: {em}/{len(rows)} ({em / len(rows) * 100:.2f}%)",
              flush=True)
        if best is None or em > best[0]:
            best = (em, thr)
    em, thr = best
    print(f"선택: cells_thr {thr} (OOF {em}/{len(rows)} = {em / len(rows) * 100:.2f}%)")
    CFG["cells_thr_final"] = thr
    with open(os.path.join(OUT_DIR, "config.json"), "w", encoding="utf-8") as f:
        json.dump(CFG, f, indent=2, ensure_ascii=False)
    print("완료. eval_test.py 로 2단계 파이프라인 test 평가를 실행하세요.")


if __name__ == "__main__":
    main()
