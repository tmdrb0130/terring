# -*- coding: utf-8 -*-
"""터링 채점기 추론 코어 — 학습 코드/데이터 없이 체크포인트만으로 동작하는 독립 모듈.

파이프라인 (eval_all_models.py 의 model_005 와 동일):
  1단계 model_004 5-fold 앙상블 + 좌우반전 TTA : 스톤 위치·색 탐지
  2단계 model_005 5-fold 앙상블 + 좌우반전 TTA : 스톤 주변 384x384 원본 해상도 crop → 닿은 칸 판정

model_004/train_cnn.py, model_004/infer.py, model_005/train_crop.py, model_005/eval_test.py 의
추론 부분만 옮겨 온 것이다. 차이는 두 가지뿐:
  · ResNet18 을 weights=None 으로 생성 (ImageNet 가중치 다운로드 없음 — 체크포인트가 전부 덮어씀)
  · 1050x1400 이 아닌 사진은 2단계 crop 전에 1050x1400 으로 리사이즈 (학습 해상도에 맞춤)
verify_core.py 가 원본 코드와 출력이 일치하는지 검사한다.
"""
import json
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

# ---------- 상수 (model_004/train_cnn.py CFG, infer.py 와 동일) ----------
IW, IH = 576, 768
STRIDE = 4
HW, HH = IW // STRIDE, IH // STRIDE
ORIG_W, ORIG_H = 1050, 1400
SX, SY = IW / float(ORIG_W), IH / float(ORIG_H)
NMS_K = 9
MERGE_DIST_HM = 12.0
MAX_PER_COLOR = 3
N_FOLDS = 5
THR = 0.05          # 스톤 피크 임계값 (infer.py)
CELLS_THR = 0.65    # 1단계 cells 헤드 임계값 (infer.py)
CROP = 384          # 2단계 crop 크기 (train_crop.py)
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

CELL_NAMES = {0: "보드 밖", 1: "1점 칸", 2: "2점 칸", 3: "3점 칸", 4: "4점 칸", 5: "중앙 5원"}
COLOR_NAMES = {"red": "빨강", "blue": "파랑"}


def resource_dir():
    """체크포인트가 들어 있는 기준 폴더. exe 로 묶였으면 번들 폴더, 아니면 저장소 루트."""
    if getattr(sys, "frozen", False):
        return os.path.join(getattr(sys, "_MEIPASS", os.path.dirname(sys.executable)), "models")
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def checkpoint_dirs():
    base = resource_dir()
    if getattr(sys, "frozen", False):
        return os.path.join(base, "model_004"), os.path.join(base, "model_005")
    return os.path.join(base, "model_004", "output"), os.path.join(base, "model_005", "output")


# ---------- 모델 ----------
class DetNet(nn.Module):
    def __init__(self):
        super().__init__()
        r = models.resnet18(weights=None)
        self.stem = nn.Sequential(r.conv1, r.bn1, r.relu, r.maxpool)
        self.l1, self.l2, self.l3, self.l4 = r.layer1, r.layer2, r.layer3, r.layer4
        self.lat4 = nn.Conv2d(512, 128, 1)
        self.lat3 = nn.Conv2d(256, 128, 1)
        self.lat2 = nn.Conv2d(128, 128, 1)
        self.lat1 = nn.Conv2d(64, 128, 1)
        self.trunk = nn.Sequential(nn.Conv2d(128, 64, 3, padding=1), nn.ReLU())
        self.hm_stone = nn.Conv2d(64, 2, 1)
        self.hm_corner = nn.Conv2d(64, 4, 1)
        self.offset = nn.Conv2d(64, 2, 1)
        self.cells = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 5, 3, padding=1))
        self.corner_reg = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, 8))

    def forward(self, x):
        c1 = self.l1(self.stem(x))
        c2 = self.l2(c1)
        c3 = self.l3(c2)
        c4 = self.l4(c3)
        p = self.lat4(c4)
        p = F.interpolate(p, size=c3.shape[2:], mode="nearest") + self.lat3(c3)
        p = F.interpolate(p, size=c2.shape[2:], mode="nearest") + self.lat2(c2)
        p = F.interpolate(p, size=c1.shape[2:], mode="nearest") + self.lat1(c1)
        t = self.trunk(p)
        return (torch.sigmoid(self.hm_stone(t)), torch.sigmoid(self.hm_corner(t)),
                torch.sigmoid(self.offset(t)), torch.sigmoid(self.cells(t)),
                self.corner_reg(c4))


class CropNet(nn.Module):
    def __init__(self):
        super().__init__()
        r = models.resnet18(weights=None)
        r.fc = nn.Linear(512, 5)
        self.net = r

    def forward(self, x):
        return self.net(x)   # logits


def _load_fold(cls, path, device):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"모델 파일이 없습니다: {path}")
    ck = torch.load(path, map_location=device, weights_only=False)
    m = cls().to(device)
    m.load_state_dict(ck["state_dict"])
    m.eval()
    return m


class Scorer:
    """모델 10개(1단계 5 + 2단계 5)를 한 번 로드해 두고 사진을 채점한다."""

    def __init__(self, device=None, progress=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        d4, d5 = checkpoint_dirs()
        self.det, self.crop = [], []
        for i in range(N_FOLDS):
            self.det.append(_load_fold(DetNet, os.path.join(d4, f"fold{i}_best.pt"), self.device))
            if progress:
                progress(i + 1, 2 * N_FOLDS)
        for i in range(N_FOLDS):
            self.crop.append(_load_fold(CropNet, os.path.join(d5, f"fold{i}_best.pt"), self.device))
            if progress:
                progress(N_FOLDS + i + 1, 2 * N_FOLDS)
        with open(os.path.join(d5, "config.json"), encoding="utf-8") as f:
            self.crop_thr = float(json.load(f)["cells_thr_final"])

    # ----- 1단계 -----
    @torch.no_grad()
    def predict_maps(self, img_bgr):
        """train_cnn.predict_maps(use_tta=True, tta='flip') 와 동일"""
        small = cv2.resize(img_bgr, (IW, IH), interpolation=cv2.INTER_AREA)
        img = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        x = (img.astype(np.float32) / 255.0 - MEAN) / STD
        x = torch.from_numpy(x.transpose(2, 0, 1))[None].to(self.device)
        xf = torch.flip(x, dims=[3])
        hs_l, hc_l, off_l, cl_l = [], [], [], []
        for m in self.det:
            hs, hc, off, cl, _reg = m(x)
            hs_l.append(hs[0].cpu())
            hc_l.append(hc[0].cpu())
            off_l.append(off[0].cpu())
            cl_l.append(cl[0].cpu())
            hs2, hc2, _, cl2, _ = m(xf)
            hs_l.append(torch.flip(hs2[0], dims=[2]).cpu())
            hc_l.append(torch.flip(hc2[0], dims=[2])[[1, 0, 3, 2]].cpu())
            cl_l.append(torch.flip(cl2[0], dims=[2]).cpu())
        return (torch.stack(hs_l).mean(0), torch.stack(hc_l).mean(0),
                torch.stack(off_l).mean(0), torch.stack(cl_l).mean(0))

    # ----- 2단계 -----
    @torch.no_grad()
    def crop_cells(self, img_rgb_orig, stones):
        """eval_test.crop_cells 와 동일. stones: [(x, y, color, cells)] 입력 좌표계"""
        if not stones:
            return []
        batch = []
        for (x, y, _c, _cs) in stones:
            c = crop_region(img_rgb_orig, x / SX, y / SY, CROP)
            batch.append(normalize(np.ascontiguousarray(c)))
            batch.append(normalize(np.ascontiguousarray(c[:, ::-1])))
        xb = torch.stack(batch).to(self.device)
        probs = torch.stack([torch.sigmoid(m(xb)) for m in self.crop]).mean(0)
        probs = probs.view(len(stones), 2, 5).mean(1).cpu().numpy()
        return [frozenset(v + 1 for v in range(5) if p[v] >= self.crop_thr) for p in probs]

    # ----- 전체 -----
    def score_image(self, img_bgr):
        """BGR 이미지 → 결과 dict. 최종 칸 판정은 2단계(model_005)."""
        hm_s, hm_c, off, cells_map = self.predict_maps(img_bgr)
        stones = decode_outputs(hm_s, off, cells_map, THR, CELLS_THR)
        h, w = img_bgr.shape[:2]
        if (w, h) == (ORIG_W, ORIG_H):
            orig = img_bgr
        else:
            interp = cv2.INTER_AREA if w > ORIG_W else cv2.INTER_LINEAR
            orig = cv2.resize(img_bgr, (ORIG_W, ORIG_H), interpolation=interp)
        cs2 = self.crop_cells(cv2.cvtColor(orig, cv2.COLOR_BGR2RGB), stones)
        out = []
        for i, ((x, y, color, cs1), cs) in enumerate(zip(stones, cs2), 1):
            out.append({
                "index": i,
                "color": color,
                "x": x / SX / ORIG_W,       # 사진 너비 대비 0~1
                "y": y / SY / ORIG_H,       # 사진 높이 대비 0~1
                "cells": sorted(cs),        # 빈 목록 = 보드 밖
                "score": max(cs) if cs else 0,
                "stage1_cells": sorted(cs1),
                "uncertain": frozenset(cs1) != cs,   # 두 모델의 판정이 다름 = 경계에 걸침
            })
        corners, conf = decode_corners(hm_c, off)
        return {
            "stones": out,
            "total": sum(s["score"] for s in out),
            "corners": (corners / [SX, SY] / [ORIG_W, ORIG_H]).tolist(),
            "corner_conf": conf,
            "landscape": w > h,
        }


# ---------- 디코딩 (train_cnn.decode_outputs / infer.decode_corners 와 동일) ----------
def decode_outputs(hm_s, off, cells_map, thr, cells_thr):
    hm_t = hm_s[None]
    pool = F.max_pool2d(hm_t, NMS_K, stride=1, padding=NMS_K // 2)
    peaks = (hm_t == pool) & (hm_t > thr)
    cands = []
    for ch, color in [(0, "red"), (1, "blue")]:
        ys, xs = torch.nonzero(peaks[0, ch], as_tuple=True)
        scored = sorted(zip(hm_s[ch, ys, xs].tolist(), ys.tolist(), xs.tolist()), reverse=True)
        for score, y, x in scored[:MAX_PER_COLOR]:
            ox = float(off[0, y, x]) - 0.5
            oy = float(off[1, y, x]) - 0.5
            y0, y1 = max(0, y - 1), min(HH, y + 2)
            x0, x1 = max(0, x - 1), min(HW, x + 2)
            prob = cells_map[:, y0:y1, x0:x1].mean(dim=(1, 2))
            cs = frozenset(v + 1 for v in range(5) if float(prob[v]) >= cells_thr)
            cands.append((score, (x + 0.5 + ox) * STRIDE, (y + 0.5 + oy) * STRIDE, color, cs))
    cands.sort(key=lambda t: t[0], reverse=True)
    stones = []
    for (score, x, y, color, cs) in cands:
        if all((x - sx) ** 2 + (y - sy) ** 2 > (MERGE_DIST_HM * STRIDE) ** 2
               for sx, sy, *_ in stones):
            stones.append((x, y, color, cs))
    return stones


def decode_corners(hm_c, off):
    corners = np.zeros((4, 2))
    conf = []
    for i in range(4):
        flat = torch.argmax(hm_c[i])
        y, x = int(flat) // HW, int(flat) % HW
        conf.append(float(hm_c[i, y, x]))
        ox = float(off[0, y, x]) - 0.5
        oy = float(off[1, y, x]) - 0.5
        corners[i] = [(x + 0.5 + ox) * STRIDE, (y + 0.5 + oy) * STRIDE]
    return corners, min(conf)


def crop_region(img, cx, cy, size):
    """(cx,cy) 중심 size x size crop — 경계 밖은 회색(127) 패딩 (train_crop.crop_region)"""
    h, w = img.shape[:2]
    half = size // 2
    x0, y0 = int(round(cx)) - half, int(round(cy)) - half
    x1, y1 = x0 + size, y0 + size
    pl, pt = max(0, -x0), max(0, -y0)
    pr, pb = max(0, x1 - w), max(0, y1 - h)
    c = img[max(0, y0):min(h, y1), max(0, x0):min(w, x1)]
    if pl or pt or pr or pb:
        c = cv2.copyMakeBorder(c, pt, pb, pl, pr, cv2.BORDER_CONSTANT, value=(127, 127, 127))
    return c


def normalize(img):
    x = (img.astype(np.float32) / 255.0 - MEAN) / STD
    return torch.from_numpy(x.transpose(2, 0, 1))


# ---------- 입출력 ----------
def read_image(path):
    """한글 경로 지원. OpenCV 5 의 imdecode 가 EXIF 회전을 자동 적용한다."""
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"이미지를 읽을 수 없습니다: {path}")
    return img


def write_image(path, img_bgr):
    ext = os.path.splitext(path)[1].lower() or ".jpg"
    ok, buf = cv2.imencode(ext, img_bgr)
    if not ok:
        raise ValueError(f"이미지를 저장할 수 없습니다: {path}")
    buf.tofile(path)


def annotate(img_bgr, result):
    """원본 사진 위에 보드 외곽선 + 스톤별 원·점수를 그려 반환 (BGR)"""
    img = img_bgr.copy()
    h, w = img.shape[:2]
    k = w / float(ORIG_W)                      # 글자·선 굵기를 사진 크기에 맞춤
    if result["corner_conf"] >= 0.15:
        quad = (np.array(result["corners"]) * [w, h]).astype(np.int32)
        cv2.polylines(img, [quad.reshape(-1, 1, 2)], True, (0, 255, 255), max(2, int(3 * k)))
    r = int(80 * k)
    for s in result["stones"]:
        cx, cy = int(s["x"] * w), int(s["y"] * h)
        bgr = (0, 0, 255) if s["color"] == "red" else (255, 80, 0)
        cv2.circle(img, (cx, cy), r, bgr, max(2, int(3 * k)))
        label = f"S{s['index']}:{s['score']}" + ("?" if s["uncertain"] else "")
        org = (cx - int(60 * k), cy - r - int(8 * k))
        cv2.putText(img, label, org, cv2.FONT_HERSHEY_SIMPLEX, 1.1 * k,
                    (255, 255, 255), max(3, int(6 * k)))
        cv2.putText(img, label, org, cv2.FONT_HERSHEY_SIMPLEX, 1.1 * k,
                    bgr, max(1, int(2 * k)))
    return img


def cells_text(cells):
    return ", ".join(CELL_NAMES[c] for c in (cells or [0]))
