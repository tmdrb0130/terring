# -*- coding: utf-8 -*-
"""
model_004 (구 model_005): 칸 판정을 사람 라벨로 직접 학습 — 기하 공식 제거

model_003(구 model_004)와의 차이:
  · 새 헤드 cells 5ch: 각 위치에 스톤이 있다면 어느 점수값 칸(1~5)에 닿는지 (multi-label)
    지도 = data/stone_map.json (스톤 좌표 ↔ 사람 검수 cells 매핑, 전량 확정본)
  · 추론: 스톤 피크 위치에서 cells 5비트를 읽어 닿은 칸 집합 산출 (빈 집합 = 보드 밖 0)
    → 호모그래피/접촉원 계산 없음. model_002 기하 코드는 어디에도 안 쓰임
  · 코너 히트맵/회귀는 보조 손실로만 유지 (트렁크가 보드 기하를 배우도록)

나머지(구조·증강·5-fold·앙상블+TTA·데이터 583장 8:2 분할)는 model_003과 동일.
학습: train 473장 (fold 분할·증강 루프), 임계값(피크/셀)은 train 스윕, test 110장 최종 평가.
"""
import csv
import io
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
M4 = os.path.join(ROOT, "model_003")   # 분할 CSV·labels.json 보유 폴더 (변수명 M4 는 구 번호 흔적)

OUT_DIR = os.path.join(MODEL_DIR, "output")
IMG_DIRS = [os.path.join(ROOT, "data", "001~085 (08.16)"),
            os.path.join(ROOT, "data", "086~583 (08.19)")]

CFG = {
    "seed": 42,
    "img_w": 576, "img_h": 768,
    "stride": 4,
    "sigma_stone": 3.0,
    "sigma_corner": 2.0,
    "max_epochs": 90,
    "patience": 15,
    "batch_size": 6,
    "lr_backbone": 1e-4,
    "lr_head": 1e-3,
    "weight_decay": 1e-4,
    "repeat_per_epoch": 2,
    "corner_reg_w": 2.0,
    "offset_w": 1.0,
    "cells_w": 2.0,            # 칸 판정 BCE 가중치
    "peak_thr_train": 0.35,
    "cells_thr": 0.5,          # 학습 중 val 측정용 (최종은 스윕)
    "n_folds": 5,
    "max_per_color": 3,
}
random.seed(CFG["seed"])
np.random.seed(CFG["seed"])
torch.manual_seed(CFG["seed"])
torch.cuda.manual_seed_all(CFG["seed"])

IW, IH = CFG["img_w"], CFG["img_h"]
HW, HH = IW // CFG["stride"], IH // CFG["stride"]
SX, SY = IW / 1050.0, IH / 1400.0
NMS_K = 9
MERGE_DIST_HM = 12.0


def img_path(name):
    for d in IMG_DIRS:
        p = os.path.join(d, name)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(name)


# ---------- 라벨 ----------
def load_stone_map():
    """이미지 → [(x, y, color, cells 집합)] (입력 좌표계). cells '0' → 빈 집합"""
    raw = json.load(open(os.path.join(ROOT, "data", "stone_map.json"), encoding="utf-8"))
    out = {}
    for name, rows in raw.items():
        st = []
        for s in rows:
            cs = frozenset() if s["cells"] == "0" else \
                frozenset(int(c) for c in s["cells"])
            st.append((float(s["x"]) * SX, float(s["y"]) * SY, s["color"], cs))
        out[name] = st
    return out


def load_quads():
    raw = json.load(open(os.path.join(M4, "labels.json"), encoding="utf-8"))
    return {n: np.array(v["quad"], dtype=np.float64) * [SX, SY] for n, v in raw.items()}


def load_human_labels():
    path = os.path.join(ROOT, "data", "terring_1_583.csv")
    out = {}
    with open(path, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
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
            out[r["image"]] = {"stones": stones, "total": int(r["total_score"])}
    return out


def load_split(name):
    rows = []
    with open(os.path.join(M4, name), encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            counts = [0] * 6
            overlap = 0
            for i in range(1, 7):
                color = (r[f"s{i}_color"] or "").strip()
                cells = (r[f"s{i}_cells"] or "").strip()
                if not color:
                    continue
                s = max(int(c) for c in cells)
                counts[s] += 1
                if s > 0:
                    overlap += len(cells) - 1
            rows.append({"image": r["image"], "on_board": sum(counts[1:]), "overlap": overlap})
    return rows


def make_folds(rows, k, seed):
    def stratum(r):
        s = "S1" if r["on_board"] <= 1 else ("S2-3" if r["on_board"] <= 3 else "S4-6")
        o = "O0" if r["overlap"] == 0 else ("O1-2" if r["overlap"] <= 2 else "O3+")
        return s + "/" + o
    groups = {}
    for r in rows:
        groups.setdefault(stratum(r), []).append(r["image"])
    rng = random.Random(seed)
    folds = [[] for _ in range(k)]
    i = 0
    for key in sorted(groups):
        g = groups[key][:]
        rng.shuffle(g)
        for name in g:
            folds[i % k].append(name)
            i += 1
    return folds


# ---------- 데이터셋 ----------
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
_IMG_CACHE = {}


def load_input_image(name):
    if name not in _IMG_CACHE:
        img = cv2.imdecode(np.fromfile(img_path(name), dtype=np.uint8), cv2.IMREAD_COLOR)
        img = cv2.resize(img, (IW, IH), interpolation=cv2.INTER_AREA)
        _IMG_CACHE[name] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return _IMG_CACHE[name]


def render_point(hm, off, mask, ch, px, py, sigma):
    s = CFG["stride"]
    cxh, cyh = px / s, py / s
    rx, ry = int(round(cxh)), int(round(cyh))
    if not (0 <= rx < HW and 0 <= ry < HH):
        return None
    x0, y0 = max(0, rx - 8), max(0, ry - 8)
    x1, y1 = min(HW, rx + 9), min(HH, ry + 9)
    ys, xs = np.mgrid[y0:y1, x0:x1]
    g = np.exp(-((xs - cxh) ** 2 + (ys - cyh) ** 2) / (2 * sigma ** 2))
    hm[ch, y0:y1, x0:x1] = np.maximum(hm[ch, y0:y1, x0:x1], g)
    hm[ch, ry, rx] = 1.0
    off[0, ry, rx] = cxh - rx + 0.5
    off[1, ry, rx] = cyh - ry + 0.5
    mask[ry, rx] = 1.0
    return rx, ry


class TerringDet(Dataset):
    def __init__(self, names, stone_map, quads, train, repeat=1):
        self.names = names
        self.stone_map = stone_map
        self.quads = quads
        self.train = train
        self.repeat = repeat

    def __len__(self):
        return len(self.names) * self.repeat

    def __getitem__(self, idx):
        name = self.names[idx % len(self.names)]
        img = load_input_image(name).copy()
        quad = self.quads[name].copy()
        stones = list(self.stone_map.get(name, []))   # 빈 보드(086~092)는 스톤 0개

        if self.train:
            if random.random() < 0.5:
                # 좌우반전: 보드 값 배치가 좌우 대칭(4-2-4 / 3-5-3 / 2-1-2)이라 cells 불변
                img = img[:, ::-1].copy()
                quad[:, 0] = IW - 1 - quad[:, 0]
                quad = quad[[1, 0, 3, 2]]
                stones = [(IW - 1 - x, y, c, cs) for x, y, c, cs in stones]
            ang = random.uniform(-20, 20)
            sc = random.uniform(0.7, 1.15)
            tx = random.uniform(-0.08, 0.08) * IW
            ty = random.uniform(-0.08, 0.08) * IH
            Mm = cv2.getRotationMatrix2D((IW / 2, IH / 2), ang, sc)
            Mm[0, 2] += tx
            Mm[1, 2] += ty
            img = cv2.warpAffine(img, Mm, (IW, IH), borderValue=(127, 127, 127))
            quad = np.hstack([quad, np.ones((4, 1))]) @ Mm.T
            stones = [(nx, ny, c, cs) for (x, y, c, cs) in stones
                      for nx, ny in [Mm @ np.array([x, y, 1.0])]
                      if -10 <= nx < IW + 10 and -10 <= ny < IH + 10]
            f = np.random.uniform(0.7, 1.3, size=3).astype(np.float32)
            img = np.clip(img.astype(np.float32) * f, 0, 255)
            img = np.clip(img + np.random.uniform(-25, 25), 0, 255).astype(np.uint8)

        x = (img.astype(np.float32) / 255.0 - MEAN) / STD
        x = torch.from_numpy(x.transpose(2, 0, 1))

        hm_s = np.zeros((2, HH, HW), dtype=np.float32)
        hm_c = np.zeros((4, HH, HW), dtype=np.float32)
        off = np.zeros((2, HH, HW), dtype=np.float32)
        mask = np.zeros((HH, HW), dtype=np.float32)       # 오프셋용 (스톤+코너)
        cells_t = np.zeros((5, HH, HW), dtype=np.float32)  # 칸 판정 타깃
        mask_st = np.zeros((HH, HW), dtype=np.float32)     # 칸 판정용 (스톤만)
        for (px, py, c, cs) in stones:
            pos = render_point(hm_s, off, mask, 0 if c == "red" else 1, px, py,
                               CFG["sigma_stone"])
            if pos is not None:
                rx, ry = pos
                mask_st[ry, rx] = 1.0
                for v in cs:
                    cells_t[v - 1, ry, rx] = 1.0
        for i in range(4):
            render_point(hm_c, off, mask, i, quad[i, 0], quad[i, 1], CFG["sigma_corner"])
        corners = torch.tensor((quad / [IW, IH]).reshape(-1), dtype=torch.float32)
        return (x, torch.from_numpy(hm_s), torch.from_numpy(hm_c),
                torch.from_numpy(off), torch.from_numpy(mask),
                torch.from_numpy(cells_t), torch.from_numpy(mask_st), corners, name)


# ---------- 모델 ----------
class DetNet(nn.Module):
    def __init__(self):
        super().__init__()
        r = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
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
        # 칸 판정 헤드: 경계 판단에 넓은 문맥이 필요해 3x3 두 층
        self.cells = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 5, 3, padding=1))
        nn.init.constant_(self.hm_stone.bias, -2.19)
        nn.init.constant_(self.hm_corner.bias, -2.19)
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


def focal_heatmap_loss(pred, target, alpha=2, beta=4):
    pred = pred.clamp(1e-4, 1 - 1e-4)
    pos = (target > 0.99).float()
    neg = 1 - pos
    pos_loss = -((1 - pred) ** alpha) * torch.log(pred) * pos
    neg_loss = -((1 - target) ** beta) * (pred ** alpha) * torch.log(1 - pred) * neg
    return (pos_loss.sum() + neg_loss.sum()) / pos.sum().clamp(min=1)


# ---------- 디코딩 ----------
def decode_outputs(hm_s, hm_c, off, cells_map, thr, cells_thr):
    """→ [(x, y, color, cells 집합)] — 기하 계산 없이 헤드 출력만 사용"""
    hm_t = hm_s[None]
    pool = F.max_pool2d(hm_t, NMS_K, stride=1, padding=NMS_K // 2)
    peaks = (hm_t == pool) & (hm_t > thr)
    cands = []
    for ch, color in [(0, "red"), (1, "blue")]:
        ys, xs = torch.nonzero(peaks[0, ch], as_tuple=True)
        scored = sorted(zip(hm_s[ch, ys, xs].tolist(), ys.tolist(), xs.tolist()), reverse=True)
        for score, y, x in scored[:CFG["max_per_color"]]:
            ox = float(off[0, y, x]) - 0.5
            oy = float(off[1, y, x]) - 0.5
            # 칸 판정: 피크 픽셀 3x3 평균 확률
            y0, y1 = max(0, y - 1), min(HH, y + 2)
            x0, x1 = max(0, x - 1), min(HW, x + 2)
            prob = cells_map[:, y0:y1, x0:x1].mean(dim=(1, 2))
            cs = frozenset(v + 1 for v in range(5) if float(prob[v]) >= cells_thr)
            cands.append((score, (x + 0.5 + ox) * CFG["stride"],
                          (y + 0.5 + oy) * CFG["stride"], color, cs))
    cands.sort(key=lambda t: t[0], reverse=True)
    stones = []
    for (score, x, y, color, cs) in cands:
        if all((x - sx) ** 2 + (y - sy) ** 2 > (MERGE_DIST_HM * CFG["stride"]) ** 2
               for sx, sy, *_ in stones):
            stones.append((x, y, color, cs))
    return stones


def stones_to_preds(stones):
    """(색, cells 집합) 목록 — 빈 집합은 보드 밖 {0} 으로 표기 (라벨 형식과 동일)"""
    return [(c, cs if cs else frozenset({0})) for _x, _y, c, cs in stones]


def compare(preds, label_stones):
    total_p = sum(max(c) for _, c in preds)
    total_l = sum(max(c) for _, c in label_stones)
    rest = list(preds)
    matched = 0
    for ls in label_stones:
        if ls in rest:
            rest.remove(ls)
            matched += 1
    full = (len(preds) == len(label_stones)) and (matched == len(label_stones))
    return full, total_p == total_l, matched, len(label_stones), total_p, total_l


def _warp_back(t, Minv):
    """TTA 뷰의 출력맵을 원좌표계로 역워핑 (채널별 warpAffine, 경계 0)"""
    a = t.numpy()
    out = np.stack([cv2.warpAffine(a[c], Minv, (HW, HH), flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
                    for c in range(a.shape[0])])
    return torch.from_numpy(out)


# 확장 TTA 뷰: (회전각, 스케일) — 회전/스케일은 되돌릴 수 있어 cells 라벨 불변
TTA_FULL_VIEWS = [(10, 1.0), (-10, 1.0), (0, 0.92), (0, 1.08)]


@torch.no_grad()
def predict_maps(models_list, name, device, use_tta=True, tta="flip"):
    """tta='flip': 좌우반전만(기존). 'full': +회전 ±10도, 스케일 0.92/1.08 뷰 평균."""
    img = load_input_image(name)
    x = (img.astype(np.float32) / 255.0 - MEAN) / STD
    x = torch.from_numpy(x.transpose(2, 0, 1))[None].to(device)
    xf = torch.flip(x, dims=[3])
    views = []   # (입력 텐서, 히트맵 좌표계 역변환 행렬)
    if use_tta and tta == "full":
        for ang, sc in TTA_FULL_VIEWS:
            Mv = cv2.getRotationMatrix2D((IW / 2, IH / 2), ang, sc)
            iv = cv2.warpAffine(img, Mv, (IW, IH), borderValue=(127, 127, 127))
            xv = (iv.astype(np.float32) / 255.0 - MEAN) / STD
            views.append((torch.from_numpy(xv.transpose(2, 0, 1))[None].to(device),
                          cv2.getRotationMatrix2D((HW / 2, HH / 2), -ang, 1.0 / sc)))
    hs_l, hc_l, off_l, cl_l = [], [], [], []
    for m in models_list:
        hs, hc, off, cl, _reg = m(x)
        hs_l.append(hs[0].cpu())
        hc_l.append(hc[0].cpu())
        off_l.append(off[0].cpu())
        cl_l.append(cl[0].cpu())
        if use_tta:  # 좌우반전 TTA — 보드 값 배치가 대칭이라 cells 채널 그대로 되뒤집기만
            hs2, hc2, _, cl2, _ = m(xf)
            hs_l.append(torch.flip(hs2[0], dims=[2]).cpu())
            hc_l.append(torch.flip(hc2[0], dims=[2])[[1, 0, 3, 2]].cpu())
            cl_l.append(torch.flip(cl2[0], dims=[2]).cpu())
        for xv, Minv in views:  # 회전/스케일 뷰 — 출력맵을 역워핑해 평균 (오프셋은 원뷰만)
            hsv, hcv, _, clv, _ = m(xv)
            hs_l.append(_warp_back(hsv[0].cpu(), Minv))
            hc_l.append(_warp_back(hcv[0].cpu(), Minv))
            cl_l.append(_warp_back(clv[0].cpu(), Minv))
    hm_s = torch.stack(hs_l).mean(0)
    hm_c = torch.stack(hc_l).mean(0)
    off = torch.stack(off_l).mean(0)
    cells_map = torch.stack(cl_l).mean(0)
    return hm_s, hm_c, off, cells_map


def eval_maps(maps, names, human, thr, cells_thr, detail=False):
    full = tot = ms = ls = 0
    abs_err = []
    rows = []
    for name in names:
        hm_s, hm_c, off, cells_map = maps[name]
        stones = decode_outputs(hm_s, hm_c, off, cells_map, thr, cells_thr)
        preds = stones_to_preds(stones)
        f_, t_, m_, l_, tp, tl = compare(preds, human[name]["stones"])
        full += f_
        tot += t_
        ms += m_
        ls += l_
        abs_err.append(abs(tp - tl))
        if detail:
            rows.append((name, tl, tp,
                         sorted("".join(map(str, sorted(c))) + col[0] for col, c in preds)))
    return {"full": full, "tot": tot, "ms": ms, "ls": ls, "n": len(names),
            "mae": sum(abs_err) / max(1, len(abs_err)), "rows": rows}


@torch.no_grad()
def compute_maps(models_list, names, device, use_tta=True, tta="flip"):
    for m in models_list:
        m.eval()
    return {name: predict_maps(models_list, name, device, use_tta, tta) for name in names}


def evaluate_names(models_list, names, human, device, thr, cells_thr,
                   use_tta=True, detail=False, tta="flip"):
    maps = compute_maps(models_list, names, device, use_tta, tta)
    return eval_maps(maps, names, human, thr, cells_thr, detail)


def train_fold(fold_i, train_names, val_names, stone_map, quads, human, device, log_writer):
    ds = TerringDet(train_names, stone_map, quads, train=True, repeat=CFG["repeat_per_epoch"])
    dl = DataLoader(ds, batch_size=CFG["batch_size"], shuffle=True,
                    num_workers=2, persistent_workers=True)
    model = DetNet().to(device)
    bb_names = ("stem", "l1", "l2", "l3", "l4")
    bb = [p for n, p in model.named_parameters() if n.startswith(bb_names)]
    hd = [p for n, p in model.named_parameters() if not n.startswith(bb_names)]
    opt = torch.optim.AdamW([{"params": bb, "lr": CFG["lr_backbone"]},
                             {"params": hd, "lr": CFG["lr_head"]}],
                            weight_decay=CFG["weight_decay"])
    ckpt = os.path.join(OUT_DIR, f"fold{fold_i}_best.pt")
    best_key = (-1, -1)
    best_epoch = 0
    epoch = 0
    while True:
        epoch += 1
        model.train()
        tr_loss = 0.0
        for x, hm_s, hm_c, off_t, mask, cells_t, mask_st, corners_t, _ in dl:
            x = x.to(device)
            hm_s, hm_c = hm_s.to(device), hm_c.to(device)
            off_t, mask = off_t.to(device), mask.to(device)
            cells_t, mask_st = cells_t.to(device), mask_st.to(device)
            corners_t = corners_t.to(device)
            ps, pc, poff, pcell, preg = model(x)
            cells_bce = F.binary_cross_entropy(
                pcell.clamp(1e-4, 1 - 1e-4), cells_t, reduction="none")
            loss = (focal_heatmap_loss(ps, hm_s) + focal_heatmap_loss(pc, hm_c)
                    + CFG["offset_w"] * (torch.abs(poff - off_t) * mask[:, None]).sum()
                      / mask.sum().clamp(min=1) / 2
                    + CFG["cells_w"] * (cells_bce * mask_st[:, None]).sum()
                      / mask_st.sum().clamp(min=1) / 5
                    + CFG["corner_reg_w"] * F.smooth_l1_loss(preg, corners_t) * 100)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tr_loss += loss.item() * x.size(0)
        tr_loss /= len(ds)

        ev = evaluate_names([model], val_names, human, device,
                            CFG["peak_thr_train"], CFG["cells_thr"], use_tta=False)
        key = (ev["ms"], ev["tot"])
        is_best = key > best_key
        if is_best:
            best_key, best_epoch = key, epoch
            torch.save({"epoch": epoch, "config": CFG, "state_dict": model.state_dict()}, ckpt)
        log_writer.writerow([fold_i, epoch, f"{tr_loss:.4f}", f"{ev['ms']}/{ev['ls']}",
                             f"{ev['tot']}/{ev['n']}", ev["full"], int(is_best)])
        if epoch % 10 == 0 or is_best:
            print(f"[fold {fold_i}] ep {epoch:>3} | loss {tr_loss:.3f} | "
                  f"val 스톤 {ev['ms']}/{ev['ls']} 총점 {ev['tot']}/{ev['n']} "
                  f"완전 {ev['full']}/{ev['n']}" + ("  * best" if is_best else ""), flush=True)
        if epoch - best_epoch >= CFG["patience"] or epoch >= CFG["max_epochs"]:
            print(f"[fold {fold_i}] 종료 (best ep {best_epoch}: 스톤 {best_key[0]})", flush=True)
            break
    return ckpt


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device, flush=True)
    with open(os.path.join(OUT_DIR, "config.json"), "w", encoding="utf-8") as f:
        json.dump(CFG, f, indent=2, ensure_ascii=False)

    stone_map = load_stone_map()
    quads = load_quads()
    human = load_human_labels()
    train_all = load_split("train.csv")
    test_names = [r["image"] for r in load_split("test.csv")]

    folds = make_folds(train_all, CFG["n_folds"], CFG["seed"])
    all_train_names = [n for f_ in folds for n in f_]
    print(f"train {len(all_train_names)} → {CFG['n_folds']}-fold / test {len(test_names)} "
          f"(사람 cells 라벨로 칸 판정 직접 학습)", flush=True)

    log_f = open(os.path.join(OUT_DIR, "train_log.csv"), "w", newline="", encoding="utf-8-sig")
    log_writer = csv.writer(log_f)
    log_writer.writerow(["fold", "epoch", "loss", "val_stone", "val_total", "val_full", "best"])

    ckpts = []
    for i in range(CFG["n_folds"]):
        val_names = folds[i]
        train_names = [n for j, f_ in enumerate(folds) if j != i for n in f_]
        print(f"\n===== fold {i}: train {len(train_names)} / val {len(val_names)} =====", flush=True)
        ckpts.append(train_fold(i, train_names, val_names, stone_map, quads,
                                human, device, log_writer))
        log_f.flush()
    log_f.close()

    # ---------- 앙상블 + 임계값 스윕 (train 기준, test 미접촉) ----------
    models_list = []
    for p in ckpts:
        ck = torch.load(p, weights_only=False)
        m = DetNet().to(device)
        m.load_state_dict(ck["state_dict"])
        m.eval()
        models_list.append(m)

    print("\n임계값 스윕 (train 473장, 앙상블+TTA):", flush=True)
    maps = compute_maps(models_list, all_train_names, device)
    best = None
    # 0.05 까지 내려서 스윕 — 잘린 스톤은 피크가 약해 0.2 로는 놓친다 (08-24 발견)
    for thr in [0.05, 0.08, 0.10, 0.15, 0.20, 0.30]:
        for cthr in [0.35, 0.45, 0.50, 0.55, 0.65]:
            ev = eval_maps(maps, all_train_names, human, thr, cthr)
            key = (ev["ms"], ev["tot"], ev["full"])
            print(f"  thr {thr:.2f}/cells {cthr:.2f}: 스톤 {ev['ms']}/{ev['ls']} "
                  f"총점 {ev['tot']}/{ev['n']} 완전 {ev['full']}/{ev['n']}", flush=True)
            if best is None or key > best[0]:
                best = (key, thr, cthr)
    _, thr, cthr = best
    print(f"선택: peak {thr} / cells {cthr}", flush=True)
    del maps

    ev = evaluate_names(models_list, test_names, human, device, thr, cthr, detail=True)
    n, ls = ev["n"], ev["ls"]
    print(f"\n=== 최종 평가: 5-fold 앙상블 + TTA (thr={thr}, cells={cthr}) / test {n}장 ===")
    print(f"[스톤 단위] {ev['ms']}/{ls} ({ev['ms']/ls*100:.0f}%)")
    print(f"[총점 일치] {ev['tot']}/{n} ({ev['tot']/n*100:.0f}%) | MAE {ev['mae']:.2f}")
    print(f"[완전 일치] {ev['full']}/{n} ({ev['full']/n*100:.0f}%)")


    print(f"\n{'image':<10}{'정답':>5}{'예측':>5}  스톤별 판정 (오답만)")
    for name, tl, tp, cells in ev["rows"]:
        if tp != tl:
            mark = "  <<<" if abs(tp - tl) >= 3 else "  <"
            print(f"{name:<10}{tl:>5}{tp:>5}  {cells}{mark}")


if __name__ == "__main__":
    main()
