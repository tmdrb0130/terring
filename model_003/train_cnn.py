# -*- coding: utf-8 -*-
"""
model_003 (구 model_004): 구 model_003 v2 구조 그대로 + 데이터 583장 (기존 85 + 신규 498)

구 model_003(85장 학습판, 폴더 제거됨) 대비 변경점:
  · 데이터: data/terring_1_583.csv (사람 검수 라벨), 이미지 폴더 2개
  · 분할: 이 폴더의 split_data.py 의 난이도 층화 8:2 (train 473 / test 110)
  · 좌표 지도(supervision): labels.json (CV 자동 좌표) — 단, CV 검출의 색별 스톤 수가
    사람 라벨과 다른 이미지는 좌표를 신뢰할 수 없으므로 학습 지도에서 제외
    (fold 검증/테스트 평가는 사람 라벨 기준이므로 그대로 포함)
  · 데이터 8배 증가에 맞춰 repeat_per_epoch 6→2, max_epochs 120→90, patience 20→15

구조 (구 model_003 v2와 동일):
  입력 576x768 → ResNet18 + FPN (stride 4)
    ├ 스톤 히트맵 2ch (red/blue)   ├ 코너 히트맵 4ch
    ├ 오프셋 2ch (서브픽셀)        └ 코너 회귀 8값 (폴백)
  디코딩: NMS → 색당 최대 3개 → 중복 병합 → 호모그래피 → model_002 기하로 칸 판정
학습: train 473장 난이도 층화 5-fold, 증강(반전/회전/스케일/이동/색) 루프
최종: 5모델 앙상블 + 반전 TTA, 임계값은 train에서 스윕, test 110장 평가
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
sys.path.insert(0, os.path.join(ROOT, "model_002"))
import terring_cv as tc  # noqa: E402

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
    "peak_thr_train": 0.35,
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
def load_coord_labels():
    with open(os.path.join(MODEL_DIR, "labels.json"), encoding="utf-8") as f:
        raw = json.load(f)
    out = {}
    for name, v in raw.items():
        out[name] = {
            "quad": np.array(v["quad"], dtype=np.float64) * [SX, SY],
            "stones": [(x * SX, y * SY, c) for x, y, c in v["stones"]],
        }
    return out


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


def coord_trusted(name, coord, human):
    """CV 좌표 라벨을 학습 지도로 써도 되는가: 색별 스톤 수가 사람 라벨과 일치"""
    if name not in coord:
        return False
    cv_cnt = {"red": 0, "blue": 0}
    for _x, _y, c in coord[name]["stones"]:
        cv_cnt[c] += 1
    hu_cnt = {"red": 0, "blue": 0}
    for c, _cells in human[name]["stones"]:
        hu_cnt[c] += 1
    return cv_cnt == hu_cnt


def load_split(name):
    rows = []
    with open(os.path.join(MODEL_DIR, name), encoding="utf-8-sig") as f:
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
        return
    x0, y0 = max(0, rx - 8), max(0, ry - 8)
    x1, y1 = min(HW, rx + 9), min(HH, ry + 9)
    ys, xs = np.mgrid[y0:y1, x0:x1]
    g = np.exp(-((xs - cxh) ** 2 + (ys - cyh) ** 2) / (2 * sigma ** 2))
    hm[ch, y0:y1, x0:x1] = np.maximum(hm[ch, y0:y1, x0:x1], g)
    hm[ch, ry, rx] = 1.0
    off[0, ry, rx] = cxh - rx + 0.5
    off[1, ry, rx] = cyh - ry + 0.5
    mask[ry, rx] = 1.0


class TerringDet(Dataset):
    def __init__(self, names, coord, train, repeat=1):
        self.names = names
        self.coord = coord
        self.train = train
        self.repeat = repeat

    def __len__(self):
        return len(self.names) * self.repeat

    def __getitem__(self, idx):
        name = self.names[idx % len(self.names)]
        img = load_input_image(name).copy()
        lab = self.coord[name]
        quad = lab["quad"].copy()
        stones = list(lab["stones"])

        if self.train:
            if random.random() < 0.5:
                img = img[:, ::-1].copy()
                quad[:, 0] = IW - 1 - quad[:, 0]
                quad = quad[[1, 0, 3, 2]]
                stones = [(IW - 1 - x, y, c) for x, y, c in stones]
            ang = random.uniform(-20, 20)
            sc = random.uniform(0.7, 1.15)
            tx = random.uniform(-0.08, 0.08) * IW
            ty = random.uniform(-0.08, 0.08) * IH
            Mm = cv2.getRotationMatrix2D((IW / 2, IH / 2), ang, sc)
            Mm[0, 2] += tx
            Mm[1, 2] += ty
            img = cv2.warpAffine(img, Mm, (IW, IH), borderValue=(127, 127, 127))
            quad = np.hstack([quad, np.ones((4, 1))]) @ Mm.T
            stones = [(nx, ny, c) for (x, y, c) in stones
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
        mask = np.zeros((HH, HW), dtype=np.float32)
        for (px, py, c) in stones:
            render_point(hm_s, off, mask, 0 if c == "red" else 1, px, py, CFG["sigma_stone"])
        for i in range(4):
            render_point(hm_c, off, mask, i, quad[i, 0], quad[i, 1], CFG["sigma_corner"])
        corners = torch.tensor((quad / [IW, IH]).reshape(-1), dtype=torch.float32)
        return (x, torch.from_numpy(hm_s), torch.from_numpy(hm_c),
                torch.from_numpy(off), torch.from_numpy(mask), corners, name)


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
                torch.sigmoid(self.offset(t)), self.corner_reg(c4))


def focal_heatmap_loss(pred, target, alpha=2, beta=4):
    pred = pred.clamp(1e-4, 1 - 1e-4)
    pos = (target > 0.99).float()
    neg = 1 - pos
    pos_loss = -((1 - pred) ** alpha) * torch.log(pred) * pos
    neg_loss = -((1 - target) ** beta) * (pred ** alpha) * torch.log(1 - pred) * neg
    return (pos_loss.sum() + neg_loss.sum()) / pos.sum().clamp(min=1)


# ---------- 디코딩 ----------
def decode_outputs(hm_s, hm_c, off, corner_reg, thr):
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
            cands.append((score, (x + 0.5 + ox) * CFG["stride"], (y + 0.5 + oy) * CFG["stride"], color))
    cands.sort(reverse=True)
    stones = []
    for (score, x, y, color) in cands:
        if all((x - sx) ** 2 + (y - sy) ** 2 > (MERGE_DIST_HM * CFG["stride"]) ** 2
               for sx, sy, _ in stones):
            stones.append((x, y, color))
    corners = np.zeros((4, 2))
    reg = corner_reg.reshape(4, 2)
    for i in range(4):
        flat = torch.argmax(hm_c[i])
        y, x = int(flat) // HW, int(flat) % HW
        if float(hm_c[i, y, x]) >= 0.15:
            ox = float(off[0, y, x]) - 0.5
            oy = float(off[1, y, x]) - 0.5
            corners[i] = [(x + 0.5 + ox) * CFG["stride"], (y + 0.5 + oy) * CFG["stride"]]
        else:
            corners[i] = [float(reg[i, 0]) * IW, float(reg[i, 1]) * IH]
    return stones, corners


def to_cells(stones, corners):
    quad = corners / [SX, SY]
    Hm = tc.get_homography(quad.astype(np.float64))
    stone_r = tc.GEO["stone_r_ratio"] * tc.BW
    preds = []
    for (x, y, color) in stones:
        p = Hm @ np.array([x / SX, y / SY, 1.0])
        preds.append((color, frozenset(tc.touched_cells(p[0] / p[2], p[1] / p[2], stone_r))))
    return preds


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


@torch.no_grad()
def predict_maps(models_list, name, device, use_tta=True):
    img = load_input_image(name)
    x = (img.astype(np.float32) / 255.0 - MEAN) / STD
    x = torch.from_numpy(x.transpose(2, 0, 1))[None].to(device)
    xf = torch.flip(x, dims=[3])
    hs_l, hc_l, off_l, reg_l = [], [], [], []
    for m in models_list:
        hs, hc, off, reg = m(x)
        hs_l.append(hs[0].cpu())
        hc_l.append(hc[0].cpu())
        off_l.append(off[0].cpu())
        reg_l.append(reg[0].cpu())
        if use_tta:
            hs2, hc2, _, _ = m(xf)
            hs_l.append(torch.flip(hs2[0], dims=[2]).cpu())
            hc_l.append(torch.flip(hc2[0], dims=[2])[[1, 0, 3, 2]].cpu())
    hm_s = torch.stack(hs_l).mean(0)
    hm_c = torch.stack(hc_l).mean(0)
    off = torch.stack(off_l).mean(0)
    reg = torch.stack(reg_l).mean(0).numpy()
    return hm_s, hm_c, off, reg


def eval_maps(maps, names, human, thr, detail=False):
    """미리 계산한 맵으로 평가 — 임계값 스윕 시 forward 재실행 방지"""
    full = tot = ms = ls = 0
    abs_err = []
    rows = []
    for name in names:
        hm_s, hm_c, off, reg = maps[name]
        stones, corners = decode_outputs(hm_s, hm_c, off, reg, thr)
        preds = to_cells(stones, corners)
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
def compute_maps(models_list, names, device, use_tta=True):
    for m in models_list:
        m.eval()
    return {name: predict_maps(models_list, name, device, use_tta) for name in names}


def evaluate_names(models_list, names, human, device, thr, use_tta=True, detail=False):
    maps = compute_maps(models_list, names, device, use_tta)
    return eval_maps(maps, names, human, thr, detail)


def train_fold(fold_i, train_names, val_names, coord, human, device, log_writer):
    ds = TerringDet(train_names, coord, train=True, repeat=CFG["repeat_per_epoch"])
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
        for x, hm_s, hm_c, off_t, mask, corners_t, _ in dl:
            x = x.to(device)
            hm_s, hm_c = hm_s.to(device), hm_c.to(device)
            off_t, mask, corners_t = off_t.to(device), mask.to(device), corners_t.to(device)
            ps, pc, poff, preg = model(x)
            loss = (focal_heatmap_loss(ps, hm_s) + focal_heatmap_loss(pc, hm_c)
                    + CFG["offset_w"] * (torch.abs(poff - off_t) * mask[:, None]).sum()
                      / mask.sum().clamp(min=1) / 2
                    + CFG["corner_reg_w"] * F.smooth_l1_loss(preg, corners_t) * 100)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tr_loss += loss.item() * x.size(0)
        tr_loss /= len(ds)

        ev = evaluate_names([model], val_names, human, device,
                            CFG["peak_thr_train"], use_tta=False)
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

    coord = load_coord_labels()
    human = load_human_labels()
    train_all = load_split("train.csv")
    test_names = [r["image"] for r in load_split("test.csv")]

    # 좌표 지도 신뢰 필터: CV 검출(색별 개수)이 사람 라벨과 다르면 학습 지도에서 제외
    trusted = {r["image"] for r in train_all if coord_trusted(r["image"], coord, human)}
    dropped = [r["image"] for r in train_all if r["image"] not in trusted]
    print(f"train {len(train_all)}장 중 좌표 신뢰 {len(trusted)} / 지도 제외 {len(dropped)}", flush=True)
    with open(os.path.join(OUT_DIR, "untrusted_coords.json"), "w", encoding="utf-8") as f:
        json.dump(sorted(dropped), f, indent=1)

    folds = make_folds(train_all, CFG["n_folds"], CFG["seed"])
    all_train_names = [n for f_ in folds for n in f_]
    print(f"train {len(all_train_names)} → {CFG['n_folds']}-fold / test {len(test_names)} (최종 전용)",
          flush=True)

    log_f = open(os.path.join(OUT_DIR, "train_log.csv"), "w", newline="", encoding="utf-8-sig")
    log_writer = csv.writer(log_f)
    log_writer.writerow(["fold", "epoch", "loss", "val_stone", "val_total", "val_full", "best"])

    ckpts = []
    for i in range(CFG["n_folds"]):
        val_names = folds[i]
        train_names = [n for j, f_ in enumerate(folds) if j != i for n in f_ if n in trusted]
        print(f"\n===== fold {i}: train {len(train_names)} / val {len(val_names)} =====", flush=True)
        ckpts.append(train_fold(i, train_names, val_names, coord, human, device, log_writer))
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
    for thr in [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]:
        ev = eval_maps(maps, all_train_names, human, thr)
        key = (ev["ms"], ev["tot"], ev["full"])
        print(f"  thr {thr:.2f}: 스톤 {ev['ms']}/{ev['ls']} 총점 {ev['tot']}/{ev['n']} "
              f"완전 {ev['full']}/{ev['n']}", flush=True)
        if best is None or key > best[0]:
            best = (key, thr)
    thr = best[1]
    print(f"선택된 임계값: {thr}", flush=True)
    del maps

    ev = evaluate_names(models_list, test_names, human, device, thr, detail=True)
    n, ls = ev["n"], ev["ls"]
    print(f"\n=== 최종 평가: 5-fold 앙상블 + TTA (thr={thr}) / test {n}장 ===")
    print(f"[스톤 단위] {ev['ms']}/{ls} ({ev['ms']/ls*100:.0f}%)")
    print(f"[총점 일치] {ev['tot']}/{n} ({ev['tot']/n*100:.0f}%) | MAE {ev['mae']:.2f}")
    print(f"[완전 일치] {ev['full']}/{n} ({ev['full']/n*100:.0f}%)")
    print(f"\n{'image':<10}{'정답':>5}{'예측':>5}  스톤별 판정")
    for name, tl, tp, cells in ev["rows"]:
        if tp != tl:
            mark = "  <<<" if abs(tp - tl) >= 3 else "  <"
            print(f"{name:<10}{tl:>5}{tp:>5}  {cells}{mark}")


if __name__ == "__main__":
    main()
