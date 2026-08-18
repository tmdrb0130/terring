# -*- coding: utf-8 -*-
"""
model_001 v4: 전역 CNN 스톤 구성(멀티셋) 직접 분류 — 완전일치 산출 가능 구조

v3(총점 회귀)은 숫자 하나만 출력해서 완전일치를 잴 수 없었다. v4는 모델의 정체성
(위치 탐지 없이 이미지 통째로 판정하는 전역 CNN)은 유지하되, 출력을 바꾼다:

  각 (색, cells 조합) 클래스마다 그 스톤이 몇 개 있는지(0~3) 분류
  → 예측 = {(색, cells): 개수} 멀티셋 → 라벨 멀티셋과 통째 비교 = 완전일치

cells 조합 사전은 전체 라벨 CSV 에 등장한 조합으로 구성 (게임 기하상 가능한 조합).
분할은 model_003(구 004)의 train 473 / test 110 을 그대로 사용 — 전 모델 통일 조건.
증강: 멀티셋 라벨은 회전·스케일·이동에 불변, 좌우반전도 보드 값 배치가 대칭이라 불변.
평가: 5-fold 앙상블(확률 평균) + 좌우반전 TTA, 완전일치/총점/MAE.
"""
import csv
import os
import random
from collections import Counter

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(MODEL_DIR)
IMG_DIRS = [os.path.join(ROOT, "data", "001~085 (08.16)"),
            os.path.join(ROOT, "data", "086~583 (08.19)")]
OUT_DIR = os.path.join(MODEL_DIR, "output")

# 분할 소스: 이름 정리 전후 모두 대응 (model_003 = 구 model_004)
for _d in ("model_003", "model_004"):
    SPLIT_DIR = os.path.join(ROOT, _d)
    if os.path.exists(os.path.join(SPLIT_DIR, "train.csv")):
        break

CFG = {
    "seed": 42,
    "img_w": 384, "img_h": 512,     # 3:4 원본 비율 유지
    "backbone": "resnet34",
    "max_count": 3,                  # 색당 스톤 최대 3개
    "max_epochs": 90,
    "patience": 15,
    "batch_size": 12,
    "lr_backbone": 1e-4,
    "lr_head": 1e-3,
    "weight_decay": 1e-4,
    "repeat_per_epoch": 2,
    "n_folds": 5,
}
random.seed(CFG["seed"])
np.random.seed(CFG["seed"])
torch.manual_seed(CFG["seed"])
torch.cuda.manual_seed_all(CFG["seed"])

IW, IH = CFG["img_w"], CFG["img_h"]
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
COLORS = ["red", "blue"]


def img_path(name):
    for d in IMG_DIRS:
        p = os.path.join(d, name)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(name)


# ---------- 라벨: 멀티셋 ----------
def load_labels():
    """image -> Counter{(color, cells문자열): 개수}, cells 는 정렬된 문자열('0'=보드밖)"""
    path = os.path.join(ROOT, "data", "terring_1_583.csv")
    out = {}
    for r in csv.DictReader(open(path, encoding="utf-8-sig")):
        c = Counter()
        for i in range(1, 7):
            color = (r[f"s{i}_color"] or "").strip()
            cells = (r[f"s{i}_cells"] or "").strip()
            if not color:
                continue
            cells = "".join(sorted(set(cells)))
            if cells != "0":
                cells = cells.replace("0", "")
            c[(color, cells)] += 1
        out[r["image"]] = c
    return out


def build_vocab(labels):
    vocab = sorted({cells for c in labels.values() for (_col, cells) in c})
    return vocab, {v: i for i, v in enumerate(vocab)}


def load_split(name):
    rows = []
    with open(os.path.join(SPLIT_DIR, name), encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            counts = [0] * 6
            overlap = 0
            for i in range(1, 7):
                color = (r[f"s{i}_color"] or "").strip()
                cells = (r[f"s{i}_cells"] or "").strip()
                if not color:
                    continue
                s = max(int(ch) for ch in cells)
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
_IMG_CACHE = {}


def load_input_image(name):
    if name not in _IMG_CACHE:
        img = cv2.imdecode(np.fromfile(img_path(name), dtype=np.uint8), cv2.IMREAD_COLOR)
        img = cv2.resize(img, (IW, IH), interpolation=cv2.INTER_AREA)
        _IMG_CACHE[name] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return _IMG_CACHE[name]


def target_of(counter, v2i):
    """(2, C) int64 — 색·cells 클래스별 개수 (max_count 로 클립)"""
    t = np.zeros((2, len(v2i)), dtype=np.int64)
    for (color, cells), n in counter.items():
        t[COLORS.index(color), v2i[cells]] = min(n, CFG["max_count"])
    return t


class TerringSet(Dataset):
    def __init__(self, names, labels, v2i, train, repeat=1):
        self.names = names
        self.labels = labels
        self.v2i = v2i
        self.train = train
        self.repeat = repeat

    def __len__(self):
        return len(self.names) * self.repeat

    def __getitem__(self, idx):
        name = self.names[idx % len(self.names)]
        img = load_input_image(name).copy()
        if self.train:
            if random.random() < 0.5:
                img = img[:, ::-1].copy()   # 좌우반전: 보드 값 배치 대칭이라 멀티셋 불변
            ang = random.uniform(-20, 20)
            sc = random.uniform(0.75, 1.15)
            tx = random.uniform(-0.08, 0.08) * IW
            ty = random.uniform(-0.08, 0.08) * IH
            Mm = cv2.getRotationMatrix2D((IW / 2, IH / 2), ang, sc)
            Mm[0, 2] += tx
            Mm[1, 2] += ty
            img = cv2.warpAffine(img, Mm, (IW, IH), borderValue=(127, 127, 127))
            f = np.random.uniform(0.7, 1.3, size=3).astype(np.float32)
            img = np.clip(img.astype(np.float32) * f, 0, 255)
            img = np.clip(img + np.random.uniform(-25, 25), 0, 255).astype(np.uint8)
        x = (img.astype(np.float32) / 255.0 - MEAN) / STD
        x = torch.from_numpy(x.transpose(2, 0, 1))
        t = torch.from_numpy(target_of(self.labels[name], self.v2i))
        return x, t, name


# ---------- 모델 ----------
class SetNet(nn.Module):
    """ResNet34 → (2색 x C클래스 x 4개수) 로짓"""
    def __init__(self, n_classes):
        super().__init__()
        r = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(r.children())[:-1])
        self.n_classes = n_classes
        self.head = nn.Sequential(
            nn.Flatten(), nn.Dropout(0.2),
            nn.Linear(512, 512), nn.ReLU(),
            nn.Linear(512, 2 * n_classes * (CFG["max_count"] + 1)))

    def forward(self, x):
        out = self.head(self.backbone(x))
        return out.view(-1, 2, self.n_classes, CFG["max_count"] + 1)


@torch.no_grad()
def predict_probs(models_list, names, device, use_tta=True, batch=16):
    """앙상블+반전TTA 확률 → {name: (2, C, 4)}"""
    for m in models_list:
        m.eval()
    out = {}
    for s in range(0, len(names), batch):
        chunk = names[s:s + batch]
        xs = []
        for n in chunk:
            img = load_input_image(n)
            x = (img.astype(np.float32) / 255.0 - MEAN) / STD
            xs.append(torch.from_numpy(x.transpose(2, 0, 1)))
        xb = torch.stack(xs).to(device)
        xf = torch.flip(xb, dims=[3])
        acc = None
        k = 0
        for m in models_list:
            for xin in ([xb, xf] if use_tta else [xb]):
                p = torch.softmax(m(xin), dim=-1)
                acc = p if acc is None else acc + p
                k += 1
        acc = (acc / k).cpu().numpy()
        for i, n in enumerate(chunk):
            out[n] = acc[i]
    return out


def decode(prob, vocab):
    """(2,C,4) 확률 → Counter{(color, cells): n}"""
    c = Counter()
    for ci, color in enumerate(COLORS):
        for vi, cells in enumerate(vocab):
            n = int(prob[ci, vi].argmax())
            if n > 0:
                c[(color, cells)] = n
    return c


def score_of(counter):
    return sum(n * (0 if cells == "0" else max(int(ch) for ch in cells))
               for (_c, cells), n in counter.items())


def evaluate(probs, names, labels, vocab, detail=False):
    full = tot = 0
    abs_err = []
    rows = []
    for n in names:
        pred = decode(probs[n], vocab)
        lab = labels[n]
        f_ = pred == lab
        tp, tl = score_of(pred), score_of(lab)
        full += f_
        tot += tp == tl
        abs_err.append(abs(tp - tl))
        if detail and not f_:
            rows.append((n, tl, tp, sorted(lab.items()), sorted(pred.items())))
    return {"full": full, "tot": tot, "n": len(names),
            "mae": sum(abs_err) / max(1, len(abs_err)), "rows": rows}


def train_fold(fold_i, tr_names, va_names, labels, vocab, v2i, device, log_writer):
    ds = TerringSet(tr_names, labels, v2i, train=True, repeat=CFG["repeat_per_epoch"])
    dl = DataLoader(ds, batch_size=CFG["batch_size"], shuffle=True, num_workers=0)
    model = SetNet(len(vocab)).to(device)
    bb = list(model.backbone.parameters())
    hd = list(model.head.parameters())
    opt = torch.optim.AdamW([{"params": bb, "lr": CFG["lr_backbone"]},
                             {"params": hd, "lr": CFG["lr_head"]}],
                            weight_decay=CFG["weight_decay"])
    crit = nn.CrossEntropyLoss()
    ckpt = os.path.join(OUT_DIR, f"fold{fold_i}_best.pt")
    best, best_epoch, epoch = (-1, -1), 0, 0
    while True:
        epoch += 1
        model.train()
        tr_loss = 0.0
        for x, t, _n in dl:
            x, t = x.to(device), t.to(device)
            logits = model(x)                        # (B,2,C,4)
            loss = crit(logits.reshape(-1, CFG["max_count"] + 1), t.reshape(-1))
            opt.zero_grad()
            loss.backward()
            opt.step()
            tr_loss += loss.item() * x.size(0)
        tr_loss /= len(ds)

        probs = predict_probs([model], va_names, device, use_tta=False)
        ev = evaluate(probs, va_names, labels, vocab)
        key = (ev["full"], ev["tot"])
        is_best = key > best
        if is_best:
            best, best_epoch = key, epoch
            torch.save({"epoch": epoch, "config": CFG, "vocab": vocab,
                        "state_dict": model.state_dict()}, ckpt)
        log_writer.writerow([fold_i, epoch, f"{tr_loss:.4f}", ev["full"], ev["tot"],
                             int(is_best)])
        if epoch % 10 == 0 or is_best:
            print(f"[fold {fold_i}] ep {epoch:>3} | loss {tr_loss:.4f} | "
                  f"val 완전 {ev['full']}/{ev['n']} 총점 {ev['tot']}/{ev['n']}"
                  + ("  * best" if is_best else ""), flush=True)
        if epoch - best_epoch >= CFG["patience"] or epoch >= CFG["max_epochs"]:
            print(f"[fold {fold_i}] 종료 (best ep {best_epoch}: 완전 {best[0]})", flush=True)
            break
    return ckpt


def load_ensemble(device):
    ck0 = torch.load(os.path.join(OUT_DIR, "fold0_best.pt"),
                     map_location=device, weights_only=False)
    vocab = ck0["vocab"]
    out = []
    for i in range(CFG["n_folds"]):
        ck = torch.load(os.path.join(OUT_DIR, f"fold{i}_best.pt"),
                        map_location=device, weights_only=False)
        m = SetNet(len(vocab)).to(device)
        m.load_state_dict(ck["state_dict"])
        m.eval()
        out.append(m)
    return out, vocab


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    labels = load_labels()
    vocab, v2i = build_vocab(labels)
    train_all = load_split("train.csv")
    test_names = [r["image"] for r in load_split("test.csv")]
    folds = make_folds(train_all, CFG["n_folds"], CFG["seed"])
    print(f"device: {device} | cells 조합 {len(vocab)}종 {vocab}", flush=True)
    print(f"train {sum(len(f) for f in folds)} -> {CFG['n_folds']}-fold / "
          f"test {len(test_names)} (분할: {os.path.basename(SPLIT_DIR)})", flush=True)

    log_f = open(os.path.join(OUT_DIR, "train_log.csv"), "w",
                 newline="", encoding="utf-8-sig")
    log_writer = csv.writer(log_f)
    log_writer.writerow(["fold", "epoch", "loss", "val_full", "val_total", "best"])
    for i in range(CFG["n_folds"]):
        va = folds[i]
        tr = [n for j, f_ in enumerate(folds) if j != i for n in f_]
        print(f"\n===== fold {i}: train {len(tr)} / val {len(va)} =====", flush=True)
        train_fold(i, tr, va, labels, vocab, v2i, device, log_writer)
        log_f.flush()
    log_f.close()

    models_list, vocab = load_ensemble(device)
    probs = predict_probs(models_list, test_names, device)
    ev = evaluate(probs, test_names, labels, vocab, detail=True)
    n = ev["n"]
    print(f"\n=== 최종 평가: 앙상블+TTA / test {n}장 ===")
    print(f"[완전 일치] {ev['full']}/{n} ({ev['full']/n*100:.1f}%)")
    print(f"[총점 일치] {ev['tot']}/{n} ({ev['tot']/n*100:.1f}%) | MAE {ev['mae']:.2f}")
    for name, tl, tp, lab, pred in ev["rows"][:20]:
        print(f"  {name} 라벨총점 {tl} 예측 {tp} | {lab} -> {pred}")


if __name__ == "__main__":
    main()
