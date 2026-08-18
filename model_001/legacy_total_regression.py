# -*- coding: utf-8 -*-
"""
terring 점수 예측 CNN (구 model_001 v3 총점 회귀 — legacy 보존본.
현행 model_001 은 train_cnn.py 의 멀티셋 분류 v4가 대체)
(v3: 무한 on-the-fly 증강 + 과적합 감지 중단 + 5-fold 앙상블)

학습 방식:
- 증강은 학습 루프 안에서 실시간(on-the-fly): 매 배치마다 원본에 새로운 랜덤 변형이
  걸리므로 같은 이미지를 두 번 똑같이 보는 일이 없음 (사전 생성 아님)
- 에폭 상한 없음: val MAE가 PATIENCE 에폭 동안 개선되지 않으면(=과적합 시작) 중단하고
  과적합 이전의 best 모델을 채택
- train.csv 58장을 난이도 층화 5-fold로 나눠 모델 5개를 학습 → 예측 평균(앙상블)
- test.csv 27장은 모든 학습이 끝난 뒤 최종 평가에만 사용
"""
import csv
import glob
import json
import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(MODEL_DIR)
DATA_DIR = os.path.join(ROOT, "data")
OUT_DIR = os.path.join(MODEL_DIR, "output")

CFG = {
    "seed": 42,
    "max_epochs": 500,        # 안전장치일 뿐, 실질 종료는 얼리스탑
    "patience": 30,           # val MAE 개선 없이 버티는 에폭 수 = 과적합 판정
    "batch_size": 12,
    "lr_head": 1e-3,
    "lr_backbone": 1e-4,
    "weight_decay": 1e-4,
    "img_size": 384,
    "backbone": "resnet34",
    "total_loss_weight": 0.1,
    "n_folds": 5,
    "repeat_per_epoch": 8,    # 에폭당 train 반복 배수 (매번 다른 증강)
    "tta_rounds": 6,          # 최종 예측 시 모델당 TTA 횟수
}
SCORE_VALUES = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])

random.seed(CFG["seed"])
np.random.seed(CFG["seed"])
torch.manual_seed(CFG["seed"])
torch.cuda.manual_seed_all(CFG["seed"])


# ---------- 데이터 ----------
def build_image_index():
    index = {}
    for p in glob.glob(os.path.join(DATA_DIR, "**", "*.jpg"), recursive=True):
        name = os.path.basename(p)
        if name in index:
            raise RuntimeError(f"이미지 파일명 중복: {name}\n  {index[name]}\n  {p}")
        index[name] = p
    return index


def load_rows(name):
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
            rows.append({
                "image": r["image"], "counts": counts, "total": int(r["total_score"]),
                "on_board": sum(counts[1:]), "overlap": overlap,
            })
    return rows


def stratum(r):
    s = "S1" if r["on_board"] <= 1 else ("S2-3" if r["on_board"] <= 3 else "S4-6")
    o = "O0" if r["overlap"] == 0 else ("O1-2" if r["overlap"] <= 2 else "O3+")
    return s + "/" + o


def make_folds(rows, k, seed):
    """난이도 층별로 섞어서 k개 fold에 라운드로빈 분배 → 모든 fold가 비슷한 난이도 구성"""
    groups = {}
    for r in rows:
        groups.setdefault(stratum(r), []).append(r)
    rng = random.Random(seed)
    folds = [[] for _ in range(k)]
    i = 0
    for key in sorted(groups):
        g = groups[key][:]
        rng.shuffle(g)
        for r in g:
            folds[i % k].append(r)
            i += 1
    return folds


# ---------- 변환 (전부 on-the-fly, 호출될 때마다 다른 랜덤 변형) ----------
NORM = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

def make_train_tf(size):
    return transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomPerspective(distortion_scale=0.3, p=0.5, fill=127),
        transforms.RandomAffine(degrees=25, translate=(0.08, 0.08),
                                scale=(0.65, 1.2), fill=127),
        transforms.RandomResizedCrop(size, scale=(0.7, 1.0), ratio=(0.85, 1.18)),
        transforms.ColorJitter(brightness=0.35, contrast=0.35, saturation=0.3, hue=0.04),
        transforms.ToTensor(),
        NORM,
    ])

def make_eval_tf(size):
    return transforms.Compose([transforms.Resize((size, size)), transforms.ToTensor(), NORM])

def make_tta_tf(size):
    return transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(degrees=10, translate=(0.04, 0.04), scale=(0.9, 1.1), fill=127),
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        NORM,
    ])


class TerringDataset(Dataset):
    def __init__(self, rows, img_index, tf, repeat=1):
        self.rows = rows
        self.img_index = img_index
        self.tf = tf
        self.repeat = repeat

    def __len__(self):
        return len(self.rows) * self.repeat

    def __getitem__(self, idx):
        r = self.rows[idx % len(self.rows)]
        img = Image.open(self.img_index[r["image"]]).convert("RGB")
        return (self.tf(img),
                torch.tensor(r["counts"], dtype=torch.float32),
                torch.tensor(r["total"], dtype=torch.float32))


def build_model():
    weights = {"resnet18": models.ResNet18_Weights.IMAGENET1K_V1,
               "resnet34": models.ResNet34_Weights.IMAGENET1K_V1}[CFG["backbone"]]
    m = {"resnet18": models.resnet18, "resnet34": models.resnet34}[CFG["backbone"]](weights=weights)
    m.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(m.fc.in_features, 128),
        nn.ReLU(),
        nn.Linear(128, 6),
    )
    return m


@torch.no_grad()
def evaluate(model, dl, device):
    model.eval()
    sv = SCORE_VALUES.to(device)
    abs_err, exact, n = [], 0, 0
    for x, counts, total in dl:
        x = x.to(device)
        pred_total = (model(x) * sv).sum(dim=1).cpu()
        err = (pred_total.round().clamp(0, 30) - total).abs()
        abs_err += err.tolist()
        exact += (err == 0).sum().item()
        n += x.size(0)
    return sum(abs_err) / n, exact, n


def train_one_fold(fold_i, train_rows, val_rows, img_index, device, log_writer):
    size = CFG["img_size"]
    train_ds = TerringDataset(train_rows, img_index, make_train_tf(size), repeat=CFG["repeat_per_epoch"])
    val_ds = TerringDataset(val_rows, img_index, make_eval_tf(size))
    train_dl = DataLoader(train_ds, batch_size=CFG["batch_size"], shuffle=True,
                          num_workers=4, persistent_workers=True)
    val_dl = DataLoader(val_ds, batch_size=CFG["batch_size"], shuffle=False,
                        num_workers=2, persistent_workers=True)

    model = build_model().to(device)
    sv = SCORE_VALUES.to(device)
    backbone = [p for n_, p in model.named_parameters() if not n_.startswith("fc.")]
    head = [p for n_, p in model.named_parameters() if n_.startswith("fc.")]
    opt = torch.optim.AdamW([
        {"params": backbone, "lr": CFG["lr_backbone"]},
        {"params": head, "lr": CFG["lr_head"]},
    ], weight_decay=CFG["weight_decay"])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5,
                                                       patience=8, min_lr=1e-6)
    mse = nn.MSELoss()

    ckpt_path = os.path.join(OUT_DIR, f"fold{fold_i}_best.pt")
    best_mae = float("inf")
    best_epoch = 0
    epoch = 0
    while True:  # 무한 학습 — 과적합 감지로만 종료
        epoch += 1
        model.train()
        tr_loss = 0.0
        for x, counts, total in train_dl:
            x, counts, total = x.to(device), counts.to(device), total.to(device)
            pred_counts = model(x)
            pred_total = (pred_counts * sv).sum(dim=1)
            loss = mse(pred_counts, counts) + CFG["total_loss_weight"] * mse(pred_total, total)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tr_loss += loss.item() * x.size(0)
        tr_loss /= len(train_ds)

        mae, exact, n = evaluate(model, val_dl, device)
        sched.step(mae)
        is_best = mae < best_mae
        if is_best:
            best_mae, best_epoch = mae, epoch
            torch.save({"fold": fold_i, "epoch": epoch, "val_mae": mae, "config": CFG,
                        "state_dict": model.state_dict()}, ckpt_path)
        log_writer.writerow([fold_i, epoch, f"{tr_loss:.4f}", f"{mae:.3f}", exact, int(is_best)])
        if epoch % 10 == 0 or is_best:
            print(f"[fold {fold_i}] epoch {epoch:>3} | loss {tr_loss:.3f} | val MAE {mae:.2f} "
                  f"| 정확 {exact}/{n}" + ("  * best" if is_best else ""), flush=True)

        if epoch - best_epoch >= CFG["patience"]:
            print(f"[fold {fold_i}] 과적합 감지 → 중단 (best: epoch {best_epoch}, val MAE {best_mae:.2f})", flush=True)
            break
        if epoch >= CFG["max_epochs"]:
            print(f"[fold {fold_i}] max_epochs 도달", flush=True)
            break
    return ckpt_path, best_mae


@torch.no_grad()
def predict_total(model, img, tf, device, sv):
    x = tf(img).unsqueeze(0).to(device)
    return float((model(x)[0] * sv).sum())


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}", flush=True)

    with open(os.path.join(OUT_DIR, "config.json"), "w", encoding="utf-8") as f:
        json.dump(CFG, f, indent=2, ensure_ascii=False)

    img_index = build_image_index()
    all_train = load_rows("train.csv")
    test_rows = load_rows("test.csv")
    folds = make_folds(all_train, CFG["n_folds"], CFG["seed"])
    print(f"train 전체 {len(all_train)}장 → {CFG['n_folds']}-fold "
          f"({', '.join(str(len(f)) for f in folds)}) / test {len(test_rows)}장 (최종 전용)", flush=True)

    log_path = os.path.join(OUT_DIR, "train_log.csv")
    log_f = open(log_path, "w", newline="", encoding="utf-8-sig")
    log_writer = csv.writer(log_f)
    log_writer.writerow(["fold", "epoch", "train_loss", "val_mae", "val_exact", "best"])

    ckpts = []
    for i in range(CFG["n_folds"]):
        val_rows = folds[i]
        train_rows = [r for j, f_ in enumerate(folds) if j != i for r in f_]
        print(f"\n===== fold {i}: train {len(train_rows)} / val {len(val_rows)} =====", flush=True)
        ckpt_path, best_mae = train_one_fold(i, train_rows, val_rows, img_index, device, log_writer)
        log_f.flush()
        ckpts.append(ckpt_path)
    log_f.close()

    # ---------- 최종 평가: 5모델 앙상블 (+TTA) ----------
    size = CFG["img_size"]
    eval_tf = make_eval_tf(size)
    tta_tf = make_tta_tf(size)
    sv = SCORE_VALUES.to(device)
    models_list = []
    for p in ckpts:
        ck = torch.load(p, weights_only=False)
        m = build_model().to(device)
        m.load_state_dict(ck["state_dict"])
        m.eval()
        models_list.append(m)
        print(f"{os.path.basename(p)}: epoch {ck['epoch']}, val MAE {ck['val_mae']:.2f}")

    print(f"\n=== 최종 평가: {len(models_list)}-모델 앙상블 / test {len(test_rows)}장 ===")
    results = []
    for r in test_rows:
        img = Image.open(img_index[r["image"]]).convert("RGB")
        plain_preds, tta_preds = [], []
        for m in models_list:
            p0 = predict_total(m, img, eval_tf, device, sv)
            plain_preds.append(p0)
            tta_preds.append(p0)
            for _ in range(CFG["tta_rounds"]):
                tta_preds.append(predict_total(m, img, tta_tf, device, sv))
        results.append((r["image"], r["total"],
                        round(sum(plain_preds) / len(plain_preds)),
                        round(sum(tta_preds) / len(tta_preds))))

    for label, col in [("앙상블", 2), ("앙상블+TTA", 3)]:
        errs = [abs(row[col] - row[1]) for row in results]
        n = len(errs)
        print(f"[{label}] MAE {sum(errs)/n:.2f} | 정확 {sum(e == 0 for e in errs)}/{n} "
              f"({sum(e == 0 for e in errs)/n*100:.0f}%) | ±1 {sum(e <= 1 for e in errs)}/{n} "
              f"({sum(e <= 1 for e in errs)/n*100:.0f}%) | ±2 {sum(e <= 2 for e in errs)}/{n}")
    print(f"\n{'image':<10}{'정답':>5}{'앙상블':>7}{'+TTA':>6}")
    for img_name, t, p, pt in results:
        flag = "   <<<" if min(abs(p - t), abs(pt - t)) >= 3 else ""
        print(f"{img_name:<10}{t:>5}{p:>7}{pt:>6}{flag}")


if __name__ == "__main__":
    main()
