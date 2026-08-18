# -*- coding: utf-8 -*-
r"""단일 이미지 추론 데모 (model_004, 구 model_005): 5-fold 앙상블 + TTA.

model_003(구 model_004)와 달리 기하 계산 없이 CNN의 cells 헤드가 닿은 칸까지 직접 판정한다.

사용법:
  python infer.py 033.jpg                      # data 폴더 안 파일명
  python infer.py C:\...\아무경로.jpg           # 임의 경로 이미지도 가능
  옵션: [--thr 0.2] [--cells-thr 0.65] [--no-save]
결과: 콘솔에 스톤별 판정 + output/infer_<이미지>.jpg 시각화 저장
"""
import argparse
import os

import cv2
import numpy as np
import torch

import train_cnn as t5
from train_cnn import (CFG, OUT_DIR, IW, IH, HW, SX, SY, DetNet, compare, decode_outputs,
                       load_human_labels, predict_maps, stones_to_preds)

# train 스윕 선택값. 2026-08-24 재스윕에서 0.2 -> 0.05 로 하향:
# 기존 그리드가 [0.2, 0.3, 0.4, 0.5] 라 최저값이 선택된 상태였고, 그 아래를 시험한 적이 없었다.
# 프레임에 잘린 스톤은 피크가 약해(080 의 경우 0.106) 0.2 에 걸려 버려지고 있었음.
THR = 0.05
CELLS_THR = 0.65
CELL_NAMES = {0: "보드 밖", 1: "1점 칸", 2: "2점 칸", 3: "3점 칸", 4: "4점 칸", 5: "중앙 5원"}


def load_ensemble(device):
    models_list = []
    for i in range(CFG["n_folds"]):
        ck = torch.load(os.path.join(OUT_DIR, f"fold{i}_best.pt"),
                        map_location=device, weights_only=False)
        m = DetNet().to(device)
        m.load_state_dict(ck["state_dict"])
        m.eval()
        models_list.append(m)
    return models_list


def resolve_image(arg):
    """인자가 실제 파일 경로면 그대로, 아니면 data 폴더에서 파일명으로 검색.
    반환: (캐시 키, 원본 경로) — 임의 경로 이미지는 _IMG_CACHE 에 직접 등록"""
    if os.path.isfile(arg):
        path = os.path.abspath(arg)
    else:
        path = t5.img_path(arg)
    key = os.path.basename(path)
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit(f"이미지를 읽을 수 없습니다: {path}")
    small = cv2.resize(img, (IW, IH), interpolation=cv2.INTER_AREA)
    t5._IMG_CACHE[key] = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
    return key, path


def decode_corners(hm_c, off):
    """코너 히트맵 argmax + 오프셋 — 시각화 전용 (판정에는 안 쓰임)"""
    corners = np.zeros((4, 2))
    conf = []
    for i in range(4):
        flat = torch.argmax(hm_c[i])
        y, x = int(flat) // HW, int(flat) % HW
        conf.append(float(hm_c[i, y, x]))
        ox = float(off[0, y, x]) - 0.5
        oy = float(off[1, y, x]) - 0.5
        corners[i] = [(x + 0.5 + ox) * CFG["stride"], (y + 0.5 + oy) * CFG["stride"]]
    return corners, min(conf)


def annotate(path, stones, corners, corner_conf):
    """원본 해상도 이미지 위에 보드 외곽선 + 스톤별 색·점수 라벨을 그려 반환 (BGR)"""
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    fx, fy = img.shape[1] / 1050.0, img.shape[0] / 1400.0  # 라벨 좌표계 → 원본 해상도
    if corner_conf >= 0.15:
        quad = (corners / [SX, SY] * [fx, fy]).astype(int)
        cv2.polylines(img, [quad.reshape(-1, 1, 2)], True, (0, 255, 255), 3)
    r = int(80 * fx)
    for i, (x, y, color, cells) in enumerate(stones, 1):
        cx, cy = int(x / SX * fx), int(y / SY * fy)
        bgr = (0, 0, 255) if color == "red" else (255, 80, 0)
        cv2.circle(img, (cx, cy), r, bgr, 3)
        score = max(cells) if cells else 0
        label = f"S_{i}:{score}"
        cv2.putText(img, label, (cx - 60, cy - r - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 6)
        cv2.putText(img, label, (cx - 60, cy - r - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, bgr, 2)
    return img


def draw_result(name, path, stones, corners, corner_conf):
    img = annotate(path, stones, corners, corner_conf)
    out = os.path.join(OUT_DIR, f"infer_{os.path.splitext(name)[0]}.jpg")
    cv2.imencode(".jpg", img)[1].tofile(out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image", help="data 폴더 안 파일명(예: 033.jpg) 또는 이미지 파일 경로")
    ap.add_argument("--thr", type=float, default=THR, help="스톤 피크 임계값")
    ap.add_argument("--cells-thr", type=float, default=CELLS_THR, help="칸 판정 임계값")
    ap.add_argument("--no-save", action="store_true")
    args = ap.parse_args()

    key, path = resolve_image(args.image)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device} | 앙상블 {CFG['n_folds']}개 fold + 좌우반전 TTA | "
          f"thr={args.thr} cells={args.cells_thr}")
    models_list = load_ensemble(device)

    with torch.no_grad():
        hm_s, hm_c, off, cells_map = predict_maps(models_list, key, device, use_tta=True)
    stones = decode_outputs(hm_s, hm_c, off, cells_map, args.thr, args.cells_thr)
    preds = stones_to_preds(stones)

    print(f"\n=== {key} 판정 결과 ===")
    print(f"[1] 스톤 검출: {len(stones)}개 (CNN이 위치·색·닿은 칸을 직접 판정)")
    total = 0
    for i, (x, y, color, cells) in enumerate(stones, 1):
        score = max(cells) if cells else 0
        total += score
        show = sorted(cells) if cells else [0]
        cell_str = ", ".join(CELL_NAMES[c] for c in show)
        ox, oy = x / SX, y / SY
        print(f"    스톤{i} {color:<4} @ 원본좌표 ({ox:4.0f},{oy:4.0f}) → "
              f"닿은 칸 {{{cell_str}}} → max = {score}점")
    print(f"[2] 총점: {total}점")

    human = load_human_labels()
    if key in human:
        lab = human[key]
        full, tot_ok, matched, n_lab, _tp, _tl = compare(preds, lab["stones"])
        print(f"\n사람 라벨(정답): 총점 {lab['total']}점")
        for i, (color, cells) in enumerate(lab["stones"], 1):
            cs = sorted(cells) if cells != frozenset({0}) else [0]
            print(f"    정답 스톤{i} {color:<4} 닿은 칸 {cs} → {max(cells)}점")
        print(f"판정: 완전 일치(스톤 수+색+닿은 칸 전부) "
              f"{'O' if full else 'X'} — 스톤 {matched}/{n_lab} 일치 "
              f"| 총점 일치 {'O' if tot_ok else 'X'}")

    if not args.no_save:
        corners, conf = decode_corners(hm_c, off)
        out = draw_result(key, path, stones, corners, conf)
        print(f"\n시각화 저장: {out}")


if __name__ == "__main__":
    main()
