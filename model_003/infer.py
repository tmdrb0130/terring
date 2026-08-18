# -*- coding: utf-8 -*-
"""단일 이미지 추론 데모: 5-fold 앙상블 + TTA로 스톤 검출 → 칸 판정 → 점수 출력.

사용법:
  python infer.py 033.jpg                      # data 폴더 안 파일명
  python infer.py C:\사진\새로찍은거.jpg        # 임의 경로 이미지도 가능
  옵션: [--thr 0.35] [--no-save]
결과: 콘솔에 스톤별 판정 + output/infer_<이미지>.jpg 시각화 저장
"""
import argparse
import os

import cv2
import numpy as np
import torch

import train_cnn as t4
from train_cnn import (CFG, OUT_DIR, IW, IH, SX, SY, DetNet, compare, decode_outputs,
                       load_human_labels, predict_maps, to_cells)
import terring_cv as tc

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
        path = t4.img_path(arg)
    key = os.path.basename(path)
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit(f"이미지를 읽을 수 없습니다: {path}")
    small = cv2.resize(img, (IW, IH), interpolation=cv2.INTER_AREA)
    t4._IMG_CACHE[key] = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
    return key, path


def annotate(path, stones, corners, preds):
    """원본 해상도 이미지 위에 보드 외곽선 + 스톤별 색·점수 라벨을 그려 반환 (BGR)"""
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    fx, fy = img.shape[1] / 1050.0, img.shape[0] / 1400.0  # 라벨 좌표계 → 원본 해상도
    quad = (corners / [SX, SY] * [fx, fy]).astype(int)
    cv2.polylines(img, [quad.reshape(-1, 1, 2)], True, (0, 255, 255), 3)
    r = int(tc.GEO["stone_r_ratio"] * tc.BW * 0.9 * fx)
    for i, ((x, y, color), (_, cells)) in enumerate(zip(stones, preds), 1):
        cx, cy = int(x / SX * fx), int(y / SY * fy)
        bgr = (0, 0, 255) if color == "red" else (255, 80, 0)
        cv2.circle(img, (cx, cy), r, bgr, 3)
        score = max(cells)
        label = f"S_{i}:{score}"
        cv2.putText(img, label, (cx - 60, cy - r - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 6)
        cv2.putText(img, label, (cx - 60, cy - r - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, bgr, 2)
    return img


def draw_result(name, path, stones, corners, preds):
    img = annotate(path, stones, corners, preds)
    out = os.path.join(OUT_DIR, f"infer_{os.path.splitext(name)[0]}.jpg")
    cv2.imencode(".jpg", img)[1].tofile(out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image", help="data 폴더 안 파일명(예: 033.jpg) 또는 이미지 파일 경로")
    ap.add_argument("--thr", type=float, default=0.35, help="히트맵 피크 임계값")
    ap.add_argument("--no-save", action="store_true")
    args = ap.parse_args()

    key, path = resolve_image(args.image)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device} | 앙상블 {CFG['n_folds']}개 fold + 좌우반전 TTA | thr={args.thr}")
    models_list = load_ensemble(device)

    with torch.no_grad():
        hm_s, hm_c, off, reg = predict_maps(models_list, key, device, use_tta=True)
    stones, corners = decode_outputs(hm_s, hm_c, off, reg, args.thr)
    preds = to_cells(stones, corners)

    print(f"\n=== {key} 판정 결과 ===")
    print(f"[1] 보드 코너 4점 검출 → 정면 보정(호모그래피) 완료")
    print(f"[2] 스톤 검출: {len(stones)}개")
    total = 0
    for i, ((x, y, color), (_, cells)) in enumerate(zip(stones, preds), 1):
        score = max(cells)
        total += score
        cell_str = ", ".join(CELL_NAMES[c] for c in sorted(cells))
        ox, oy = x / SX, y / SY
        print(f"    스톤{i} {color:<4} @ 원본좌표 ({ox:4.0f},{oy:4.0f}) → "
              f"닿은 칸 {{{cell_str}}} → max = {score}점")
    print(f"[3] 총점: {total}점")

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
        out = draw_result(key, path, stones, corners, preds)
        print(f"\n시각화 저장: {out}")


if __name__ == "__main__":
    main()
