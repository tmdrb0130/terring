# -*- coding: utf-8 -*-
"""오답 이미지의 분쟁 스톤을 확대 시각화 — 사람이 직접 보고 판정하기 위한 카드 생성.

각 오답 이미지마다 카드 1장(JPG)을 만든다:
  왼쪽  = 전체 보드 (검출된 스톤 전부 표시, 분쟁 스톤만 노란 굵은 원)
  오른쪽 = 분쟁 스톤 확대 (원본 해상도, 접촉 여부를 눈으로 확인할 수 있게)
  상단  = 이미지명 / 사람 라벨 vs 모델 예측 / 쟁점이 되는 칸

사용법: python review_disputes.py [--out ../review_disputes]
"""
import argparse
import os

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from train_cnn import SX, SY, compute_maps, decode_outputs, load_human_labels, load_split
from infer import THR, CELLS_THR, load_ensemble

FONT_PATH = r"C:\Windows\Fonts\malgun.ttf"
CELL_NAME = {1: "1점칸", 2: "2점칸", 3: "3점칸", 4: "4점칸", 5: "중앙5원", 0: "보드밖"}
ZOOM = 560          # 확대 패널 한 변 (원본 픽셀 기준 crop 크기)
PANEL_H = 1000      # 카드 높이
HEADER_H = 190


def fmt_cells(cs):
    if not cs or cs == frozenset({0}):
        return "0"
    return "".join(str(c) for c in sorted(cs))


def score_of(cs):
    return max(cs) if cs and cs != frozenset({0}) else 0


def match_and_find_disputes(dets, label_stones):
    """예측 스톤과 라벨을 (색, cells) 동일한 것끼리 먼저 짝짓고, 남은 것이 분쟁.
    반환: [(det 또는 None, label 또는 None)] — 짝이 안 맞으면 한쪽이 None"""
    rest_det = list(dets)
    rest_lab = list(label_stones)
    for lab in list(rest_lab):
        color, cs = lab
        lab_cs = frozenset() if cs == frozenset({0}) else cs
        for d in list(rest_det):
            if d[2] == color and d[3] == lab_cs:
                rest_det.remove(d)
                rest_lab.remove(lab)
                break
    # 남은 것끼리 같은 색으로 짝지음 (분쟁 후보)
    pairs = []
    for lab in list(rest_lab):
        color, cs = lab
        for d in list(rest_det):
            if d[2] == color:
                pairs.append((d, lab))
                rest_det.remove(d)
                rest_lab.remove(lab)
                break
        else:
            pairs.append((None, lab))
            rest_lab.remove(lab)
    for d in rest_det:
        pairs.append((d, None))
    return pairs


def draw_text(img_bgr, lines, box):
    """PIL 로 한글 텍스트 그리기 (cv2.putText 는 한글 불가)"""
    x0, y0, w, h = box
    pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    dr = ImageDraw.Draw(pil)
    y = y0
    for text, size, color in lines:
        try:
            font = ImageFont.truetype(FONT_PATH, size)
        except OSError:
            font = ImageFont.load_default()
        dr.text((x0, y), text, font=font, fill=color)
        y += int(size * 1.45)
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def make_card(name, path, dets, label_stones, out_path, disputes=None):
    """disputes 를 직접 주면 (좌표 기반 매칭 등) 내부 매칭을 생략한다.
    형식: [(det 또는 None, (색, cells frozenset) 또는 None)]"""
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    H, W = img.shape[:2]
    fx, fy = W / 1050.0, H / 1400.0
    if disputes is None:
        pairs = match_and_find_disputes(dets, label_stones)
        disputes = [(d, l) for d, l in pairs if d is None or l is None
                    or d[3] != (frozenset() if l[1] == frozenset({0}) else l[1])]

    # ---- 왼쪽: 전체 보드 ----
    board = img.copy()
    for (x, y, color, cs) in dets:
        cx, cy = int(x / SX * fx), int(y / SY * fy)
        bgr = (0, 0, 255) if color == "red" else (255, 100, 0)
        cv2.circle(board, (cx, cy), int(78 * fx), bgr, 3)
    for d, _l in disputes:
        if d is None:
            continue
        cx, cy = int(d[0] / SX * fx), int(d[1] / SY * fy)
        cv2.circle(board, (cx, cy), int(95 * fx), (0, 255, 255), 8)
    bh = PANEL_H - HEADER_H
    bw = int(board.shape[1] * bh / board.shape[0])
    board = cv2.resize(board, (bw, bh))

    # ---- 오른쪽: 분쟁 스톤 확대 ----
    # 미검출(예측 없음)이면 그릴 좌표가 없다 — 중앙에 원을 그리면 오검출로 오해되므로 표시하지 않음
    has_pos = bool(disputes) and disputes[0][0] is not None
    if has_pos:
        d = disputes[0][0]
        cx, cy = int(d[0] / SX * fx), int(d[1] / SY * fy)
    else:
        cx, cy = W // 2, H // 2
    half = ZOOM // 2
    x0, y0 = max(0, min(W - ZOOM, cx - half)), max(0, min(H - ZOOM, cy - half))
    zoom = img[y0:y0 + ZOOM, x0:x0 + ZOOM].copy()
    if has_pos:
        cv2.circle(zoom, (cx - x0, cy - y0), int(78 * fx), (0, 255, 255), 3)
    zoom = cv2.resize(zoom, (bh, bh), interpolation=cv2.INTER_CUBIC)

    card = np.full((PANEL_H, bw + bh + 30, 3), 245, dtype=np.uint8)
    card[HEADER_H:, :bw] = board
    card[HEADER_H:, bw + 30:] = zoom

    # ---- 상단 텍스트 ----
    tl = sum(score_of(cs) for _c, cs in label_stones)
    tp = sum(score_of(cs) for _x, _y, _c, cs in dets)
    lines = [(f"{name}    사람 라벨 총점 {tl}점  vs  모델 예측 총점 {tp}점"
              + ("   (총점 일치)" if tl == tp else "   (총점 불일치)"), 40, (20, 20, 20))]
    for d, l in disputes:
        lab_s = fmt_cells(l[1]) if l else "(라벨 없음)"
        pred_s = fmt_cells(d[3]) if d else "(미검출)"
        color = (l[0] if l else d[2])
        diff = ""
        if d and l:
            a = frozenset() if l[1] == frozenset({0}) else l[1]
            b = d[3]
            only_lab = sorted(a - b)
            only_pred = sorted(b - a)
            parts = []
            if only_lab:
                parts.append("라벨만: " + ", ".join(CELL_NAME[c] for c in only_lab))
            if only_pred:
                parts.append("모델만: " + ", ".join(CELL_NAME[c] for c in only_pred))
            diff = "   쟁점 -> " + " / ".join(parts)
        lines.append((f"분쟁 스톤({color}):  라벨 {lab_s} ({score_of(l[1]) if l else 0}점)"
                      f"   모델 {pred_s} ({score_of(d[3]) if d else 0}점){diff}",
                      34, (0, 0, 160)))
    lines.append(("노란 원 = 분쟁 스톤 / 오른쪽은 원본 해상도 확대", 28, (110, 110, 110)))
    card = draw_text(card, lines, (24, 18, card.shape[1], HEADER_H))
    cv2.imencode(".jpg", card, [cv2.IMWRITE_JPEG_QUALITY, 88])[1].tofile(out_path)
    return disputes, tl, tp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))), "review_disputes"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    import train_cnn as t5
    device = "cuda" if torch.cuda.is_available() else "cpu"
    human = load_human_labels()
    test_names = [r["image"] for r in load_split("test.csv")]
    models_list = load_ensemble(device)
    print(f"device: {device} | test {len(test_names)}장에서 오답 탐색", flush=True)

    maps = compute_maps(models_list, test_names, device)
    wrong = []
    for name in test_names:
        dets = decode_outputs(*maps[name], THR, CELLS_THR)
        preds = [(c, cs if cs else frozenset({0})) for _x, _y, c, cs in dets]
        lab = human[name]["stones"]
        rest = list(preds)
        matched = 0
        for ls in lab:
            if ls in rest:
                rest.remove(ls)
                matched += 1
        if not (len(preds) == len(lab) and matched == len(lab)):
            wrong.append((name, dets, lab))
    print(f"완전일치 실패 {len(wrong)}장: {[w[0] for w in wrong]}", flush=True)

    summary = []
    for name, dets, lab in wrong:
        path = t5.img_path(name)
        out_path = os.path.join(args.out, f"{os.path.splitext(name)[0]}.jpg")
        disputes, tl, tp = make_card(name, path, dets, lab, out_path)
        summary.append((name, tl, tp, len(disputes)))
        print(f"  {name}: 라벨 {tl}점 / 예측 {tp}점 / 분쟁 스톤 {len(disputes)}개 -> "
              f"{os.path.basename(out_path)}", flush=True)
    print(f"\n{len(summary)}장 저장 완료: {args.out}")


if __name__ == "__main__":
    main()
