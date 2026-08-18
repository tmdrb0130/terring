# -*- coding: utf-8 -*-
"""분쟁 스톤 검토 시트 생성: 자동 판정과 사람 라벨이 다른 스톤마다
[원본 확대 | 정면 보정 + 칸 경계] 카드를 만들어 review/ 폴더에 저장"""
import os

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import terring_cv as tc
from eval_cv import load_labels

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
REVIEW = os.path.join(MODEL_DIR, "review")
os.makedirs(REVIEW, exist_ok=True)

FONT = ImageFont.truetype(r"C:\Windows\Fonts\malgun.ttf", 30)
FONT_S = ImageFont.truetype(r"C:\Windows\Fonts\malgun.ttf", 24)


def cells_str(cs):
    return "".join(map(str, sorted(cs))) if cs else "-"


def make_card(name, img, res, stone, label_cands, idx):
    x, y = int(stone["x"]), int(stone["y"])
    wx, wy = int(stone["wx"]), int(stone["wy"])

    # 왼쪽: 원본 확대 (스톤 주변)
    r = 230
    H, W = img.shape[:2]
    x0, y0 = max(0, x - r), max(0, y - r)
    x1, y1 = min(W, x + r), min(H, y + r)
    crop_o = img[y0:y1, x0:x1].copy()
    cv2.circle(crop_o, (x - x0, y - y0), 12, (0, 255, 255), 3)
    crop_o = cv2.resize(crop_o, (460, int(crop_o.shape[0] * 460 / crop_o.shape[1])))

    # 오른쪽: 정면 보정 + 칸 경계 + 접촉원
    size = (tc.BW + 2 * tc.MARGIN, tc.BH + 2 * tc.MARGIN)
    warp = cv2.warpPerspective(img, res["H"], size)
    for (cx0, cy0, cx1, cy1, v) in tc._cell_rects():
        cv2.rectangle(warp, (int(cx0), int(cy0)), (int(cx1), int(cy1)), (0, 255, 0), 3)
    g = tc.GEO
    cv2.circle(warp, (int(tc.MARGIN + g["circle_cx"] * tc.BW), int(tc.MARGIN + g["circle_cy"] * tc.BH)),
               int(g["circle_r"] * min(tc.BW, tc.BH)), (0, 255, 0), 3)
    stone_r = int(g["stone_r_ratio"] * tc.BW)
    cv2.circle(warp, (wx, wy), stone_r, (0, 255, 255), 4)   # 자동이 계산한 접촉원
    cv2.circle(warp, (wx, wy), 6, (0, 255, 255), -1)
    rw = 280
    wx0, wy0 = max(0, wx - rw), max(0, wy - rw)
    wx1, wy1 = min(size[0], wx + rw), min(size[1], wy + rw)
    crop_w = warp[wy0:wy1, wx0:wx1].copy()
    crop_w = cv2.resize(crop_w, (460, int(crop_w.shape[0] * 460 / crop_w.shape[1])))

    h = max(crop_o.shape[0], crop_w.shape[0])
    banner = 120
    canvas = np.full((h + banner, 940, 3), 255, np.uint8)
    canvas[banner:banner + crop_o.shape[0], 0:460] = crop_o
    canvas[banner:banner + crop_w.shape[0], 480:940] = crop_w

    pil = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(pil)
    auto = cells_str(stone["cells"])
    lab = " 또는 ".join(cells_str(c) for c in label_cands) if label_cands else "?"
    color_kr = "빨강" if stone["color"] == "red" else "파랑"
    d.text((12, 8), f"[{idx}] {name}  ({color_kr} 스톤)", font=FONT, fill=(0, 0, 0))
    d.text((12, 52), f"자동 판정: {auto} ({max(stone['cells'])}점)   내 라벨: {lab}"
                     f"   → 어느 쪽이 맞나요?", font=FONT_S, fill=(200, 0, 0))
    d.text((12, 86), "왼쪽=원본 확대(노란점=검출 중심) / 오른쪽=정면 보정(노란원=계산된 접촉 범위, 초록=칸 경계)",
           font=FONT_S, fill=(90, 90, 90))
    out = os.path.join(REVIEW, f"case{idx:02d}_{name.replace('.jpg', '')}_{stone['color']}.jpg")
    pil.save(out, quality=90)
    return out


def main():
    labels = load_labels()
    idx = 0
    seen_keys = set()   # 같은 장면 반복 방지: (색, 자동cells, 라벨cells) 단위로 1건만
    for name in sorted(labels):
        img = cv2.imread(os.path.join(tc.IMG_DIR, name))
        res = tc.process_image(img)
        if res is None:
            continue
        lab_stones = list(labels[name]["stones"])
        # 매칭: 완전 일치 쌍 제거
        rest = lab_stones[:]
        disputed = []
        for s in res["stones"]:
            key = (s["color"], frozenset(s["cells"]))
            if key in rest:
                rest.remove(key)
            else:
                disputed.append(s)
        for s in disputed:
            cands = [c for col, c in rest if col == s["color"]]
            dkey = (s["color"], frozenset(s["cells"]), tuple(sorted(map(tuple, map(sorted, cands)))))
            if dkey in seen_keys:
                continue
            seen_keys.add(dkey)
            idx += 1
            out = make_card(name, img, res, s, cands, idx)
            print(f"case{idx:02d}: {name} {s['color']} 자동={cells_str(s['cells'])} "
                  f"라벨후보={[cells_str(c) for c in cands]} -> {os.path.basename(out)}")
    print(f"\n총 {idx}건 저장: {REVIEW}")


if __name__ == "__main__":
    main()
