# -*- coding: utf-8 -*-
"""
model_002: 고전 CV 기반 terring 점수 판정 파이프라인

이미지 → ① 보드(선명한 3x3) 검출: 색 마스크 → 모서리 라인 피팅 → 호모그래피 정면 보정
      → ② 스톤 검출: 분홍 꼭지 블롭 + 주변 원판 색(red/blue) 분류
      → ③ 기하 계산: 스톤 접촉원 vs 칸 경계/중앙 원 교차 → 닿은 칸 → max → 총점

보드 실제 구조 (실측 기반):
  좌열: 4(빨) / 3(파) / 2(파)  — 경계 h1L, h2L
  중열: 2(빨) 위 / 1(파) 아래  — 경계 hm, 그 위에 5 원이 얹힘
  우열: 4(빨) / 3(빨) / 2(파)  — 경계 h1R, h2R
기하 상수는 geo.json(캘리브레이션 산출물)이 있으면 그것을 사용.
"""
import json
import os
import cv2
import numpy as np

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(MODEL_DIR)
IMG_DIR = os.path.join(ROOT, "data", "001~085 (08.16)")
DEBUG = os.path.join(MODEL_DIR, "debug")

BW, BH = 900, 900
MARGIN = 220

# 기하 상수 (0~1 비율, 보드 좌표계). calibrate.py가 geo.json으로 갱신
GEO = {
    "c1": 1 / 3, "c2": 2 / 3,          # 세로 칸 경계 x
    "h1L": 1 / 3, "h2L": 2 / 3,        # 좌열 가로 경계 y
    "h1R": 1 / 3, "h2R": 2 / 3,        # 우열 가로 경계 y
    "hm": 0.5,                         # 중열 2/1 경계 y
    "circle_cx": 0.5, "circle_cy": 0.5, "circle_r": 0.21,
    "stone_r_ratio": 0.078,            # 스톤 접촉원 반지름 / 보드 폭
    "base_offset": 0.25,               # 꼭지 중심 → 바닥 중심 보정 (스톤 반지름 배수, 이미지 아래 방향)
    "out_margin": None,                # 보드 밖 허용 범위(보드 폭 비율). None=끔. 라벨이 보드 밖
                                       # 스톤도 0점으로 기록하므로 기본은 끄는 게 맞음
    "knob_rim_min": 0.10,              # 꼭지 단독 검출을 인정할 최소 나무 테두리 비율
}
_geo_path = os.path.join(MODEL_DIR, "geo.json")
if os.path.exists(_geo_path):
    GEO.update(json.load(open(_geo_path, encoding="utf-8")))


# ---------- ① 보드 검출 ----------
def vivid_mask(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    red = ((H <= 10) | (H >= 168)) & (S > 110) & (V > 90)
    blue = (H >= 95) & (H <= 130) & (S > 110) & (V > 60)
    yellow = (H >= 18) & (H <= 40) & (S > 110) & (V > 140)
    m = (red | blue | yellow).astype(np.uint8) * 255
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((31, 31), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((15, 15), np.uint8))
    return m


def find_board_quad(img):
    m = vivid_mask(img)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) < 100000:
        return None
    hull = cv2.convexHull(c)
    peri = cv2.arcLength(hull, True)
    quad0 = None
    for eps in np.arange(0.01, 0.12, 0.01):
        ap = cv2.approxPolyDP(hull, eps * peri, True)
        if len(ap) == 4:
            quad0 = order_quad(ap.reshape(4, 2).astype(np.float64))
            break
    if quad0 is None:
        return None
    # 모서리 라인 피팅으로 정밀화: 컨투어 점을 가까운 변에 배정 후 fitLine
    pts = c.reshape(-1, 2).astype(np.float64)
    lines = []
    for i in range(4):
        a, b = quad0[i], quad0[(i + 1) % 4]
        ab = b - a
        L = np.linalg.norm(ab)
        u = ab / L
        rel = pts - a
        t = rel @ u
        dist = np.abs(rel[:, 0] * u[1] - rel[:, 1] * u[0])
        sel = pts[(dist < 18) & (t > 0.08 * L) & (t < 0.92 * L)]
        if len(sel) < 30:
            lines.append((a, b))
            continue
        vx, vy, x0, y0 = cv2.fitLine(sel.astype(np.float32), cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
        p = np.array([x0, y0], dtype=np.float64)
        d = np.array([vx, vy], dtype=np.float64)
        lines.append((p, p + d * 100))
    quad = []
    for i in range(4):
        p = intersect(lines[(i - 1) % 4], lines[i])
        if p is None:
            return quad0
        quad.append(p)
    return order_quad(np.array(quad))


def order_quad(q):
    s = q.sum(axis=1)
    d = q[:, 0] - q[:, 1]
    return np.array([q[np.argmin(s)], q[np.argmax(d)], q[np.argmax(s)], q[np.argmin(d)]])


def intersect(l1, l2):
    (p1, p2), (p3, p4) = l1, l2
    d1, d2 = p2 - p1, p4 - p3
    denom = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(denom) < 1e-9:
        return None
    t = ((p3[0] - p1[0]) * d2[1] - (p3[1] - p1[1]) * d2[0]) / denom
    return p1 + t * d1


def get_homography(quad):
    dst = np.array([[MARGIN, MARGIN], [MARGIN + BW, MARGIN],
                    [MARGIN + BW, MARGIN + BH], [MARGIN, MARGIN + BH]], dtype=np.float64)
    Hm, _ = cv2.findHomography(quad, dst)
    return Hm


# ---------- ② 스톤 검출 ----------
# 두 경로 병행:
#  (a) 분홍 꼭지 블롭 — 대부분의 스톤을 정확한 중심으로 잡음
#  (b) Hough 원 + 나무 림/꼭지 검증 — 조명에 색이 날아간 스톤도 원형 테두리로 잡음
def _annulus_px(hsv, x, y, r0, r1):
    Himg, W = hsv.shape[:2]
    x, y = int(x), int(y)
    r1i = int(np.ceil(r1))
    ys, xs = np.mgrid[max(0, y - r1i):min(Himg, y + r1i + 1),
                      max(0, x - r1i):min(W, x + r1i + 1)]
    d2 = (xs - x) ** 2 + (ys - y) ** 2
    sel = (d2 >= r0 * r0) & (d2 <= r1 * r1)
    return hsv[max(0, y - r1i):min(Himg, y + r1i + 1),
               max(0, x - r1i):min(W, x + r1i + 1)][sel]


def _color_vote(px, min_frac=0.22):
    """픽셀 집합에서 red/blue 판정 (색조 기반 — 밝기 무관)"""
    if len(px) < 40:
        return None
    Hc, Sc = px[:, 0].astype(int), px[:, 1].astype(int)
    red = (((Hc <= 8) | (Hc >= 160)) & (Sc > 55)).mean()
    blue = ((Hc >= 95) & (Hc <= 140) & (Sc > 40)).mean()
    if max(red, blue) < min_frac:
        return None
    return "red" if red > blue else "blue"


def _knob_blobs(hsv):
    H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    # 채도 상한이 핵심: 선명한 빨간 칸(S>=120)과 분홍 꼭지(S 40~90)를 가른다
    pink = ((H >= 140) & (H <= 176) & (S >= 25) & (S <= 110) & (V >= 165)).astype(np.uint8) * 255
    pink = cv2.morphologyEx(pink, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    pink = cv2.morphologyEx(pink, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    n, _, stats, centroids = cv2.connectedComponentsWithStats(pink)
    out = []
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        w, h = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        if not (80 <= area <= 2500):
            continue
        if max(w, h) > 60 or max(w, h) / max(1, min(w, h)) > 2.5:
            continue
        out.append((float(centroids[i][0]), float(centroids[i][1])))
    return out


def _wood_rim_frac(hsv, x, y, r):
    """스톤의 나무 테두리는 Hough 원 안쪽 가장자리(0.75~1.05r)에 있다"""
    px = _annulus_px(hsv, x, y, r * 0.75, r * 1.05)
    if len(px) < 40:
        return 0.0
    Hc, Sc, Vc = px[:, 0].astype(int), px[:, 1].astype(int), px[:, 2].astype(int)
    wood = ((Hc >= 9) & (Hc <= 32) & (Sc >= 30) & (Vc >= 80)).mean()
    return wood


def _dark_side_frac(hsv, x, y, r):
    """스톤 측면의 그림자 진 크레센트 — 평평한 칸에는 없다"""
    px = _annulus_px(hsv, x, y, r * 0.80, r * 1.18)
    if len(px) < 40:
        return 0.0
    return (px[:, 2].astype(int) < 75).mean()


def _outside_board(x, y, Hm, margin):
    """워프 보드 좌표로 옮겨, 보드 사각형 밖으로 margin(보드 폭 비율)을 넘는지"""
    p = cv2.perspectiveTransform(np.array([[[x, y]]], dtype=np.float64), Hm)[0, 0]
    u, v = (p[0] - MARGIN) / BW, (p[1] - MARGIN) / BH
    return not (-margin <= u <= 1 + margin and -margin <= v <= 1 + margin)


def find_stones(img, Hm=None, out_margin=None):
    """Hm(보드 호모그래피)을 주면 선명한 보드에서 out_margin 이상 떨어진 검출
    (파스텔 보드/바닥의 오검출)을 폐기한다. Hm=None이면 기존 동작 그대로."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    knobs = _knob_blobs(hsv)
    gray = cv2.medianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 5)
    cc = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1, minDist=85,
                          param1=110, param2=28, minRadius=48, maxRadius=82)
    circles = [] if cc is None else list(cc[0])

    stones = []          # (x, y, color)
    used_knob = [False] * len(knobs)

    def near_knob(x, y, r):
        for i, (kx, ky) in enumerate(knobs):
            if not used_knob[i] and (kx - x) ** 2 + (ky - y) ** 2 <= (0.65 * r) ** 2:
                return i
        return None

    # (b) Hough 원 경로
    for (cx, cy, r) in circles:
        ki = near_knob(cx, cy, r)
        rim = _wood_rim_frac(hsv, cx, cy, r)
        interior = _annulus_px(hsv, cx, cy, 0, r * 0.62)
        if ki is not None:
            # 꼭지+원이 겹치면 스톤 확정 → 색은 미약한 우세라도 인정 (조명 날림 대응)
            color = _color_vote(interior, min_frac=0.04)
            if color is None:
                color = classify_ring(hsv, knobs[ki][0], knobs[ki][1], min_frac=0.04)
        else:
            # 원만 있으면: (나무 링 + 색 근거) 또는 (매우 강한 색 + 측면 그림자).
            # 단, 내부가 나무색이면 장애물(기둥) 단면이므로 제외
            Hi = interior[:, 0].astype(int)
            Si = interior[:, 1].astype(int)
            wood_int = ((Hi >= 9) & (Hi <= 32) & (Si >= 30)).mean() if len(interior) else 1.0
            if wood_int > 0.35:
                continue
            strong = _color_vote(interior, min_frac=0.55)
            bright = float(np.median(interior[:, 2])) >= 90 if len(interior) else False
            if rim >= 0.20:
                color = _color_vote(interior, min_frac=0.25)
            elif strong is not None and (strong == "blue" or bright) and \
                    _dark_side_frac(hsv, cx, cy, r) >= 0.08:
                # 강한 색 근거 경로 — 어두운 기둥 단면과 혼동되는 건 빨강뿐이므로
                # 빨강은 밝기(조명 날림) 조건을 추가로 요구
                color = strong
            elif rim >= 0.10 and _color_vote(interior, min_frac=0.70) is not None:
                # 나무 링이 어느 정도 보이고 내부 색이 압도적이면 스톤
                # (기둥 단면은 링이 없어 rim≈0)
                color = _color_vote(interior, min_frac=0.70)
            else:
                continue
        if color is None:
            continue
        if ki is not None:
            used_knob[ki] = True
            cx, cy = knobs[ki]  # 꼭지 중심이 더 정확
        stones.append((float(cx), float(cy), color))

    # (a) 남은 꼭지 경로 (Hough가 놓친 스톤)
    # 넓은 프레이밍에서는 파스텔 보드의 장식 점이 꼭지로 잡히므로 나무 테두리를 요구한다.
    # 진짜 스톤 0.20~0.35 vs 오검출 0.00~0.03으로 확실히 갈린다.
    rim_min = GEO["knob_rim_min"]
    Himg, Wimg = hsv.shape[:2]
    for i, (kx, ky) in enumerate(knobs):
        if used_knob[i]:
            continue
        color = classify_ring(hsv, kx, ky)
        if not color:
            continue
        # 프레임에 잘린 스톤은 테두리 고리가 화면 밖이라 rim이 0 → 이때는 검사를 면제
        clipped = (kx < 68 or ky < 68 or kx > Wimg - 68 or ky > Himg - 68)
        if clipped or _wood_rim_frac(hsv, kx, ky, 65) >= rim_min:
            stones.append((kx, ky, color))

    # 중복 제거 (실제 스톤 간 최소 중심 거리는 지름 ~130px → 105px 이내는 같은 스톤)
    dedup = []
    for s in stones:
        if all((s[0] - t[0]) ** 2 + (s[1] - t[1]) ** 2 > 105 ** 2 for t in dedup):
            dedup.append(s)

    if Hm is not None:
        m = GEO["out_margin"] if out_margin is None else out_margin
        if m is not None:
            dedup = [t for t in dedup if not _outside_board(t[0], t[1], Hm, m)]
    return dedup


def silhouette_bottom(hsv, x, y, color, r=66):
    """스톤 실루엣(윗면 색 + 나무 옆면 + 그림자)의 최하단점 y — 접촉점 보정용.
    중심 수직 띠(|dx|<0.45r)에서 스톤 픽셀의 y 상위 97퍼센타일."""
    Himg, W = hsv.shape[:2]
    x, y = int(x), int(y)
    x0, x1 = max(0, x - int(0.45 * r)), min(W, x + int(0.45 * r) + 1)
    y0, y1 = max(0, y - int(0.3 * r)), min(Himg, y + int(1.7 * r) + 1)
    if x1 <= x0 or y1 <= y0:
        return float(y)
    patch = hsv[y0:y1, x0:x1]
    Hc = patch[..., 0].astype(int)
    Sc = patch[..., 1].astype(int)
    Vc = patch[..., 2].astype(int)
    if color == "red":
        top = ((Hc <= 8) | (Hc >= 160)) & (Sc > 55)
    else:
        top = (Hc >= 95) & (Hc <= 140) & (Sc > 40)
    wood = (Hc >= 9) & (Hc <= 32) & (Sc >= 30) & (Vc >= 80)
    shadow = Vc < 75
    mask = top | wood | shadow
    ys = np.nonzero(mask.any(axis=1))[0]
    if len(ys) == 0:
        return float(y)
    # 아래에서부터 연결된 스톤 영역의 끝: 행별 채움 비율이 유지되는 마지막 행
    rows = mask.mean(axis=1)
    solid = np.nonzero(rows > 0.25)[0]
    if len(solid) == 0:
        return float(y)
    return float(y0 + solid[-1])


def classify_ring(hsv, x, y, r_in=20, r_out=55, min_frac=0.22):
    """꼭지 주변 고리 색으로 판정"""
    return _color_vote(_annulus_px(hsv, int(x), int(y), r_in, r_out), min_frac)


def _border_stones(hsv, existing, band=8):
    """이미지 테두리에 걸려 잘린 스톤: 테두리에 닿은 진한 원판색 블롭을 찾는다"""
    Himg, W = hsv.shape[:2]
    H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    navy = ((H >= 95) & (H <= 135) & (S > 60) & (V >= 90) & (V < 200))
    maroon = (((H <= 8) | (H >= 165)) & (S > 80) & (V >= 40) & (V < 150))
    out = []
    for mask, color in [(navy, "blue"), (maroon, "red")]:
        m = cv2.morphologyEx(mask.astype(np.uint8) * 255, cv2.MORPH_OPEN,
                             np.ones((7, 7), np.uint8))
        n, _, stats, centroids = cv2.connectedComponentsWithStats(m)
        for i in range(1, n):
            x0, y0 = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP]
            w, h = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
            area = stats[i, cv2.CC_STAT_AREA]
            touches = x0 <= band or y0 <= band or x0 + w >= W - band or y0 + h >= Himg - band
            if not touches:
                continue
            if not (2500 <= area <= 20000) or max(w, h) > 170:
                continue
            if area / (w * h) < 0.35:
                continue
            cx, cy = centroids[i]
            if any((cx - s[0]) ** 2 + (cy - s[1]) ** 2 <= 105 ** 2 for s in existing + out):
                continue
            out.append((float(cx), float(cy), color))
    return out


# ---------- ③ 기하 점수 계산 ----------
def _cell_rects():
    """(x0,y0,x1,y1,value) 목록 — 보드 좌표(픽셀, 워프 공간)"""
    g = GEO
    bx, by = MARGIN, MARGIN
    c1, c2 = bx + g["c1"] * BW, bx + g["c2"] * BW
    rects = []
    # 좌열
    rects.append((bx, by, c1, by + g["h1L"] * BH, 4))
    rects.append((bx, by + g["h1L"] * BH, c1, by + g["h2L"] * BH, 3))
    rects.append((bx, by + g["h2L"] * BH, c1, by + BH, 2))
    # 중열 (2 위 / 1 아래)
    rects.append((c1, by, c2, by + g["hm"] * BH, 2))
    rects.append((c1, by + g["hm"] * BH, c2, by + BH, 1))
    # 우열
    rects.append((bx + g["c2"] * BW, by, bx + BW, by + g["h1R"] * BH, 4))
    rects.append((bx + g["c2"] * BW, by + g["h1R"] * BH, bx + BW, by + g["h2R"] * BH, 3))
    rects.append((bx + g["c2"] * BW, by + g["h2R"] * BH, bx + BW, by + BH, 2))
    return rects


def circle_rect_nearest(cx, cy, x0, y0, x1, y1):
    return min(max(cx, x0), x1), min(max(cy, y0), y1)


def touched_cells(wx, wy, stone_r):
    g = GEO
    ccx, ccy = MARGIN + g["circle_cx"] * BW, MARGIN + g["circle_cy"] * BH
    cr = g["circle_r"] * min(BW, BH)
    vals = set()
    if (wx - ccx) ** 2 + (wy - ccy) ** 2 <= (stone_r + cr) ** 2:
        vals.add(5)
    for (x0, y0, x1, y1, v) in _cell_rects():
        nx, ny = circle_rect_nearest(wx, wy, x0, y0, x1, y1)
        if (wx - nx) ** 2 + (wy - ny) ** 2 > stone_r ** 2:
            continue
        # 이 칸과의 실질 접촉점이 중앙 원 안이면 그 부분은 5 영역 → 칸 접촉으로 안 침
        if (nx - ccx) ** 2 + (ny - ccy) ** 2 <= cr ** 2:
            # 원 밖의 칸 영역과도 닿는지 대략 확인: 접촉원 위 8방향 샘플
            hit = False
            for a in np.linspace(0, 2 * np.pi, 12, endpoint=False):
                px_, py_ = wx + stone_r * np.cos(a), wy + stone_r * np.sin(a)
                if x0 <= px_ <= x1 and y0 <= py_ <= y1 and \
                   (px_ - ccx) ** 2 + (py_ - ccy) ** 2 > cr ** 2:
                    hit = True
                    break
            if not hit:
                continue
        vals.add(v)
    if not vals:
        return {0}
    return vals


# ---------- 통합 ----------
def process_image(img):
    quad = find_board_quad(img)
    if quad is None:
        return None
    Hm = get_homography(quad)
    stone_r = GEO["stone_r_ratio"] * BW
    stones = []
    total = 0
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    alpha = GEO.get("alpha", 0.0)
    for (x, y, color) in find_stones(img, Hm):
        # 윗면 중심 → 바닥 접촉 중심: 실루엣 최하단점 방향으로 α만큼 보정
        sb = silhouette_bottom(hsv, x, y, color)
        p = cv2.perspectiveTransform(
            np.array([[[x, y + alpha * (sb - y)]]], dtype=np.float64), Hm)[0, 0]
        wx, wy = float(p[0]), float(p[1])
        vals = touched_cells(wx, wy, stone_r)
        sc = max(vals)
        stones.append({"x": x, "y": y, "color": color, "wx": wx, "wy": wy,
                       "cells": vals, "score": sc})
        total += sc
    return {"quad": quad, "H": Hm, "stones": stones, "total": total}


def estimate_stone_radius_img(Hm, x, y, stone_r_warp):
    inv = np.linalg.inv(Hm)
    p0 = cv2.perspectiveTransform(np.array([[[x, y]]], dtype=np.float64), Hm)[0, 0]
    b = cv2.perspectiveTransform(
        np.array([[[p0[0] + stone_r_warp, p0[1]]]], dtype=np.float64), inv)[0, 0]
    return float(np.hypot(b[0] - x, b[1] - y))


# ---------- 시각화 ----------
def visualize(img, res, out_path):
    vis = img.copy()
    if res is None:
        cv2.putText(vis, "BOARD NOT FOUND", (30, 60), 0, 1.5, (0, 0, 255), 3)
        cv2.imwrite(out_path, vis)
        return
    cv2.polylines(vis, [res["quad"].astype(int)], True, (0, 255, 0), 3)
    for s in res["stones"]:
        c = (0, 0, 255) if s["color"] == "red" else (255, 100, 0)
        cv2.circle(vis, (int(s["x"]), int(s["y"])), 8, c, -1)
        cells = "".join(map(str, sorted(s["cells"], reverse=True)))
        for th, cc in [(4, (255, 255, 255)), (2, c)]:
            cv2.putText(vis, f"{cells}>{s['score']}", (int(s["x"]) - 40, int(s["y"]) - 20),
                        0, 0.9, cc, th)
    for th, cc in [(6, (255, 255, 255)), (3, (0, 200, 0))]:
        cv2.putText(vis, f"total={res['total']}", (30, 70), 0, 2, cc, th)

    size = (BW + 2 * MARGIN, BH + 2 * MARGIN)
    warp = cv2.warpPerspective(img, res["H"], size)
    for (x0, y0, x1, y1, v) in _cell_rects():
        cv2.rectangle(warp, (int(x0), int(y0)), (int(x1), int(y1)), (0, 255, 0), 2)
        cv2.putText(warp, str(v), (int(x0) + 8, int(y0) + 30), 0, 0.9, (0, 255, 0), 2)
    g = GEO
    cv2.circle(warp, (int(MARGIN + g["circle_cx"] * BW), int(MARGIN + g["circle_cy"] * BH)),
               int(g["circle_r"] * min(BW, BH)), (0, 255, 0), 2)
    for s in res["stones"]:
        c = (0, 0, 255) if s["color"] == "red" else (255, 100, 0)
        cv2.circle(warp, (int(s["wx"]), int(s["wy"])), int(g["stone_r_ratio"] * BW), c, 3)
        cv2.circle(warp, (int(s["wx"]), int(s["wy"])), 5, c, -1)
    h = max(vis.shape[0], warp.shape[0])
    canvas = np.zeros((h, vis.shape[1] + warp.shape[1], 3), np.uint8)
    canvas[:vis.shape[0], :vis.shape[1]] = vis
    canvas[:warp.shape[0], vis.shape[1]:] = warp
    cv2.imwrite(out_path, canvas)


if __name__ == "__main__":
    for name in ["001.jpg", "017.jpg", "025.jpg", "042.jpg", "085.jpg"]:
        img = cv2.imread(os.path.join(IMG_DIR, name))
        res = process_image(img)
        visualize(img, res, os.path.join(DEBUG, f"cv_{name}"))
        if res:
            print(f"{name}: total={res['total']}, stones={len(res['stones'])}, "
                  + "; ".join(f"{s['color']}:{sorted(s['cells'])}" for s in res["stones"]))
        else:
            print(f"{name}: board not found")
