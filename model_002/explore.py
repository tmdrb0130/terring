# -*- coding: utf-8 -*-
"""색상 프로브 + 보드 검출 1차 실험"""
import os
import cv2
import numpy as np

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(MODEL_DIR)
IMG_DIR = os.path.join(ROOT, "data", "001~085 (08.16)")
DEBUG = os.path.join(MODEL_DIR, "debug")

img = cv2.imread(os.path.join(IMG_DIR, "001.jpg"))
print("size:", img.shape)  # (H, W, 3)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# 원본이 1050x1400이라 가정하고 001.jpg에서 눈으로 짚은 지점들 (x, y)
points = {
    "red cell(4)": (255, 340),
    "red cell(2)": (565, 385),
    "blue cell(3)": (255, 600),
    "yellow circle(5)": (560, 620),
    "blue cell(1)": (560, 850),
    "pastel pink(2)": (545, 1080),
    "pastel purple(3)": (150, 1300),
    "stone1 maroon top": (370, 470),
    "stone1 pink knob": (396, 493),
    "stone2 navy top": (718, 480),
    "stone2 pink knob": (735, 492),
    "wood pin": (400, 900),
    "wood table": (900, 150),
}
print(f"{'name':<22}{'BGR':<18}{'HSV(med 9x9)':<16}")
for name, (x, y) in points.items():
    win_hsv = hsv[max(0,y-4):y+5, max(0,x-4):x+5].reshape(-1, 3)
    win_bgr = img[max(0,y-4):y+5, max(0,x-4):x+5].reshape(-1, 3)
    mh = np.median(win_hsv, axis=0).astype(int)
    mb = np.median(win_bgr, axis=0).astype(int)
    print(f"{name:<22}{str(mb):<18}{str(mh):<16}")

# ---- 1차 보드(선명한 3x3 영역) 마스크 ----
H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
red = ((H <= 10) | (H >= 168)) & (S > 110) & (V > 90)
blue = (H >= 95) & (H <= 130) & (S > 110) & (V > 60)
yellow = (H >= 18) & (H <= 40) & (S > 110) & (V > 140)
vivid = (red | blue | yellow).astype(np.uint8) * 255
vivid = cv2.morphologyEx(vivid, cv2.MORPH_CLOSE, np.ones((31, 31), np.uint8))
vivid = cv2.morphologyEx(vivid, cv2.MORPH_OPEN, np.ones((15, 15), np.uint8))

cnts, _ = cv2.findContours(vivid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
c = max(cnts, key=cv2.contourArea)
hull = cv2.convexHull(c)
peri = cv2.arcLength(hull, True)
quad = cv2.approxPolyDP(hull, 0.02 * peri, True)
print("\nboard contour area:", cv2.contourArea(c), "quad pts:", quad.reshape(-1, 2).tolist())

vis = img.copy()
cv2.drawContours(vis, [hull], -1, (0, 255, 0), 4)
for p in quad.reshape(-1, 2):
    cv2.circle(vis, tuple(p), 14, (0, 0, 255), -1)
cv2.imwrite(os.path.join(DEBUG, "explore_board.jpg"), vis)
cv2.imwrite(os.path.join(DEBUG, "explore_mask.jpg"), vivid)
print("saved: debug/explore_board.jpg, explore_mask.jpg")
