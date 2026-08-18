# -*- coding: utf-8 -*-
"""검출 후보 오버레이: Hough 원(초록=채택 경로 후보), 꼭지(십자), 최종 스톤(굵은 원)"""
import os
import sys
import cv2
import terring_cv as tc

for name in sys.argv[1:]:
    img = cv2.imread(os.path.join(tc.IMG_DIR, name))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gray = cv2.medianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 5)
    cc = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1, minDist=85,
                          param1=110, param2=28, minRadius=48, maxRadius=82)
    vis = img.copy()
    knobs = tc._knob_blobs(hsv)
    for kx, ky in knobs:
        cv2.drawMarker(vis, (int(kx), int(ky)), (255, 0, 255), cv2.MARKER_CROSS, 34, 4)
    if cc is not None:
        for (cx, cy, r) in cc[0]:
            rim = tc._wood_rim_frac(hsv, cx, cy, r)
            col = tc._color_vote(tc._annulus_px(hsv, cx, cy, 0, r * 0.62), 0.04)
            cv2.circle(vis, (int(cx), int(cy)), int(r), (0, 255, 255), 2)
            lab = f"{rim:.2f}" + (col[0] if col else "-")
            cv2.putText(vis, lab, (int(cx) - 35, int(cy) - int(r) - 6), 0, 0.7, (0, 255, 255), 2)
    stones = tc.find_stones(img)
    for (x, y, color) in stones:
        c = (0, 0, 255) if color == "red" else (255, 120, 0)
        cv2.circle(vis, (int(x), int(y)), 66, c, 4)
    cv2.imwrite(os.path.join(tc.DEBUG, "det_" + name), vis)
    print(name, "hough:", 0 if cc is None else len(cc[0]), "knobs:", len(knobs),
          "stones:", len(stones))
