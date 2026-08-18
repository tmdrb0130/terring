# -*- coding: utf-8 -*-
import os
import cv2
import terring_cv as tc

img = cv2.imread(os.path.join(tc.IMG_DIR, "054.jpg"))
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
gray = cv2.medianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 5)
cc = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1, minDist=85,
                      param1=110, param2=28, minRadius=48, maxRadius=82)
vis = img.copy()
knobs = tc._knob_blobs(hsv)
for kx, ky in knobs:
    cv2.drawMarker(vis, (int(kx), int(ky)), (255, 0, 255), cv2.MARKER_CROSS, 30, 4)
n = 0 if cc is None else len(cc[0])
print("hough circles:", n, "/ knobs:", len(knobs))
if cc is not None:
    for (cx, cy, r) in cc[0]:
        rim = tc._wood_rim_frac(hsv, cx, cy, r)
        col = tc._color_vote(tc._annulus_px(hsv, cx, cy, 0, r * 0.62))
        ok = rim >= 0.18
        c = (0, 255, 0) if ok else (0, 0, 255)
        cv2.circle(vis, (int(cx), int(cy)), int(r), c, 3)
        label = f"{rim:.2f}" + (col[0] if col else "-")
        cv2.putText(vis, label, (int(cx) - 35, int(cy)), 0, 0.7, (255, 255, 255), 2)
cv2.imwrite(os.path.join(tc.DEBUG, "hough_054.jpg"), vis)
print("saved")
