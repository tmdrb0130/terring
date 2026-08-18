# -*- coding: utf-8 -*-
import os
import sys
import cv2
import terring_cv as tc

for name in sys.argv[1:]:
    img = cv2.imread(os.path.join(tc.IMG_DIR, name))
    res = tc.process_image(img)
    tc.visualize(img, res, os.path.join(tc.DEBUG, "chk_" + name))
    parts = "; ".join(f"{s['color']}:{sorted(s['cells'])}" for s in res["stones"])
    print(name, "total:", res["total"], "|", parts)
