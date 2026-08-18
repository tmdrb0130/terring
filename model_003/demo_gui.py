# -*- coding: utf-8 -*-
r"""테링 자동 채점 GUI 데모 (model_003, 구 model_004)

이미지를 선택하면 5-fold 앙상블 + TTA로 추론해서
보드/스톤 시각화와 점수 판정을 화면에 바로 보여준다.

실행: ..\terring\Scripts\python.exe demo_gui.py
"""
import os
import threading
import tkinter as tk
from tkinter import filedialog

import cv2
import numpy as np
import torch
from PIL import Image, ImageTk

from train_cnn import (CFG, compare, load_human_labels, predict_maps, decode_outputs,
                       to_cells)
from infer import CELL_NAMES, annotate, load_ensemble, resolve_image

THR = 0.35
VIEW_W, VIEW_H = 420, 560  # 이미지 표시 영역 픽셀 크기 (3:4)


class App:
    def __init__(self, root):
        self.root = root
        root.title("테링 자동 채점 데모 — model_003 (CNN)")
        root.configure(bg="#1e1e2e")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.models = None
        self.human = load_human_labels()
        self.busy = False

        # ── 상단 바 ──
        top = tk.Frame(root, bg="#1e1e2e")
        top.pack(fill="x", padx=12, pady=(12, 6))
        self.btn = tk.Button(top, text="이미지 선택하고 채점하기", font=("맑은 고딕", 13, "bold"),
                             bg="#89b4fa", fg="#11111b", padx=16, pady=6,
                             state="disabled", command=self.pick_image)
        self.btn.pack(side="left")
        self.status = tk.Label(top, text=f"모델 로딩 중... (device: {self.device})",
                               font=("맑은 고딕", 11), bg="#1e1e2e", fg="#f9e2af")
        self.status.pack(side="left", padx=14)

        # ── 본문: 왼쪽 이미지 / 오른쪽 판정 결과 ──
        body = tk.Frame(root, bg="#1e1e2e")
        body.pack(fill="both", expand=True, padx=12, pady=(0, 12))
        holder = tk.Frame(body, bg="#11111b", width=VIEW_W, height=VIEW_H)
        holder.pack_propagate(False)  # Label 글자수 단위가 아니라 픽셀 크기로 고정
        holder.pack(side="left")
        self.canvas = tk.Label(holder, bg="#11111b", text="이미지를 선택하세요",
                               fg="#6c7086", font=("맑은 고딕", 12))
        self.canvas.pack(fill="both", expand=True)
        self.text = tk.Text(body, width=40, font=("맑은 고딕", 10), bg="#11111b",
                            fg="#cdd6f4", relief="flat", padx=12, pady=10, wrap="word")
        self.text.pack(side="left", fill="both", expand=True, padx=(12, 0))
        for tag, color, size, bold in [("h", "#89b4fa", 12, True), ("big", "#a6e3a1", 16, True),
                                       ("red", "#f38ba8", 10, True), ("blue", "#74c7ec", 10, True),
                                       ("dim", "#6c7086", 9, False), ("ok", "#a6e3a1", 11, True),
                                       ("bad", "#f38ba8", 11, True)]:
            self.text.tag_configure(tag, foreground=color,
                                    font=("맑은 고딕", size, "bold" if bold else "normal"))
        self.text.configure(state="disabled")

        threading.Thread(target=self.load_models, daemon=True).start()

    # ── 모델 로딩 (백그라운드) ──
    def load_models(self):
        models = load_ensemble(self.device)
        def done():
            self.models = models
            self.btn.configure(state="normal")
            self.status.configure(text=f"준비 완료 — 앙상블 {CFG['n_folds']}개 fold + TTA "
                                       f"(device: {self.device})", fg="#a6e3a1")
        self.root.after(0, done)

    # ── 이미지 선택 → 추론 ──
    def pick_image(self):
        if self.busy:
            return
        path = filedialog.askopenfilename(
            title="채점할 이미지 선택",
            initialdir=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                    "data"),
            filetypes=[("이미지", "*.jpg *.jpeg *.png"), ("모든 파일", "*.*")])
        if not path:
            return
        self.busy = True
        self.btn.configure(state="disabled")
        self.status.configure(text="추론 중... (CNN 앙상블 forward)", fg="#f9e2af")
        self.show_image(cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR))
        threading.Thread(target=self.run_infer, args=(path,), daemon=True).start()

    def run_infer(self, path):
        try:
            key, path = resolve_image(path)
            with torch.no_grad():
                hm_s, hm_c, off, reg = predict_maps(self.models, key, self.device, use_tta=True)
            stones, corners = decode_outputs(hm_s, hm_c, off, reg, THR)
            preds = to_cells(stones, corners)
            vis = annotate(path, stones, corners, preds)
            self.root.after(0, lambda: self.show_result(key, stones, preds, vis))
        except Exception as e:
            self.root.after(0, lambda: self.fail(e))

    def fail(self, e):
        self.busy = False
        self.btn.configure(state="normal")
        self.status.configure(text=f"오류: {e}", fg="#f38ba8")

    # ── 표시 ──
    def show_image(self, bgr):
        img = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        img.thumbnail((VIEW_W, VIEW_H))
        self.tkimg = ImageTk.PhotoImage(img)
        self.canvas.configure(image=self.tkimg, text="")

    def show_result(self, key, stones, preds, vis):
        self.show_image(vis)
        t = self.text
        t.configure(state="normal")
        t.delete("1.0", "end")
        t.insert("end", f"판정 결과 — {key}\n\n", "h")
        t.insert("end", "① 보드 코너 4점 검출 → 정면 보정\n", "dim")
        t.insert("end", f"② 스톤 검출: {len(stones)}개\n\n")
        total = 0
        for i, ((_x, _y, color), (_, cells)) in enumerate(zip(stones, preds), 1):
            score = max(cells)
            total += score
            cell_str = ", ".join(CELL_NAMES[c] for c in sorted(cells))
            t.insert("end", f"  ● 스톤{i} ", color if color in ("red", "blue") else None)
            t.insert("end", f"{color}\n     닿은 칸 {{{cell_str}}} → {score}점\n")
        t.insert("end", "\n③ 총점  ")
        t.insert("end", f"{total}점\n", "big")

        if key in self.human:
            lab = self.human[key]
            full, tot_ok, matched, n_lab, _tp, _tl = compare(preds, lab["stones"])
            t.insert("end", f"\n사람 라벨(정답): {lab['total']}점\n", "dim")
            t.insert("end", "완전 일치 (스톤 수+색+닿은 칸 전부): ")
            t.insert("end", f"{'O' if full else 'X'}", "ok" if full else "bad")
            t.insert("end", f"  (스톤 {matched}/{n_lab})\n", "dim")
            t.insert("end", "총점 일치: ", "dim")
            t.insert("end", f"{'O' if tot_ok else 'X'}\n", "ok" if tot_ok else "bad")
        else:
            t.insert("end", "\n(새 이미지 — 정답 라벨 없음)\n", "dim")
        t.configure(state="disabled")

        self.busy = False
        self.btn.configure(state="normal")
        self.status.configure(text="완료 — 다른 이미지도 선택해 보세요", fg="#a6e3a1")


if __name__ == "__main__":
    root = tk.Tk()
    App(root)
    root.mainloop()
