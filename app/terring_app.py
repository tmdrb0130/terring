# -*- coding: utf-8 -*-
"""터링 채점기 — 사진 한 장을 고르면 스톤별 점수와 총점을 보여 주는 실행 프로그램.

실행:
  터링채점기.exe                               GUI
  터링채점기.exe --cli 사진.jpg 결과.json       창 없이 채점 (결과 JSON + 표시 이미지 저장)
  터링채점기.exe --gui-test 사진.jpg 화면.png   GUI 를 띄워 자동 채점 후 창을 캡처하고 종료 (빌드 점검용)
"""
import json
import os
import sys
import threading
import traceback


def app_dir():
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))


def log_error(text):
    try:
        with open(os.path.join(app_dir(), "error_log.txt"), "a", encoding="utf-8") as f:
            f.write(text + "\n")
    except OSError:
        pass


def run_cli(image, out_json):
    import terring_core as core
    scorer = core.Scorer()
    img = core.read_image(image)
    res = scorer.score_image(img)
    vis_path = os.path.splitext(out_json)[0] + "_result.jpg"
    core.write_image(vis_path, core.annotate(img, res))
    res["device"] = scorer.device
    res["result_image"] = vis_path
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)


def run_gui(test_image=None, test_shot=None):
    import tkinter as tk
    from tkinter import filedialog, messagebox

    import cv2
    from PIL import Image, ImageTk

    BG, PANEL, FG = "#1e1e2e", "#11111b", "#cdd6f4"
    FONT = "맑은 고딕"

    # 고해상도(배율 125~200%) 화면에서 흐릿하지 않게 DPI 인식을 켜고, 사진 영역을 배율에 맞춘다.
    try:
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass
    root = tk.Tk()
    scale = root.winfo_fpixels("1i") / 96.0
    VIEW_H = int(min(640 * scale, root.winfo_screenheight() - 260 * scale))
    VIEW_W = VIEW_H * 3 // 4

    class App:
        def __init__(self, root):
            self.root = root
            root.title("터링 자동 채점기")
            root.configure(bg=BG)
            self.scorer = None
            self.core = None
            self.busy = False
            self.picked_once = False
            self.last_vis = None
            self.last_name = None

            top = tk.Frame(root, bg=BG)
            top.pack(fill="x", padx=12, pady=(12, 6))
            self.btn = tk.Button(top, text="사진 선택하고 채점하기", font=(FONT, 13, "bold"),
                                 bg="#89b4fa", fg="#11111b", padx=16, pady=6,
                                 state="disabled", command=self.pick_image)
            self.btn.pack(side="left")
            self.save_btn = tk.Button(top, text="결과 사진 저장", font=(FONT, 11),
                                      bg="#313244", fg=FG, padx=12, pady=6,
                                      state="disabled", command=self.save_result)
            self.save_btn.pack(side="left", padx=(8, 0))
            self.status = tk.Label(top, text="모델을 불러오는 중입니다... (처음 한 번, 수십 초)",
                                   font=(FONT, 11), bg=BG, fg="#f9e2af")
            self.status.pack(side="left", padx=14)

            body = tk.Frame(root, bg=BG)
            body.pack(fill="both", expand=True, padx=12, pady=(0, 12))
            holder = tk.Frame(body, bg=PANEL, width=VIEW_W, height=VIEW_H)
            holder.pack_propagate(False)
            holder.pack(side="left")
            self.canvas = tk.Label(holder, bg=PANEL, fg="#6c7086", font=(FONT, 12),
                                   text="보드 전체가 보이게 세로로 찍은 사진을\n선택하세요")
            self.canvas.pack(fill="both", expand=True)
            self.text = tk.Text(body, width=42, font=(FONT, 10), bg=PANEL, fg=FG,
                                relief="flat", padx=12, pady=10, wrap="word")
            self.text.pack(side="left", fill="both", expand=True, padx=(12, 0))
            for tag, color, size, bold in [("h", "#89b4fa", 12, True), ("big", "#a6e3a1", 20, True),
                                           ("red", "#f38ba8", 11, True), ("blue", "#74c7ec", 11, True),
                                           ("dim", "#6c7086", 9, False), ("warn", "#f9e2af", 10, False)]:
                self.text.tag_configure(tag, foreground=color,
                                        font=(FONT, size, "bold" if bold else "normal"))
            self.text.configure(state="disabled")
            threading.Thread(target=self.load_models, daemon=True).start()

        # ── 모델 로딩 (백그라운드) ──
        def load_models(self):
            try:
                import terring_core as core

                def progress(i, n):
                    self.root.after(0, lambda: self.status.configure(
                        text=f"모델을 불러오는 중입니다... ({i}/{n})"))
                scorer = core.Scorer(progress=progress)
            except Exception as e:
                err = traceback.format_exc()
                log_error(err)
                self.root.after(0, lambda: self.fatal(e))
                return

            def done():
                self.core, self.scorer = core, scorer
                self.btn.configure(state="normal")
                dev = "그래픽카드(GPU)" if scorer.device == "cuda" else "CPU"
                self.status.configure(text=f"준비 완료 ({dev} 사용)", fg="#a6e3a1")
                if test_image:
                    self.start(test_image)
            self.root.after(0, done)

        def fatal(self, e):
            self.status.configure(text="모델을 불러오지 못했습니다", fg="#f38ba8")
            messagebox.showerror("터링 자동 채점기",
                                 f"모델을 불러오지 못했습니다.\n\n{e}\n\n"
                                 "프로그램 폴더를 통째로 다시 복사해 보세요.")

        # ── 사진 선택 → 채점 ──
        def pick_image(self):
            if self.busy:
                return
            opts = {}
            sample = os.path.join(app_dir(), "테스트용 이미지")
            if not self.picked_once and os.path.isdir(sample):
                opts["initialdir"] = sample      # 처음 한 번은 같이 들어 있는 샘플 사진 폴더를 연다
            path = filedialog.askopenfilename(
                title="채점할 사진 선택",
                filetypes=[("사진", "*.jpg *.jpeg *.png *.bmp"), ("모든 파일", "*.*")], **opts)
            if path:
                self.picked_once = True
                self.start(path)

        def start(self, path):
            self.busy = True
            self.btn.configure(state="disabled")
            self.save_btn.configure(state="disabled")
            self.status.configure(text="채점 중입니다...", fg="#f9e2af")
            threading.Thread(target=self.run_infer, args=(path,), daemon=True).start()

        def run_infer(self, path):
            try:
                img = self.core.read_image(path)
                self.root.after(0, lambda: self.show_image(img))
                res = self.scorer.score_image(img)
                vis = self.core.annotate(img, res)
                self.root.after(0, lambda: self.show_result(os.path.basename(path), res, vis))
            except Exception as e:
                log_error(traceback.format_exc())
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

        def show_result(self, name, res, vis):
            core = self.core
            self.last_vis, self.last_name = vis, name
            self.show_image(vis)
            t = self.text
            t.configure(state="normal")
            t.delete("1.0", "end")
            t.insert("end", f"{name}\n\n", "h")
            t.insert("end", "총점  ")
            t.insert("end", f"{res['total']}점\n\n", "big")
            stones = res["stones"]
            t.insert("end", f"스톤 {len(stones)}개\n\n", "dim")
            if not stones:
                t.insert("end", "스톤을 찾지 못했습니다.\n", "warn")
            for s in stones:
                t.insert("end", f"● S{s['index']} {core.COLOR_NAMES[s['color']]}", s["color"])
                t.insert("end", f"   {s['score']}점\n")
                t.insert("end", f"    닿은 칸: {core.cells_text(s['cells'])}\n", "dim")
                if s["uncertain"]:
                    t.insert("end", "    ※ 칸 경계에 걸쳐 판정이 애매합니다. 직접 확인하세요.\n",
                             "warn")
                t.insert("end", "\n")
            if res["landscape"]:
                t.insert("end", "※ 가로로 찍힌 사진입니다. 이 프로그램은 세로 사진으로 "
                                "학습되어 결과가 부정확할 수 있습니다.\n\n", "warn")
            t.insert("end", "한 스톤이 여러 칸에 닿으면 가장 높은 점수를 줍니다.\n"
                            "사진의 S1, S2... 가 위 목록의 번호입니다.\n", "dim")
            t.configure(state="disabled")
            self.busy = False
            self.btn.configure(state="normal")
            self.save_btn.configure(state="normal")
            self.status.configure(text="채점 완료", fg="#a6e3a1")
            if test_shot:
                self.root.after(800, self.capture_and_quit)

        def save_result(self):
            if self.last_vis is None:
                return
            base = os.path.splitext(self.last_name)[0]
            path = filedialog.asksaveasfilename(
                title="결과 사진 저장", defaultextension=".jpg",
                initialfile=f"{base}_채점결과.jpg", filetypes=[("JPEG", "*.jpg"), ("PNG", "*.png")])
            if not path:
                return
            try:
                self.core.write_image(path, self.last_vis)
                self.status.configure(text="결과 사진을 저장했습니다", fg="#a6e3a1")
            except Exception as e:
                messagebox.showerror("터링 자동 채점기", f"저장하지 못했습니다.\n\n{e}")

        def capture_and_quit(self):
            from PIL import ImageGrab
            r = self.root
            r.lift()
            r.attributes("-topmost", True)
            r.update()
            x, y = r.winfo_rootx(), r.winfo_rooty()
            ImageGrab.grab(bbox=(x, y, x + r.winfo_width(), y + r.winfo_height())).save(test_shot)
            r.destroy()

    App(root)
    root.mainloop()


def main():
    args = sys.argv[1:]
    try:
        if len(args) == 3 and args[0] == "--cli":
            run_cli(args[1], args[2])
        elif len(args) == 3 and args[0] == "--gui-test":
            run_gui(args[1], args[2])
        else:
            run_gui()
    except Exception:
        log_error(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
