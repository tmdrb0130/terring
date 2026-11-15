# -*- coding: utf-8 -*-
r"""터링채점기.exe 빌드 스크립트.

준비 (저장소 루트에서 한 번):
  py -3.12 -m venv .venv_build
  .venv_build\Scripts\python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cpu
  .venv_build\Scripts\python -m pip install numpy==2.5.2 opencv-python==5.0.0.93 pillow==12.3.0 pyinstaller

빌드:
  .venv_build\Scripts\python app\build_exe.py

결과: dist\터링채점기\ 폴더 (이 폴더를 통째로 전달). CPU 전용 torch 로 묶어 그래픽카드 없이 동작한다.
"""
import os
import shutil

import PyInstaller.__main__

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
NAME = "터링채점기"

args = [
    os.path.join(HERE, "terring_app.py"),
    "--name", NAME,
    "--noconfirm", "--clean", "--windowed",
    "--paths", HERE,
    "--distpath", os.path.join(ROOT, "dist"),
    "--workpath", os.path.join(ROOT, "build"),
    "--specpath", os.path.join(ROOT, "build"),
]
for model in ("model_004", "model_005"):
    out = os.path.join(ROOT, model, "output")
    for i in range(5):
        args += ["--add-data", f"{os.path.join(out, f'fold{i}_best.pt')};models/{model}"]
    args += ["--add-data", f"{os.path.join(out, 'config.json')};models/{model}"]

PyInstaller.__main__.run(args)

shutil.copy(os.path.join(HERE, "사용법.txt"), os.path.join(ROOT, "dist", NAME, "사용법.txt"))
# 받는 사람이 바로 넣어 볼 수 있게 샘플 사진을 같이 넣는다.
shutil.copytree(os.path.join(HERE, "sample"), os.path.join(ROOT, "dist", NAME, "테스트용 이미지"),
                dirs_exist_ok=True)
# 중간 작업 폴더에도 같은 이름의 exe 가 생기는데 그건 실행되지 않는다 — 헷갈리지 않게 지운다.
shutil.rmtree(os.path.join(ROOT, "build"), ignore_errors=True)
print("완료:", os.path.join(ROOT, "dist", NAME))
