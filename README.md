# terring — 점수판 자동 채점 AI

terring 보드(3x3 점수판 + 중앙 5원 x2)에 스톤(red/blue 최대 6개)을 굴린 사진을 찍으면
스톤별 닿은 칸과 총점을 자동 판정하는 프로젝트.

## 저장소 구성

| 위치 | 내용 | git 포함 |
|---|---|---|
| `model_001`~`model_005` | 모델별 코드 + 학습된 체크포인트(`output/*.pt`) | O |
| `data/*.csv`, `data/stone_map.json` | 사람 검수 라벨 (583장 + 빈보드 10장) | O |
| `model_003/train.csv`, `test.csv`, `labels.json` | 학습/평가 분할과 좌표 라벨 (전 모델 공용) | O |
| `docs/` | 프로젝트 문서 (개요·모델 상세·작업일지·로드맵·촬영 가이드) | O |
| `data/**/*.jpg` | 원본 이미지 (약 460MB) | **X — USB로 복사** |

모델 번호는 현행 폴더명이 기준 (2026-08-25 재정리: 구 004→003, 구 005→004, 구 006→005).

## 새 컴퓨터에서 시작하기

1. 저장소 클론 후 **USB의 `data` 이미지 폴더들을 그대로 `data/` 아래에 복사**
   (`001~085 (08.16)`, `086~583 (08.19)`, `584~593 (08.25)` — 이미지가 라벨 CSV와 같은 위치에 있어야 함)
2. 가상환경 생성 (venv 이름 `terring` — 코드가 `..\terring\Scripts\python.exe`를 가정):
   ```
   python -m venv terring
   terring\Scripts\pip install -r requirements.txt
   ```
3. 동작 확인:
   ```
   cd model_004
   ..\terring\Scripts\python.exe eval_test.py       # test 110장 평가 (체크포인트 사용, 재학습 없음)
   ..\terring\Scripts\python.exe infer.py 033.jpg   # 단일 이미지 추론
   ..\terring\Scripts\python.exe demo_gui.py        # GUI 데모
   ```
4. 전 모델 비교: 루트에서 `terring\Scripts\python.exe eval_all_models.py`

## 현행 성능 (test 117장, 완전 일치 기준)

- model_004 (탐지 CNN + cells 직접 학습): 89.7%
- model_005 (2단계 고해상도 crop): 91.5% — fold 편차 고려 시 004와 동급
- 남은 오답은 대부분 스톤-칸 경계 분쟁 (docs/04 로드맵 참고)

## 주의

- OpenCV 5.0은 `imdecode`가 EXIF 회전을 자동 적용함. 이미지의 EXIF Orientation 태그를
  초기화/제거하지 말 것 (태그가 회전 정보를 담당).
- 콘솔 출력은 cp949 환경 기준 — 스크립트 출력에 특수문자 금지.
