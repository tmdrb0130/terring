# -*- coding: utf-8 -*-
"""분쟁 판정용 리뷰 페이지(HTML) 생성 — review_disputes 카드 이미지를 embed.

사용법: python build_review_page.py
출력: ../review_disputes/review.html  (Artifact 로 발행해서 브라우저에서 판정)
"""
import base64
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CARD_DIR = os.path.join(ROOT, "review_disputes")
OUT = os.path.join(CARD_DIR, "review.html")

# 2026-08-24 사용자 판정 반영 후 남은 오답 (model_004 앙상블+TTA, thr 0.05 / cells 0.65)
# 전부 사람 라벨이 맞다고 확정된 케이스 = 순수한 모델 오류
DISPUTES = [
    # (이미지, 라벨총점, 예측총점, 스톤색, 라벨cells, 예측cells, 쟁점, 분류)
    ("165", 14, 14, "red", "23", "123", "모델이 1점칸을 덧붙임", "tie"),
    ("216", 13, 13, "blue", "4", "24", "모델이 2점칸을 덧붙임", "tie"),
    ("329", 9, 9, "red", "12", "2", "모델이 1점칸을 빠뜨림", "tie"),
    ("437", 16, 16, "red", "2345", "235", "모델이 4점칸을 빠뜨림", "tie"),
    ("440", 16, 16, "red", "235", "1235", "모델이 1점칸을 덧붙임", "tie"),
    ("484", 4, 4, "blue", "234", "24", "모델이 3점칸을 빠뜨림", "tie"),
    ("578", 5, 5, "blue", "1235", "135", "모델이 2점칸을 빠뜨림", "tie"),
]

RESOLVED = [
    ("297", "모델이 옳음 &mdash; 라벨을 <span class='mono'>0</span> &rarr; "
            "<span class='mono'>2</span> 로 수정 (총점 18 &rarr; 20)"),
    ("412", "모델이 옳음 &mdash; 라벨을 <span class='mono'>1</span> &rarr; "
            "<span class='mono'>15</span> 로 수정 (총점 12 &rarr; 16)"),
    ("475", "모델이 옳음 &mdash; 라벨을 <span class='mono'>3</span> &rarr; "
            "<span class='mono'>23</span> 로 수정 (총점 불변)"),
    ("080", "모델 오류 &mdash; 프레임에 절반 잘린 스톤의 피크가 0.106 이라 "
            "임계값 0.2 에 걸려 버려졌다. 임계값을 0.05 로 낮춰 해결"),
]

KIND = {
    "score": ("총점까지 틀림", "critical"),
    "tie": ("총점은 정답", "warn"),
    "miss": ("스톤 미검출", "info"),
}

# 전부 2026-08-24 현재 라벨 기준으로 재측정 (model_003/eval_all.py + model_005/eval_test.py)
MODELS = [
    # (이름, 설명, 완전일치 순수91장, 비고) — 2026-08-25 폴더 재정리 후 현행 번호 기준
    ("model_002", "OpenCV + 기하 계산 (학습 없음)", 38.5, "규칙 기반 출발점"),
    ("구 model_003", "탐지 CNN + 기하 공식 (85장 학습, 폴더 제거)", 45.1, "CV를 스승으로 학습"),
    ("model_003", "구 003 + 데이터 583장", 47.3, "데이터 8배, +2.2%p뿐"),
    ("model_004", "칸 판정을 사람 라벨로 직접 학습", 89.0, "구조 전환 &mdash; 현행 최고"),
    ("model_005", "model_004 + 2단계 고해상도 crop", 89.0, "개선 없음"),
]


def b64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")


def main():
    cards = []
    for (name, tl, tp, color, lab, pred, issue, kind) in DISPUTES:
        p = os.path.join(CARD_DIR, f"{name}.jpg")
        if not os.path.exists(p):
            print("건너뜀(카드 없음):", name)
            continue
        kind_label, kind_cls = KIND[kind]
        color_ko = "빨강" if color == "red" else "파랑"
        cards.append(f"""
    <article class="card" id="img{name}">
      <header class="card-head">
        <div class="card-id">
          <span class="num">{name}</span>
          <span class="chip chip-{kind_cls}">{kind_label}</span>
        </div>
        <dl class="verdict">
          <div><dt>사람 라벨</dt><dd class="mono">{lab}</dd></div>
          <div class="arrow" aria-hidden="true">vs</div>
          <div><dt>모델 예측</dt><dd class="mono">{pred}</dd></div>
          <div class="tot"><dt>총점</dt><dd class="mono">{tl} / {tp}</dd></div>
        </dl>
        <p class="issue"><span class="dot dot-{color}"></span>{color_ko} 스톤 &mdash; {issue}</p>
      </header>
      <figure>
        <img src="data:image/jpeg;base64,{b64(p)}" alt="{name}.jpg 분쟁 스톤 확대" loading="lazy">
      </figure>
      <footer class="judge">
        <span>판정:</span>
        <span class="opt">라벨이 맞다 &rarr; 모델 오류</span>
        <span class="opt">모델이 맞다 &rarr; 라벨 수정</span>
        <span class="opt">애매 &rarr; 기준 정의 필요</span>
      </footer>
    </article>""")

    rows = []
    best = max(m[2] for m in MODELS)
    for (nm, desc, full, note) in MODELS:
        cls = " row-best" if full == best else ""
        # 막대 = 성능 비율 (표를 눈으로 훑을 때 격차가 바로 보이도록)
        bar = f'<span class="bar" style="--w:{full:.0f}%"></span>'
        rows.append(f"""<tr class="{cls.strip()}"><th scope="row">{nm}</th><td>{desc}</td>"""
                    f"""<td class="num-cell"><span class="mono">{full:.1f}%</span>{bar}</td>"""
                    f"""<td class="note">{note}</td></tr>""")

    html = TEMPLATE.replace("{{CARDS}}", "\n".join(cards)) \
                   .replace("{{ROWS}}", "\n".join(rows))
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(html)
    size = os.path.getsize(OUT) / 1024 / 1024
    print(f"생성 완료: {OUT} ({size:.1f} MB, 카드 {len(cards)}장)")


TEMPLATE = r"""<title>터링 라벨 분쟁 11건</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Gowun+Batang:wght@400;700&family=IBM+Plex+Sans+KR:wght@300;400;500;600&family=IBM+Plex+Mono:wght@400;600&display=swap">
<style>
  :root {
    --ground: #F5F1E8;
    --surface: #FFFFFF;
    --surface-2: #EFE9DC;
    --ink: #241C15;
    --ink-2: #5A4F45;
    --ink-3: #8B8178;
    --rule: #DDD4C4;
    --red: #B5352C;
    --blue: #2A548F;
    --gold: #B8912A;
    --wood: #A8794C;
    --crit-bg: #F7E2DE;  --crit-ink: #92281F;
    --warn-bg: #F2EAD3;  --warn-ink: #7A5F14;
    --info-bg: #E3E9F2;  --info-ink: #2A548F;
  }
  @media (prefers-color-scheme: dark) {
    :root:not([data-theme="light"]) {
      --ground: #161210;
      --surface: #1F1A16;
      --surface-2: #29221C;
      --ink: #F0E8DC;
      --ink-2: #C0B4A6;
      --ink-3: #8D8175;
      --rule: #3A302A;
      --red: #E0776B;
      --blue: #7FA6DC;
      --gold: #D9B85C;
      --wood: #C79A6B;
      --crit-bg: #3B211D;  --crit-ink: #F0A79B;
      --warn-bg: #362D18;  --warn-ink: #DCC172;
      --info-bg: #1F2A38;  --info-ink: #9DBCE4;
    }
  }
  :root[data-theme="dark"] {
    --ground: #161210;
    --surface: #1F1A16;
    --surface-2: #29221C;
    --ink: #F0E8DC;
    --ink-2: #C0B4A6;
    --ink-3: #8D8175;
    --rule: #3A302A;
    --red: #E0776B;
    --blue: #7FA6DC;
    --gold: #D9B85C;
    --wood: #C79A6B;
    --crit-bg: #3B211D;  --crit-ink: #F0A79B;
    --warn-bg: #362D18;  --warn-ink: #DCC172;
    --info-bg: #1F2A38;  --info-ink: #9DBCE4;
  }

  * { box-sizing: border-box; }
  body {
    margin: 0;
    background: var(--ground);
    color: var(--ink);
    font-family: "IBM Plex Sans KR", system-ui, sans-serif;
    font-weight: 300;
    line-height: 1.7;
    -webkit-font-smoothing: antialiased;
  }
  .wrap { max-width: 1180px; margin: 0 auto; padding: 0 24px 96px; }
  .mono { font-family: "IBM Plex Mono", ui-monospace, monospace; font-variant-numeric: tabular-nums; }

  header.page {
    padding: 72px 0 40px;
    border-bottom: 2px solid var(--ink);
  }
  .eyebrow {
    font-size: .78rem; letter-spacing: .16em; text-transform: uppercase;
    color: var(--ink-3); font-weight: 500; margin: 0 0 14px;
  }
  h1 {
    font-family: "Gowun Batang", Georgia, serif;
    font-weight: 700; font-size: clamp(2.1rem, 5vw, 3.2rem);
    line-height: 1.15; margin: 0 0 18px; text-wrap: balance;
  }
  .lede {
    font-size: 1.06rem; color: var(--ink-2); max-width: 62ch; margin: 0;
  }

  h2 {
    font-family: "Gowun Batang", Georgia, serif;
    font-size: 1.7rem; font-weight: 700; margin: 64px 0 6px;
  }
  .sub { color: var(--ink-3); font-size: .95rem; margin: 0 0 24px; }

  .tablewrap { overflow-x: auto; }
  table { border-collapse: collapse; width: 100%; font-size: .95rem; }
  th, td { text-align: left; padding: 13px 16px; border-bottom: 1px solid var(--rule); }
  thead th {
    font-size: .76rem; letter-spacing: .1em; text-transform: uppercase;
    color: var(--ink-3); font-weight: 600; border-bottom: 1.5px solid var(--ink);
  }
  tbody th { font-weight: 600; white-space: nowrap; }
  .num-cell { text-align: right; font-weight: 600; white-space: nowrap; }
  .num-cell .mono { font-size: 1.02rem; }
  .bar {
    display: block; height: 3px; margin-top: 5px; margin-left: auto;
    width: var(--w); max-width: 100%; background: var(--wood); opacity: .5;
  }
  .row-best .bar { background: var(--gold); opacity: 1; }
  .row-best th, .row-best td { background: var(--surface-2); }
  .note { color: var(--ink-3); font-size: .88rem; }
  tbody tr:last-child td, tbody tr:last-child th { border-bottom: none; }

  .grid-cards { display: flex; flex-direction: column; gap: 40px; margin-top: 8px; }
  .card {
    background: var(--surface);
    border: 1px solid var(--rule);
    border-radius: 3px;
    overflow: hidden;
  }
  .card-head { padding: 22px 26px 18px; border-bottom: 1px solid var(--rule); }
  .card-id { display: flex; align-items: baseline; gap: 14px; flex-wrap: wrap; }
  .num {
    font-family: "Gowun Batang", Georgia, serif;
    font-size: 1.9rem; font-weight: 700; letter-spacing: .02em;
  }
  .chip {
    font-size: .74rem; font-weight: 600; letter-spacing: .06em;
    padding: 4px 11px; border-radius: 2px;
  }
  .chip-critical { background: var(--crit-bg); color: var(--crit-ink); }
  .chip-warn { background: var(--warn-bg); color: var(--warn-ink); }
  .chip-info { background: var(--info-bg); color: var(--info-ink); }

  .verdict {
    display: flex; align-items: flex-end; gap: 22px; flex-wrap: wrap;
    margin: 16px 0 12px;
  }
  .verdict div { display: flex; flex-direction: column; gap: 2px; }
  .verdict dt {
    font-size: .72rem; letter-spacing: .1em; text-transform: uppercase;
    color: var(--ink-3); font-weight: 600;
  }
  .verdict dd { margin: 0; font-size: 1.5rem; font-weight: 600; }
  .verdict .arrow {
    color: var(--ink-3); font-size: .85rem; padding-bottom: 8px;
    font-style: italic;
  }
  .verdict .tot { margin-left: auto; }
  .verdict .tot dd { font-size: 1.1rem; color: var(--ink-2); }

  .issue { margin: 0; font-size: .97rem; color: var(--ink-2); display: flex;
           align-items: center; gap: 9px; }
  .dot { width: 11px; height: 11px; border-radius: 50%; flex: none;
         border: 1.5px solid rgba(0,0,0,.25); }
  .dot-red { background: var(--red); }
  .dot-blue { background: var(--blue); }

  figure { margin: 0; background: var(--surface-2); }
  figure img { display: block; width: 100%; height: auto; }

  .judge {
    display: flex; gap: 10px; flex-wrap: wrap; align-items: center;
    padding: 16px 26px; font-size: .87rem; color: var(--ink-3);
    border-top: 1px solid var(--rule);
  }
  .judge span:first-child { font-weight: 600; color: var(--ink-2); }
  .opt {
    border: 1px solid var(--rule); border-radius: 2px; padding: 5px 12px;
  }

  .callout {
    background: var(--surface-2); border-left: 3px solid var(--gold);
    padding: 20px 24px; margin: 28px 0 0; font-size: .97rem; color: var(--ink-2);
  }
  .callout strong { color: var(--ink); font-weight: 600; }

  @media (max-width: 640px) {
    .verdict .tot { margin-left: 0; }
    .card-head { padding: 18px 18px 14px; }
    .judge { padding: 14px 18px; }
  }
</style>

<div class="wrap">
  <header class="page">
    <p class="eyebrow">2026년 8월 24일 &middot; test 110장</p>
    <h1>모델과 사람이 엇갈린 11장</h1>
    <p class="lede">
      model_004가 test 110장 중 99장을 완전히 맞혔습니다. 남은 11장은 모두
      <strong>스톤 하나가 인접한 한 칸에 닿았는지</strong>를 두고 모델과 라벨이 갈린 경우입니다.
      해상도를 올려도, TTA를 늘려도, 임계값을 조정해도 풀리지 않았습니다.
      각 카드의 확대 사진을 보고 어느 쪽이 맞는지 판정해 주세요.
    </p>
  </header>

  <h2>모델별 성능</h2>
  <p class="sub">
    완전 일치 = 스톤 수 + 각 스톤의 색 + 닿은 칸 집합이 전부 일치.
    학습에 한 번도 쓰지 않은 순수 test 91장 기준, 2026-08-24 현재 라벨로 전부 재측정.
  </p>
  <div class="tablewrap">
    <table>
      <thead>
        <tr><th>모델</th><th>방식</th><th class="num-cell">완전 일치</th><th>비고</th></tr>
      </thead>
      <tbody>
        {{ROWS}}
      </tbody>
    </table>
  </div>
  <div class="callout">
    <strong>읽는 법.</strong> 완전 일치는 사진 한 장의 스톤을 전부 맞혀야 1점입니다.
    스톤 하나만 어긋나도 그 사진이 통째로 탈락하므로, 스톤 단위 정확도(97.7%)보다 훨씬 낮게 나옵니다.
    실제 채점에 쓰이는 <strong>총점 일치는 별개로 98.2%</strong>(test 110장 중 108장)입니다 &mdash;
    아래 11건 중 9건은 닿은 칸이 갈렸어도 최고점이 같아 총점은 정답이기 때문입니다.
  </div>

  <h2>분쟁 11건</h2>
  <p class="sub">
    노란 원이 분쟁 스톤입니다. 오른쪽은 그 스톤을 원본 해상도로 확대한 것이고,
    왼쪽은 전체 보드에서의 위치입니다.
  </p>
  <div class="grid-cards">
    {{CARDS}}
  </div>
</div>
"""


if __name__ == "__main__":
    main()
