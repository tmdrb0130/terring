# -*- coding: utf-8 -*-
"""train 473장의 경계 분쟁 스톤 검토 카드 생성 (좌표 기반 매칭).

각 fold 의 val 이미지를 그 fold 모델(out-of-fold, 학습 때 못 본 모델)로 예측해서
사람 라벨과 cells 가 갈리는 스톤을 찾는다.

매칭: stone_map 의 사람 검수 좌표와 검출 좌표를 같은 색끼리 거리 기반으로 짝지음.
  (이전 버전은 색+cells 로만 짝을 지어서, 같은 색 스톤 2개의 예측이 겹치면
   분쟁 표시가 엉뚱한 스톤에 가는 버그가 있었다 — 425 에서 사용자 발견)

검출 개수 차이(가짜 피크/미검출)는 단일 모델 노이즈라 제외 — cells 분쟁만.
피크 임계값은 0.2 사용 (0.05 는 앙상블+TTA 전제라 단일 모델에선 가짜 피크가 섞임).

사용법: python review_train_disputes.py
출력: ../review_train_disputes/*.jpg + 콘솔 목록 + 구버전 대비 원 위치 변경 목록
"""
import json
import os

import torch

import train_cnn as t5
from train_cnn import (CFG, OUT_DIR, ROOT, SX, SY, DetNet, compute_maps, decode_outputs,
                       load_human_labels, load_split, make_folds)
from infer import CELLS_THR
from review_disputes import fmt_cells, make_card, match_and_find_disputes

AUDIT_THR = 0.2
MAX_DIST = 70.0   # 검출↔stone_map 매칭 허용 거리 (원본 px, 스톤 반지름 ~80)
CARD_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "review_train_disputes")


def load_stone_map():
    raw = json.load(open(os.path.join(ROOT, "data", "stone_map.json"), encoding="utf-8"))
    out = {}
    for name, rows in raw.items():
        out[name] = [(s["s_idx"], float(s["x"]), float(s["y"]), s["color"],
                      frozenset({0}) if s["cells"] == "0"
                      else frozenset(int(c) for c in s["cells"]))
                     for s in rows]
    return out


def match_by_coord(dets, map_rows):
    """같은 색끼리 거리순 그리디 매칭 → [(det, map_row)]"""
    cands = []
    for i, (x, y, c, _cs) in enumerate(dets):
        ox, oy = x / SX, y / SY
        for j, (_s, mx, my, mc, _mcs) in enumerate(map_rows):
            if c != mc:
                continue
            d = ((ox - mx) ** 2 + (oy - my) ** 2) ** 0.5
            if d <= MAX_DIST:
                cands.append((d, i, j))
    cands.sort()
    used_i, used_j, pairs = set(), set(), []
    for d, i, j in cands:
        if i in used_i or j in used_j:
            continue
        used_i.add(i)
        used_j.add(j)
        pairs.append((dets[i], map_rows[j]))
    return pairs


def main():
    os.makedirs(CARD_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    human = load_human_labels()
    stone_map = load_stone_map()
    folds = make_folds(load_split("train.csv"), CFG["n_folds"], CFG["seed"])
    print(f"device: {device} | OOF thr={AUDIT_THR} cells={CELLS_THR} | 좌표 매칭", flush=True)

    found = []
    for i in range(CFG["n_folds"]):
        ck = torch.load(os.path.join(OUT_DIR, f"fold{i}_best.pt"),
                        map_location=device, weights_only=False)
        m = DetNet().to(device)
        m.load_state_dict(ck["state_dict"])
        m.eval()
        maps = compute_maps([m], folds[i], device, use_tta=False)
        for name in folds[i]:
            if name not in stone_map:
                continue
            dets = decode_outputs(*maps[name], AUDIT_THR, CELLS_THR)
            pairs = match_by_coord(dets, stone_map[name])
            disp = []
            for det, (sidx, mx, my, mc, mcs) in pairs:
                lab_cs = frozenset() if mcs == frozenset({0}) else mcs
                if det[3] != lab_cs:
                    disp.append((det, (mc, mcs), sidx))
            if disp:
                found.append((name, dets, disp))
        del maps

    print(f"\ncells 분쟁 {len(found)}장:", flush=True)
    moved = []
    for name, dets, disp in sorted(found):
        # 구버전(색+cells 매칭)이라면 어떤 스톤에 원을 그렸을지 계산해 비교
        old_pairs = match_and_find_disputes(dets, human[name]["stones"])
        old_disp = [(d, l) for d, l in old_pairs if d is None or l is None
                    or d[3] != (frozenset() if l[1] == frozenset({0}) else l[1])]
        old_pos = {(round(d[0]), round(d[1])) for d, _l in old_disp if d is not None}
        new_pos = {(round(d[0]), round(d[1])) for d, _l, _s in disp}
        if old_pos != new_pos:
            moved.append(name)

        out = os.path.join(CARD_DIR, f"{os.path.splitext(name)[0]}.jpg")
        make_card(name, t5.img_path(name), dets, human[name]["stones"], out,
                  disputes=[(d, l) for d, l, _s in disp])
        for det, (mc, mcs), sidx in disp:
            a = frozenset() if mcs == frozenset({0}) else mcs
            diff = "".join(map(str, sorted(a ^ det[3]))) or "-"
            one = "한칸" if len(a ^ det[3]) == 1 else "복수"
            print(f"  {name:<9} {sidx:<3} {mc:<5} 라벨 {fmt_cells(mcs):<5} "
                  f"모델 {fmt_cells(det[3]):<5} 쟁점칸 {diff:<3} [{one}]", flush=True)

    print(f"\n구버전 카드에서 원 위치가 틀렸던 이미지 {len(moved)}장: {moved}")
    print(f"카드 저장: {CARD_DIR}")


if __name__ == "__main__":
    main()
