# -*- coding: utf-8 -*-
"""583장 전체를 다기준 균형 8:2 분할 → model_003/train.csv, test.csv

기존 분할의 문제:
  · 1차(001~085)/2차(086~583) 촬영분이 섞여 있고, 옛 모델들이 1차를 학습에 썼던 탓에
    "순수 test" 같은 별도 집계가 필요했다. 전 모델을 재학습하면 그 구분은 불필요.
  · 층화 기준이 (보드 위 스톤 수, 걸침 수) 둘뿐이라 스톤 개수 등 다른 축이 틀어졌다.

이 스크립트는 여러 축의 주변분포를 동시에 맞춘다:
  스톤 개수 / 보드 위 스톤 수 / 보드 밖 스톤 수 / 걸침 총수 /
  한 스톤이 걸치는 최대 칸 수 / 중앙 5원 접촉 여부 / 총점 구간 / 촬영 회차

방법: 복합 층화로 초기 분할 → 교환(swap) 언덕오르기로 전 축의 분포 격차 합을 최소화.
seed 고정이라 재현 가능. test 비율 20%.

사용법: python make_split.py [--test-ratio 0.2] [--seed 42] [--iters 200000]
"""
import argparse
import csv
import os
import random
import shutil
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "data", "terring_1_583.csv")
OUT_DIR = os.path.join(ROOT, "model_003")     # 전 모델이 load_split 으로 읽는 위치


def feats(r):
    n = on = off = overlap = total = 0
    maxcells = 0
    five = 0
    for i in range(1, 7):
        color = (r[f"s{i}_color"] or "").strip()
        cells = (r[f"s{i}_cells"] or "").strip()
        if not color:
            continue
        n += 1
        st = set(int(c) for c in cells)
        s = max(st)
        total += s
        if s > 0:
            on += 1
            overlap += len(st) - 1
            maxcells = max(maxcells, len(st))
            if 5 in st:
                five = 1
        else:
            off += 1
    return {"n": n, "on": on, "off": off, "overlap": overlap,
            "total": total, "maxcells": maxcells, "five": five}


def axes(r):
    """균형을 맞출 축들 → [(축이름, 값)]"""
    f = feats(r)
    ov = "0" if f["overlap"] == 0 else ("1-2" if f["overlap"] <= 2 else "3+")
    tb = "0-5" if f["total"] <= 5 else ("6-12" if f["total"] <= 12 else "13+")
    batch = "1차" if int(r["image"].split(".")[0]) < 86 else "2차"
    return [("스톤개수", f["n"]), ("보드위", f["on"]), ("보드밖", f["off"]),
            ("걸침", ov), ("최대칸수", f["maxcells"]), ("5원접촉", f["five"]),
            ("총점", tb), ("회차", batch)]


def stratum(r):
    f = feats(r)
    nb = "N0-3" if f["n"] <= 3 else ("N4-5" if f["n"] <= 5 else "N6")
    ob = "B0-1" if f["on"] <= 1 else ("B2-3" if f["on"] <= 3 else "B4+")
    vb = "V0" if f["overlap"] == 0 else ("V1-2" if f["overlap"] <= 2 else "V3+")
    return f"{nb}/{ob}/{vb}"


def divergence(tr_c, te_c, ntr, nte, keys):
    """전 축 주변분포의 비율 격차 합 (작을수록 균형)"""
    d = 0.0
    for k in keys:
        d += abs(tr_c[k] / ntr - te_c[k] / nte)
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--iters", type=int, default=200000)
    args = ap.parse_args()

    rows = list(csv.DictReader(open(SRC, encoding="utf-8-sig")))
    fields = list(rows[0].keys())
    n_test = round(len(rows) * args.test_ratio)
    rng = random.Random(args.seed)
    print(f"전체 {len(rows)}장 -> train {len(rows) - n_test} / test {n_test}", flush=True)

    ax = {r["image"]: axes(r) for r in rows}
    keys = sorted({kv for a in ax.values() for kv in a})

    # 1) 복합 층화 초기 분할 (큰 잔여부터 채워 정확히 n_test 확보)
    groups = defaultdict(list)
    for r in rows:
        groups[stratum(r)].append(r)
    frac = n_test / len(rows)
    test, leftovers = [], []
    for k in sorted(groups):
        g = groups[k][:]
        rng.shuffle(g)
        exact = len(g) * frac
        take = int(exact)
        test += g[:take]
        leftovers.append((exact - take, g[take:]))
    leftovers.sort(reverse=True, key=lambda t: t[0])
    i = 0
    while len(test) < n_test and i < len(leftovers):
        if leftovers[i][1]:
            test.append(leftovers[i][1][0])
        i += 1
    test_set = {r["image"] for r in test}
    train = [r for r in rows if r["image"] not in test_set]

    tr_c = Counter(kv for r in train for kv in ax[r["image"]])
    te_c = Counter(kv for r in test for kv in ax[r["image"]])
    ntr, nte = len(train), len(test)
    cur = divergence(tr_c, te_c, ntr, nte, keys)
    print(f"초기 분포 격차 {cur:.4f}", flush=True)

    # 2) 교환 언덕오르기 — train/test 한 장씩 맞바꿔 격차가 줄면 채택
    improved = 0
    for _ in range(args.iters):
        ti = rng.randrange(ntr)
        si = rng.randrange(nte)
        a, b = train[ti], test[si]
        if a["image"] == b["image"]:
            continue
        for kv in ax[a["image"]]:
            tr_c[kv] -= 1
            te_c[kv] += 1
        for kv in ax[b["image"]]:
            te_c[kv] -= 1
            tr_c[kv] += 1
        new = divergence(tr_c, te_c, ntr, nte, keys)
        if new < cur - 1e-12:
            train[ti], test[si] = b, a
            cur = new
            improved += 1
        else:                       # 되돌리기
            for kv in ax[a["image"]]:
                tr_c[kv] += 1
                te_c[kv] -= 1
            for kv in ax[b["image"]]:
                te_c[kv] += 1
                tr_c[kv] -= 1
    print(f"교환 {improved}회 채택 -> 최종 격차 {cur:.4f}", flush=True)

    train.sort(key=lambda r: int(r["image"].split(".")[0]))
    test.sort(key=lambda r: int(r["image"].split(".")[0]))

    for f in ("train.csv", "test.csv"):
        p = os.path.join(OUT_DIR, f)
        if os.path.exists(p):
            for i in range(1, 50):
                b = f"{p}.bak{i}"
                if not os.path.exists(b):
                    shutil.copy2(p, b)
                    break
    for f, rr in [("train.csv", train), ("test.csv", test)]:
        with open(os.path.join(OUT_DIR, f), "w", newline="", encoding="utf-8-sig") as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            w.writerows(rr)

    # 3) 축별 분포 검증
    print(f"\n{'축':<10}{'값':<8}{'train%':>9}{'test%':>9}{'차이':>9}")
    worst = 0.0
    for axis in ["스톤개수", "보드위", "보드밖", "걸침", "최대칸수", "5원접촉", "총점", "회차"]:
        vals = sorted({v for (a, v) in keys if a == axis}, key=str)
        for v in vals:
            x = tr_c[(axis, v)] / ntr * 100
            y = te_c[(axis, v)] / nte * 100
            worst = max(worst, abs(y - x))
            flag = " <<" if abs(y - x) >= 5 else ""
            print(f"{axis:<10}{str(v):<8}{x:>8.1f}%{y:>8.1f}%{y - x:>+8.1f}%p{flag}")
    print(f"\n최대 격차 {worst:.1f}%p  (5%p 미만이면 균형으로 간주)")
    print(f"저장: {OUT_DIR}/train.csv ({ntr}장), test.csv ({nte}장)")


if __name__ == "__main__":
    main()
