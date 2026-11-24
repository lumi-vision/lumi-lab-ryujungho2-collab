"""패턴 공간 주파수(cpd) 대역 비교 — 사전 등록(이 머리말이 판단 규칙, 실행 전 커밋, 2026-11-25).

질문: 방사형 무늬 등이 '1–10 cpd' 대역 때문에 놓인다(진단: val 015/018 바깥 살 0.3–0.9 cpd). 대역을 무엇으로 둘 것인가?
변형(모두 G2 = P1 흐름 면제 + P2, 다른 조건 동일)
  A  1–10 cpd   팀원 값 — 출처 없음(PATTERN_LITERATURE.md). 이전 결과와의 비교용 참고.
  B  1–8 cpd    Fisher 외 2026 EF 합의("about 50% of patients", Wilkins 1979 인용) — 'Fisher 축'(단, 면적·쌍 수는 BT.1702 그대로인 혼합)
  C  하한 없음   BT.1702-3 Att.1 원문(cpd 조건 없음) — 'BT.1702 축'. 구현 하한 0.15 cpd = 56 cm·15인치 화면 너비(≈33°)에 6 쌍이
                 들어갈 수 있는 최저 주파수(0.18)보다 조금 아래 → 실제 하한은 쌍 수 규칙이 정한다. 상한도 없음(표시 해상도 0.9 나이퀴스트까지).
판단 규칙(결과를 보고 바꾸지 않음)
  1. 대역은 **근거로** 정한다: 1 층은 C(BT.1702 축)와 B(Fisher 축)를 **둘 다** 보고하고, 출처 없는 A 는 버린다(참고로만 표시).
     실험은 대역을 고르는 데 쓰지 않고, 각 축의 **비용(오탐)과 이득(회복)을 보이는** 데 쓴다.
  2. 일상 영상(진단 바탕 82 + South 안전 297)에서 C 가 B 보다 더 낸 '위험'은 전부 뷰어로 사람이 정지 화면을 확인해
     (가) 규격상 실제 줄무늬(> 5 쌍·ΔL ≥ 20·면적) → C 가 맞음(사람 라벨 '안전'과 규격이 다름을 기록)
     (나) 아님 → 판정기 결함으로 원인을 찾아 고친다(고친 뒤 다시 전체 측정, 변경 기록).
  3. 적합성: 규격 21 사례 + cpd 사례(0.3·0.5·0.8·9 cpd, 면적 60 % 직선) — 정답은 축마다 다름(C: 모두 위험, B: 1–8 밖은 안전).
세트: 적합성 · 합성 정지 무늬 위험(train 진단 22 + 조정에 안 쓴 val/test 20, 정답 = 생성 의도) · 일상 379.
출력: checks/judge/cpd_sweep.md · data/viewer/cpd/ (판정이 갈린 클립의 정지 프레임 + 근거 칸, 로컬 전용 — South 재배포 금지라 깃에 넣지 않음)
"""
from __future__ import annotations

import csv
import glob
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from lumi_judge.cpu_limit import limit_cpu  # noqa: E402
limit_cpu()
import torch  # noqa: E402
from lumi_judge.grid_motion import judge_pattern_motion  # noqa: E402
from lumi_judge.grid_pattern import GridPatternJudge  # noqa: E402

VARIANTS = {"A": (1.0, 10.0), "B": (1.0, 8.0), "C": (0.15, 1e3)}
VIEW = ROOT / "data" / "viewer" / "cpd"
_G: dict = {}


def G_of(v, h, w):
    k = (v, h, w)
    if k not in _G:
        if len(_G) > 6:
            _G.clear(); torch.cuda.empty_cache()
        _G[k] = GridPatternJudge(h, w, cpd_lo=VARIANTS[v][0], cpd_hi=VARIANTS[v][1])
    return _G[k]


def judge_all(x, fps):
    out = {}
    h, w = x.shape[1:3]
    for v in VARIANTS:
        r = judge_pattern_motion(x, fps, G=G_of(v, h, w))
        b = r["best1"] or r["best2"] or r["best_any"]
        out[v] = {"danger": bool(r["p1"] or r["p2"]), "p1": r["p1"], "p2": r["p2"],
                  "t": None if b is None else b["t"], "cpd": None if b is None else round(b["cpd"], 2),
                  "pairs": None if b is None else b["pairs"],
                  "area": None if b is None else round(b["area_static"] + b["area_change"], 3),
                  "cells": [] if b is None else b["cells"]["static"] + b["cells"]["change"], "shape": [G_of(v, h, w).rows, 32]}
    return out


def save_view(cid, x, fps, res, why):
    d = VIEW / cid.replace("/", "_")
    d.mkdir(parents=True, exist_ok=True)
    step = max(1, int(round(fps / 4)))                                 # 초당 4 장 정지 프레임(자동 재생 없음)
    idx = list(range(0, len(x), step))[:64]
    for i in idx:
        cv2.imwrite(str(d / f"f{i:04d}.jpg"), x[i], [cv2.IMWRITE_JPEG_QUALITY, 85])
    meta = {"id": cid, "fps": fps, "frames": idx, "w": int(x.shape[2]), "h": int(x.shape[1]), "why": why,
            "res": {v: {k: r[k] for k in ("danger", "p1", "p2", "t", "cpd", "pairs", "area", "cells", "shape")} for v, r in res.items()}}
    (d / "meta.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")


def conformance():
    import pattern_conformance as PC
    rows = []
    cases = [(n, kw) for n, _, kw in PC.cases()] + [(f"cpd {c}", dict(cpd=c, area=0.6)) for c in (0.3, 0.5, 0.8, 9.0)]
    for name, kw in cases:
        L, ar = PC.stripes(**kw)
        u8 = PC.to_u8(L)
        x = np.repeat(u8[None], int(PC.FPS), 0)
        res = judge_all(x, PC.FPS)
        base = PC.truth_of(kw, ar)
        c = kw.get("cpd", 3.0)
        truth = {"A": base and 1.0 <= c <= 10.0, "B": base and 1.0 <= c <= 8.0, "C": base}
        rows.append({"case": f"{name} [실면적 {100 * ar:.0f}%]", "truth": truth, "res": {v: res[v]["danger"] for v in VARIANTS},
                     "cpd_meas": {v: res[v]["cpd"] for v in VARIANTS}})
        print(" ", rows[-1]["case"], {v: (truth[v], res[v]["danger"]) for v in VARIANTS}, flush=True)
    return rows


def synth_static():
    from make_hazard7_dataset import FPS, decode
    jobs = []
    for split in ("train", "val", "test"):
        rec = {}
        for f in glob.glob(str(ROOT / f"data/hazard7/big1/{split}/manifest.s*.jsonl")):
            for l in open(f, encoding="utf-8"):
                r = json.loads(l); rc = r.get("recipe", r)
                rec[str(r.get("id") or r.get("name") or r.get("file", "")).replace(".mp4", "")] = rc
        diag = None
        if split == "train":                                           # train 은 진단 세트(합성 양성 41)에 든 것만
            dr = [json.loads(l) for l in (ROOT / "data/judge/pattern_diag_rows.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
            diag = {r["id"].replace(".mp4", "") for r in dr if r["grp"] == "pos"}
        for cid, rc in sorted(rec.items()):
            if rc.get("kind") != "pattern" or rc.get("intent") != "hazard" or any(rc.get("pat_drift", [0, 0])):
                continue
            if diag is not None and cid not in diag:
                continue
            p = ROOT / f"data/hazard7/big1/{split}/{cid}.mp4"
            if p.exists():
                jobs.append((f"{split}/{cid}", lambda p=p: (decode(p), FPS), rc.get("pattern")))
    return jobs


def daily():
    from eval_real_hazards import decode as dec_any
    from make_hazard7_dataset import FPS, decode
    jobs = []
    dr = [json.loads(l) for l in (ROOT / "data/judge/pattern_diag_rows.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    seen = set()
    for r in dr:
        if r["grp"] in ("fp", "tn") and r["id"] not in seen:
            seen.add(r["id"])
            jobs.append((f"bg/{r['id']}", lambda rid=r["id"]: (decode(ROOT / "data/hazard7/big1/train" / rid), FPS), "bg"))
    src = ROOT / "data/real/south2021/social-media-study-gifs"
    files = {p.stem: p for p in src.iterdir()}
    lab = [r for r in csv.DictReader(open(ROOT / "data/real/south2021/social-media-study.csv", encoding="utf-8", errors="ignore"))
           if r["dangerous"] == "safe" and r["label"] in files]
    rng = np.random.default_rng(0)                                     # grid_pattern_eval --set south_safe 와 같은 300
    for i in rng.choice(len(lab), 300, replace=False):
        k = lab[i]["label"]
        jobs.append((f"south/{k}", lambda k=k: dec_any(files[k]), "south"))
    return jobs


def run(jobs, name, out_rows):
    cache = ROOT / "data" / "judge" / f"cpd_sweep_{name}.jsonl"
    done = {json.loads(l)["id"] for l in cache.read_text(encoding="utf-8").splitlines() if l.strip()} if cache.exists() else set()
    with cache.open("a", encoding="utf-8") as f:
        for i, (cid, load, tag) in enumerate(jobs, 1):
            if cid in done:
                continue
            try:
                x, fps = load()
                res = judge_all(x, fps)
                row = {"id": cid, "tag": tag, "res": {v: {k: r[k] for k in ("danger", "p1", "p2", "t", "cpd", "pairs", "area")} for v, r in res.items()}}
                ds = {v: res[v]["danger"] for v in VARIANTS}
                if len(set(ds.values())) > 1 or name == "synth":           # 판정이 갈렸거나 합성 위험 → 뷰어
                    save_view(cid, x, fps, res, f"{name}: " + " ".join(f"{v}={'위험' if d else '안전'}" for v, d in ds.items()))
            except Exception as e:  # noqa: BLE001
                row = {"id": cid, "tag": tag, "error": f"{type(e).__name__}: {e}"}
            f.write(json.dumps(row, ensure_ascii=False) + "\n"); f.flush()
            torch.cuda.empty_cache()
            if i % 25 == 0:
                print(f"  {name} {i}/{len(jobs)}", flush=True)
    out_rows[name] = [json.loads(l) for l in cache.read_text(encoding="utf-8").splitlines() if l.strip()]


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    conf = conformance()
    R = {}
    run(synth_static(), "synth", R)
    run(daily(), "daily", R)
    yn = lambda b: "위험" if b else "안전"
    L = ["# 패턴 cpd 대역 비교 (사전 등록 = scripts/cpd_sweep.py 머리말)", "",
         "> A 1–10(출처 없음, 참고) · **B 1–8 (Fisher 2026 EF 합의 축)** · **C 하한 없음 (BT.1702-3 원문 축)** — 나머지 조건 동일(G2)", "",
         "## 1. 적합성 (정답은 축마다: C = 원문, B = 1–8 밖 안전)", "",
         "| 사례 | A | B | C | 측정 cpd (C) |", "|---|---|---|---|---|"]
    for r in conf:
        mk = lambda v: yn(r["res"][v]) + ("" if r["res"][v] == r["truth"][v] else " ✗")
        L.append(f"| {r['case']} | {mk('A')} | {mk('B')} | {mk('C')} | {r['cpd_meas']['C']} |")
    acc = {v: sum(r["res"][v] == r["truth"][v] for r in conf) for v in VARIANTS}
    L += ["", f"정답 수: A {acc['A']}/{len(conf)} · B {acc['B']}/{len(conf)} · C {acc['C']}/{len(conf)}", ""]
    for name, title in (("synth", "2. 합성 정지 무늬 위험 (정답 = 위험, train 진단 + val/test)"), ("daily", "3. 일상 영상 (정답 = 안전; C 의 추가 '위험'은 뷰어로 사람 확인)")):
        rs = [r for r in R[name] if "error" not in r]
        L += [f"## {title}", "", f"n = {len(rs)} (오류 {len(R[name]) - len(rs)})", "", "| 묶음 | n | A 위험 | B 위험 | C 위험 |", "|---|---|---|---|---|"]
        tags = sorted({r["tag"] for r in rs}) + ["전체"]
        for t in tags:
            q = rs if t == "전체" else [r for r in rs if r["tag"] == t]
            L.append(f"| {t} | {len(q)} | " + " | ".join(str(sum(r["res"][v]["danger"] for r in q)) for v in VARIANTS) + " |")
        diff = [r for r in rs if len({r["res"][v]["danger"] for v in VARIANTS}) > 1]
        L += ["", f"판정이 갈린 클립 {len(diff)}: " + ", ".join(f"`{r['id']}`(" + "/".join(f"{v}{'●' if r['res'][v]['danger'] else '○'}" for v in VARIANTS)
                                                         + f", C cpd {r['res']['C']['cpd']})" for r in diff[:40]), ""]
    (ROOT / "checks" / "judge" / "cpd_sweep.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
# 변경 기록
# 2026-11-25 1 차 실행 후: C(하한 없음)가 적합성 '어두운 막대 155' 를 놓침 — 넓힌 필터 묶음에서 저주파 필터가 줄무늬 영역의 밝기 계단에 반응해
#   줄무늬 대신 덩어리를 고름(확인: 패턴 안 선택 cpd 중앙 0.15). → GridPatternJudge 가 하한 < 1 이면 두 패스(1 cpd 이상 필터 · 저주파 필터,
#   둘 다 쌍은 넓은 범위로 셈)로 찾아 면적이 큰 쪽(grid_pattern.py). 결과를 본 뒤의 변경이므로 1 차 결과(data/judge/cpd_v1)와 함께 보고.
