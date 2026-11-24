"""격자 판정기 G 의 정지 패턴(P1)을 실제·합성 클립에 — v1(팀원 원본)·v2(면적 수정) 판정과 비교. 판정만, 메모리에서만.

세트
  diag  : big1 합성 전 안전 바탕 중 v1 오탐 41 · v1 정상 41 · 합성 패턴 41 (data/judge/pattern_diag_rows.jsonl)
  south : South 2021 중 v1 이 패턴이라 한 것 + 사람 패턴 라벨 (data/judge/south_partial/rows.jsonl) + 안전·무판정 무작위 200(대조)
프레임 표본: 초당 8 프레임만 판정(정지 패턴의 0.5 초 지속 조건을 재기에 충분 — 구현 선택), 판정 프레임 사이는 같은 상태로 봄.
클립 판정: 아무 1 초 창 안에서 패턴 프레임이 0.5 초 이상이면 위험(규격 해설 Harding FPA 0.5 s).
출력: data/judge/grid_pattern_<세트>.jsonl, checks/judge/grid_pattern_<세트>.md

    python scripts/grid_pattern_eval.py --set diag|south
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from lumi_judge.cpu_limit import limit_cpu  # noqa: E402
limit_cpu()
from lumi_judge.grid_flash import luminance_cd  # noqa: E402
from lumi_judge.grid_pattern import GridPatternJudge  # noqa: E402

SAMPLE_FPS = 8.0
_J = {}


def judge_clip(x: np.ndarray, fps: float) -> dict:
    h, w = x.shape[1:3]
    if (h, w) not in _J:
        _J[(h, w)] = GridPatternJudge(h, w)
    G = _J[(h, w)]
    step = max(1, int(round(fps / SAMPLE_FPS)))
    idx = list(range(0, len(x), step))
    pat = np.zeros(len(x), bool)
    best = None
    for i in idx:
        lum = luminance_cd(torch.from_numpy(np.ascontiguousarray(x[i:i + 1])).cuda())[0]
        r = G.frame(lum)
        pat[i:i + step] = r["is_pat"]
        if r["is_pat"] and (best is None or r["area"] > best["area"]):
            best = dict(r, t=round(i / fps, 2))
    win = max(1, int(math.ceil(fps - 1e-9)))
    need = int(math.ceil(0.5 * fps - 1e-9))
    cs = np.concatenate([[0], np.cumsum(pat)])
    n = len(pat)
    counts = cs[np.minimum(np.arange(n) + win, n)] - cs[:n]
    danger = bool((counts >= need).any()) if n else False
    return {"danger": danger, "pat_frames": int(pat.sum()), "best": best}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="diag", choices=("diag", "south", "south_safe"))
    ap.add_argument("--judge", default="p1", choices=("p1", "motion"))   # motion = P1(흐름 면제) + P2(변화 패턴)
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    jobs = []
    if a.set == "diag":
        from make_hazard7_dataset import FPS, decode
        rows = [json.loads(l) for l in (ROOT / "data/judge/pattern_diag_rows.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
        seen = set()
        for r in rows:
            if r["cfg"] == "default" and r["id"] not in seen:
                seen.add(r["id"])
                v2 = next((q["pat_frames"] >= 12 for q in rows if q["id"] == r["id"] and q["cfg"] == "maskarea"), None)
                jobs.append((r["grp"], r["id"], lambda rid=r["id"]: (decode(ROOT / "data/hazard7/big1/train" / rid), FPS), r["pat_frames"] >= 12, v2))
    elif a.set == "south_safe":                                       # 일상 오탐률: South 사람 '안전' 무작위 300 (시드 0)
        from eval_real_hazards import decode as dec_any
        src = ROOT / "data/real/south2021/social-media-study-gifs"
        files = {p.stem: p for p in src.iterdir()}
        lab = [r for r in csv.DictReader(open(ROOT / "data/real/south2021/social-media-study.csv", encoding="utf-8", errors="ignore"))
               if r["dangerous"] == "safe" and r["label"] in files]
        rng = np.random.default_rng(0)
        for i in rng.choice(len(lab), 300, replace=False):
            k = lab[i]["label"]
            jobs.append(("safe", k, lambda k=k: dec_any(files[k]), None, None))
    else:
        from eval_real_hazards import decode as dec_any
        src = ROOT / "data/real/south2021/social-media-study-gifs"
        files = {p.stem: p for p in src.iterdir()}
        R = [json.loads(l) for l in (ROOT / "data/judge/south_partial/rows.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
        R = [r for r in R if "error" not in r]
        pick = [r for r in R if "정적 공간 패턴" in r["labels"] or r["human"]["pattern"] == "pattern"]
        rest = [r for r in R if r not in pick and r["human"]["dangerous"] == "safe"]
        rng = np.random.default_rng(0)
        pick += [rest[i] for i in rng.choice(len(rest), min(200, len(rest)), replace=False)]
        for r in pick:
            grp = "human_pattern" if r["human"]["pattern"] == "pattern" else ("safe" if r["human"]["dangerous"] == "safe" else "danger_other")
            jobs.append((grp, r["label"], lambda k=r["label"]: dec_any(files[k]), "정적 공간 패턴" in r["labels"], None))
    suf = "" if a.judge == "p1" else "_motion"
    out = ROOT / "data" / "judge" / f"grid_pattern_{a.set}{suf}.jsonl"
    done = {json.loads(l)["id"] for l in out.read_text(encoding="utf-8").splitlines() if l.strip()} if out.exists() else set()
    t0 = time.time()
    with out.open("a", encoding="utf-8") as f:
        for i, (grp, cid, load, v1, v2) in enumerate(jobs, 1):
            if cid in done:
                continue
            try:
                x, fps = load()
                if a.judge == "p1":
                    r = judge_clip(x, fps)
                    row = {"id": cid, "grp": grp, "v1": v1, "v2": v2, "G": r["danger"], "G_frames": r["pat_frames"], "G_best": r["best"]}
                else:
                    from lumi_judge.grid_motion import judge_pattern_motion
                    h, w = x.shape[1:3]
                    if (h, w) not in _J:
                        _J[(h, w)] = GridPatternJudge(h, w)
                    r = judge_pattern_motion(x, fps, G=_J[(h, w)])
                    p1 = judge_clip(x, fps)["danger"]                         # 같은 클립의 이전 G(P1 만) — 짝지은 비교
                    strip = lambda b: None if b is None else {k: v for k, v in b.items() if k != "cells"}
                    row = {"id": cid, "grp": grp, "v1": v1, "v2": v2, "G": p1, "G2": r["p1"] or r["p2"], "G2_p1": r["p1"], "G2_p2": r["p2"],
                           "best1": strip(r["best1"]), "best2": strip(r["best2"])}
            except Exception as e:  # noqa: BLE001
                row = {"id": cid, "grp": grp, "error": f"{type(e).__name__}: {e}"}
            f.write(json.dumps(row, ensure_ascii=False) + "\n"); f.flush()
            torch.cuda.empty_cache()
            if i % 20 == 0:
                print(f"  {i}/{len(jobs)} ({time.time() - t0:.0f}s)", flush=True)
    report(out, a.set + suf)


def report(out, name):
    R = [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines() if l.strip()]
    R = [r for r in R if "error" not in r]
    groups = sorted({r["grp"] for r in R})
    L = [f"# 격자 판정기 G — 정지 패턴, {name} 세트", "", "> 셀 = 그 그룹에서 '패턴 위험'으로 판정한 클립 수 / 전체. v2 = 면적 수정판(diag 만).", "",
         "| 그룹 | n | v1 원본 | v2 면적 수정 | G 격자(P1) | G2 (P1 흐름 면제 + P2) |", "|---|---|---|---|---|---|"]
    for g in groups:
        rs = [r for r in R if r["grp"] == g]
        v2 = [r for r in rs if r["v2"] is not None]
        g2 = [r for r in rs if "G2" in r]
        v1s = [r for r in rs if r["v1"] is not None]
        L.append(f"| {g} | {len(rs)} | {sum(r['v1'] for r in v1s) if v1s else '-'} | {sum(r['v2'] for r in v2) if v2 else '-'} | {sum(r['G'] for r in rs)} | "
                 f"{sum(r['G2'] for r in g2) if g2 else '-'} |")
    p = ROOT / "checks" / "judge" / f"grid_pattern_{name}.md"
    p.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
