"""정지 패턴(P1) 씨앗 수·고르는 방식 민감도 — 사전 등록(이 머리말이 판단 규칙, 실행 전 커밋, 2026-11-26).

배경: G 는 영역마다 씨앗 ≤ n 개(기본 12, 구현 선택 — 근거 없음)로 경로를 그려 '확인 비율'을 구하고, 면적 = 원면적 × 확인 비율.
  확인 비율은 n 개 표본의 비율이라 표본 오차가 있고(진짜 비율 0.6 이면 n = 12 에서 표준오차 ≈ 0.14), 씨앗은 칸의 진폭 최대 픽셀(치우침).
  예비 확인(실행 전, 한 프레임): 실제 동심원 줄무늬 원면적 54.5 % 가 n = 6 '최대' 에서 확인 4/6 → 36 % → 안전(틀림), n = 12 에서 위험.
판정기: BT.1702 축(cpd 하한 없음, 두 패스) G P1, 초당 8 프레임 표본, 1 초 창 0.5 초 지속.
변형: n ∈ {6, 12, 24, 48} × 방식 ∈ {max(칸 고르게 + 진폭 최대 픽셀, 현재), random(칸 무작위 + 칸 안 줄무늬 픽셀 무작위)}.
세트: 적합성 21(정지) · 합성 정지 무늬 위험 42(cpd 실험과 같은 목록) · 일상 379(진단 바탕 82 + South 안전 297).
효율: 씨앗은 원면적 > 40 % 인 영역이 있는 프레임에서만 결과에 영향 → 기준(12, max)으로 '씨앗 민감 프레임'을 찾고 그 프레임만 변형으로 다시 판정
      (민감하지 않은 프레임은 어느 변형이든 '패턴 아님' — 코드상 동일).
측정
  A 결정적: 변형마다 적합성 정답 수, 합성 탐지 수, 일상 위험 수, 프레임당 시간.
  B 안정성: random 방식으로 난수 시드 0–9 의 10 번 반복 → 클립 판정이 반복마다 갈리는 클립 비율(뒤집힘률, 씨앗 민감 클립 대상),
            프레임별 면적 추정의 표준편차(원면적 > 40 % 프레임).
판단 규칙(결과 보고 바꾸지 않음)
  1. 후보: 적합성 21/21 이고, 일상 위험 수 ≤ 기준(12, max) 이고, 합성 탐지 ≥ 기준 − 1 인 변형.
  2. 그중 random 방식 뒤집힘률 ≤ 5 % 를 만족하는 가장 작은 n 을 채택(같은 n 이면 치우침 없는 random 우선, 단 결정성이 필요하면 고정 난수 시드).
  3. 48 에서도 뒤집힘률 > 5 % 면 고정 n 은 기각하고 '적응형'(확인 비율 Wilson 95 % 구간이 40 % 문턱을 걸치면 씨앗 추가)을 다음 실험으로.
출력: checks/judge/seeds_sweep.md · data/judge/seeds_sweep.json
"""
from __future__ import annotations

import csv
import glob
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

NS = (6, 12, 24, 48)
MODES = ("max", "random")
SAMPLE_FPS = 8.0
R = 10
_G: dict = {}


def G_of(h, w):
    if (h, w) not in _G:
        if len(_G) > 3:
            _G.clear(); torch.cuda.empty_cache()
        _G[(h, w)] = GridPatternJudge(h, w, cpd_lo=0.15, cpd_hi=1e3)
    return _G[(h, w)]


def clip_verdict(flags, fps, step):
    T = len(flags) * step
    pat = np.zeros(T, bool)
    for k, f in enumerate(flags):
        pat[k * step:(k + 1) * step] = f
    win = max(1, int(math.ceil(fps - 1e-9))); need = int(math.ceil(0.5 * fps - 1e-9))
    cs = np.concatenate([[0], np.cumsum(pat)])
    return bool(((cs[np.minimum(np.arange(T) + win, T)] - cs[:T]) >= need).any())


def run_clip(x, fps):
    h, w = x.shape[1:3]
    G = G_of(h, w)
    step = max(1, int(round(fps / SAMPLE_FPS)))
    idx = list(range(0, len(x), step))
    lums, base, sens = [], [], []
    t0 = time.time()
    for i in idx:
        lum = luminance_cd(torch.from_numpy(np.ascontiguousarray(x[i:i + 1])).cuda())[0]
        b = G.frame(lum)                                               # 기준 (12, max)
        base.append(b["is_pat"])
        if b["regions"]:
            sens.append(len(lums)); lums.append((len(base) - 1, lum))
    out = {"n_frames": len(idx), "n_sens": len(sens), "base": clip_verdict(base, fps, step), "det": {}, "rand": {}, "area_sd": {}, "sec": {}}
    if not sens:
        for n in NS:
            for m in MODES:
                out["det"][f"{n}_{m}"] = out["base"]
            out["rand"][str(n)] = [out["base"]] * R
        return out
    for n in NS:
        for m in MODES:
            flags = list(base); t1 = time.time()
            for k, lum in lums:
                flags[k] = G.frame(lum, n_seeds=n, seed_mode=m, rng=np.random.default_rng(0))["is_pat"]
            out["det"][f"{n}_{m}"] = clip_verdict(flags, fps, step)
            out["sec"][f"{n}_{m}"] = (time.time() - t1) / len(lums)
        verd, sds = [], []
        areas = {k: [] for k, _ in lums}
        for r in range(R):
            rng = np.random.default_rng(1000 + r)
            flags = list(base)
            for k, lum in lums:
                f = G.frame(lum, n_seeds=n, seed_mode="random", rng=rng)
                flags[k] = f["is_pat"]
                ra = [a * v for a, v, _ in f["regions"]]
                areas[k].append(max(ra) if ra else 0.0)
            verd.append(clip_verdict(flags, fps, step))
        out["rand"][str(n)] = verd
        sds = [float(np.std(v)) for v in areas.values() if len(v) == R]
        out["area_sd"][str(n)] = float(np.median(sds)) if sds else 0.0
    return out


def jobs():
    from eval_real_hazards import decode as dec_any
    from make_hazard7_dataset import FPS, decode
    J = []
    for l in open(ROOT / "data/judge/cpd_sweep_synth.jsonl", encoding="utf-8"):
        r = json.loads(l); sp, cid = r["id"].split("/")
        J.append(("synth", r["id"], lambda sp=sp, cid=cid: (decode(ROOT / f"data/hazard7/big1/{sp}/{cid}.mp4"), FPS)))
    for l in open(ROOT / "data/judge/cpd_sweep_daily.jsonl", encoding="utf-8"):
        r = json.loads(l)
        if "error" in r:
            continue
        kind, key = r["id"].split("/", 1)
        if kind == "bg":
            J.append(("daily", r["id"], lambda key=key: (decode(ROOT / "data/hazard7/big1/train" / key), FPS)))
        else:
            src = ROOT / "data/real/south2021/social-media-study-gifs"
            p = next(src.glob(key + ".*"))
            J.append(("daily", r["id"], lambda p=p: dec_any(p)))
    return J


def conformance():
    import pattern_conformance as PC
    res = {f"{n}_{m}": 0 for n in NS for m in MODES}
    G = GridPatternJudge(PC.H, PC.W, cpd_lo=0.15, cpd_hi=1e3)
    cases = PC.cases()
    for name, _, kw in cases:
        L, ar = PC.stripes(**kw)
        truth = PC.truth_of(kw, ar)
        lum = luminance_cd(torch.from_numpy(PC.to_u8(L)[None]).cuda())[0]
        for n in NS:
            for m in MODES:
                res[f"{n}_{m}"] += int(G.frame(lum, n_seeds=n, seed_mode=m, rng=np.random.default_rng(0))["is_pat"] == truth)
    return res, len(cases)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    cache = ROOT / "data/judge/seeds_sweep.jsonl"
    done = {json.loads(l)["id"] for l in cache.read_text(encoding="utf-8").splitlines() if l.strip()} if cache.exists() else set()
    J = jobs()
    with cache.open("a", encoding="utf-8") as f:
        for i, (grp, cid, load) in enumerate(J, 1):
            if cid in done:
                continue
            try:
                x, fps = load()
                o = run_clip(x, fps); o.update(id=cid, grp=grp)
            except Exception as e:  # noqa: BLE001
                o = {"id": cid, "grp": grp, "error": f"{type(e).__name__}: {e}"}
            f.write(json.dumps(o, ensure_ascii=False) + "\n"); f.flush()
            torch.cuda.empty_cache()
            if i % 20 == 0:
                print(f"  {i}/{len(J)}", flush=True)
    conf, nconf = conformance()
    rows = [json.loads(l) for l in cache.read_text(encoding="utf-8").splitlines() if l.strip()]
    rows = [r for r in rows if "error" not in r]
    sens = [r for r in rows if r["n_sens"] > 0]
    base_daily = sum(r["base"] for r in rows if r["grp"] == "daily"); base_syn = sum(r["base"] for r in rows if r["grp"] == "synth")
    L = ["# 정지 패턴 씨앗 민감도 (사전 등록 = scripts/seeds_sweep.py 머리말)", "",
         f"> 클립 {len(rows)} (합성 정지 무늬 위험 {sum(r['grp'] == 'synth' for r in rows)} · 일상 {sum(r['grp'] == 'daily' for r in rows)}) · "
         f"씨앗 민감 클립(원면적 > 40 % 영역이 있는 프레임 보유) {len(sens)} · 기준 = (12, max)", "",
         "## A. 결정적 비교", "", "| 변형 | 적합성 | 합성 탐지 | 일상 위험 | 프레임당 시간(민감 프레임) |", "|---|---|---|---|---|"]
    summary = {}
    for n in NS:
        for m in MODES:
            k = f"{n}_{m}"
            syn = sum(r["det"][k] for r in rows if r["grp"] == "synth"); day = sum(r["det"][k] for r in rows if r["grp"] == "daily")
            sec = np.mean([r["sec"][k] for r in sens if k in r["sec"]]) if sens else 0.0
            summary[k] = {"conf": conf[k], "synth": syn, "daily": day, "sec": float(sec)}
            L.append(f"| n = {n}, {m} | {conf[k]}/{nconf} | {syn} | {day} | {sec:.2f} s |")
    L += ["", "## B. 안정성 (random, 난수 10 번)", "", "| n | 뒤집힘률(민감 클립 중 판정이 갈린 비율) | 면적 추정 표준편차(중앙값, %p) |", "|---|---|---|"]
    for n in NS:
        flips = [len(set(r["rand"][str(n)])) > 1 for r in sens]
        fr = float(np.mean(flips)) if flips else 0.0
        sd = float(np.median([r["area_sd"].get(str(n), 0.0) for r in sens])) if sens else 0.0
        summary[f"flip_{n}"] = fr; summary[f"sd_{n}"] = sd
        L.append(f"| {n} | {100 * fr:.1f} % ({sum(flips)}/{len(flips)}) | {100 * sd:.1f} |")
    # 판단 규칙
    cands = [(n, m) for n in NS for m in MODES
             if summary[f"{n}_{m}"]["conf"] == nconf and summary[f"{n}_{m}"]["daily"] <= base_daily and summary[f"{n}_{m}"]["synth"] >= base_syn - 1]
    ok = [(n, m) for n, m in cands if summary[f"flip_{n}"] <= 0.05]
    ok.sort(key=lambda t: (t[0], 0 if t[1] == "random" else 1))
    verdict = (f"채택: n = {ok[0][0]}, {ok[0][1]}" if ok else "고정 n 기각 → 적응형(Wilson 구간) 실험으로 (규칙 3)")
    L += ["", "## 판정 (사전 규칙)", "", f"- 기준(12, max): 합성 {base_syn} · 일상 {base_daily}",
          f"- 규칙 1 통과 후보: {', '.join(f'{n}/{m}' for n, m in cands) or '없음'}", f"- **{verdict}**"]
    (ROOT / "checks/judge/seeds_sweep.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    (ROOT / "data/judge/seeds_sweep.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
