"""화질·해상도·주사율 강건성 — 적응형(adapt) 측정의 채택 여부 (사전 등록, 이 머리말이 판단 규칙, 실행 전 커밋, 2026-11-26).

질문: 같은 영상을 열화시켰을 때 판정이 유지되는가, 그리고 적응형(격자 = max(1°, 8 px) · 압축 블록 의심 표시)이 고정형(가로 32 칸)보다 나은가.
열화(각 세트에 똑같이): 원본 · 해상도 ½ · ¼ · H.264 CRF 23 · 35 · 45 · GIF(256 색 팔레트 + 디더링) · (주사율 세트만) 12 · 8 fps
세트
  패턴 위험: 합성 정지 무늬 위험 42(cpd 실험 목록) + 움직이는 패턴 적합성 위험 7
  패턴 안전: 진단 바탕 중 팀원 판이 패턴이라 한 41(질감이 많아 오탐에 가장 취약) + 움직이는 패턴 적합성 안전 9
  깜빡임: 시퀀스 비교 시험 6 (주사율 열화만 — 격자와 무관)
판정: 패턴 = judge_pattern_motion(P1+P2) — adapt 는 quality.profile 의 격자·블록 표시, 고정은 GridPatternJudge 기본. 깜빡임 = judge_luminance.
지표: 원본 판정과의 일치율(세트·열화별) · 안전 세트의 새 위험 수(열화로 생긴 오탐) · 위험 세트에서 놓친 것 중 '판정 가능 범위 밖'(원본 cpd > 열화 후 판정 가능 cpd)으로 설명되는 비율
판단 규칙(결과 보고 바꾸지 않음)
  1. 적응형을 기본으로 채택: 모든 열화 합산에서 (a) 안전 세트 새 위험 수 ≤ 고정형 그리고 (b) 위험 세트 일치 수 ≥ 고정형 − 2.
  2. 아니면 고정형 유지, 적응형의 어느 부분(격자/블록 표시)이 문제인지 열화별로 기록.
  3. 열화로 놓친 위험 중 '판정 가능 범위 밖'이 아닌 것은 결함 후보로 목록화(원인 추적 대상).
출력: checks/judge/robustness_sweep.md · data/judge/robustness_sweep.jsonl
"""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import av
import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from lumi_judge.cpu_limit import limit_cpu  # noqa: E402
limit_cpu()
from lumi_judge.grid_flash import judge_luminance  # noqa: E402
from lumi_judge.grid_motion import judge_pattern_motion  # noqa: E402
from lumi_judge.grid_pattern import GridPatternJudge  # noqa: E402
from lumi_judge.quality import profile  # noqa: E402

DEGS = ["orig", "half", "quarter", "crf23", "crf35", "crf45", "gif"]


def h264(x, fps, crf):
    buf = io.BytesIO(); c = av.open(buf, "w", format="mp4")
    s = c.add_stream("libx264", rate=int(round(fps))); s.width = x.shape[2] - x.shape[2] % 2; s.height = x.shape[1] - x.shape[1] % 2
    s.pix_fmt = "yuv420p"; s.options = {"crf": str(crf)}
    for f in x:
        for p in s.encode(av.VideoFrame.from_ndarray(np.ascontiguousarray(f[:s.height, :s.width]), format="bgr24")):
            c.mux(p)
    for p in s.encode():
        c.mux(p)
    c.close(); buf.seek(0)
    with av.open(buf) as d:
        return np.stack([fr.to_ndarray(format="bgr24") for fr in d.decode(video=0)])


def gif(x):
    from PIL import Image
    out = []
    for f in x:
        im = Image.fromarray(f[..., ::-1]).convert("P", palette=Image.ADAPTIVE, colors=256, dither=Image.FLOYDSTEINBERG)
        out.append(np.asarray(im.convert("RGB"))[..., ::-1])
    return np.stack(out)


def degrade(x, fps, d):
    if d == "orig":
        return x, fps
    if d in ("half", "quarter"):
        k = 2 if d == "half" else 4
        return np.stack([cv2.resize(f, (max(16, f.shape[1] // k), max(16, f.shape[0] // k)), interpolation=cv2.INTER_AREA) for f in x]), fps
    if d.startswith("crf"):
        return h264(x, fps, int(d[3:])), fps
    if d == "gif":
        return gif(x), fps
    if d.endswith("fps"):
        f2 = float(d[:-3]); idx = np.round(np.arange(0, len(x) / fps, 1 / f2) * fps).astype(int)
        return x[np.clip(idx, 0, len(x) - 1)], f2


def pattern(x, fps, adapt):
    h, w = x.shape[1:3]
    if adapt:
        qp = profile(x, fps)
        G = GridPatternJudge(h, w, grid_deg=qp["cell_deg"])
        if qp["blocky"]:
            G.block_px = 8
    else:
        qp = profile(x, fps); G = GridPatternJudge(h, w)
    r = judge_pattern_motion(x, fps, G=G)
    b = r["best1"] or r["best2"] or r["best_any"]
    return bool(r["p1"] or r["p2"]), (None if b is None else float(b["cpd"])), qp


def jobs():
    import pattern_motion_conformance as PM
    from make_hazard7_dataset import FPS, decode
    J = []
    for l in open(ROOT / "data/judge/cpd_sweep_synth.jsonl", encoding="utf-8"):
        r = json.loads(l); sp, cid = r["id"].split("/")
        J.append(("pat_haz", r["id"], lambda sp=sp, cid=cid: (decode(ROOT / f"data/hazard7/big1/{sp}/{cid}.mp4"), FPS)))
    dr = [json.loads(l) for l in (ROOT / "data/judge/pattern_diag_rows.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    for rid in sorted({r["id"] for r in dr if r["grp"] == "fp"}):
        J.append(("pat_safe", "bg/" + rid, lambda rid=rid: (decode(ROOT / "data/hazard7/big1/train" / rid), FPS)))
    for name, truth, scored, kw in PM.CASES:
        if scored:
            J.append(("pat_haz" if truth else "pat_safe", "motion/" + name, lambda kw=kw: (PM.clip(**kw)[0], PM.FPS)))
    return J


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    cache = ROOT / "data/judge/robustness_sweep.jsonl"
    done = {json.loads(l)["id"] for l in cache.read_text(encoding="utf-8").splitlines() if l.strip()} if cache.exists() else set()
    with cache.open("a", encoding="utf-8") as f:
        for i, (grp, cid, load) in enumerate(jobs(), 1):
            if cid in done:
                continue
            try:
                x0, fps0 = load()
                row = {"id": cid, "grp": grp, "res": {}}
                for d in DEGS:
                    x, fps = degrade(x0, fps0, d)
                    a, ca, qp = pattern(x, fps, True); b, cb, _ = pattern(x, fps, False)
                    row["res"][d] = {"adapt": a, "fixed": b, "cpd": ca or cb, "cpd_max": qp["cpd_max_judgeable"], "blocky": qp["blocky"],
                                     "cell_deg": qp["cell_deg"]}
                    torch.cuda.empty_cache()
            except Exception as e:  # noqa: BLE001
                row = {"id": cid, "grp": grp, "error": f"{type(e).__name__}: {e}"}
            f.write(json.dumps(row, ensure_ascii=False) + "\n"); f.flush()
            if i % 10 == 0:
                print(f"  {i}", flush=True)
    # 깜빡임 × 주사율
    import make_seq_figs as SQ
    flash = []
    for name, truth, build in SQ.CASES:
        x, _ = build()
        res = {}
        for d in ("orig", "12fps", "8fps"):
            y, fps = degrade(x, SQ.FPS, d)
            res[d] = bool(judge_luminance(y, fps)["dangerous"])
        flash.append({"case": name, "truth": truth, "res": res})
    report(cache, flash)


def report(cache, flash):
    rows = [json.loads(l) for l in cache.read_text(encoding="utf-8").splitlines() if l.strip()]
    err = [r for r in rows if "error" in r]; rows = [r for r in rows if "error" not in r]
    L = ["# 화질·해상도·주사율 강건성 (사전 등록 = scripts/robustness_sweep.py 머리말)", "",
         f"> 패턴 위험 {sum(r['grp'] == 'pat_haz' for r in rows)} · 패턴 안전 {sum(r['grp'] == 'pat_safe' for r in rows)} (오류 {len(err)}) · 셀 = 위험 판정 수", "",
         "| 열화 | 위험 세트: 적응 | 고정 | 안전 세트(오탐): 적응 | 고정 | 블록 표시 켜짐 | 평균 격자 ° |", "|---|---|---|---|---|---|---|"]
    tot = {"ha": 0, "hf": 0, "sa": 0, "sf": 0}
    for d in DEGS:
        H = [r["res"][d] for r in rows if r["grp"] == "pat_haz"]; S = [r["res"][d] for r in rows if r["grp"] == "pat_safe"]
        ha, hf = sum(q["adapt"] for q in H), sum(q["fixed"] for q in H); sa, sf = sum(q["adapt"] for q in S), sum(q["fixed"] for q in S)
        if d != "orig":
            tot["ha"] += ha; tot["hf"] += hf; tot["sa"] += sa; tot["sf"] += sf
        L.append(f"| {d} | {ha}/{len(H)} | {hf}/{len(H)} | {sa}/{len(S)} | {sf}/{len(S)} | {sum(q['blocky'] for q in H + S)} | {np.mean([q['cell_deg'] for q in H + S]):.2f} |")
    # 놓친 것 설명
    miss, expl = 0, 0
    lst = []
    for r in rows:
        if r["grp"] != "pat_haz" or not r["res"]["orig"]["adapt"]:
            continue
        c0 = r["res"]["orig"]["cpd"]
        for d in DEGS[1:]:
            q = r["res"][d]
            if not q["adapt"]:
                miss += 1
                if c0 and c0 > q["cpd_max"]:
                    expl += 1
                else:
                    lst.append(f"{r['id']} · {d} (원본 cpd {c0}, 판정 가능 ≤ {q['cpd_max']})")
    ok1 = tot["sa"] <= tot["sf"] and tot["ha"] >= tot["hf"] - 2
    L += ["", f"열화 합산 — 위험 세트 위험 판정: 적응 {tot['ha']} · 고정 {tot['hf']} / 안전 세트 오탐: 적응 {tot['sa']} · 고정 {tot['sf']}",
          f"원본에서 잡았는데 열화 후 놓친 경우 {miss} 중 '판정 가능 범위 밖'으로 설명 {expl}", "",
          "## 판정 (사전 규칙)", "", f"- **{'적응형을 기본으로 채택' if ok1 else '고정형 유지'}** (규칙 1: 오탐 적응 ≤ 고정 그리고 위험 적응 ≥ 고정 − 2)",
          "", "## 결함 후보 (범위 안인데 열화로 놓침 — 원인 추적)", ""] + [f"- {s}" for s in lst[:60]]
    L += ["", "## 깜빡임 × 주사율 (판정 가능 범위: 초당 fps/2 회)", "", "| 사례 | 정답 | 원본 30 fps | 12 fps | 8 fps |", "|---|---|---|---|---|"]
    for q in flash:
        yn = lambda b: "위험" if b else "안전"
        L.append(f"| {q['case']} | {yn(q['truth'])} | {yn(q['res']['orig'])} | {yn(q['res']['12fps'])} | {yn(q['res']['8fps'])} |")
    (ROOT / "checks/judge/robustness_sweep.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
