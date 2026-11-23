"""판정기 v2 2 단계 — 패턴 오탐 진단(설정 절제). 판정만, 영상은 메모리에서만.

표본: big1 의 합성 전 바탕(`<id>.clean.mp4`, 안전한 실제·그래픽 영상)
  - fp : 원본 판정기가 바탕에서 '정적 공간 패턴'을 낸 클립(labels.npz 의 clean 마스크, 12 프레임 이상)
  - tn : 패턴을 내지 않은 클립
  - pos: hazard7 의 pattern/hazard 합성 중 패턴이 새로 잡힌 클립(합성 양성 — 참고)
설정: default · no_gabor(Gabor 대체 끔) · fft06(FFT 대체 문턱 0.35→0.6) · no_gabor_fft06 · enhanced(향상 모드)
출력: checks/judge/pattern_diag.md, data/judge/pattern_diag_rows.jsonl

    python scripts/judge_v2_pattern_diag.py --n 40
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
import lumi_judge as J  # noqa: E402
from make_hazard7_dataset import FPS, decode  # noqa: E402

PAT = "정적 공간 패턴"
LABELS = ["휘도 플래시", "교집합", "빠른 장면 전환", "적색 플래시", "휘도 연장", "적색 연장", "정적 공간 패턴"]


def configs():
    m0, p0 = J.PartialMaskConfig(), J.PSEConfig()
    rep = dataclasses.replace
    return {
        "default": (m0, p0),
        "no_gabor": (rep(m0, enable_gabor_fallback=False), p0),
        "fft06": (m0, rep(p0, pattern_fft_energy_ratio_min=0.6)),
        "no_gabor_fft06": (rep(m0, enable_gabor_fallback=False), rep(p0, pattern_fft_energy_ratio_min=0.6)),
        "enhanced": (rep(m0, enhanced_spatial_pattern=True), p0),
    }


def pick(split: str, n: int, seed: int = 0):
    d = ROOT / "data" / "hazard7" / "big1" / split
    fp, tn, pos = [], [], []
    for q in sorted(d.glob("*.labels.npz")):
        sid = q.name[: -len(".labels.npz")]
        with np.load(q) as z:
            labs = list(z["labels"])
            k = labs.index(PAT)
            c = int(z["clean"][k].sum())
            a = int(z["added"][k].sum()) if "added" in z else 0
        (fp if c >= 12 else tn).append(sid)
        if "_pattern_hazard_" in sid and a >= 12:
            pos.append(sid)
    rng = np.random.default_rng(seed)
    f = lambda L: [L[i] for i in sorted(rng.choice(len(L), min(n, len(L)), replace=False))]
    return d, f(fp), f(tn), f(pos)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--split", default="train")
    a = ap.parse_args()
    from lumi_judge.cpu_limit import limit_cpu
    limit_cpu()
    sys.stdout.reconfigure(encoding="utf-8")
    d, fp, tn, pos = pick(a.split, a.n)
    out = ROOT / "data" / "judge" / "pattern_diag_rows.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {(json.loads(l)["id"], json.loads(l)["cfg"]) for l in out.read_text(encoding="utf-8").splitlines() if l.strip()} if out.exists() else set()
    C = configs()
    jobs = [("fp", s, ".clean.mp4") for s in fp] + [("tn", s, ".clean.mp4") for s in tn] + [("pos", s, ".mp4") for s in pos]
    print(f"fp {len(fp)} tn {len(tn)} pos {len(pos)} × 설정 {len(C)}", flush=True)
    t0 = time.time()
    with out.open("a", encoding="utf-8") as f:
        for i, (grp, sid, ext) in enumerate(jobs, 1):
            todo = [c for c in C if (sid + ext, c) not in done]
            if not todo:
                continue
            x = decode(d / f"{sid}{ext}")
            for c in todo:
                m, p = C[c]
                ts = time.time()
                r = J.analyze_frames(list(x), FPS, device="cuda", mask_config=m, pse=p)
                pm = np.asarray(r.frame_masks.get(PAT, np.zeros(len(x), bool)), bool)
                f.write(json.dumps({"id": sid + ext, "grp": grp, "cfg": c, "pat_frames": int(pm.sum()),
                                    "labels": sorted(r.active_labels), "sec": round(time.time() - ts, 2)}, ensure_ascii=False) + "\n")
                f.flush()
            if i % 10 == 0:
                print(f"  {i}/{len(jobs)} ({time.time() - t0:.0f}초)", flush=True)
    report(out)


def report(out: Path):
    rows = [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines() if l.strip()]
    cfgs = list(configs())
    L = ["# 패턴 오탐 진단 — 설정 절제 (판정기 v2 2 단계)", "",
         "> 표본: big1 합성 전 바탕(안전) 중 원본 판정기가 패턴을 낸 것(fp) / 안 낸 것(tn), pattern/hazard 합성(pos, 참고). 패턴 = 12 프레임(0.5 s) 이상.", "",
         "| 설정 | fp 중 여전히 패턴 | tn 중 새로 패턴 | pos 중 패턴 유지 | 초/클립 |", "|---|---|---|---|---|"]
    for c in cfgs:
        R = [r for r in rows if r["cfg"] == c]
        g = lambda k: [r for r in R if r["grp"] == k]
        hit = lambda rs: sum(r["pat_frames"] >= 12 for r in rs)
        L.append(f"| {c} | {hit(g('fp'))}/{len(g('fp'))} | {hit(g('tn'))}/{len(g('tn'))} | {hit(g('pos'))}/{len(g('pos'))} | "
                 f"{np.mean([r['sec'] for r in R]) if R else float('nan'):.1f} |")
    p = ROOT / "checks" / "judge" / "pattern_diag.md"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
