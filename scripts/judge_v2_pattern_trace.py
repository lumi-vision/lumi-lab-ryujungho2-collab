"""판정기 v2 — 남은 패턴 오탐(향상 모드) 프레임 단위 추적. 판정만, 영상은 메모리에서만.

가설: 프레임 판정의 면적 조건이 '줄무늬가 차지한 면적'이 아니라 화면 전체 국소 대비 면적(fine_area)이라,
질감이 많은 장면은 면적 조건을 쉽게 넘고 어딘가 단면 하나의 6 쌍만으로 패턴이 된다.
기록(프레임마다): 판정, fine_area, 최종 쌍 수, 단계별 최대 쌍 수(선형·평행검증·Sobel·링·스포크·FFT·Gabor),
                 실제 패턴 마스크 면적(inspect_frame_with_mask 의 마스크).
대상: data/judge/pattern_diag_rows.jsonl 에서 향상 모드로도 패턴이 남은 안전 바탕(fp) 과 합성 패턴(pos).

    python scripts/judge_v2_pattern_trace.py
"""
from __future__ import annotations

import dataclasses
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from lumi_judge.cpu_limit import limit_cpu  # noqa: E402
limit_cpu()
import lumi_judge as J  # noqa: E402
from lumi_judge import spatial_mask as SM  # noqa: E402
from make_hazard7_dataset import FPS, decode  # noqa: E402

D = ROOT / "data" / "hazard7" / "big1" / "train"
REC: list = []
CUR: dict = {}


def wrap_step(name, fn, pairs_of):
    def w(self, *a, **k):
        r = fn(self, *a, **k)
        try:
            CUR.setdefault("steps", {})[name] = max(CUR.get("steps", {}).get(name, 0), int(pairs_of(r)))
        except Exception:  # noqa: BLE001
            pass
        return r
    return w


def install():
    C = SM.PartialStaticPatternDetector
    C._scan_sobel_targeted_bundles = wrap_step("sobel", C._scan_sobel_targeted_bundles, lambda r: r[0])
    C._scan_rings_enhanced = wrap_step("rings", C._scan_rings_enhanced, lambda r: r[0])
    C._scan_spokes_enhanced = wrap_step("spokes", C._scan_spokes_enhanced, lambda r: r[0])
    C._evaluate_2d_fft_fallback = wrap_step("fft", C._evaluate_2d_fft_fallback, lambda r: r[0])
    C._evaluate_gabor_fallback = wrap_step("gabor", C._evaluate_gabor_fallback, lambda r: r[0])
    C._verify_fine_parallel_neighbors = wrap_step("parallel_verify", C._verify_fine_parallel_neighbors,
                                                  lambda r: max([x[4] for x in r] or [0]))
    orig_enh = C._inspect_frame_enhanced

    def enh(self, lum):
        CUR.clear()
        r = orig_enh(self, lum)
        CUR["res"] = (bool(r[0]), float(r[1]), int(r[2]), float(r[3]))
        return r
    C._inspect_frame_enhanced = enh
    orig_mask = C.inspect_frame_with_mask

    def wm(self, lum, close_scale=0.50):
        r = orig_mask(self, lum, close_scale)
        res = CUR.get("res") or (bool(r[0]), float(r[1]), int(r[2]), float(r[3]))
        REC.append({"is_pat": bool(r[0]), "fine_area": round(res[1], 4), "pairs": int(r[2]),
                    "mask_area": round(float(r[4].float().mean().item()), 4), "steps": dict(CUR.get("steps", {}))})
        return r
    C.inspect_frame_with_mask = wm


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    rows = [json.loads(l) for l in (ROOT / "data/judge/pattern_diag_rows.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    tgt = [(r["grp"], r["id"]) for r in rows if r["cfg"] == "enhanced" and r["pat_frames"] >= 12 and r["grp"] in ("fp", "pos")]
    print("대상", len(tgt), {g: sum(1 for x, _ in tgt if x == g) for g in ("fp", "pos")}, flush=True)
    install()
    m = dataclasses.replace(J.PartialMaskConfig(), enhanced_spatial_pattern=True)
    out = ROOT / "data" / "judge" / "pattern_trace.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for i, (grp, cid) in enumerate(tgt, 1):
            REC.clear()
            x = decode(D / cid)
            J.analyze_frames(list(x), FPS, device="cuda", mask_config=m)
            pat = [r for r in REC if r["is_pat"]]
            win = defaultdict(int)
            for r in pat:
                st = r["steps"]
                best = max(st.values()) if st else 0
                src = [k for k, v in st.items() if v == r["pairs"]] or (["linear"] if r["pairs"] > best else ["?"])
                win[src[0]] += 1
            f.write(json.dumps({"grp": grp, "id": cid, "frames": len(REC), "pat_frames": len(pat),
                                "fine_area_med": float(np.median([r["fine_area"] for r in pat])) if pat else None,
                                "mask_area_med": float(np.median([r["mask_area"] for r in pat])) if pat else None,
                                "mask_area_max": float(max([r["mask_area"] for r in pat])) if pat else None,
                                "pairs_med": float(np.median([r["pairs"] for r in pat])) if pat else None,
                                "winner": dict(win)}, ensure_ascii=False) + "\n")
            f.flush()
            print(f"  {i}/{len(tgt)} {grp} {cid}", flush=True)
    summarize(out)


def summarize(out):
    R = [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines() if l.strip()]
    L = ["# 남은 패턴 오탐 추적 (향상 모드) — 프레임 단위", "",
         "> 패턴 프레임만 집계. fine_area = 판정이 쓰는 면적(화면 전체 국소 대비 면적), mask_area = 실제 패턴 마스크 면적. 규격: 패턴이 화면 > 40 %.", "",
         "| 그룹 | n | fine_area 중앙값 | 패턴 마스크 면적 중앙값 | 마스크 면적 최대 > 40 % 인 클립 | 쌍 수 중앙값 | 패턴을 만든 단계(프레임 수 합) |", "|---|---|---|---|---|---|---|"]
    for g in ("fp", "pos"):
        rs = [r for r in R if r["grp"] == g and r["pat_frames"]]
        if not rs:
            continue
        win = defaultdict(int)
        for r in rs:
            for k, v in r["winner"].items():
                win[k] += v
        L.append(f"| {g} | {len(rs)} | {np.median([r['fine_area_med'] for r in rs]):.2f} | {np.median([r['mask_area_med'] for r in rs]):.3f} | "
                 f"{sum(r['mask_area_max'] > 0.40 for r in rs)}/{len(rs)} | {np.median([r['pairs_med'] for r in rs]):.0f} | "
                 + ", ".join(f"{k} {v}" for k, v in sorted(win.items(), key=lambda z: -z[1])) + " |")
    p = ROOT / "checks" / "judge" / "pattern_trace.md"
    p.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
