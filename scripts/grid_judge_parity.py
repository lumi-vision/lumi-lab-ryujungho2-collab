"""격자 판정기 G 의 F1(휘도 깜빡임) GPU 판이 `lumi_synth.measure.analyze` 와 같은 판정을 내는가.

비교: 화면 전이 열 e[t], 1 초 최대 깜빡임, 위험 여부, 위험 픽셀 지도. 속도도 함께.
    python scripts/grid_judge_parity.py [--n 12]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from lumi_judge.cpu_limit import limit_cpu  # noqa: E402
limit_cpu()
from lumi_judge.grid_flash import judge_luminance  # noqa: E402
from lumi_synth import measure  # noqa: E402
from make_hazard7_dataset import FPS, decode  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12)
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    d = ROOT / "data/hazard7/big1/val"
    kinds = ["lum_flash_hazard", "lum_flash_near", "intersection_hazard", "scene_hazard", "lum_ext_hazard", "pattern_hazard"]
    clips = []
    for k in kinds:
        clips += sorted(d.glob(f"val_{k}_*.mp4"))[: max(1, a.n // len(kinds))]
    clips = [c for c in clips if ".clean" not in c.name][: a.n]
    bad = 0; tg = tc = 0.0
    for p in clips:
        x = decode(p)
        t0 = time.time(); g = judge_luminance(x, FPS); tg += time.time() - t0
        t0 = time.time(); r = measure.analyze(x, FPS); tc += time.time() - t0
        same = (np.array_equal(g["transitions"], r["transitions"]) and g["dangerous"] == r["dangerous"]
                and g["max_flashes"] == r["max_flashes"] and np.array_equal(g["hazard_mask"], r["hazard_mask"]))
        bad += not same
        print(f"  {p.name}: {'같음' if same else '다름'} · 위험 {g['dangerous']}/{r['dangerous']} · 깜빡임 {g['max_flashes']}/{r['max_flashes']}"
              f" · 근거 {g['evidence']['segments'][:1]} 면적 {g['evidence']['hazard_area_pct']}% 칸 {len(g['evidence']['cells'])}", flush=True)
    print(f"불일치 {bad}/{len(clips)} · GPU {tg:.1f}s vs numpy {tc:.1f}s")


if __name__ == "__main__":
    main()
