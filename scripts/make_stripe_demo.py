"""줄무늬 탐지를 플레이어로 직접 보기 위한 시연 영상 (Elephants Dream 장면 왼쪽 55% 에 줄무늬 합성).

⚠️ 고대비 줄무늬는 정지해 있어도 광과민성 위험 자극이다. 이 PC 에서만, 0.25× 로 본다. 명암 반전·깜빡임은 넣지 않았다
   (정지 또는 느린 흐름만).
출력: data/stripe_demo/*.mkv (FFV1 무손실, Git 제외) — LUMI_플레이어.bat → 폴더 열기로 이 폴더를 연다.
사용: python scripts\\make_stripe_demo.py [--width 768 --height 432]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import av
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from lumi_synth import measure, stripe_freq as sf  # noqa: E402

ED = ROOT / "data/raw/elephantsdream-480-h264-st-aac.mov"
FPS, SECONDS, REGION = 30.0, 2.0, 0.55

DEMOS = [  # (파일 이름, 종류, cpd, 흐름 °/s, 추가 인자) — 흐름은 위상을 시간에 따라 옮김(한 방향, 깜빡임 아님)
    ("01_straight_1.2cpd", "straight", 1.2, 0.0, {}),
    ("02_straight_3cpd_drift", "straight", 3.0, 2.0, {"angle": 30.0}),
    ("03_concentric_3cpd", "concentric", 3.0, 0.0, {"center": (0.5, 0.25)}),
    ("04_radial_40", "radial", 3.0, 0.0, {"spokes": 40, "center": (0.5, 0.27)}),
    ("05_spiral_2cpd", "spiral", 2.0, 0.0, {"spokes": 12, "center": (0.5, 0.27)}),
    ("06_chirp_2to6cpd", "chirp", 2.0, 0.0, {"chirp": 2.0}),
    ("07_checker_3cpd", "checker", 3.0, 0.0, {}),
    ("08_straight_12cpd_out_of_range", "straight", 12.0, 0.0, {}),
]


def background(w: int, h: int, n: int, start: int = 4200) -> np.ndarray:
    out = []
    with av.open(str(ED)) as c:
        for i, fr in enumerate(c.decode(video=0)):
            if i >= start:
                out.append(cv2.resize(fr.to_ndarray(format="bgr24"), (w, h), interpolation=cv2.INTER_AREA))
            if len(out) >= n:
                break
    return np.stack(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--width", type=int, default=768)
    ap.add_argument("--height", type=int, default=432)
    args = ap.parse_args()
    w, h = args.width, args.height
    n = int(FPS * SECONDS)
    ppd = measure.pixels_per_degree(w, h)
    out = ROOT / "data/stripe_demo"
    out.mkdir(parents=True, exist_ok=True)
    bg = background(w, h, n)
    region = np.zeros((h, w), bool)
    region[:, : int(round(w * REGION))] = True
    for name, kind, cpd, dps, kw in DEMOS:
        wr = cv2.VideoWriter(str(out / f"{name}.mkv"), cv2.VideoWriter_fourcc(*"FFV1"), FPS, (w, h))
        for t in range(n):
            phase_shift = dps * cpd * 360.0 * t / FPS          # °/s × cpd = 초당 주기 수 → 위상(도)
            pat, _ = sf.render_pattern(kind, h, w, ppd, cpd, 10, 110, **kw)
            if phase_shift:
                phi, _ = sf.pattern_phase(kind, h, w, ppd, cpd, **kw)
                pat = np.where(np.sin(phi - np.radians(phase_shift)) >= 0, 110.0, 10.0).astype(np.float32)
            frame = bg[t].copy()
            frame[region] = sf.cd_to_bgr(pat)[region]
            wr.write(frame)
        wr.release()
        print(name, flush=True)
    print("출력:", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
