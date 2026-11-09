"""해상도 민감도: 같은 자극을 해상도만 바꿔 판정하면 정답(위험/안전)이 그대로인가.

왜 재나
    노트북 트랙(128×72), 집 PC 트랙(1920×1080), 팀원 판정기(긴 변 720 으로 축소 → 1080p 는 720×405)가
    서로 다른 해상도에서 판정한다. 판정 규칙의 면적(화면 비율)과 줄무늬 공간 주파수(cpd)는 원리상 해상도와
    무관해야 하지만, 축소(INTER_AREA 평균)가 작은 영역·가는 줄의 밝기 차이를 깎으면 판정이 바뀐다.

실험
    A. 깜빡임: 1920×1080 에서 만들고(24 fps, 24프레임, 5 Hz 사각파, 바탕 코드 60 / 켜짐 코드 200)
       1280×720, 720×405, 640×360, 128×72 로 INTER_AREA 축소 후 analyze().
       - 원 하나, 면적 24% / 26% / 30%
       - 작은 점 여러 개(지름 8 px @1080p, 합계 면적 약 30%)
    B. 줄무늬: 1280×720 에서 화면 전체 정지 사각 줄무늬(코드 40 / 200, 13프레임)를 1.5 / 3 / 6 / 9 cpd 로 만들고
       720×405, 640×360, 128×72 로 축소 후 analyze_pattern(). (1080p 는 노트북 메모리 7.6 GB 로 무리라 제외)
결과: checks/rules/resolution_sensitivity.md
주의: 자극은 메모리에서만 만들고 화면에 띄우지 않는다.
"""
from __future__ import annotations

import gc
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from lumi_synth import measure  # noqa: E402

OUT = ROOT / "checks/rules/resolution_sensitivity.md"

# 메모리 절약: luminance_cd 는 원소별 계산이라 프레임 묶음으로 나눠도 결과가 같다(validate_rules_pse.py 와 같은 방식).
_LUM = measure.luminance_cd


def _chunked_luminance(frames_bgr: np.ndarray, chunk: int = 2) -> np.ndarray:
    out = np.empty(frames_bgr.shape[:-1], np.float32)
    for i in range(0, len(frames_bgr), chunk):
        out[i:i + chunk] = _LUM(frames_bgr[i:i + chunk])
    return out


measure.luminance_cd = _chunked_luminance
FPS = 24.0
FLASH_SIZES = [(1920, 1080), (1280, 720), (720, 405), (640, 360), (128, 72)]
STRIPE_SIZES = [(1280, 720), (720, 405), (640, 360), (128, 72)]


def flash_clip(h: int, w: int, mask: np.ndarray, n: int = 24, hz: float = 5.0) -> np.ndarray:
    frames = np.full((n, h, w, 3), 60, np.uint8)
    for t in range(n):
        if int(2 * hz * t / FPS) % 2 == 0:
            frames[t][mask] = 200
    return frames


def disc_mask(h, w, area):
    r = np.sqrt(area * h * w / np.pi)
    yy, xx = np.mgrid[0:h, 0:w]
    return (yy - h / 2) ** 2 + (xx - w / 2) ** 2 <= r * r


def dots_mask(h, w, area, d=8):
    m = np.zeros((h, w), bool)
    step = int(round(d / np.sqrt(area * 4 / np.pi)))     # 정사각 격자 간격: 점 면적 / 칸 면적 = area
    yy, xx = np.mgrid[0:h, 0:w]
    cy, cx = (yy % step) - step / 2, (xx % step) - step / 2
    m |= cy ** 2 + cx ** 2 <= (d / 2) ** 2
    return m


def resize(frames, size):
    w, h = size
    if frames.shape[2] == w and frames.shape[1] == h:
        return frames
    return np.stack([cv2.resize(f, (w, h), interpolation=cv2.INTER_AREA) for f in frames])


def stripe_clip(h, w, cpd, n=13):
    ppd = measure.pixels_per_degree(w, h)
    bar = ppd / (2 * cpd)
    g = (np.floor(np.arange(w) / bar) % 2).astype(bool)
    img = np.where(g, 200, 40).astype(np.uint8)
    return np.broadcast_to(img[None, None, :, None], (n, h, w, 3)).copy(), bar


def main():
    lines = ["# 해상도 민감도 — 같은 자극을 해상도만 바꿔 판정", "",
             "> 스크립트: `scripts/resolution_sensitivity.py`. 축소는 OpenCV INTER_AREA(영역 평균). "
             "판정기는 현재 `measure.py`(깜빡임 `analyze`, 줄무늬 `analyze_pattern`).", "",
             "## A. 깜빡임 (1920×1080 에서 생성, 5 Hz, 24프레임)", "",
             "| 자극 | " + " | ".join(f"{w}×{h}" for w, h in FLASH_SIZES) + " |",
             "|---|" + "---|" * len(FLASH_SIZES)]
    cases = [("원 24%", lambda h, w: disc_mask(h, w, 0.24)), ("원 26%", lambda h, w: disc_mask(h, w, 0.26)),
             ("원 30%", lambda h, w: disc_mask(h, w, 0.30)), ("점 지름 8px 합계 ~30%", lambda h, w: dots_mask(h, w, 0.30))]
    for name, mk in cases:
        m = mk(1080, 1920)
        base = flash_clip(1080, 1920, m)
        cells = []
        for size in FLASH_SIZES:
            f = resize(base, size)
            r = measure.analyze(f, FPS)
            cells.append(f"{'위험' if r['dangerous'] else '안전'} ({r['peak_area']:.3f}, {r['max_flashes']:g}/s)")
            del f, r
            gc.collect()
        lines.append(f"| {name} (설계 면적 {m.mean():.3f}) | " + " | ".join(cells) + " |")
        del base
        gc.collect()
        print(name, cells, flush=True)
    lines += ["", "칸 = 판정 (최대 같은 방향 동시 면적, 1초 최대 깜빡임).", "",
              "## B. 줄무늬 (1280×720 에서 생성, 정지, 화면 전체, 13프레임)", "",
              f"시청 조건 56 cm·15인치: 1도당 픽셀 = " + ", ".join(
                  f"{w}×{h} → {measure.pixels_per_degree(w, h):.1f}" for w, h in STRIPE_SIZES)
              + ". 표현 가능한 최고 주파수(2픽셀 주기) = 1도당 픽셀 / 2.", "",
              "| 자극 | " + " | ".join(f"{w}×{h}" for w, h in STRIPE_SIZES) + " |",
              "|---|" + "---|" * len(STRIPE_SIZES)]
    for cpd in (1.5, 3.0, 6.0, 9.0):
        base, bar = stripe_clip(720, 1280, cpd)
        cells = []
        for size in STRIPE_SIZES:
            f = resize(base, size)
            t0 = time.time()
            r = measure.analyze_pattern(f, FPS)
            cells.append(f"{'위험' if r['dangerous'] else '안전'} ({r['peak_stripe_area']:.2f}, {time.time() - t0:.0f}s)")
            del f, r
            gc.collect()
        lines.append(f"| {cpd:g} cpd (줄 폭 {bar:.1f}px @1280) | " + " | ".join(cells) + " |")
        print(cpd, cells, flush=True)
        del base
        gc.collect()
    lines += ["", "칸 = 판정 (최대 줄무늬 면적, 판정 시간 13프레임 기준, 노트북 CPU)."]
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("보고서:", OUT)


if __name__ == "__main__":
    main()
