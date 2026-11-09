"""줄무늬 판정: 공간 주파수 1~8 cpd, 움직임 3분류(정지·반전·흐름), 분류별 면적 기준.

시험 영상: 160×90, 30 fps, 1초(30프레임). 화면 왼쪽 일부에만 세로 줄무늬(회색 30 ↔ 200,
2.6 ↔ 115.5 cd/m²), 나머지는 회색 120. 계산을 빠르게 하려고 작은 화면을 쓰고, 시청 거리를 늘려
1도당 픽셀 수를 실제 화면과 비슷하게(약 20 px/도) 맞춘다.

움직임 조건은 Binnie, Findlay & Wilkins 1985 의 세 조건(정지 / 진동 / 한 방향 흐름)을 30 fps 로
표현할 수 있는 범위에서 재현한다. 30 fps 의 나이퀴스트 한계(15 Hz) 때문에 진동은 5 Hz 로 둔다.
"""
import numpy as np
import pytest

from lumi_synth import measure

FPS = 30
H, W = 90, 160
DARK, BRIGHT, BG = 30, 200, 120


@pytest.fixture(autouse=True)
def about_20_px_per_degree(monkeypatch):
    # 15인치 16:9 화면의 가로 폭 약 33.2 cm 가 약 8°가 되도록 시청 거리를 늘림 → 160 px / 8° ≈ 20 px/도
    monkeypatch.setattr(measure, "VIEW_DISTANCE_CM", 237.0)
    yield


def stripes(period=8, area=0.55, motion="static", n=30, hz=5.0):
    """왼쪽 area 비율 폭에 세로 줄무늬. motion: static | drift | oscillate | reverse | flash."""
    x = np.arange(W)
    width = int(round(W * area))
    v = np.full((n, H, W, 3), BG, np.uint8)
    for t in range(n):
        if motion == "drift":
            off = 2.0 * t                                    # 프레임당 2 px 한 방향
        elif motion == "oscillate":
            off = 2.0 * np.sin(2 * np.pi * hz * t / FPS)     # 진폭 2 px, 5 Hz 왕복
        else:
            off = 0.0
        bright = ((x - off) % period) < period / 2
        if motion == "reverse" and int(t * hz * 2 / FPS) % 2 == 1:
            bright = ~bright                                 # 5 Hz 명암 반전
        row = np.where(bright, BRIGHT, DARK).astype(np.uint8)
        if motion == "flash" and int(t * hz * 2 / FPS) % 2 == 1:
            row[:] = BG                                      # 5 Hz 로 줄무늬가 나타났다 사라짐
        v[t, :, :width] = row[:width, None]
    return v


def cpd_of(period):
    return measure.pixels_per_degree(W, H, 0.0) / period


def test_setup_is_about_20_px_per_degree():
    assert 18 <= measure.pixels_per_degree(W, H, 0.0) <= 22


@pytest.mark.parametrize("motion, expected", [
    ("static", "static"),
    ("drift", "drifting"),
    ("oscillate", "reversing"),
    ("reverse", "reversing"),
    ("flash", "reversing"),
])
def test_motion_three_classes(motion, expected):
    r = measure.analyze_pattern(stripes(motion=motion), FPS)
    labels = [m for m in r["motion"] if m]
    assert labels, "줄무늬를 찾지 못함"
    share = labels.count(expected) / len(labels)
    assert share >= 0.8, (motion, expected, labels, r["time_ratio"].max(), r["direction_index"])


def test_binnie_order_oscillation_static_drift():
    """Binnie 1985 순서: 진동(위험) > 정지(면적에 따라) ≫ 흐름(제외). 면적 30%에서:
    반전 > 25% → 위험, 정지 ≤ 40% → 안전, 흐름 → 제외라 안전."""
    osc = measure.analyze_pattern(stripes(area=0.30, motion="oscillate"), FPS)
    sta = measure.analyze_pattern(stripes(area=0.30, motion="static"), FPS)
    dri = measure.analyze_pattern(stripes(area=0.30, motion="drift"), FPS)
    assert 0.25 < osc["peak_stripe_area"] < 0.40
    assert osc["dangerous"] is True
    assert sta["dangerous"] is False
    assert dri["dangerous"] is False


def test_static_large_area_is_dangerous():
    r = measure.analyze_pattern(stripes(area=0.55, motion="static"), FPS)
    assert r["peak_stripe_area"] > 0.40 and r["dangerous"] is True


def test_drift_large_area_is_exempt():
    r = measure.analyze_pattern(stripes(area=0.55, motion="drift"), FPS)
    assert r["dangerous"] is False


def test_drift_not_exempt_when_switched_off(monkeypatch):
    monkeypatch.setattr(measure, "PATTERN_DRIFT_EXEMPT", False)
    r = measure.analyze_pattern(stripes(area=0.55, motion="drift"), FPS)
    assert r["dangerous"] is True


def test_cpd_band_1_to_8():
    # 주기 8 px ≈ 2.5 cpd (범위 안), 주기 2 px ≈ 10 cpd (1~8 밖 — 예전 1~10 기준이면 안)
    assert 1.0 <= cpd_of(8) <= 8.0
    assert cpd_of(2) > 8.0
    inside = measure.analyze_pattern(stripes(period=8, area=0.55), FPS)
    outside = measure.analyze_pattern(stripes(period=2, area=0.55), FPS)
    assert inside["peak_stripe_area"] > 0.40
    # 2026-11-10 측정: 방향별로 상한을 자르던 이전 구현은 56% (비스듬한 스캔이 10 cpd 를 2.6 cpd 로 잼),
    # 최댓값 방식은 2.75% — 줄무늬 영역 가장자리에서 비스듬한 스캔만 닿는 픽셀이 남음(알려진 한계)
    assert outside["peak_stripe_area"] < 0.05 and outside["dangerous"] is False
