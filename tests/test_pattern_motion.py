"""줄무늬 판정: 공간 주파수 1~10 cpd, 움직임 3분류(정지·반전·흐름), 분류별 면적 기준.

기본 설정은 팀원 기준(2026-11-12): 움직임과 상관없이 면적 > 40%, 흐름 제외 없음, 6쌍 이상.
ITU-R BT.1702-3 원문 설정(반전 > 25%, 흐름 제외)은 itu 픽스처로 켜서 따로 시험한다.

시험 영상: 160×90, 30 fps, 1초(30프레임). 화면 왼쪽 일부에만 세로 줄무늬(회색 30 ↔ 200,
2.6 ↔ 115.5 cd/m²), 나머지는 회색 120. 계산을 빠르게 하려고 작은 화면을 쓰고, 시청 거리를 늘려
1도당 픽셀 수를 실제 화면과 비슷하게(약 20 px/도) 맞춘다.

움직임 조건은 Binnie, Findlay & Wilkins 1985 의 세 조건(정지 / 진동 / 한 방향 흐름)을 30 fps 로
표현할 수 있는 범위에서 재현한다. 30 fps 의 나이퀴스트 한계(15 Hz) 때문에 진동은 5 Hz 로 둔다.
"""
import numpy as np
import pytest

from lumi_synth import measure, oracle

FPS = 30
H, W = 90, 160
DARK, BRIGHT, BG = 30, 200, 120


@pytest.fixture(autouse=True)
def about_20_px_per_degree(monkeypatch):
    # 15인치 16:9 화면의 가로 폭 약 33.2 cm 가 약 8°가 되도록 시청 거리를 늘림 → 160 px / 8° ≈ 20 px/도
    monkeypatch.setattr(measure, "VIEW_DISTANCE_CM", 237.0)
    yield


@pytest.fixture
def itu(monkeypatch):
    """ITU-R BT.1702-3 Attachment 1 원문 설정: 반전 > 25%, 한 방향 흐름 제외(휘도 판정에서도 제외 = 결정 A)."""
    monkeypatch.setattr(measure, "PATTERN_AREA_DYNAMIC", 0.25)
    monkeypatch.setattr(measure, "PATTERN_DRIFT_EXEMPT", True)


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


def test_team_default_area_40_regardless_of_motion():
    """팀원 기준(기본): 움직임과 상관없이 30%는 안전, 55%는 위험. 흐름 제외 없음."""
    for motion in ("oscillate", "static", "drift"):
        assert measure.analyze_pattern(stripes(area=0.30, motion=motion), FPS)["dangerous"] is False, motion
        assert measure.analyze_pattern(stripes(area=0.55, motion=motion), FPS)["dangerous"] is True, motion


def test_binnie_order_oscillation_static_drift(itu):
    """(ITU 원문 설정) Binnie 1985 순서: 진동(위험) > 정지(면적에 따라) ≫ 흐름(제외). 면적 30%에서:
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


def test_drift_large_area_is_exempt(itu):
    r = measure.analyze_pattern(stripes(area=0.55, motion="drift"), FPS)
    assert r["dangerous"] is False


def test_drift_not_exempt_when_switched_off(monkeypatch):
    monkeypatch.setattr(measure, "PATTERN_DRIFT_EXEMPT", False)
    r = measure.analyze_pattern(stripes(area=0.55, motion="drift"), FPS)
    assert r["dangerous"] is True


def test_cpd_band_1_to_10(monkeypatch):
    # 주기 8 px ≈ 2.5 cpd (범위 안). 범위 밖은 시청 거리를 300 cm 로 늘려 주기 2 px ≈ 12.6 cpd 로 만든다
    # (기본 거리에서는 주기 2 px ≈ 9.98 cpd 라 1~10 안 — 2026-11-12 에 상한을 8 → 10 으로 되돌림)
    assert 1.0 <= cpd_of(8) <= 10.0
    inside = measure.analyze_pattern(stripes(period=8, area=0.55), FPS)
    monkeypatch.setattr(measure, "VIEW_DISTANCE_CM", 300.0)
    assert cpd_of(2) > 10.0
    outside = measure.analyze_pattern(stripes(period=2, area=0.55), FPS)
    assert inside["peak_stripe_area"] > 0.40
    # 2026-11-10 측정: 방향별로 상한을 자르던 이전 구현은 56% (비스듬한 스캔이 10 cpd 를 2.6 cpd 로 잼),
    # 최댓값 방식은 2.75% — 줄무늬 영역 가장자리에서 비스듬한 스캔만 닿는 픽셀이 남음(알려진 한계)
    assert outside["peak_stripe_area"] < 0.05 and outside["dangerous"] is False


@pytest.mark.parametrize("area", [0.30, 0.55])
@pytest.mark.parametrize("period", [6, 8])
def test_area_accuracy_within_1_point(area, period):
    """면적 측정 오차 고정 (2026-11-10): 이전 구현은 +4.4%p 과대 측정.
    지금은 비스듬한 스캔이 경계 밖으로 몇 픽셀 넘쳐 +0.6%p 안팎(보수적 방향)."""
    r = measure.analyze_pattern(stripes(motion="static", area=area, period=period, n=3), FPS)
    intended = round(W * area) / W
    assert intended <= r["peak_stripe_area"] <= intended + 0.01, (area, period, r["peak_stripe_area"])


# ---- 결정 A (2026-11-10): 흐름 줄무늬는 휘도 깜빡임에서도 제외 -----------------------------------
def test_drift_stripes_dangerous_by_default():
    r = measure.analyze_all(stripes(area=0.55, motion="drift"), FPS)
    assert r["pattern"]["dangerous"] is True                     # 팀원 기준: 흐름 제외 없음, > 40%
    assert r["flash"]["drift_exempt_frames"] == 0                # 휘도 판정에서도 빼지 않음
    assert r["dangerous"] is True


def test_drift_stripes_flash_alone_is_dangerous_but_analyze_all_exempts(itu):
    v = stripes(area=0.55, motion="drift")
    # 2026-11-14 (D3 = 20 ms): 흐름 2 px/프레임·주기 8 px 이면 한 프레임에 같은 방향으로 바뀌는 픽셀은 줄무늬 영역의
    # 25% = 화면의 13.75% 라 휘도만 봐도 안전이다. 이전 D3(2프레임 합침)는 27.5% 로 합쳐 위험으로 봤다.
    assert measure.analyze(v, FPS)["dangerous"] is False
    r = measure.analyze_all(v, FPS)
    assert r["pattern"]["dangerous"] is False                    # 줄무늬: 흐름 제외
    assert r["flash"]["dangerous"] is False                      # 휘도: 흐름 줄무늬 영역 제외
    assert r["flash"]["drift_exempt_frames"] > 0
    assert r["dangerous"] is False


def test_reversing_stripes_still_dangerous_in_analyze_all(itu):
    r = measure.analyze_all(stripes(area=0.30, motion="reverse"), FPS)
    assert r["pattern"]["dangerous"] is True and r["dangerous"] is True


def test_reversing_stripes_30_percent_safe_by_default():
    """팀원 기준(기본): 반전 30%는 줄무늬 ≤ 40%, 휘도는 방향별 면적 15% ≤ 25% → 안전."""
    r = measure.analyze_all(stripes(area=0.30, motion="reverse"), FPS)
    assert r["pattern"]["dangerous"] is False and r["dangerous"] is False


def test_flash_outside_drift_region_still_counts():
    """흐름 줄무늬(왼쪽 30%) + 오른쪽 40% 휘도 깜빡임 5 Hz → 휘도 위험은 그대로 잡혀야 함."""
    v = stripes(area=0.30, motion="drift")
    t = np.arange(len(v)) / FPS
    on = (t * 5 * 2).astype(int) % 2 == 1
    v[on, :, 96:] = 220
    v[~on, :, 96:] = 40
    r = measure.analyze_all(v, FPS)
    assert r["flash"]["dangerous"] is True


def _bars(regular: bool, h: int = 180, w: int = 320, cpd: float = 2.0, area: float = 0.55) -> np.ndarray:
    """세로 막대 한 프레임(1×H×W 휘도). regular=False 면 막대 폭이 0.3~1.7배로 제각각(평균은 같음)."""
    half = measure.pixels_per_degree(w, h, 0.0) / cpd / 2.0
    xs = np.arange(w, dtype=np.float64)
    if regular:
        bright = (xs // half) % 2 == 0
    else:
        edges = np.cumsum(np.random.default_rng(7).uniform(0.3, 1.7, 2000) * half)
        bright = np.searchsorted(edges, xs) % 2 == 0
    lum = np.full((h, w), 40.0, np.float32)
    m = oracle.region_mask("rectangle", area, h, w)
    lum[m] = np.broadcast_to(np.where(bright, 110.0, 10.0), (h, w))[m]
    return lum[None]


def test_regularity_keeps_regular_stripes_and_rejects_irregular_bars():
    """규칙성(2026-11-12): 간격이 일정한 줄무늬는 면적 그대로, 폭이 제각각인 막대와 잡음은 40% 아래로 재진다."""
    assert measure.stripe_masks(_bars(True)).mean() > 0.50
    assert measure.stripe_masks(_bars(False)).mean() < 0.40
    noise = np.full((180, 320), 40.0, np.float32)
    m = oracle.region_mask("rectangle", 0.55, 180, 320)
    noise[m] = np.random.default_rng(3).uniform(10, 110, (180, 320)).astype(np.float32)[m]
    assert measure.stripe_masks(noise[None]).mean() < 0.40


def test_regularity_off_restores_old_behaviour(monkeypatch):
    monkeypatch.setattr(measure, "PATTERN_REGULARITY_CV", None)
    assert measure.stripe_masks(_bars(False)).mean() > 0.45
