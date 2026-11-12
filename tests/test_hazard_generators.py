"""생성 계열 확장(oracle.render_* · red_pair)이 조건을 의도대로 만드는지 확인한다."""
import numpy as np
import pytest

from lumi_synth import measure, oracle

FPS = 30.0


@pytest.mark.parametrize("rho,duv", [(0.85, 0.25), (0.85, 0.19), (0.6, 0.25), (0.95, 0.32)])
def test_red_pair_controls_ratio_duv_and_keeps_luminance(rho, duv):
    p = oracle.red_pair(rho, duv)
    assert abs(p["delta_uv"] - duv) < 0.005
    assert abs(p["ratio_a"] - rho) < 0.01
    assert abs(p["lum_a_cd"] - p["lum_b_cd"]) < 0.3          # 휘도 깜빡임이 생기지 않을 만큼 같음
    assert p["sat_a"] == (p["ratio_a"] >= 0.8)
    assert not p["sat_b"]


def test_render_red_judged_by_duv_and_leaves_outside_untouched():
    bg = np.full((30, 54, 96, 3), 128, np.uint8)
    m = oracle.region_mask("rectangle", 0.4, 54, 96)
    w = oracle.wave("square", 5, 30, FPS)
    hit = oracle.red_pair(0.85, 0.25)
    miss = oracle.red_pair(0.85, 0.19)
    x = oracle.render_red(bg, m, w, hit["a_rgb"], hit["b_rgb"])
    assert (x[:, ~m] == 128).all()
    assert measure.analyze_red(x, FPS)["dangerous"]
    assert not measure.analyze(x, FPS)["dangerous"]             # 같은 휘도 → 휘도 깜빡임 없음
    y = oracle.render_red(bg, m, w, miss["a_rgb"], miss["b_rgb"])
    assert not measure.analyze_red(y, FPS)["dangerous"]


def test_render_stripes_period_and_reversal():
    h, w = 108, 192
    bg = np.full((4, h, w, 3), 128, np.uint8)
    m = np.ones((h, w), bool)
    cpd = 1.0
    x = oracle.render_stripes(bg, m, FPS, cpd, 0.0, 10, 110, "reversing", hz=7.5)   # 반주기 = 2프레임
    row = measure.luminance_cd(x[0])[h // 2]
    spec = np.abs(np.fft.rfft(row - row.mean()))
    period = len(row) / np.argmax(spec)
    assert abs(period - oracle.stripe_period_px(w, h, cpd, 0.0)) < 1.0
    assert (x[0] == x[1]).all() and not (x[0] == x[2]).all()
    lum0, lum2 = measure.luminance_cd(x[0]), measure.luminance_cd(x[2])
    assert np.corrcoef(lum0.ravel(), lum2.ravel())[0, 1] < -0.9   # 명암 반전


def test_render_stripes_outside_mask_untouched_and_pairs():
    h, w = 108, 192
    bg = np.random.default_rng(0).integers(0, 256, (2, h, w, 3), dtype=np.uint8)
    m = oracle.region_mask("rectangle", 0.3, h, w)
    x = oracle.render_stripes(bg, m, FPS, 2.0)
    assert (x[:, ~m] == bg[:, ~m]).all()
    period = oracle.stripe_period_px(w, h, 2.0, 0.0)
    assert abs(oracle.stripe_pairs_in_mask(m, period, 0.0) - m.any(axis=0).sum() / period) < 1e-6


# ---- 설계상 기대: 팀원 보고서(stage3) 기준과 우리 기준 (scripts/make_hazard_sweep.py) ----------------------
def _sweep():
    import importlib
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("make_hazard_sweep")


def test_team_flash_count_and_safe_interval():
    s = _sweep()
    def lum(hz):
        return s.expect_lum(0.4, 50.0 * oracle.wave("square", hz, 60, FPS), 0.0)
    assert lum(3.0)["expected"] is False and lum(3.0)["broken"] == "횟수"      # 같은 방향 간격 333.33 ms ≥ 333.3 ms
    assert lum(3.5)["expected"] is True
    assert lum(3.0)["expected_ours"] is False and lum(3.5)["expected_ours"] is True
    # 66.8 ms(2프레임) 안의 변화가 20 cd/m² 에 못 미치는 느린 변화: 극값 차는 50 이어도 팀원 기준은 '밝기' 미달
    slow = s.expect_lum(0.4, 50.0 * oracle.wave("sine", 1.0, 60, FPS), 0.0)
    assert slow["broken"] == "밝기" and "밝기" not in slow["broken_ours"]


def test_stripe_expectations_team_default_and_itu_setting(monkeypatch):
    s = _sweep()
    static = s.expect_stripe(0.5, 3.0, 10, 110, "static", 60)
    assert static["expected"] and static["expected_ours"]
    # 기본(2026-11-12): 판정기 상수가 팀원 기준과 같아 두 기대가 일치
    rev = s.expect_stripe(0.32, 3.0, 10, 110, "reversing", 40)
    assert rev["expected"] is False and rev["expected_ours"] is False         # 움직임과 상관없이 > 40%
    drift = s.expect_stripe(0.5, 3.0, 10, 110, "drifting", 60)
    assert drift["expected"] is True and drift["expected_ours"] is True       # 흐름 제외 없음
    assert s.expect_stripe(0.5, 9.0, 10, 110, "static", 60)["expected_ours"]  # 1~10 cpd 로 되돌림
    assert not s.expect_stripe(0.5, 10.5, 10, 110, "static", 60)["expected"]
    five = s.expect_stripe(0.5, 3.0, 10, 110, "static", 5)
    assert five["expected"] is False and five["expected_ours"] is False       # ≥ 6쌍
    # 판정기를 ITU 원문 설정으로 바꾸면 expected_ours 만 따라 바뀐다 (expected 는 팀원 기준 고정)
    monkeypatch.setattr(measure, "PATTERN_AREA_DYNAMIC", 0.25)
    monkeypatch.setattr(measure, "PATTERN_DRIFT_EXEMPT", True)
    rev = s.expect_stripe(0.32, 3.0, 10, 110, "reversing", 40)
    assert rev["expected"] is False and rev["expected_ours"] is True
    drift = s.expect_stripe(0.5, 3.0, 10, 110, "drifting", 60)
    assert drift["expected"] is True and drift["expected_ours"] is False


def test_team_red_floor_at_470000():
    s = _sweep()
    w = oracle.wave("square", 5, 60, FPS)
    below = s.expect_red(0.4, w, oracle.red_pair_from_colors((70, 0, 0), (0, 0, 0)))
    above = s.expect_red(0.4, w, oracle.red_pair_from_colors((71, 0, 0), (0, 0, 0)))
    assert below["expected"] is False and below["expected_ours"] is False     # 판정기에도 같은 하한 (2026-11-12)
    assert above["expected"] is True and above["expected_ours"] is True
    assert oracle.red_pair(0.85, 0.25)["excess_a"] >= 0.0625                  # 기본 적색 쌍은 팀원 하한도 넘음


def test_red_floor_in_detector():
    """판정기: 순수 빨강 R 71(#470000) ↔ 검정은 적색 위험, R 70 은 포화 빨강 하한 미달이라 안전. 하한을 끄면 둘 다 위험."""
    bg = np.full((30, 54, 96, 3), 128, np.uint8)
    m = oracle.region_mask("rectangle", 0.4, 54, 96)
    w = oracle.wave("square", 5, 30, FPS)
    hit = oracle.render_red(bg, m, w, (71, 0, 0), (0, 0, 0))
    miss = oracle.render_red(bg, m, w, (70, 0, 0), (0, 0, 0))
    assert measure.analyze_red(hit, FPS)["dangerous"] is True
    assert measure.analyze_red(miss, FPS)["dangerous"] is False


def test_red_floor_off_restores_old_behaviour(monkeypatch):
    monkeypatch.setattr(measure, "RED_MIN_EXCESS", 0.0)
    bg = np.full((30, 54, 96, 3), 128, np.uint8)
    m = oracle.region_mask("rectangle", 0.4, 54, 96)
    miss = oracle.render_red(bg, m, oracle.wave("square", 5, 30, FPS), (70, 0, 0), (0, 0, 0))
    assert measure.analyze_red(miss, FPS)["dangerous"] is True
