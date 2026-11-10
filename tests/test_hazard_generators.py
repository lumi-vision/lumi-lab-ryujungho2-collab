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
