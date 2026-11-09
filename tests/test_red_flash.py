"""적색 깜빡임 판정 (WCAG 2.2 적색 정의 + ITU 면적·횟수 규칙).

색은 pse-test-media red_flash_patterns README 색표에서 가져왔다 (휘도 Y ≈ 0.19 로 같아 휘도 깜빡임이 섞이지 않음).
  (205,78,78) 포화 빨강(선형 비율 0.8003) ↔ (31,134,134): Δu′v′ 0.2046 → 위험 쪽
  (205,78,78) ↔ (60,132,132): Δu′v′ 0.1948 → 색 차이 미달
"""
import numpy as np

from lumi_synth import measure

FPS = 30
H, W = 36, 64
RED, CYAN_F, CYAN_D = (205, 78, 78), (31, 134, 134), (60, 132, 132)


def flicker(a, b, hz=5.0, secs=2, area=1.0, bg=(120, 120, 120)):
    """화면 왼쪽 area 폭이 색 a ↔ b 를 hz 로 오가는 영상 (RGB 로 받아 BGR 로 저장)."""
    n = int(FPS * secs)
    v = np.empty((n, H, W, 3), np.uint8)
    v[:] = bg[::-1]
    cols = int(round(W * area))
    for t in range(n):
        r, g, b_ = b if int(t * hz * 2 / FPS) % 2 else a
        v[t, :, :cols] = (b_, g, r)
    return v


def test_saturated_red_uses_linear_values():
    sat, u, v = measure.red_state(np.array([[[78, 78, 205]]], np.uint8))   # BGR
    assert sat.item()                                                      # 선형 0.8003 ≥ 0.8 (감마 값이면 0.568)


def test_pse_colors_delta_uv_matches_readme():
    _, u1, v1 = measure.red_state(np.array([[RED[::-1], CYAN_F[::-1], CYAN_D[::-1]]], np.uint8))
    d_f = float(np.hypot(u1[0, 0] - u1[0, 1], v1[0, 0] - v1[0, 1]))
    d_d = float(np.hypot(u1[0, 0] - u1[0, 2], v1[0, 0] - v1[0, 2]))
    assert abs(d_f - 0.2046) < 0.001 and abs(d_d - 0.1948) < 0.001


def test_red_flash_5hz_fullscreen_is_dangerous():
    r = measure.analyze_red(flicker(RED, CYAN_F), FPS)
    assert r["dangerous"] and 4.5 <= r["max_flashes"] <= 6 and r["max_delta_uv"] > 0.2


def test_red_flash_is_not_a_luminance_flash():
    # 같은 휘도로 설계된 색 쌍이라 휘도 판정은 안전이어야 함 (적색만 따로 잡혀야 함)
    assert measure.analyze(flicker(RED, CYAN_F), FPS)["dangerous"] is False


def test_chromaticity_difference_below_threshold_is_safe():
    r = measure.analyze_red(flicker(RED, CYAN_D), FPS)
    assert r["max_flashes"] == 0 and not r["dangerous"]


def test_count_below_threshold_is_safe():
    r = measure.analyze_red(flicker(RED, CYAN_F, hz=2), FPS)
    assert r["max_flashes"] <= 3 and not r["dangerous"]


def test_area_rule():
    assert not measure.analyze_red(flicker(RED, CYAN_F, area=0.20), FPS)["dangerous"]
    assert measure.analyze_red(flicker(RED, CYAN_F, area=0.30), FPS)["dangerous"]


def test_non_red_large_chromaticity_change_is_not_red_flash():
    # 보라(129,107,193) ↔ 청록(31,134,134): 색이 크게 바뀌어도 포화 빨강이 없으면 적색 깜빡임 아님
    r = measure.analyze_red(flicker((129, 107, 193), (31, 134, 134)), FPS)
    assert r["max_flashes"] == 0


def test_black_to_red_counts():
    # 검정의 색도를 D65 로 두면 빨강(255,0,0)과 Δu′v′ ≈ 0.26 → 적색 깜빡임 (구현 선택, W3C #4285)
    r = measure.analyze_red(flicker((255, 0, 0), (0, 0, 0)), FPS)
    assert r["dangerous"]


def test_analyze_all_includes_red():
    r = measure.analyze_all(flicker(RED, CYAN_F), FPS)
    assert r["red"]["dangerous"] and r["dangerous"]


# ---- pse-test-media 적색 패턴 208개 (설계: f 위험, c·s·d 안전) ----------------------------------
import csv  # noqa: E402
from pathlib import Path  # noqa: E402

import pytest  # noqa: E402

PSE_RED = Path(__file__).resolve().parents[2] / "tools/pse-test-media/red_flash_patterns/30fps"


def _pse_frames(path):
    with path.open(newline="") as f:
        rgb = np.array([[int(r["r"]), int(r["g"]), int(r["b"])] for r in csv.DictReader(f)], np.uint8)
    v = np.empty((len(rgb), 4, 4, 3), np.uint8)
    v[:] = rgb[:, None, None, ::-1]
    return v


@pytest.mark.skipif(not PSE_RED.exists(), reason="pse-test-media 가 tools/ 에 없음")
def test_all_pse_red_patterns_match_design():
    wrong = []
    for p in sorted(PSE_RED.glob("*.csv")):
        got = measure.analyze_red(_pse_frames(p), FPS)["dangerous"]
        if got != (p.name[0] == "f"):
            wrong.append(p.name)
    assert not wrong, wrong


@pytest.mark.skipif(not PSE_RED.exists(), reason="pse-test-media 가 tools/ 에 없음")
@pytest.mark.parametrize("name, why", [
    ("d015_multi.csv", "첫 쌍의 한쪽이 Δ0.195 → 쌍 전체 무효 (2026-11-10 이전 과잉 검출)"),
    ("dr011_multi.csv", "마지막 쌍의 돌아오는 변화가 Δ0.198 → 그 쌍 무효"),
    ("s014_multi.csv", "포화 빨강에 못 미치는 빨강 → 포화 빨강(Δ0.001)은 변화가 아님"),
])
def test_flash_pair_rule_regressions(name, why):
    assert not measure.analyze_red(_pse_frames(PSE_RED / name), FPS)["dangerous"], why
