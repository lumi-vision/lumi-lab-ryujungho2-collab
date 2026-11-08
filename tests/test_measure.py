"""측정 함수(ITU 방식)가 정답이 뻔한 영상에서 맞게 재는지 확인한다.

시험 영상은 48×64 회색 화면. 회색값 30 ≈ 2.6 cd/m²(어두움), 200 ≈ 116 cd/m²(밝음).
"""
import numpy as np

from lumi_synth.measure import analyze, criteria_report, luminance_cd

FPS = 24
H, W = 48, 64
DARK, BRIGHT = 30, 200


def flicker_video(hz=5, secs=3, area=1.0, lo=DARK, hi=BRIGHT):
    """화면 왼쪽부터 area 비율만큼의 띠가 hz로 lo↔hi를 오가는 영상. 나머지는 lo로 고정."""
    t = np.arange(FPS * secs) / FPS
    on = (t * hz * 2).astype(int) % 2 == 1
    v = np.full((len(t), H, W, 3), lo, dtype=np.uint8)
    cols = int(round(W * area))
    v[on, :, :cols] = hi
    return v


def test_luminance_black_white_is_0_and_200():
    px = np.array([[0, 0, 0], [255, 255, 255]], dtype=np.uint8)
    y = luminance_cd(px)
    assert y[0] == 0.0 and abs(y[1] - 200.0) < 1e-3


def test_static_video_has_no_flash():
    r = analyze(np.full((72, H, W, 3), 120, dtype=np.uint8), FPS)
    assert r["max_flashes"] == 0 and not r["dangerous"]


def test_fullscreen_5hz_is_dangerous():
    r = analyze(flicker_video(hz=5), FPS)
    assert 4.5 <= r["max_flashes"] <= 6 and r["dangerous"]


def test_fullscreen_2hz_is_safe():
    r = analyze(flicker_video(hz=2), FPS)
    assert 1.5 <= r["max_flashes"] <= 2.5 and not r["dangerous"]


def test_area_30_percent_is_dangerous():
    r = analyze(flicker_video(hz=5, area=0.30), FPS)
    assert abs(r["peak_area"] - 0.30) < 0.02 and r["dangerous"]


def test_area_20_percent_is_safe():
    # 깜빡임은 빠르지만 면적이 25% 이하라 화면 전이로 세지 않음
    r = analyze(flicker_video(hz=5, area=0.20), FPS)
    assert abs(r["peak_area"] - 0.20) < 0.02 and r["max_flashes"] == 0 and not r["dangerous"]


def test_small_change_is_ignored():
    # 회색 120 ↔ 128: 약 6 cd/m² 차이로 20 미만
    r = analyze(flicker_video(hz=5, lo=120, hi=128), FPS)
    assert r["peak_area"] == 0


def test_bright_side_only_is_ignored():
    # 회색 235 ↔ 255: 어두운 쪽도 약 166 cd/m²로 160 이상
    r = analyze(flicker_video(hz=5, lo=235, hi=255), FPS)
    assert r["peak_area"] == 0


# ---- 기준별 판정표: pse-test-media의 f / a / c / l 설계를 흉내 낸 영상으로, 무너진 조건이 정확히 짚이는지 ----

def passed(report):
    return [row["넘음"] for row in report]


def test_report_f_all_conditions_exceeded():
    # 회색 64 ↔ 109 (10.3 ↔ 30.6 cd/m², 변화 20.3), 면적 30%, 초당 5번
    r = criteria_report(flicker_video(hz=5, area=0.30, lo=64, hi=109), FPS)
    assert passed(r) == [True, True, True, True]


def test_report_a_area_fails():
    r = criteria_report(flicker_video(hz=5, area=0.20, lo=64, hi=109), FPS)
    assert passed(r) == [True, True, False, False]


def test_report_c_count_fails():
    r = criteria_report(flicker_video(hz=2, area=0.30, lo=64, hi=109), FPS)
    assert passed(r) == [True, False, True, False]


def test_report_l_luminance_fails():
    # 회색 64 ↔ 100: 약 25.4 - 10.3 = 15 cd/m² 변화로 20 미만
    r = criteria_report(flicker_video(hz=5, area=0.30, lo=64, hi=100), FPS)
    assert passed(r)[0] is False and passed(r)[3] is False
