"""지속(연장) 플래시 판정 시험. 화면 전이 열 e[t] 로 직접 시험한다."""
import numpy as np

from lumi_synth import measure

FPS = 30.0


def flashes(hz: float, seconds: float, total_s: float = 8.0, start_s: float = 0.5) -> np.ndarray:
    """hz 플래시/초(전이 2·hz 번/초)를 start_s 부터 seconds 동안 넣은 화면 전이 열."""
    e = np.zeros(int(total_s * FPS), np.int8)
    period = FPS / (2 * hz)                       # 전이 간격(프레임)
    t, sign = start_s * FPS, 1
    while t < (start_s + seconds) * FPS and int(t) < len(e):
        e[int(round(t))] = sign
        sign, t = -sign, t + period
    return e


def test_iris_2_5hz_for_6s_fails_but_3s_passes():
    assert measure.extended_flash(flashes(2.5, 6.0), FPS)["extended_dangerous"]
    assert not measure.extended_flash(flashes(2.5, 3.0), FPS)["extended_dangerous"]


def test_iris_fails_after_about_4s():
    r = measure.extended_flash(flashes(2.5, 6.0), FPS)
    assert 4.0 <= r["first_fail_s"] <= 5.6          # 시작 0.5초 + 1초 창 채우기 + 4초 누적


def test_continuous_needs_more_than_5s():
    assert not measure.extended_flash(flashes(2.5, 4.5), FPS, mode="continuous")["extended_dangerous"]
    assert not measure.extended_flash(flashes(2.5, 5.0), FPS, mode="continuous")["extended_dangerous"]  # 정확히 5초 이하
    assert measure.extended_flash(flashes(2.5, 7.0), FPS, mode="continuous")["extended_dangerous"]


def test_frequency_band():
    assert not measure.extended_flash(flashes(1.5, 7.0), FPS)["extended_dangerous"]              # IRIS: 2~3 Hz 만
    assert measure.extended_flash(flashes(1.5, 7.0), FPS, min_transitions=2)["extended_dangerous"]  # 1~3 Hz 변형
    assert not measure.extended_flash(flashes(4.0, 7.0), FPS)["extended_dangerous"]              # 4 Hz 는 일반 위험 쪽
