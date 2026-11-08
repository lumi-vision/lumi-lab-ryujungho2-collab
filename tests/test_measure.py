"""측정 함수가 정답이 뻔한 신호에서 맞게 세는지 확인한다."""
import numpy as np

from lumi_synth.measure import count_transitions, max_flashes_per_second, relative_luminance_bgr


def square_wave(fps=24, secs=3, hz=5, lo=0.05, hi=0.6):
    t = np.arange(fps * secs) / fps
    return np.where((t * hz * 2).astype(int) % 2 == 0, lo, hi)


def test_luminance_black_white():
    px = np.array([[0, 0, 0], [255, 255, 255]], dtype=np.uint8)
    y = relative_luminance_bgr(px)
    assert y[0] == 0.0 and abs(y[1] - 1.0) < 1e-6


def test_flat_signal_has_no_flash():
    assert max_flashes_per_second(np.full(120, 0.3), 24) == 0


def test_square_wave_5hz_counts_5_flashes():
    # 24fps에서 정확히 5Hz는 안 나눠떨어지므로 근사: 4.5~5.5 사이
    f = max_flashes_per_second(square_wave(hz=5), 24)
    assert 4.5 <= f <= 6


def test_small_change_is_ignored():
    assert len(count_transitions(square_wave(lo=0.30, hi=0.35))) == 0


def test_bright_side_only_is_ignored():
    # 두 상태 모두 0.8 이상(어두운 쪽이 0.8 이상)이면 기준상 전이가 아님
    assert len(count_transitions(square_wave(lo=0.85, hi=0.99))) == 0
