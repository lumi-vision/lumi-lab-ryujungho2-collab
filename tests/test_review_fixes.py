"""2026-11-13 코드 리뷰에서 찾은 판정기 버그의 회귀 테스트. 각 테스트는 리뷰의 재현 입력을 그대로 쓴다.

고치기 전에는 모두 실패했다(놓침 5건, 과대 계수 1건).
"""
import numpy as np

from lumi_synth import measure as m


def _flat(values, h=36, w=64):
    return np.stack([np.full((h, w, 3), v, np.uint8) for v in values])


def test_first_frame_midpoint_still_counts():
    """첫 프레임이 깜빡임의 중간 밝기(188)여도 174 ↔ 201(ΔL 32) 5 Hz 를 센다. 전에는 0회."""
    seq = [188] + [174 if (t // 3) % 2 == 0 else 201 for t in range(1, 60)]
    r = m.analyze(_flat(seq), 30.0)
    assert r["max_flashes"] == 5.0 and r["dangerous"]


def test_slow_start_then_flash_uses_running_extremes():
    """첫 전이 전에는 처음부터의 최저점에서 잰다: 100 → 90 → 110(최저점에서 +20)이면 밝아짐 전이."""
    lum = np.array([100.0, 90.0, 110.0], np.float32).reshape(3, 1, 1)
    assert m.pixel_transitions(lum)[:, 0, 0].tolist() == [0, 0, 1]


def test_d4_leftover_area_not_counted_twice():
    """체커보드를 8프레임마다 반전(2 플래시/초) + 반전 직후 그때 밝은 픽셀 하나만 한 프레임 어두워짐 → 안전.
    전에는 반전마다 남은 반대 방향 면적 + 그 픽셀 하나로 화면 전이가 한 번 더 세어져 4 플래시/초(위험)였다."""
    yy, xx = np.mgrid[:16, :16]
    board = ((yy + xx) % 2).astype(bool)
    frames = []
    for t in range(60):
        b = board if (t // 8) % 2 == 0 else ~board
        f = np.where(b[..., None], 200, 30).astype(np.uint8).repeat(3, axis=2)
        if t >= 8 and t % 8 == 1:
            f[(0, 0) if b[0, 0] else (0, 1)] = 0
        frames.append(f)
    r = m.analyze(np.stack(frames), 30.0)
    assert r["max_flashes"] <= 2.5 and not r["dangerous"]


def test_red_flash_survives_near_black_color_noise():
    """빨강 ↔ 검정 5 Hz, 검정 프레임마다 (1,0,0)/(0,1,0)/(0,0,1) 색 캐스트 → 적색 깜빡임 5회. 전에는 0회."""
    casts = [(0, 0, 1), (0, 1, 0), (1, 0, 0)]          # BGR
    frames, k = [], 0
    for t in range(60):
        if (t // 3) % 2 == 0:
            frames.append(np.full((36, 64, 3), (0, 0, 255), np.uint8))
        else:
            frames.append(np.full((36, 64, 3), casts[k % 3], np.uint8))
            k += 1
    r = m.analyze_red(np.stack(frames), 30.0)
    assert r["max_flashes"] == 5.0 and r["dangerous"]


def test_non_integer_fps_window_uses_ceil():
    """12.5 fps 에서 2프레임마다 반전(3.125 Hz = 1초에 전이 7번) → 위험. 전에는 W = round(12.5) = 12 라 안전."""
    assert m.window_frames(12.5) == 13 and m.window_frames(29.97) == 30 and m.window_frames(24.0) == 24
    seq = [30 if (t // 2) % 2 == 0 else 200 for t in range(50)]
    assert m.analyze(_flat(seq), 12.5)["dangerous"]


def test_odd_period_stripes_inside_cpd_range():
    """128×72 에서 주기 3 px(1.29 cpd, 범위 안) 세로 줄무늬 → 줄무늬. 전에는 반 주기 폭 2 에서 끊겨 면적 0."""
    img = np.zeros((72, 128, 3), np.uint8)
    for x in range(128):
        img[:, x] = 200 if (x % 3) < 2 else 20
    area = m.stripe_masks(m.luminance_cd(np.stack([img] * 2)))[0].mean()
    assert area > 0.9


def test_fine_orthogonal_lines_do_not_erase_stripes():
    """세로 2.9 cpd 줄무늬의 밝은 줄에만 가로 1 px 간격 선(768×432 에서 약 11.6 cpd, 범위 밖)을 겹쳐도
    줄무늬 면적이 그대로. 전에는 그 픽셀들이 지워져 약 0.49."""
    h, w = 432, 768
    period = m.pixels_per_degree(w, h) / 2.9
    xs = np.arange(w)
    bright = (xs % period) < period / 2
    img = np.where(bright[None, :], 200, 20).astype(np.uint8)
    img = np.repeat(img, h, axis=0)
    lines = (np.arange(h) % 2 == 0)[:, None] & bright[None, :]
    img[lines] = 120                                   # 밝은 줄 안에서만 200 ↔ 120 (ΔL ≥ 20) 가로 줄
    frames = np.repeat(img[None, :, :, None], 3, axis=3)
    area = m.stripe_masks(m.luminance_cd(frames))[0].mean()
    assert area > 0.9
