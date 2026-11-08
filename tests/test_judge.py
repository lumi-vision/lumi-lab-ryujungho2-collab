"""휘도 깜빡임 판정: 알려진 자극에서 하드 판정이 맞는지, 소프트 판정이 하드와 같은 방향인지."""
import numpy as np
import pytest

from lumi.judge import judge_luminance_flash

FPS = 24


def gray(y: float) -> int:
    v = 12.92 * y if y <= 0.0031308 else 1.055 * y ** (1 / 2.4) - 0.055
    return int(round(v * 255))


def flash(hz, lo=0.0, hi=1.0, area=1.0, seconds=2.0, h=48, w=64):
    t = np.arange(int(FPS * seconds)) / FPS
    on = np.sin(2 * np.pi * hz * t) >= 0
    clip = np.full((len(t), h, w, 3), gray(lo), np.uint8)
    rh, rw = int(round(h * np.sqrt(area))), int(round(w * np.sqrt(area)))
    clip[on, :rh, :rw] = gray(hi)
    return clip


@pytest.mark.parametrize("hz,area,lo,hi,fail", [
    (4, 1.0, 0.0, 1.0, True),     # 전체 화면 흰↔검 4 Hz
    (1, 1.0, 0.0, 1.0, False),    # 1 Hz는 통과
    (4, 0.5, 0.3, 0.5, True),     # 국소(50%), ΔY 0.2 — IRIS는 놓치는 사례
    (4, 0.5, 0.44, 0.6, True),    # 국소(50%), ΔY 0.16 — IRIS는 놓치는 사례
    (4, 0.16, 0.0, 1.0, False),   # 면적 16%는 통과
    (4, 1.0, 0.5, 0.55, False),   # ΔY 0.05는 통과
    (4, 1.0, 0.85, 1.0, False),   # 어두운 쪽이 0.8 이상이면 통과
])
def test_hard_judgement(hz, area, lo, hi, fail):
    assert judge_luminance_flash(flash(hz, lo, hi, area), FPS).fail is fail


def test_static_clip_has_no_transitions():
    clip = np.full((30, 32, 32, 3), 128, np.uint8)
    r = judge_luminance_flash(clip, FPS)
    assert r.max_transitions_1s == 0 and not r.fail


def test_soft_count_tracks_hard():
    torch = pytest.importorskip("torch")
    from lumi.judge import soft_transition_count
    for hz, area, lo, hi in [(4, 1.0, 0.0, 1.0), (1, 1.0, 0.0, 1.0), (4, 0.5, 0.3, 0.5), (4, 0.16, 0.0, 1.0)]:
        clip = flash(hz, lo, hi, area)
        hard = judge_luminance_flash(clip, FPS).max_transitions_1s
        x = torch.from_numpy(clip).permute(3, 0, 1, 2).float().unsqueeze(0) / 255
        soft = float(soft_transition_count(x, FPS, tau_y=0.005, tau_area=0.005, beta=20)["count"])
        assert abs(soft - hard) < 1.0, (hz, area, soft, hard)


def test_soft_count_is_differentiable():
    torch = pytest.importorskip("torch")
    from lumi.judge import soft_transition_count
    x = (torch.from_numpy(flash(4, 0.3, 0.5, 0.5)).permute(3, 0, 1, 2).float().unsqueeze(0) / 255).requires_grad_()
    soft_transition_count(x, FPS)["count"].sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0
