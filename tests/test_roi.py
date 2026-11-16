"""탐지·보정 부품(lumi_synth.roi)의 성질: 라벨이 판정기에서 나오는가, 튜브 밖을 건드리지 않는가, 지표의 기준값.

작은 합성 바탕으로만 시험한다(data/ 불필요, 영상은 메모리에서만). 줄무늬 판정은 느려서 한 시험에서만 켠다."""
import numpy as np
import pytest
import torch

from lumi_synth import generate as gen, measure, oracle, roi

FPS = 24.0
T, H, W = 48, 96, 160


def clean_clip() -> np.ndarray:
    row = np.linspace(40, 120, W).astype(np.uint8)
    return np.broadcast_to(row[None, None, :, None], (T, H, W, 3)).copy()


def flicker_clip(freq_hz: float):
    clean = clean_clip()
    mask = oracle.region_mask("rectangle", 0.5, H, W)
    return clean, mask, oracle.render_flicker(clean, mask, oracle.wave("square", freq_hz, T, FPS), 60.0)


def test_tubes_come_from_the_judge_not_from_the_intent():
    clean, mask, hot = flicker_clip(6.0)                     # 6 Hz → 위험
    tubes, su = roi.judge_tubes(hot, FPS, stripes=False)
    assert su["lum"]["dangerous"] and su["dangerous"]
    assert tubes["lum"].shape == (T, H, W) and tubes["lum"].any()
    assert not tubes["lum"][:, ~mask].any()                  # 튜브는 빛을 넣은 영역 안에만
    assert tubes["lum"].any(axis=0)[mask].mean() > 0.9       # 그 영역 대부분을 덮는다
    assert not tubes["red"].any() and not tubes["stripe"].any()

    _, _, slow = flicker_clip(2.0)                           # 2 Hz → 같은 영역·같은 빛이어도 안전 → 튜브가 빈다
    tubes, su = roi.judge_tubes(slow, FPS, stripes=False)
    assert not su["dangerous"] and not roi.stack_tubes(tubes).any()

    tubes, su = roi.judge_tubes(clean, FPS, stripes=False)
    assert not su["dangerous"] and not roi.stack_tubes(tubes).any()


def test_stripe_tube_marks_stripe_pixels():
    clean = np.full((T, 216, 384, 3), 90, np.uint8)
    r = gen.sample_recipe("stripe", "hazard", 1, split="val", bg=0, shape="rectangle", area=0.7, cpd=3.0, angle=0.0, motion="static")
    out = gen.render(r, clean, FPS)
    tubes, su = roi.judge_tubes(out["flicker"], FPS)
    assert su["stripe"]["dangerous"]
    inside = tubes["stripe"].any(axis=0)
    assert inside[out["mask"]].mean() > 0.7 and inside[~out["mask"]].mean() < 0.15


def test_roinet_shape_and_pooling():
    net = roi.RoiNet(width=8)
    x = torch.rand(1, 3, 12, 64, 96)
    assert net(x).shape == (1, 3, 12, 16, 24)
    tube = torch.zeros(1, 3, 12, 64, 96, dtype=torch.bool)
    tube[:, 0, :, :32] = True
    pooled = roi.pool_tubes(tube)
    assert pooled.shape == (1, 3, 12, 16, 24)
    assert float(pooled[:, 0].mean()) == pytest.approx(0.5) and float(pooled[:, 1:].sum()) == 0.0
    assert float(roi.roi_loss(torch.full_like(pooled, -20.0), torch.zeros_like(pooled))) < 1e-3   # 빈 목표·빈 예측 → 0


def test_corrector_keeps_everything_outside_the_tube():
    torch.manual_seed(0)
    net = roi.Corrector(width=8)
    x = torch.rand(1, 3, 8, 32, 48)
    m = torch.zeros(1, 3, 8, 32, 48)
    m[:, 1, :, 8:20, 10:30] = 1.0
    assert torch.equal(net(x, m), x)                         # 처음에는(출력층 0) 아무것도 바꾸지 않는다
    torch.nn.init.normal_(net.out.weight, std=0.5)
    y = net(x, m)
    changed = (y != x).any(dim=1)[0]
    inside = m.amax(dim=1)[0] > 0
    assert changed.any() and not changed[~inside].any()      # 튜브 밖은 구조적으로 입력 그대로
    assert torch.equal(net(x, torch.zeros_like(m)), x)       # 튜브가 비면 입력 그대로


def test_soften_grows_and_stays_in_range():
    m = torch.zeros(1, 1, 2, 20, 20)
    m[..., 8:12, 8:12] = 1.0
    s = roi.soften(m, grow=2, blur=2)
    assert s.shape == m.shape and float(s.min()) >= 0.0 and float(s.max()) <= 1.0
    assert (s > 0).sum() > m.sum() and bool((s[m > 0] > 0.5).all())
    assert float(s[..., 0, 0].max()) == 0.0


def test_metrics_have_the_documented_reference_values():
    clean, mask, hot = flicker_clip(6.0)
    region = np.broadcast_to(mask, (T, H, W))
    assert roi.psnr(clean, clean, region) == float("inf")
    assert roi.psnr(hot, clean, region) < 30
    assert roi.texture_ratio(clean, clean, region) == pytest.approx(1.0)
    rng = np.random.default_rng(0)
    moving = np.clip(clean.astype(int) + rng.integers(-3, 4, clean.shape), 0, 255).astype(np.uint8)
    assert roi.temporal_ratio(moving, moving, region) == pytest.approx(1.0)
    assert roi.temporal_ratio(np.clip(hot.astype(int) + (moving.astype(int) - clean), 0, 255).astype(np.uint8), moving, region) > 2
    assert roi.seam_ratio(clean, clean, region) == pytest.approx(1.0)
    assert roi.seam_ratio(hot, clean, region) > 1.5          # 빛을 넣은 네모의 경계가 이음새로 잡힌다
    assert roi.tube_iou(region, region) == 1.0 and np.isnan(roi.tube_iou(~region & region, ~region & region))
    prob = np.zeros((T, 24, 40), np.float32)
    assert not roi.clip_decision(prob)
    prob[3, :12, :20] = 0.9                                  # 한 프레임에서 화면의 25%
    assert roi.clip_decision(prob)


def test_rule_correction_makes_the_clip_safe_and_stays_inside_the_tube():
    clean, mask, hot = flicker_clip(6.0)
    tubes, _ = roi.judge_tubes(hot, FPS, stripes=False)
    y = roi.rule_correct(hot, tubes, FPS)
    assert not measure.analyze(y, FPS)["dangerous"]
    outside = ~roi.stack_tubes(tubes).any(axis=0)
    assert np.array_equal(y[outside], hot[outside])
    assert roi.psnr(y, clean, ~outside) > roi.psnr(hot, clean, ~outside)
