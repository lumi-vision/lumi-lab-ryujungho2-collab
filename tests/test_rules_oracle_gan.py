"""규칙(D1~D4)·오라클·GAN 의 핵심 성질을 고정하는 테스트. 각 테스트의 주석이 '왜 이 성질이 필요한가'다."""
import numpy as np
import pytest

from lumi_synth import measure, oracle

FPS = 30
H, W = 40, 40


def gray(cd: float) -> int:
    """휘도 cd/m² 에 가장 가까운 회색 값."""
    vals = measure.luminance_cd(np.repeat(np.arange(256, dtype=np.uint8)[:, None], 3, 1)[None])[0]
    return int(np.argmin(np.abs(vals - cd)))


LO, HI = gray(64), gray(104)          # 약 40 cd/m² 차이, 어두운 쪽 < 160


def flicker(cycle: int, frames: int = 90, split: bool = False) -> np.ndarray:
    """cycle 프레임마다 화면 전체가 밝기 반전. split=True 면 위 절반 → 다음 프레임 아래 절반으로 나눠 그림."""
    top = np.zeros((H, W), bool)
    top[: H // 2] = True
    cur, st, out = np.full((H, W), LO, np.uint8), LO, []
    for t in range(frames):
        if t % cycle == 0 and t > 0:
            st = HI if st == LO else LO
            if split:
                cur[top] = st
            else:
                cur[:] = st
        elif split and t % cycle == 1 and t > 1:
            cur[~top] = st
        out.append(np.repeat(cur[..., None], 3, 2).copy())
    return np.stack(out)


# ---------------------------------------------------------------- 규칙 D1~D4
@pytest.mark.parametrize("cycle, expected", [(15, 1.0), (8, 2.0), (4, 4.0)])
def test_d1_split_transition_counts_once(cycle, expected):
    # D1: 한 번의 변화가 두 프레임에 나뉘어 그려져도 1번. 2026-11-09 이전 구현은 2배로 세어
    #     초당 1.88번(안전)을 4번(위험)으로 판정했다.
    one, split = measure.analyze(flicker(cycle), FPS), measure.analyze(flicker(cycle, split=True), FPS)
    assert one["max_flashes"] == split["max_flashes"] == expected


@pytest.mark.parametrize("fps, cycle, merged", [(30, 4, False), (60, 8, True)])
def test_d3_sync_window_is_20ms_of_real_time(fps, cycle, merged):
    # D3 (2026-11-14): '동시' = 프레임 시각 차이 ≤ 20 ms. 20% + 20% 가 연속 두 프레임에 나뉜 3.75 Hz 깜빡임은
    #   60 fps(16.7 ms 차이)에서는 40% 로 합쳐 위험, 30 fps(33 ms 차이)에서는 합치지 않아 각 20% < 25% → 전이 없음.
    #   2026-11-09~11-14 구현(c = round(fps/15) 프레임)은 30 fps 에서도 합쳐 위험으로 봤다.
    m1, m2 = np.zeros((H, W), bool), np.zeros((H, W), bool)
    m1[:8], m2[8:16] = True, True
    cur, st, out = np.full((H, W), LO, np.uint8), LO, []
    for t in range(2 * fps):
        if t % cycle == 0 and t > 0:
            st = HI if st == LO else LO
            cur[m1] = st
        elif t % cycle == 1 and t > 1:
            cur[m2] = st
        out.append(np.repeat(cur[..., None], 3, 2).copy())
    r = measure.analyze(np.stack(out), fps)
    assert r["dangerous"] == merged
    if merged:
        assert r["hazard_mask"].mean() == pytest.approx(0.4)
    else:
        assert r["max_flashes"] == 0.0


@pytest.mark.parametrize("period, flashes", [(3, 5.0), (5, 3.0)])
def test_d4_checkerboard_reversal_counts_once_per_reversal(period, flashes):
    # D4 + '새 전이' 조건: 체커보드 반전 1번 = 화면 전이 1번. (조건이 없으면 5 Hz 를 10번으로 셌다)
    cb = (np.indices((H, W)).sum(0) // 4) % 2 == 0
    v = np.stack([np.repeat(np.where(cb ^ bool((t // period) % 2), HI, LO).astype(np.uint8)[..., None], 3, 2)
                  for t in range(60)])
    assert measure.analyze(v, FPS)["max_flashes"] == flashes


def test_every_frame_reversal_is_15hz():
    v = np.stack([np.full((H, W, 3), HI if t % 2 else LO, np.uint8) for t in range(60)])
    assert measure.analyze(v, FPS)["max_flashes"] == 15.0


def test_area_boundary_exactly_25_percent_is_safe():
    # AREA_LIMIT 는 '초과'. 정확히 25.00% 는 안전, 25.06% 는 위험.
    for npx, danger in ((400, False), (401, True)):
        m = np.zeros(H * W, bool)
        m[:npx] = True
        m = m.reshape(H, W)
        v = np.stack([np.where(m[..., None], HI if t % 4 < 2 else LO, LO).astype(np.uint8).repeat(3, 2)
                      for t in range(60)])
        assert measure.analyze(v, FPS)["dangerous"] is danger


def test_six_transitions_safe_seven_dangerous():
    # 1초(30프레임) 안 화면 전이 6번 = 깜빡임 3번 → 안전, 7번 → 위험
    for n, danger in ((6, False), (7, True)):
        levels = [LO] * 60
        for k in range(n):
            for t in range(1 + 4 * k, 5 + 4 * k):
                levels[t] = HI if k % 2 == 0 else LO
        for t in range(1 + 4 * n, 60):
            levels[t] = HI if n % 2 == 1 else LO     # 마지막 상태 유지(추가 전이 없음)
        v = np.stack([np.full((H, W, 3), x, np.uint8) for x in levels])
        assert measure.analyze(v, FPS)["dangerous"] is danger


def test_analyze_rejects_bad_input():
    with pytest.raises(ValueError):
        measure.analyze(np.zeros((1, 4, 4, 3), np.uint8), FPS)          # 프레임 1장
    with pytest.raises(ValueError):
        measure.analyze(np.zeros((5, 4, 4, 3), np.float32), FPS)        # uint8 아님
    with pytest.raises(ValueError):
        measure.analyze(np.zeros((5, 4, 4, 3), np.uint8), 0)            # fps 0


# ---------------------------------------------------------------- 오라클
def _bg(t=48, h=36, w=64, level=None):
    lv = gray(50) if level is None else level
    rng = np.random.default_rng(0)
    base = np.full((h, w, 3), lv, np.uint8) + rng.integers(0, 3, (h, w, 3), dtype=np.uint8)  # 약한 정지 질감
    return np.repeat(base[None], t, 0)


@pytest.mark.parametrize("intent", oracle.INTENTS)
def test_oracle_intent_matches_judge_on_plain_background(intent):
    # 어두운 정지 바탕에서는 설계 의도와 판정이 항상 같아야 한다(의도 범위가 경계에서 충분히 떨어짐).
    rng = np.random.default_rng(1)
    for _ in range(5):
        r = oracle.synthesize(_bg(), oracle.sample_params(intent, rng), 24.0)
        assert r["intent_ok"], (intent, r["params"], r["judge"]["max_flashes"], r["judge"]["peak_area"])


def test_oracle_mask_area_and_locality():
    # 데이터셋 해상도(128×72)에서 면적 오차 ≤ 0.025 — 의도 범위가 경계(0.25)에서 0.05 이상 떨어져 있으므로 충분.
    # (blobs 는 원 중심이 크기와 함께 움직여 0.466 목표에 0.106 이 나오던 버그를 2026-11-09 에 고쳤다)
    rng = np.random.default_rng(2)
    for _ in range(20):
        p = oracle.sample_params(oracle.INTENTS[int(rng.integers(4))], rng)
        clean = _bg(t=12, h=72, w=128)
        r = oracle.synthesize(clean, p, 24.0)
        assert abs(r["mask_allow"].mean() - p.area) <= 0.025, (p.shape, p.area, r["mask_allow"].mean())
        assert (r["flicker"][:, ~r["mask_allow"]] == clean[:, ~r["mask_allow"]]).all()   # 마스크 밖 = 원본


def test_oracle_additive_light_sets_luminance_exactly():
    # 선형 RGB 세 채널에 δ 를 더하면 휘도가 정확히 ΔL 만큼 오른다(8비트 양자화 오차 ~1 cd/m² 이내).
    p = oracle.OracleParams("hazard", 1.0, "full", (0.5, 0.5), 1.0, 4.0, "square", 0.0, 40.0, 0)
    clean = _bg(level=gray(40))
    r = oracle.synthesize(clean, p, 24.0)
    on = r["w"] > 0.5
    dl = measure.luminance_cd(r["flicker"][on]).mean() - measure.luminance_cd(clean[on]).mean()
    assert dl == pytest.approx(40.0, abs=1.0)


# ---------------------------------------------------------------- GAN
torch = pytest.importorskip("torch")


def test_gan_locality_and_shapes():
    from lumi_synth.gan import LightGenerator
    g = LightGenerator(width=8, zdim=4, fps=24.0)
    clean = torch.rand(2, 3, 24, 36, 64)
    mask = torch.zeros(2, 1, 1, 36, 64)
    mask[..., 10:26, 20:44] = 1
    out, delta = g(clean, mask, torch.tensor([1.0, 0.0]), torch.randn(2, 4))
    assert out.shape == clean.shape and delta.shape == (2, 1, 24, 36, 64)
    assert torch.equal(out * (1 - mask), clean * (1 - mask))            # 마스크 밖 = 정확히 원본
    assert float(delta.min()) >= 0.0                                     # 빛은 더하기만


def test_soft_judge_tracks_hard_judge_on_simple_flicker():
    # 대리 판정기는 단순한 화면 깜빡임에서 하드 판정의 전이 수와 거의 같아야 한다.
    from lumi_synth.gan import luminance_rel, soft_judge, srgb_to_linear
    for cycle in (15, 8, 4, 3):
        v = flicker(cycle, frames=60)                                    # (T,H,W,3) BGR
        hard = 2 * measure.analyze(v, FPS)["max_flashes"]
        x = torch.from_numpy(v).permute(3, 0, 1, 2)[None].float() / 255
        soft = float(soft_judge(luminance_rel(srgb_to_linear(x)), FPS)["count"][0])
        assert abs(soft - hard) <= 1.0, (cycle, soft, hard)


def test_judge_loss_gradient_reaches_generator():
    from lumi_synth.gan import LightGenerator, judge_loss, luminance_rel, srgb_to_linear
    # 시드 고정 (2026-11-09): 고정 전에는 전체 실행 3번 중 1번 실패했다. 무작위 초기값·입력에 따라 손실이
    # 0(목표를 이미 만족)이 되어 기울기가 0인 경우가 있었기 때문이다. 이 테스트는 '기울기 경로가 이어져 있는가'를
    # 보는 것이므로 결과가 매번 같도록 시드를 고정한다.
    torch.manual_seed(0)
    g = LightGenerator(width=8, zdim=4, fps=24.0)
    clean = torch.rand(2, 3, 24, 32, 32) * 0.3
    mask = torch.ones(2, 1, 1, 32, 32)
    out, _ = g(clean, mask, torch.tensor([1.0, 0.0]), torch.randn(2, 4))
    loss, _ = judge_loss(luminance_rel(srgb_to_linear(out)), torch.tensor([1.0, 0.0]), 24.0)
    loss.backward()
    assert any(p.grad is not None and float(p.grad.abs().sum()) > 0 for p in g.parameters())


@pytest.mark.parametrize("hz", [2.0, 2.5, 4.0])
def test_soft_judge_counts_opposite_phase_once(hz):
    # 독립 리뷰(2026-11-09): 소프트 판정 v0 은 반대 위상(왼쪽 밝아짐 + 오른쪽 어두워짐)을 2번으로 세어
    # 2 Hz 안전 영상을 8.8(위험 목표 충족)로 냈다. G 가 이 빈틈으로 '가짜 위험'을 만들 수 있었다.
    from lumi_synth.gan import luminance_rel, soft_judge, srgb_to_linear
    fps, t = 24, np.arange(72) / 24
    ph = (t * hz * 2).astype(int) % 2
    v = np.zeros((72, 36, 64, 3), np.uint8)
    for i in range(72):
        v[i, :, :32], v[i, :, 32:] = (HI, LO) if ph[i] else (LO, HI)
    hard = 2 * measure.analyze(v, fps)["max_flashes"]
    x = torch.from_numpy(v).permute(3, 0, 1, 2)[None].float() / 255
    soft = float(soft_judge(luminance_rel(srgb_to_linear(x)), fps)["count"][0])
    assert abs(soft - hard) <= 0.5, (hz, soft, hard)


# ---------------------------------------------------------------- 트랙(해상도·fps)
def test_normalized_diversity_ignores_intensity_and_sees_rhythm():
    # 다양성 손실(정규화): 세기만 2배인 빛은 차이 0, 리듬(주파수)이 다른 빛은 차이가 난다.
    # 정규화하지 않으면 세기 차이만으로도 손실이 줄어든다 — λ_div 를 키울 때 빛이 세지던 원인으로 추정한 것.
    import math

    from lumi_synth.gan import spectral_diversity
    t = torch.arange(48) / 24.0
    light = lambda hz, amp: (amp * (0.5 + 0.5 * torch.sin(2 * math.pi * hz * t)))[None, None, :, None, None].expand(1, 1, 48, 4, 4)
    mask = torch.ones(1, 1, 1, 4, 4)
    z1, z2 = torch.zeros(1, 4), torch.ones(1, 4)
    same_rhythm = float(spectral_diversity(light(4, 0.2), light(4, 0.4), mask, z1, z2, normalize=True))
    other_rhythm = float(spectral_diversity(light(4, 0.2), light(7, 0.2), mask, z1, z2, normalize=True))
    assert abs(same_rhythm) < 1e-4 and other_rhythm < -0.02
    assert float(spectral_diversity(light(4, 0.2), light(4, 0.4), mask, z1, z2)) < -0.05     # 정규화 없음: 세기 차이가 보임


# ---------------------------------------------------------------- 적색·줄무늬 GAN (2026-11-14)
@pytest.mark.parametrize("hz", [2.0, 5.0])
def test_red_soft_judge_tracks_hard_judge(hz):
    # 적색 소프트 판정기: 규칙 기반 적색 깜빡임(같은 휘도의 빨강 ↔ 파랑 쪽 색)에서 하드 판정의 전이 수와 1번 이내로 맞는다.
    from lumi_synth.gan import soft_judge_red, srgb_to_linear
    clean = np.full((48, 36, 64, 3), 128, np.uint8)
    mask = oracle.region_mask("rectangle", 0.5, 36, 64)
    pair = oracle.red_pair(0.9, 0.28)
    v = oracle.render_red(clean, mask, oracle.wave("square", hz, 48, 24.0), pair["a_rgb"], pair["b_rgb"])
    hard = measure.analyze_red(v, 24.0)
    x = torch.from_numpy(v).permute(3, 0, 1, 2)[None].float() / 255
    soft = float(soft_judge_red(srgb_to_linear(x), 24.0)["count"][0])
    assert abs(soft - 2 * hard["max_flashes"]) <= 1.0
    assert (soft >= 7) == hard["dangerous"]


@pytest.mark.parametrize("hazard", ["red", "stripe"])
def test_hazard_generators_keep_outside_of_mask(hazard):
    from lumi_synth.gan import STRIPE_HI, STRIPE_SPEED_MAX, make_generator
    torch.manual_seed(0)
    g = make_generator(hazard, width=8, zdim=4, fps=24.0)
    clean = torch.rand(2, 3, 24, 36, 64)
    mask = torch.zeros(2, 1, 1, 36, 64)
    mask[..., 8:28, 16:48] = 1
    outs = g(clean, mask, torch.tensor([1.0, 0.0]), torch.randn(2, 4))
    out, field = outs[0], outs[1]
    assert out.shape == clean.shape and field.shape == (2, 1, 24, 36, 64)
    assert torch.equal(out * (1 - mask), clean * (1 - mask))            # 마스크 밖 = 정확히 원본
    assert float(field.min()) >= 0.0 and float(field.max()) <= 1.0       # 빨강 정도 ρ · 덮는 정도 a
    assert not torch.equal(out * mask, clean * mask)                     # 마스크 안은 바뀜
    if hazard == "stripe":
        cpd, theta, speed, hi = outs[2].unbind(1)
        assert outs[2].shape == (2, 4) and bool((cpd > 0).all()) and bool(((theta >= 0) & (theta < 3.1416)).all())
        assert bool((speed.abs() <= STRIPE_SPEED_MAX).all()) and bool(((hi >= STRIPE_HI[0]) & (hi <= STRIPE_HI[1])).all())
    with pytest.raises(ValueError):
        make_generator("nope")


def test_red_generator_full_blend_is_saturated_red_with_same_luminance():
    # 섞는 식: ρ = 1 이면 그 픽셀은 포화 빨강(판정기 기준)이 되고, 어두운 장면(Y ≤ 0.2126)에서는 휘도가 그대로다.
    from lumi_synth.gan import DELTA_MAX, RedGenerator, luminance_rel, srgb_to_linear

    class Full(RedGenerator):
        def light(self, clean, mask, target, z):
            return torch.full_like(clean[:, :1], DELTA_MAX)

    clean = torch.rand(1, 3, 4, 8, 8) * 0.4 + 0.05                       # 선형 휘도 약 0.003~0.17
    out, _ = Full(8, 4, 24.0)(clean, torch.ones(1, 1, 1, 8, 8), torch.ones(1), torch.zeros(1, 4))
    x = (out[0].permute(1, 2, 3, 0) * 255).round().byte().numpy()
    sat, _, _ = measure.red_state(x)
    assert sat.all()
    y0, y1 = luminance_rel(srgb_to_linear(clean)), luminance_rel(srgb_to_linear(out))
    keep = y0 >= 0.2126 * 0.10                                           # 그보다 어두운 픽셀은 빨강 하한 때문에 밝아진다
    assert float((y1 - y0)[keep].abs().max()) < 2e-3


def test_stripe_generator_draws_stripes_the_detector_finds():
    # 섞는 식: a = 1 이면 영역이 그 cpd 의 줄무늬로 덮이고, 줄무늬 탐지기(쿼드트리+가버)가 영역 면적만큼 찾는다.
    from lumi_synth import stripe_freq as sf
    from lumi_synth.gan import DELTA_MAX, StripeGenerator

    class Fixed(StripeGenerator):
        def light(self, clean, mask, target, z):
            return torch.full_like(clean[:, :1], DELTA_MAX)

        def stripe_params(self, target, z, ppd):
            return torch.tensor([[2.0, 0.6, 0.0, 0.55]])                 # 2 cpd, 약 34°, 정지, 밝은 줄 110 cd/m²

    h, w = 216, 384
    mask = oracle.region_mask("rectangle", 0.6, h, w)
    clean = torch.full((1, 3, 2, h, w), 0.5)
    out, amp, par = Fixed(8, 4, 24.0)(clean, torch.from_numpy(mask[None, None, None]).float(), torch.ones(1), torch.zeros(1, 4))
    x = (out[0].permute(1, 2, 3, 0) * 255).round().byte().numpy()
    found = sf.frame_stripes_quadtree(measure.luminance_cd(x[0]), None, refine=True)["mask"]
    assert found.mean() == pytest.approx(mask.mean(), abs=0.05)
    assert (found & ~mask).mean() < 0.03                                  # 영역 밖을 줄무늬로 칠하지 않음


def test_stripe_judge_loss_targets_and_gradient():
    from lumi_synth.gan import StripeGenerator, judge_loss_stripe, soft_judge_stripe, stripe_diversity
    full, none = torch.ones(1, 1, 48, 8, 8), torch.zeros(1, 1, 48, 8, 8)
    brief = none.clone()
    brief[:, :, :4] = 1                                                  # 4프레임(0.17초)만 화면 전체
    hi = torch.tensor([0.55])
    assert float(soft_judge_stripe(full, hi, 24.0)["duration"][0]) == pytest.approx(1.0, abs=0.02)
    assert float(judge_loss_stripe(full, hi, torch.ones(1), 24.0)[0]) < 1e-3      # 계속 덮음 = 위험 목표 충족
    assert float(judge_loss_stripe(full, hi, torch.zeros(1), 24.0)[0]) > 1.0      # 근처 안전 목표에는 어긋남
    assert float(judge_loss_stripe(brief, hi, torch.zeros(1), 24.0)[0]) < 1e-3    # 잠깐만 덮음 = 근처 안전 충족
    assert float(judge_loss_stripe(none, hi, torch.zeros(1), 24.0)[0]) > 1.0      # 아무것도 없음은 근처 안전이 아님
    torch.manual_seed(0)
    g = StripeGenerator(8, 4, 24.0)
    clean, mask = torch.rand(2, 3, 24, 32, 32), torch.ones(2, 1, 1, 32, 32)
    z1, z2 = torch.randn(2, 4), torch.randn(2, 4)
    o1, o2 = g(clean, mask, torch.tensor([1.0, 0.0]), z1), g(clean, mask, torch.tensor([1.0, 0.0]), z2)
    loss = judge_loss_stripe(mask * o1[1], o1[2][:, 3], torch.tensor([1.0, 0.0]), 24.0)[0] + stripe_diversity(o1[2], o2[2], z1, z2)
    loss.backward()
    assert g.head.weight.grad.abs().sum() > 0 and g.par[0].weight.grad.abs().sum() > 0


def test_track_tag():
    # 판정 해상도 = 학습 해상도 = 768×432 로 통일(2026-11-09). 128×72 는 예비 실험 재현용.
    from lumi_synth.data import ClipSpec
    assert ClipSpec().tag == "768x432_24fps"
    assert ClipSpec(128, 72).tag == "128x72_24fps"


def test_read_clip_refuses_upscaling(tmp_path):
    # 원본(64×36)보다 큰 목표(128×72)는 확대하지 않고 멈춘다 — 480×270 Elephants Dream 이 조용히 확대될 뻔한 일 방지.
    import cv2
    from lumi_synth.data import ClipSpec, read_clip
    path = tmp_path / "small.avi"
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 24.0, (64, 36))
    for _ in range(5):
        vw.write(np.full((36, 64, 3), 100, np.uint8))
    vw.release()
    with pytest.raises(ValueError, match="확대 금지"):
        read_clip(path, 0, ClipSpec(128, 72, 4, 24.0))
    assert read_clip(path, 0, ClipSpec(32, 18, 4, 24.0)).shape == (4, 18, 32, 3)   # 축소는 허용


def test_24_to_30_duplication_adds_no_flicker_and_keeps_rate():
    # PLAN.md 의 24→30 fps 변환(원본 4장마다 1장 복제)이 (1) 새 깜빡임을 만들지 않고 (2) 깜빡임 빈도를 유지하는지.
    from lumi_synth.data import duplicate_schedule
    idx = duplicate_schedule(150, 24.0, 30.0)
    assert idx[0] == 0 and idx[-1] == 119 and (np.diff(idx) >= 0).all() and (np.diff(idx) <= 1).all()
    assert len(set(idx.tolist())) == 120                        # 원본 120장(5초)을 빠짐없이 사용
    still = np.repeat(np.full((1, 8, 8, 3), LO, np.uint8), 120, 0)
    assert measure.analyze(still[idx], 30.0)["max_flashes"] == 0
    t = np.arange(120) / 24.0
    src = np.stack([np.full((8, 8, 3), HI if int(t_ * 5 * 2) % 2 else LO, np.uint8) for t_ in t])  # 5 Hz @24fps
    before = measure.analyze(src, 24.0)["max_flashes"]
    after = measure.analyze(src[idx], 30.0)["max_flashes"]
    assert before == after == 5.0


# ---------------------------------------------------------------- 화면 비율(여백 처리, 2026-11-09)
def test_content_box_aspect_ratios():
    # 16:9 화면(128×72)에 넣을 때 영상 영역: 비율 유지, 가운데, 남는 쪽은 여백.
    from lumi_synth.data import ClipSpec, content_box
    spec = ClipSpec(128, 72)
    assert content_box(1280, 720, spec) == (0, 0, 128, 72)        # 16:9 → 여백 없음
    assert content_box(640, 359, spec) == (0, 0, 128, 72)         # BBB 640×359: 1픽셀 이하 차이는 꽉 채움
    assert content_box(640, 480, spec) == (16, 0, 96, 72)         # 4:3 → 좌우 여백 16
    x0, y0, w, h = content_box(720, 1280, spec)                   # 9:16 세로 → 좌우 큰 여백
    assert (y0, h) == (0, 72) and w == round(720 * 72 / 1280) and x0 == (128 - w) // 2
    assert content_box(2560, 1080, spec)[1] > 0                   # 21:9 → 위아래 여백
    with pytest.raises(ValueError, match="확대 금지"):
        content_box(64, 36, spec)


def test_letterbox_keeps_content_and_black_bars():
    from lumi_synth.data import ClipSpec, content_box, letterbox
    spec = ClipSpec(128, 72)
    frame = np.full((480, 640, 3), 150, np.uint8)                 # 4:3 회색
    out = letterbox(frame, spec)
    x0, y0, w, h = content_box(640, 480, spec)
    assert out.shape == (72, 128, 3)
    assert (out[:, :x0] == 0).all() and (out[:, x0 + w:] == 0).all()   # 여백은 검정
    assert (out[y0:y0 + h, x0:x0 + w] == 150).all()                # 영상 영역은 그대로(축소만)
    same = np.random.default_rng(0).integers(0, 256, (72, 128, 3), dtype=np.uint8)
    assert letterbox(same, spec) is same                            # 크기·비율이 같으면 손대지 않음


def test_portrait_flash_area_is_measured_against_screen(tmp_path):
    # ITU 면적 = '화면에 표시된 면적' 대비. 세로 영상의 70% 가 깜빡여도 16:9 화면 기준으로는 약 22% → 안전.
    # 여백 없이 늘리면 영상 프레임 기준 70% → 위험으로 잘못 판정된다.
    import cv2
    from lumi_synth import measure
    from lumi_synth.data import ClipSpec, read_clip
    sw, sh = 144, 256                                              # 9:16 세로
    path = tmp_path / "portrait.avi"
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 24.0, (sw, sh))
    for t in range(24):
        f = np.full((sh, sw, 3), 60, np.uint8)
        if int(2 * 5 * t / 24) % 2 == 0:                           # 5 Hz
            f[: int(sh * 0.7)] = 200
        vw.write(f)
    vw.release()
    spec = ClipSpec(128, 72, 24, 24.0)
    boxed = read_clip(path, 0, spec)
    r = measure.analyze(boxed, 24.0)
    assert r["peak_area"] < measure.AREA_LIMIT and not r["dangerous"]
    assert abs(r["peak_area"] - 0.7 * (40 * 72) / (128 * 72)) < 0.03    # 영상 영역 40×72 의 70%
    stretched = np.stack([cv2.resize(f, (128, 72), interpolation=cv2.INTER_AREA) for f in _read_all(path)])
    assert measure.analyze(stretched, 24.0)["dangerous"]           # 비율 무시(이전 방식) → 위험으로 잘못 판정


def _read_all(path):
    import cv2
    cap, out = cv2.VideoCapture(str(path)), []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        out.append(f)
    cap.release()
    return out


def test_red_crisp_loss_prefers_one_frame_switches():
    """한 프레임에 켜지고 꺼지는 ρ(사각파)는 손실 0, 서서히 바뀌는 ρ 는 손실이 크다. 마스크 밖은 재지 않는다."""
    torch = pytest.importorskip("torch")
    from lumi_synth import gan
    t = torch.arange(48).float()
    shape = (1, 1, 48, 4, 4)
    square = ((t // 3) % 2).view(1, 1, 48, 1, 1).expand(shape)
    slow = (0.5 + 0.5 * torch.sin(2 * torch.pi * t / 12)).view(1, 1, 48, 1, 1).expand(shape)
    mask = torch.ones(1, 1, 1, 4, 4)
    assert float(gan.red_crisp_loss(square, mask)) == 0.0
    assert float(gan.red_crisp_loss(slow, mask)) > 0.15
    assert float(gan.red_crisp_loss(slow, torch.zeros_like(mask))) == 0.0


def test_red_palette_states_are_red_and_not_red_and_keep_outside():
    """적색 v2(palette): 빨강 상태는 포화 빨강, 빨강 아닌 상태는 아님. 평평한 빨강 ↔ 평평한 다른 색 4 Hz 사각파는 적색 위험. 마스크 밖은 원본."""
    torch = pytest.importorskip("torch")
    from lumi_synth import gan
    rng = np.random.default_rng(0)
    clean_u8 = rng.integers(30, 200, (1, 48, 36, 64, 3)).astype(np.uint8)
    clean = torch.from_numpy(clean_u8).permute(0, 4, 1, 2, 3).float() / 255
    lin = gan.srgb_to_linear(clean)
    for p in ([1, 0, 0.4, 0, 0.5, 0.5, 0.5, 0, 0.5], [0.3, 1, 0.6, 1, 1, 0, 0.6, 1, 0.3], [0.5, 1, 0.1, 0.8, 0, 1, 0, 1, 0.6]):
        other, red = gan.red_palette(lin, torch.tensor([p], dtype=torch.float32))
        ratio = red[:, 2] / red.sum(1)
        assert float(ratio.min()) >= measure.RED_SAT_RATIO and float((red[:, 2] - red[:, 0] - red[:, 1]).min()) >= measure.RED_MIN_EXCESS
        rho = ((torch.arange(48) // 3) % 2).float().view(1, 1, 48, 1, 1)
        mask = torch.zeros(1, 1, 1, 36, 64)
        mask[..., 4:32, 6:58] = 1
        out = gan.linear_to_srgb((1 - mask) * lin + mask * ((1 - rho) * other + rho * red))
        x = (out[0].permute(1, 2, 3, 0) * 255).round().clamp(0, 255).byte().numpy()
        assert measure.analyze_red(x, 24.0)["dangerous"]
        assert np.abs(x.astype(int) - clean_u8[0])[:, ~mask[0, 0, 0].bool().numpy()].max() <= 1
    g = gan.make_generator("red", 4, 8, 24.0, red_palette=True)
    o = g(clean, mask[:, :, :, :, :], torch.ones(1), torch.randn(1, 8))
    assert len(o) == 3 and o[2].shape == (1, gan.RED_PALETTE_N)
    assert torch.equal(o[0][..., ~mask[0, 0, 0].bool()], clean[..., ~mask[0, 0, 0].bool()])
