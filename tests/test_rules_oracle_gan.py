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


def test_d3_two_partial_frames_are_concurrent():
    # D3: 20% + 20% 가 연속 두 프레임에 나뉘면 동시(40%)로 본다. 3.75 Hz → 위험.
    m1, m2 = np.zeros((H, W), bool), np.zeros((H, W), bool)
    m1[:8], m2[8:16] = True, True
    cur, st, out = np.full((H, W), LO, np.uint8), LO, []
    for t in range(60):
        if t % 4 == 0 and t > 0:
            st = HI if st == LO else LO
            cur[m1] = st
        elif t % 4 == 1 and t > 1:
            cur[m2] = st
        out.append(np.repeat(cur[..., None], 3, 2).copy())
    r = measure.analyze(np.stack(out), FPS)
    assert r["dangerous"] and r["hazard_mask"].mean() == pytest.approx(0.4)


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
