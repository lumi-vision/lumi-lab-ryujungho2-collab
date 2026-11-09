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
