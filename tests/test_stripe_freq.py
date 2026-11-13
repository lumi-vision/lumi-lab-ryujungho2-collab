"""주파수 방식 줄무늬 탐지기(stripe_freq)와 E-Means 직접 구현의 기본 동작."""
import numpy as np

from lumi_synth import measure, stripe_freq as sf

W, H = 384, 216
PPD = measure.pixels_per_degree(W, H)


def _frame(kind, cpd, lo=10.0, hi=110.0, region=0.55, **kw):
    pat, _ = sf.render_pattern(kind, H, W, PPD, cpd, lo, hi, **kw)
    lum = np.full((H, W), 60.0, np.float32)
    lum[:, : int(W * region)] = pat[:, : int(W * region)]
    return measure.luminance_cd(sf.cd_to_bgr(lum)[None])[0]


def test_patch_features_find_frequency_and_direction():
    p = sf.patch_size(W, H)
    pat, _ = sf.render_pattern("straight", p, p, PPD, 3.0, 10, 110, angle=0.0)
    f = sf.patch_features(pat[None], PPD)
    assert abs(f["cpd"][0] - 3.0) < 0.6
    assert min(f["theta"][0], 180 - f["theta"][0]) < 10          # 밝기가 x 방향으로 바뀜 = 0°
    assert f["rho1"][0] > 0.6 and f["rho2"][0] < 0.1


def test_flat_noise_and_checker_are_not_stripes():
    p = sf.patch_size(W, H)
    rng = np.random.default_rng(0)
    flat = np.full((p, p), 50.0, np.float32)
    noise = (60 + rng.normal(0, 30, (p, p))).astype(np.float32)
    checker, _ = sf.render_pattern("checker", p, p, PPD, 3.0, 10, 110)
    f = sf.patch_features(np.stack([flat, noise, checker]), PPD)
    assert not sf.rule_ok(f).any()


def test_straight_stripes_area_and_out_of_range():
    inside = sf.frame_stripes(_frame("straight", 3.0), None)["mask"].mean()
    outside = sf.frame_stripes(_frame("straight", 12.0), None)["mask"].mean()
    weak = sf.frame_stripes(_frame("straight", 3.0, lo=40, hi=55), None)["mask"].mean()
    assert 0.5 <= inside <= 0.65                                    # 영역 55% (+ 경계 패치 반 칸)
    assert outside < 0.05 and weak < 0.05


def test_curved_stripes_detected():
    area = sf.frame_stripes(_frame("concentric", 3.0, center=(0.5, 0.25)), None)["mask"].mean()
    assert area > 0.4


def test_quadtree_low_frequency_and_boundary():
    """쿼드트리: 큰 블록이라 1.2 cpd(2도 패치에 2.5주기뿐)도 잡고, 영역 경계를 넘쳐 칠하는 양이 작다."""
    low = sf.frame_stripes_quadtree(_frame("straight", 1.2), None)["mask"].mean()
    mid = sf.frame_stripes_quadtree(_frame("straight", 3.0), None)["mask"].mean()
    assert 0.5 <= low <= 0.65
    assert 0.53 <= mid <= 0.59


def test_quadtree_rejects_flat_checker_and_out_of_range():
    for lum in (np.full((H, W), 60.0, np.float32), _frame("checker", 3.0), _frame("straight", 12.0),
                _frame("straight", 3.0, lo=40, hi=55)):
        assert sf.frame_stripes_quadtree(lum, None)["mask"].mean() < 0.05


def test_gabor_refine_trims_boundary_overpaint():
    """가버 다듬기: 쿼드트리가 경계 블록을 통째로 칠해 넘친 면적을 깎는다(직선·동심원 모두 정답 55% 에 가깝게)."""
    for lum in (_frame("straight", 3.0), _frame("straight", 1.2), _frame("concentric", 3.0, center=(0.5, 0.25))):
        plain = sf.frame_stripes_quadtree(lum, None)["mask"].mean()
        refined = sf.frame_stripes_quadtree(lum, None, refine=True)["mask"].mean()
        assert refined <= plain + 1e-9
        assert abs(refined - 0.55) < 0.02


def test_quadtree_concentric_center_not_dropped():
    """동심원 중심 블록(스펙트럼이 원으로 퍼짐)도 쪼개서 잡는다 — 2026-11-13 QT_SPLIT_RHO 0.2 일 때 중심 띠가 빠졌음."""
    lum = _frame("concentric", 3.0, center=(0.5, 0.25))
    assert sf.frame_stripes_quadtree(lum, None)["mask"].mean() > 0.5


def test_analyze_pattern_freq_verdict():
    lum = _frame("straight", 3.0)
    frames = np.repeat(sf.cd_to_bgr(lum)[None], 4, axis=0)
    r = sf.analyze_pattern_freq(frames, 4.0, model=None)
    assert r["dangerous"] and not r["emeans"]
    r0 = sf.analyze_pattern_freq(np.repeat(sf.cd_to_bgr(np.full((H, W), 60.0))[None], 4, axis=0), 4.0, model=None)
    assert not r0["dangerous"]


def _inertia(x, c):
    return float(((x - c[sf._assign(x, c)]) ** 2).sum(axis=1).mean())


def _lloyd(x, c, it=25):
    c = c.copy()
    for _ in range(it):
        a = sf._assign(x, c)
        for j in range(len(c)):
            if (a == j).any():
                c[j] = x[a == j].mean(0)
    return c


def test_emeans_as_good_as_lloyd_from_random_init():
    """같은 무작위 초기화의 Lloyd 25회와 비교해 군집 품질(평균 제곱 거리)이 나쁘지 않아야 한다.
    (잘 떨어진 군집도 무작위 초기화면 둘 다 국소 최솟값에 빠질 수 있다 — 2026-11-13 확인: seed 8개 중 6개가 둘 다 같음)"""
    rng = np.random.default_rng(0)
    x = np.concatenate([c + rng.normal(0, 0.5, (400, 6)) for c in rng.normal(0, 5, (8, 6))])
    worse = 0
    for seed in range(6):
        c, info = sf.emeans(x, 8, epochs=5, seed=seed)
        init = x[np.random.default_rng(seed).choice(len(x), 8, replace=False)]
        worse += _inertia(x, c) > 1.1 * _inertia(x, _lloyd(x, init))
    assert worse <= 1
    b, a0 = sf.emeans_recipe(len(x), 8, 5)
    assert info["batch"] == b and 0 < a0 <= 1.0


def test_emeans_uses_all_clusters_on_four_blobs():
    """잘 떨어진 2차원 군집 4개에서 중심 4개가 모두 쓰인다(빈 군집 없음)."""
    rng = np.random.default_rng(1)
    x = np.concatenate([c + rng.normal(0, 0.3, (300, 2)) for c in ([0, 0], [10, 0], [0, 10], [10, 10])])
    c, _ = sf.emeans(x, 4, epochs=8, seed=0, batch=200)
    assert len(np.unique(sf._assign(x, c))) == 4
