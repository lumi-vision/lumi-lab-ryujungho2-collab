"""image_discriminator v1 리뷰에서 재현한 오판 사례를 회귀 테스트로 고정한다."""
import cv2
import numpy as np
import pytest

from lumi.static_features import (ViewingGeometry, analyze_image, binnie_reference,
                                  count_stripe_pairs, pattern_visual_angle_deg)

H, W = 360, 640


def square_wave(pairs: int, width: int = 10) -> np.ndarray:
    return np.concatenate([np.full(width, 200.0 if i % 2 == 0 else 0.0) for i in range(2 * pairs)])


@pytest.mark.parametrize("pairs", [4, 5, 6, 10])
def test_exact_pair_count_full_line(pairs):
    # v1은 `경계 수 // 2`로 계산해 항상 1쌍 적게 셌다(5쌍 → 4쌍).
    assert count_stripe_pairs(square_wave(pairs)) == pairs


@pytest.mark.parametrize("pairs", [4, 5, 10])
def test_pattern_on_white_background(pairs):
    p = np.concatenate([np.full(40, 200.0), square_wave(pairs), np.full(40, 200.0)])
    assert count_stripe_pairs(p) == pairs


def test_thin_dark_lines_on_white_count():
    # 명/암 폭이 1:5로 달라도 규칙 패턴이다.
    assert count_stripe_pairs(np.tile(np.array([0.0] + [200.0] * 5), 12)) >= 10


def test_low_contrast_stripes_ignored():
    # 휘도차 10 cd/m² (<20) 줄무늬는 대비 규칙을 만족하지 않는다.
    p = np.tile(np.concatenate([np.full(8, 60.0), np.full(8, 50.0)]), 20)
    assert count_stripe_pairs(p) == 0


def _text_document() -> np.ndarray:
    img = np.full((H, W, 3), 255, np.uint8)
    rng = np.random.default_rng(1)
    words = "The quick brown fox jumps over the lazy dog LUMI research 2026".split()
    y = 30
    while y < H - 10:
        x = 20
        while True:
            word = words[rng.integers(len(words))]
            (tw, _), _ = cv2.getTextSize(word, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            if x + tw > W - 20:
                break
            cv2.putText(img, word, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 20), 1, cv2.LINE_AA)
            x += tw + 8
        y += 22
    return img


def test_text_document_is_not_a_pattern():
    # v1: 흰 배경 문서가 패턴·고대비·고휘도 3개 항목에서 '위험'.
    result = analyze_image(_text_document())
    assert result["pattern"]["area_fraction"] < 0.05
    assert not result["pattern"]["pattern_over_25pct"]
    # 글자 한 줄에서 우연히 생긴 쌍 수는 진단 필드에만 남고, 패턴 쌍 수로 보고하지 않는다.
    assert result["pattern"]["max_pairs"] < 5


def test_small_patch_area_is_not_inflated_by_surroundings():
    # v1: 작은 줄무늬 조각(4%) + 주변 글자형 블록 → 패턴 면적 99.9%, 40% 기준 '위험'.
    img = np.full((H, W, 3), 255, np.uint8)
    rng = np.random.default_rng(0)
    for _ in range(600):
        y, x = rng.integers(0, H - 6), rng.integers(0, W - 12)
        img[y:y + 4, x:x + rng.integers(3, 10)] = 0
    cy, cx = H // 2, W // 2
    img[cy - 30:cy + 30, cx - 60:cx + 60] = 255
    for i in range(0, 120, 8):
        img[cy - 30:cy + 30, cx - 60 + i:cx - 56 + i] = 0
    result = analyze_image(img)
    patch_area = 60 * 120 / (H * W)
    assert result["pattern"]["max_pairs"] >= 5
    # 조각을 실제로 검출했고(하한), 주변 블록 때문에 부풀지 않았다(상한).
    assert patch_area * 0.6 < result["pattern"]["area_fraction"] < patch_area * 2
    assert result["mask"][cy - 20:cy + 20, cx - 50:cx + 50].mean() > 0.8
    assert not result["pattern"]["pattern_over_25pct"]


def test_half_black_half_white_has_no_pattern():
    # v1: 고휘도 항목에서 '위험'. v2는 고휘도를 판정하지 않고 특징만 낸다.
    img = np.zeros((H, W, 3), np.uint8)
    img[:, W // 2:] = 255
    result = analyze_image(img)
    assert result["pattern"]["area_fraction"] == 0.0
    assert result["features"]["bright_area_ge160"] == pytest.approx(0.5, abs=0.01)
    assert not any(k.startswith("pattern_over") and v for k, v in result["pattern"].items())


def test_saturated_red_is_a_feature_not_a_verdict():
    img = np.full((H, W, 3), 255, np.uint8)
    img[:, : int(W * 0.45)] = (255, 0, 0)
    result = analyze_image(img)
    assert result["features"]["saturated_red_area"] == pytest.approx(0.45, abs=0.01)
    assert result["pattern"]["area_fraction"] == 0.0


@pytest.mark.parametrize("kind", ["vertical", "horizontal", "rings"])
def test_full_screen_patterns_detected(kind):
    yy, xx = np.mgrid[:H, :W]
    if kind == "vertical":
        g = (xx // 6) % 2
    elif kind == "horizontal":
        g = (yy // 6) % 2
    else:
        g = (np.hypot(yy - H / 2, xx - W / 2) // 10) % 2
    img = np.repeat((g * 255).astype(np.uint8)[..., None], 3, axis=2)
    result = analyze_image(img)
    assert result["pattern"]["area_fraction"] > 0.9
    assert result["pattern"]["pattern_over_40pct"]


def test_half_screen_pattern_area():
    img = np.full((H, W, 3), 128, np.uint8)
    for x in range(0, W // 2, 10):
        img[:, x:x + 5] = 255
        img[:, x + 5:x + 10] = 0
    result = analyze_image(img)
    assert result["pattern"]["area_fraction"] == pytest.approx(0.5, abs=0.03)


def test_reference_value_depends_on_geometry_and_is_bounded():
    near = pattern_visual_angle_deg(0.1, ViewingGeometry(27, distance_cm=50))
    far = pattern_visual_angle_deg(0.1, ViewingGeometry(27, distance_cm=300))
    assert near > far > 0
    for theta in (0.0, 1.0, near, 179.0):
        assert 0.0 <= binnie_reference(theta)["binnie_reference"] <= 1.0
    # 경계 연속성: Q=8.75 → 0, Q=56.37 → 1 (0.021Q 보정). θ = -ln(1 - Q/100) / 0.0574
    theta_for = lambda q: -np.log(1 - q / 100) / 0.0574
    assert binnie_reference(theta_for(8.75) + 1e-6)["binnie_reference"] == pytest.approx(0.0, abs=2e-3)
    assert binnie_reference(theta_for(56.37) - 1e-6)["binnie_reference"] == pytest.approx(1.0, abs=2e-3)
    mid = binnie_reference(theta_for(30.0))
    assert mid["Q"] == pytest.approx(30.0) and mid["binnie_reference"] == pytest.approx(0.021 * 30 - 0.1827)


def test_resize_path_keeps_mask_in_input_shape():
    h, w = 1200, 4200   # MAX_SIDE(3840) 초과 → 축소 분석
    xx = np.arange(w)[None, :].repeat(h, 0)
    img = np.repeat((((xx // 8) % 2) * 255).astype(np.uint8)[..., None], 3, axis=2)
    result = analyze_image(img)
    assert result["resized"] and result["analyzed_shape"] != (h, w)
    assert result["mask"].shape == (h, w)
    assert result["pattern"]["area_fraction"] > 0.9


@pytest.mark.parametrize("shape", [(1, 10000), (10000, 1), (1, 1), (2, 2)])
def test_extreme_shapes_do_not_crash(shape):
    result = analyze_image(np.zeros((*shape, 3), np.uint8))
    assert result["pattern"]["area_fraction"] == 0.0


def test_cli_static_outputs_json(tmp_path, capsys):
    import json
    from lumi.cli import main
    path = tmp_path / "doc.png"
    cv2.imencode(".png", cv2.cvtColor(_text_document(), cv2.COLOR_RGB2BGR))[1].tofile(str(path))
    mask = tmp_path / "mask.png"
    assert main(["static", str(path), "--mask-out", str(mask)]) == 0
    out = json.loads(capsys.readouterr().out)
    assert "mask" not in out and mask.exists()
    assert out["pattern"]["pattern_over_25pct"] is False
