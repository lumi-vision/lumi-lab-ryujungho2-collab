"""생성기 묶음(lumi_synth.generate)의 성질: 재현성, 조건 고정, 마스크 밖 보존, 의도 라벨.

GAN 체크포인트와 바탕 데이터(data/)가 없어도 돌도록 규칙 기반 종류만 작은 합성 바탕으로 시험한다.
영상은 메모리에서만 만든다(저장·표시하지 않음)."""
import numpy as np
import pytest
import json

from lumi_synth import generate as gen, measure

FPS = 24.0
T, H, W = 48, 90, 160


def clean_clip() -> np.ndarray:
    """깜빡임 없는 바탕: 가로 방향 밝기 기울기 (어두운 쪽 < 160 cd/m²)."""
    row = np.linspace(40, 120, W).astype(np.uint8)
    return np.broadcast_to(row[None, None, :, None], (T, H, W, 3)).copy()


@pytest.mark.parametrize("kind", ["lum", "red", "stripe"])
def test_recipe_is_reproducible_and_keeps_outside_of_mask(kind):
    clean = clean_clip()
    r1 = gen.sample_recipe(kind, "hazard", 3, split="val", bg=0)
    r2 = gen.sample_recipe(kind, "hazard", 3, split="val", bg=0)
    assert r1 == r2                                                    # 같은 시드 = 같은 조건
    a, b = gen.render(r1, clean, FPS), gen.render(r2, clean, FPS)
    assert np.array_equal(a["flicker"], b["flicker"])                  # 같은 조건 = 같은 영상
    assert a["expected"] is True
    assert not (a["flicker"] != clean)[:, ~a["mask"]].any()            # 마스크 밖은 원본 그대로
    assert (a["flicker"] != clean)[:, a["mask"]].any()                 # 마스크 안은 바뀜
    assert a["mask"].mean() == pytest.approx(r1["area"], abs=0.03)


def test_fixed_conditions_override_random_ones_and_irrelevant_keys_are_ignored():
    r = gen.sample_recipe("lum", "hazard", 1, split="test", bg=5, shape="rectangle", area=0.5, freq_hz=6.0, cpd=3.0)
    assert (r["shape"], r["area"], r["freq_hz"], r["split"], r["bg"]) == ("rectangle", 0.5, 6.0, "test", 5)
    assert "cpd" not in r                                              # 휘도 깜빡임에 cpd 는 없는 조건
    s = gen.sample_recipe("stripe", "hazard", 1, split="val", bg=0, freq_hz=4.0, motion="reversing")
    assert s["hz"] == 4.0 and s["motion"] == "reversing"               # 줄무늬에서 주파수 = 반전 주파수


def test_intent_sets_the_one_condition_below_the_limit():
    for seed in range(8):
        lum = gen.sample_recipe("lum", "near", seed, split="val", bg=0)
        red = gen.sample_recipe("red", "near", seed, split="val", bg=0)
        stripe = gen.sample_recipe("stripe", "near", seed, split="val", bg=0)
        assert lum["freq_hz"] <= 2.5 and red["freq_hz"] <= 2.5         # 초당 3회 초과가 기준
        assert stripe["area"] <= 0.35 < measure.PATTERN_AREA_STATIC    # 줄무늬 면적 40% 초과가 기준
        assert gen.sample_recipe("stripe", "hazard", seed, split="val", bg=0)["area"] > measure.PATTERN_AREA_STATIC


def test_rule_based_flicker_matches_its_intent_under_the_judge():
    clean = clean_clip()
    for intent, want in (("hazard", True), ("near", False)):
        r = gen.sample_recipe("lum", intent, 2, split="val", bg=0, shape="rectangle", area=0.5, waveform="square", delta_cd=60.0)
        out = gen.render(r, clean, FPS)
        assert measure.analyze(out["flicker"], FPS)["dangerous"] is want, r


def test_mix_without_checkpoint_never_picks_gan():
    kinds = {gen.sample_recipe("mix", "both", s, split="val", bg=0)["kind"] for s in range(40)}
    assert "gan" not in kinds and kinds <= set(gen.KINDS)
    assert len(kinds) == 3                                             # 휘도·적색·줄무늬가 모두 나옴


def test_pair_roundtrip_separates_intent_from_judgement(tmp_path):
    clean = clean_clip()
    # 위험 의도로 요청해도 면적을 작게 주면 실제 판정은 위험이 아니다.
    recipe = gen.sample_recipe("lum", "hazard", 5, shape="rectangle", area=0.1,
                               freq_hz=5, waveform="square", delta_cd=60)
    recipe["id"] = "pair"
    result = gen.make(recipe, tmp_path, clean=clean, fps=FPS)
    with np.load(result["path"], allow_pickle=False) as saved:
        assert int(saved["schema_version"]) == 2
        assert np.array_equal(saved["clean"], clean)
        assert np.array_equal(saved["flicker"], result["flicker"])
        assert np.array_equal(saved["mask_changed"], (saved["flicker"] != clean).any(-1))
        assert not saved["mask_changed"][:, ~saved["mask_allow"]].any()
        assert bool(saved["intent_hazard"])
        assert "hazard_frames" not in saved and "hazard_mask" not in saved
        assert json.loads(str(saved["recipe"]))["clean_sha256"]
    assert not measure.analyze(result["flicker"], FPS)["dangerous"]
    with pytest.raises(FileExistsError):
        gen.make(recipe, tmp_path, clean=clean, fps=FPS)


@pytest.mark.parametrize("override", [
    {"freq_hz": 12}, {"freq_hz": float("nan")}, {"area": 1.1},
    {"area": 0}, {"center": [2, 0]}, {"delta_cd": -1}, {"waveform": "bad"},
])
def test_invalid_conditions_are_rejected(override):
    recipe = gen.sample_recipe("lum", "hazard", 0)
    recipe.update(override)
    with pytest.raises(ValueError):
        gen.render(recipe, clean_clip(), FPS)


def test_invalid_intent_is_not_silently_randomized():
    with pytest.raises(ValueError, match="intent"):
        gen.sample_recipe("lum", "typo", 1)


def test_external_source_keeps_fps_and_reads_requested_interval(tmp_path):
    import av
    path = tmp_path / "source.mkv"
    with av.open(str(path), "w") as c:
        stream = c.add_stream("ffv1", rate=24)
        stream.width, stream.height, stream.pix_fmt = 160, 90, "bgr0"
        for i in range(48):
            frame = av.VideoFrame.from_ndarray(np.full((90, 160, 3), i, np.uint8), format="bgr24")
            for packet in stream.encode(frame):
                c.mux(packet)
        for packet in stream.encode():
            c.mux(packet)
    recipe = gen.sample_recipe("lum", "near", 1, source=str(path), start_sec=1, seconds=0.5, max_side=80)
    clip, fps = gen.load_source(recipe)
    assert fps == 24 and clip.shape == (12, 45, 80, 3)
    assert np.all(clip[0] == 24) and np.all(clip[-1] == 35)
    recipe["id"] = "external"
    result = gen.make(recipe, tmp_path)
    assert np.array_equal(result["clean"], clip)


def test_gan_uses_cpu_when_gpu_memory_is_busy(tmp_path, monkeypatch):
    import torch
    from lumi_synth.gan import LightGenerator
    checkpoint = tmp_path / "test.pt"
    checkpoint.touch()
    cfg = {"width": 8, "zdim": 4, "fps": 24, "train_hw": [36, 64], "hazard": "lum"}
    generator = LightGenerator(8, 4, 24).eval()
    monkeypatch.setattr(gen, "_load_generator", lambda *args: (generator, cfg))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda: (128 * 1024 ** 2, 16 * 1024 ** 3))
    clean = np.full((24, 36, 64, 3), 80, np.uint8)
    mask = np.zeros((36, 64), bool)
    mask[8:28, 12:52] = True
    result = gen.gan_render(checkpoint, clean, mask, 1, 14, 24)
    assert result["inference_device"] == "cpu"
    assert np.array_equal(result["flicker"][:, ~mask], clean[:, ~mask])
