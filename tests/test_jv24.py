"""Jordan & Vanderheiden 2024 시간창 조항(선택 옵션) 시험. 기본 판정(analyze)은 바뀌지 않아야 한다."""
import numpy as np

from lumi_synth import measure


def lum_video(values, h=4, w=4):
    """프레임별 휘도(cd/m²) 목록 → (T,H,W) 휘도 배열."""
    return np.stack([np.full((h, w), v, np.float32) for v in values])


def test_frame_conversions():
    assert measure.qualifying_frames(30, "strict") == 1      # 33 ms ≤ 66, 66.7 ms > 66
    assert measure.qualifying_frames(30, "lenient") == 2
    assert measure.qualifying_frames(60, "strict") == 3      # 50 ms
    assert measure.sync_frames(30) == 1                      # 같은 프레임만
    assert measure.sync_frames(60) == 2                      # 16.7 ms 까지


def test_two_step_change_is_one_transition_by_default_but_not_under_strict_66ms():
    # pse-test-media 유형 A 와 같은 모양: 19.5 + 19.2 로 두 프레임에 나뉜 하강
    lum = lum_video([87.9, 87.9, 68.4, 49.2, 49.2])
    assert (measure.pixel_transitions(lum) != 0).sum() > 0            # 기본: 극값 기준이라 전이
    assert (measure.pixel_transitions_windowed(lum, 1) != 0).sum() == 0  # strict: 한 프레임 걸음 < 20 → 전이 아님
    assert (measure.pixel_transitions_windowed(lum, 2) != 0).sum() > 0   # lenient: 2프레임(66.7 ms) 안 38.7 → 전이


def test_continuous_models_on_two_step_change():
    # 30 fps: 19.5 + 19.2 하강. 선형 보간이면 20 cd/m² 가 약 34 ms 안에 일어나므로 전이, hold 도 66 ms 안에 두 걸음이 보여 전이
    lum = lum_video([87.9, 87.9, 68.4, 49.2, 49.2])
    assert (measure.pixel_transitions_continuous(lum, 30.0, model="linear") != 0).sum() > 0
    assert (measure.pixel_transitions_continuous(lum, 30.0, model="hold") != 0).sum() > 0
    # 아주 느린 변화(프레임당 5, 30 fps → 66 ms 에 10 cd/m²)는 어느 모델에서도 전이 아님
    slow = lum_video([40 + 5 * i for i in range(20)])
    for m in ("linear", "hold"):
        assert (measure.pixel_transitions_continuous(slow, 30.0, model=m) != 0).sum() == 0


def test_continuous_is_fps_consistent():
    # 같은 물리 신호(초당 12 cd/m² 의 기울기가 아니라, 25 ms 동안 30 cd/m² 상승)를 다른 fps 로 표본
    def sample(fps):
        t = np.arange(int(0.5 * fps)) / fps
        return lum_video(list(np.clip((t - 0.2) / 0.025, 0, 1) * 30 + 50))
    for fps in (24.0, 30.0, 60.0, 120.0):
        assert (measure.pixel_transitions_continuous(sample(fps), fps, model="linear") == 1).sum() > 0


def test_fast_change_counts_under_all_modes():
    lum = lum_video([40, 40, 100, 100, 40, 40])
    for k in (1, 2):
        s = measure.pixel_transitions_windowed(lum, k)
        assert list(s[:, 0, 0]) == [0, 0, 1, 0, -1, 0]


def test_fast_flash_merge_only_matters_above_66fps():
    e = np.array([1, -1, 1, -1, 1, -1, 0, 0], np.int8)          # 매 프레임 전이
    assert np.array_equal(measure.merge_fast_flashes(e, 30.0), e)  # 30 fps: 한 칸 건너 66.7 ms > 15 ms
    m = measure.merge_fast_flashes(e, 240.0)                      # 240 fps: 한 칸 건너 8.3 ms ≤ 15 ms
    assert np.count_nonzero(m) == 2 and m[0] == 1


def test_analyze_unchanged_and_jv24_runs():
    rng = np.random.default_rng(0)
    frames = rng.integers(0, 256, (12, 8, 8, 3), dtype=np.uint8)
    a = measure.analyze(frames, 30.0)
    b = measure.flash_from_transitions(measure.pixel_transitions(measure.luminance_cd(frames)), 30.0)
    assert a["max_flashes"] == b["max_flashes"]
    j = measure.analyze_jv24(frames, 30.0)
    assert set(j) >= {"dangerous", "max_flashes", "hazard_mask"}
