"""IRIS 결과 해석·라벨 기록·무손실 저장·구간 파형 검사(IRIS 실행 파일 없이)."""
import json
from pathlib import Path

import numpy as np
import pytest

from lumi.dataset_synthesis import DatasetConfig, build_dataset, synthesize
from lumi.iris_labeling import (IrisError, IrisFrames, label_row, load_frames,
                                parse_framedata, standard_result, summarize, window_label)

HEADER = ("Frame,TimeStamp,AverageLuminance,FlashAreaLuminance,AverageLuminanceDiff,"
          "AverageLuminanceDiffAcc,AverageRed,FlashAreaRed,AverageRedDiff,AverageRedDiffAcc,"
          "LuminanceTransitions,RedTransitions,LuminanceExtendedFailCount,RedExtendedFailCount,"
          "LuminanceFrameResult,RedFrameResult,PatternArea,PatternDetectedLines,PatternFrameResult")


def fake_csv(lum, red, pat) -> str:
    lines = [HEADER]
    for i, (l, r, p) in enumerate(zip(lum, red, pat), 1):
        # IRIS는 행 끝에 NUL 문자를 씁니다(FrameData::ToCSV).
        lines.append(f"{i},00:00:00.000000,0.5,10.00%,0.1,0.1,0,0.00%,0,0,{l},0,0,0,{l},{r},"
                     f"0.00%,0,{p}\x00")
    return "\n".join(lines) + "\n"


def test_parse_strips_nul_and_reads_results():
    frames = parse_framedata(fake_csv([0, 1, 3, 3], [0, 0, 2, 0], [0, 0, 0, 1]))
    assert frames.luminance.tolist() == [0, 1, 3, 3]
    assert frames.red.tolist() == [0, 0, 2, 0]
    assert frames.pattern.tolist() == [0, 0, 0, 1]
    s = summarize(frames)
    assert s["fail_frames"] == {"luminance": [2, 3], "red": [2], "pattern": [3]}
    assert standard_result(s) == "iris:luminance_flash_fail+red_extended_fail+pattern_fail"


def test_standard_result_pass_and_warning():
    ok = summarize(parse_framedata(fake_csv([0, 0], [0, 0], [0, 0])))
    warn = summarize(parse_framedata(fake_csv([0, 1], [0, 0], [0, 0])))
    assert standard_result(ok) == "iris:pass"
    assert standard_result(warn) == "iris:pass_with_warning"


def test_parse_rejects_bad_input():
    with pytest.raises(IrisError):
        parse_framedata("a,b\n1,2\n")
    with pytest.raises(IrisError):
        parse_framedata(fake_csv([0, 7], [0, 0], [0, 0]))


def test_window_label_rules():
    frames = parse_framedata(fake_csv([0, 3, 0, 0], [0, 0, 0, 2], [0, 0, 0, 0]))
    assert window_label(frames, 0, 3, "last") == {"luminance": 0, "red": 0, "pattern": 0}
    assert window_label(frames, 0, 3, "max") == {"luminance": 3, "red": 0, "pattern": 0}
    assert window_label(frames, 1, 3, "last")["red"] == 2
    with pytest.raises(ValueError):
        window_label(frames, 2, 3, "last")


class StubRunner:
    version = "stubcommit"
    settings_sha256 = "0" * 64

    def __init__(self, csv_text=None, error=None):
        self.csv_text, self.error = csv_text, error

    def run(self, video):
        if self.error:
            raise IrisError(self.error)
        return parse_framedata(self.csv_text), self.csv_text


def test_label_row_success_and_error(tmp_path: Path):
    (tmp_path / "train").mkdir()
    (tmp_path / "train" / "x.mkv").write_bytes(b"")
    row = {"id": "x", "frames": 3, "video": "train/x.mkv", "risk_verified": False}
    out = label_row(row, tmp_path, StubRunner(fake_csv([0, 2, 2], [0, 0, 0], [0, 0, 0])))
    assert out["risk_verified"] is True
    assert out["standard_result"] == "iris:luminance_extended_fail"
    assert out["iris"]["fail_frames"]["luminance"] == [1, 2]
    assert load_frames(tmp_path, out).luminance.tolist() == [0, 2, 2]
    assert (tmp_path / "train" / "x.iris.csv").is_file()

    bad_len = label_row(row, tmp_path, StubRunner(fake_csv([0, 0], [0, 0], [0, 0])))
    assert bad_len["standard_result"] == "iris_error" and bad_len["risk_verified"] is False
    failed = label_row(row, tmp_path, StubRunner(error="crash"))
    assert failed["standard_result"] == "iris_error"
    no_video = label_row({"id": "y", "frames": 3}, tmp_path, StubRunner())
    assert no_video["standard_result"] == "iris_error"


def test_lossless_video_roundtrip(tmp_path: Path):
    pytest.importorskip("cv2")
    from lumi.lossless_video import read_video_rgb, write_lossless_video
    rgb = np.random.default_rng(0).integers(0, 256, (10, 40, 48, 3), dtype=np.uint8)
    path = write_lossless_video(tmp_path / "a.mkv", rgb, 24)
    assert np.array_equal(read_video_rgb(path), rgb)


def test_segmented_activation_and_export(tmp_path: Path):
    cv2 = pytest.importorskip("cv2")
    cfg = DatasetConfig(frames=48, width=48, height=40, fps=24, segments=3)
    clean = np.full((48, 40, 48, 3), 90, np.uint8)
    affected, mask, activation, meta = synthesize(clean, cfg, "luminance", np.random.default_rng(1))
    segs = meta["segments"]
    assert len(segs) == 3 and segs[0]["start_frame"] == 0 and segs[-1]["end_frame"] == 48
    assert all(s["end_frame"] - s["start_frame"] >= 4 for s in segs)
    for s in segs:
        if s["amplitude"] == 0:
            assert not activation[s["start_frame"]:s["end_frame"]].any()
    assert np.array_equal(clean[mask == 0], affected[mask == 0])
    with pytest.raises(ValueError):
        DatasetConfig(frames=16, segments=5).validate()

    source = tmp_path / "s.avi"
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"MJPG"), 24, (48, 40))
    for i in range(30):
        writer.write(np.full((40, 48, 3), 40 + i, np.uint8))
    writer.release()
    ds = tmp_path / "ds"
    rows = build_dataset([source], ds, samples_per_source=4,
                         cfg=DatasetConfig(frames=48, width=48, height=40, fps=24,
                                           segments=3, export_video=True))
    from lumi.lossless_video import read_video_rgb
    for row in rows:
        with np.load(ds / row["file"]) as arr:
            assert np.array_equal(read_video_rgb(ds / row["video"]), arr["affected"])
        assert json.loads((ds / row["file"]).with_suffix(".json").read_text(encoding="utf-8"))["video"]
