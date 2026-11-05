from pathlib import Path
import json
import math
import subprocess
import sys

import cv2
import numpy as np
import pytest

from lumi.analysis import frame_statistics, scan_video
from lumi.frame_buffer import FrameHistory
from lumi.video_io import VideoFrame, VideoReader


def make_video(path: Path, fps=12, frames=18):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (96, 64))
    assert writer.isOpened(), "OpenCV 영상 기록기를 사용할 수 없습니다"
    for i in range(frames):
        img = np.full((64, 96, 3), 120 + i // 6, dtype=np.uint8)
        writer.write(img)
    writer.release()
    return path


def test_opencv_reader_timestamps(tmp_path):
    source = make_video(tmp_path / "test.avi")
    with VideoReader(source, backend="opencv") as reader:
        assert reader.info.backend == "opencv"
        assert reader.info.width == 96
        assert reader.info.timestamp_precision == "fps_estimated_not_exact"
        frames = list(reader.frames())
    assert len(frames) == 18
    assert frames[0].bgr.shape == (64, 96, 3)
    assert all(frame.timestamp_source == "fps_estimate" for frame in frames)
    assert frames[-1].relative_sec == pytest.approx(17 / 12)
    assert [f.index for f in frames] == list(range(18))


def test_bounded_segment_and_limit(tmp_path):
    source = make_video(tmp_path / "test.avi")
    with VideoReader(source, "opencv") as reader:
        frames = list(reader.frames(start_sec=0.5, end_sec=1.0, max_frames=3))
    assert [f.index for f in frames] == [6, 7, 8]


def test_scan_jsonl(tmp_path):
    source = make_video(tmp_path / "test.avi")
    out = tmp_path / "out" / "frames.jsonl"
    report = scan_video(str(source), str(out), "opencv", max_frames=4)
    assert report["frames_processed"] == 4
    data = [json.loads(row) for row in out.read_text().splitlines()]
    assert len(data) == 4
    assert "pts" in data[0] and "mean_gray_8bit" in data[0]
    assert "PSE 위험 탐지 결과가 아닙니다" in report["warning"]
    summary = out.with_suffix(".summary.json")
    assert summary.is_file()


def test_history(tmp_path):
    frames = [VideoFrame(i, np.zeros((2, 2, 3), dtype=np.uint8), i / 2, i / 2,
                         None, None, "fps_estimate") for i in range(4)]
    history = FrameHistory(capacity=2)
    history.push(frames[0]); history.push(frames[1]); history.push(frames[2])
    assert history.size == 2
    assert history.previous().index == 1
    assert history.forward_cached().index == 2
    history.clear()
    assert history.current() is None


def test_invalid_ranges_and_paths(tmp_path):
    with pytest.raises(FileNotFoundError):
        VideoReader(tmp_path / "missing.mp4")
    source = make_video(tmp_path / "test.avi")
    with VideoReader(source, "opencv") as reader:
        with pytest.raises(ValueError):
            list(reader.frames(start_sec=2, end_sec=1))
        with pytest.raises(ValueError):
            list(reader.frames(max_frames=0))


def test_preview_requires_opt_in(tmp_path):
    from lumi.player import preview
    source = make_video(tmp_path / "test.avi")
    with pytest.raises(PermissionError):
        preview(str(source), allow_preview=False)


def test_cli_inspect(tmp_path):
    source = make_video(tmp_path / "test.avi")
    from lumi.cli import main
    assert main(["inspect", str(source), "--backend", "opencv"]) == 0
    assert main(["preview", str(source)]) == 2


def test_pyav_timing_when_installed(tmp_path):
    pytest.importorskip("av")
    source = make_video(tmp_path / "test.avi")
    with VideoReader(source, "pyav") as reader:
        frames = list(reader.frames())
        assert reader.info.backend == "pyav"
    assert len(frames) == 18
    assert frames[1].relative_sec > frames[0].relative_sec
    assert frames[1].timestamp_source == "pts"
