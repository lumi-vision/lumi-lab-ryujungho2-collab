"""진단·의학적 판정 목적이 아닌 프레임 통계를 계산한다. 아직 PSE 위험 판정기는 없다."""
from __future__ import annotations

import json
from pathlib import Path

import cv2

from lumi.video_io import VideoReader


def frame_statistics(bgr) -> dict:
    """기초 밝기 통계만 계산하며 광과민성 위험을 평가하지 않는다."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return {
        "mean_gray_8bit": round(float(gray.mean()), 5),
        "min_gray_8bit": int(gray.min()),
        "max_gray_8bit": int(gray.max()),
    }


def scan_video(
    input_path: str,
    output_jsonl: str,
    backend: str = "auto",
    start: float = 0.0,
    end: float | None = None,
    max_frames: int | None = None,
) -> dict:
    destination = Path(output_jsonl)
    destination.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    first = last = None
    with VideoReader(input_path, backend=backend) as reader:
        assert reader.info is not None
        info = reader.info.as_dict()
        with destination.open("w", encoding="utf-8") as out:
            for frame in reader.frames(start, end, max_frames):
                record = {**frame.metadata(), **frame_statistics(frame.bgr)}
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                first = frame.relative_sec if first is None else first
                last = frame.relative_sec
                n += 1
    report = {
        "video": info,
        "frames_processed": n,
        "first_relative_sec": first,
        "last_relative_sec": last,
        "output_jsonl": str(destination),
        "warning": "평균 회색조 밝기는 진단용 메타데이터이며 PSE 위험 탐지 결과가 아닙니다.",
    }
    summary = destination.with_suffix(".summary.json")
    summary.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    report["summary_json"] = str(summary)
    return report
