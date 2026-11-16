"""로컬 생성 작업: 원본 폴더 목록과 무손실 영상 출력. 판정은 플레이어가 담당한다."""
from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path

import av
import numpy as np

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mkv", ".mov", ".m4v", ".webm"}


def list_sources(folder: Path, recursive: bool = False) -> list[dict]:
    folder = folder.expanduser().resolve()
    if not folder.is_dir():
        raise ValueError(f"원본 폴더가 없습니다: {folder}")
    files = sorted(p for p in (folder.rglob("*") if recursive else folder.iterdir())
                   if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS)
    if len(files) > 2000:
        raise ValueError("영상이 2000개를 넘습니다. 하위 폴더를 지정하세요")
    rows = []
    for p in files:
        row = {"path": str(p), "name": p.relative_to(folder).as_posix(), "bytes": p.stat().st_size}
        try:
            with av.open(str(p)) as c:
                s = c.streams.video[0]
                rate = s.average_rate or s.guessed_rate
                duration = float(s.duration * s.time_base) if s.duration is not None else (
                    c.duration / av.time_base if c.duration is not None else None)
                row.update(width=s.width, height=s.height, fps=float(rate) if rate else None, seconds=duration)
        except Exception as exc:
            row["error"] = str(exc)
        rows.append(row)
    return rows


def write_lossless_video(path: Path, frames: np.ndarray, fps: float) -> None:
    """uint8 BGR 를 FFV1 MKV 로 저장한다. 판정용 입력은 NPZ와 동일한 픽셀을 유지한다."""
    if path.exists():
        raise FileExistsError(path)
    temp = path.with_suffix(".partial.mkv")
    try:
        with av.open(str(temp), "w") as c:
            stream = c.add_stream("ffv1", rate=Fraction(str(fps)).limit_denominator(100000))
            stream.width, stream.height, stream.pix_fmt = frames.shape[2], frames.shape[1], "bgr0"
            for frame in frames:
                for packet in stream.encode(av.VideoFrame.from_ndarray(frame, format="bgr24")):
                    c.mux(packet)
            for packet in stream.encode():
                c.mux(packet)
        temp.rename(path)
    finally:
        temp.unlink(missing_ok=True)


def list_results(folder: Path) -> list[dict]:
    """서버를 재시작해도 생성 완료 결과를 다시 찾는다."""
    rows = []
    for p in sorted(folder.glob("*.result.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            row = json.loads(p.read_text(encoding="utf-8"))
            if (folder / f"{row['id']}.npz").is_file():
                rows.append(row)
        except (OSError, ValueError, KeyError):
            continue
    return rows
