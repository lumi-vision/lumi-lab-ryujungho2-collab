"""가변 프레임 레이트(VFR)의 표시 시각을 부정확하게 확정하지 않고 프레임을 디코딩한다.

PyAV는 표시 시각(PTS)을 제공한다. OpenCV는 대체 백엔드이며 FPS를 이용해
시각을 추정한다는 사실을 명시적으로 기록한다. 원본 미디어는 수정하지 않는다.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np


@dataclass(frozen=True)
class VideoInfo:
    path: str
    backend: str
    width: int
    height: int
    fps_estimate: float | None
    duration_estimate_sec: float | None
    frame_count_reported: int | None
    timestamp_precision: str
    audio_playback_supported: bool = False

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class VideoFrame:
    index: int                      # 전체 디코딩 스트림 기준 인덱스(0부터 시작)
    bgr: np.ndarray                 # 후속 처리를 위한 축소하지 않은 uint8 영상
    timestamp_sec: float            # 원본 타임라인 기준 표시 시각(가능한 경우)
    relative_sec: float             # 첫 번째 디코딩 프레임 기준 상대 시각
    pts: int | None                 # PyAV 사용 시 원본 PTS
    time_base: str | None           # 원본 유리수 시간 단위(예: "1/15360")
    timestamp_source: str           # "pts" 또는 "fps_estimate"

    def metadata(self) -> dict:
        return {
            "index": self.index,
            "timestamp_sec": round(self.timestamp_sec, 8),
            "relative_sec": round(self.relative_sec, 8),
            "pts": self.pts,
            "time_base": self.time_base,
            "timestamp_source": self.timestamp_source,
        }


class VideoReader:
    def __init__(self, path: str | Path, backend: str = "auto") -> None:
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(f"영상 파일이 없습니다: {self.path}")
        if backend not in ("auto", "pyav", "opencv"):
            raise ValueError("backend는 auto, pyav, opencv 중 하나여야 합니다")
        self.backend = backend
        self._container = None
        self._capture = None
        self._stream = None
        self.info: VideoInfo | None = None

    def __enter__(self) -> "VideoReader":
        if self.backend in ("auto", "pyav"):
            try:
                import av  # type: ignore  # 선택적으로 설치하는 디코더
            except ImportError as ex:
                if self.backend == "pyav":
                    raise RuntimeError(
                        "PyAV가 설치되어 있지 않습니다. 설치 명령: pip install -e '.[precise]'"
                    ) from ex
            else:
                self._open_pyav(av)
                return self
        self._open_opencv()
        return self

    def _open_pyav(self, av) -> None:
        self._container = av.open(str(self.path))
        try:
            videos = self._container.streams.video
            if not videos:
                raise ValueError("파일에 영상 스트림이 없습니다")
            self._stream = videos[0]
            stream = self._stream
            fps = float(stream.average_rate) if stream.average_rate else None
            if fps is not None and fps <= 0:
                fps = None
            duration = None
            if stream.duration is not None and stream.time_base:
                duration = float(stream.duration * stream.time_base)
            elif self._container.duration is not None:
                duration = float(self._container.duration / av.time_base)
            self.info = VideoInfo(
                path=str(self.path), backend="pyav", width=stream.width,
                height=stream.height, fps_estimate=fps,
                duration_estimate_sec=duration,
                frame_count_reported=int(stream.frames) if stream.frames else None,
                timestamp_precision="frame_pts_when_available",
            )
        except Exception:
            self._container.close()
            self._container = None
            raise

    def _open_opencv(self) -> None:
        self._capture = cv2.VideoCapture(str(self.path))
        if not self._capture.isOpened():
            self._capture.release()
            self._capture = None
            raise ValueError(f"OpenCV가 해당 파일을 디코딩하지 못했습니다: {self.path}")
        cap = self._capture
        fps = cap.get(cv2.CAP_PROP_FPS)
        fps = float(fps) if np.isfinite(fps) and fps > 0 else None
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        count = count if count > 0 else None
        duration = count / fps if fps and count else None
        self.info = VideoInfo(
            path=str(self.path), backend="opencv",
            width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            fps_estimate=fps, duration_estimate_sec=duration,
            frame_count_reported=count, timestamp_precision="fps_estimated_not_exact",
        )

    def frames(
        self,
        start_sec: float = 0.0,
        end_sec: float | None = None,
        max_frames: int | None = None,
    ) -> Iterator[VideoFrame]:
        """프레임을 시간순으로 읽는다. 시작·종료 시각은 첫 프레임 기준 상대 시간이다.

        열린 리더에서는 한 번만 호출한다. 다시 읽으려면 리더를 새로 열어야 한다.
        일부 영상 컨테이너의 부정확한 탐색을 피하기 위해 처음부터 디코딩한다.
        """
        if self.info is None:
            raise RuntimeError("`with VideoReader(...) as reader:`로 리더를 먼저 여세요")
        if start_sec < 0 or (end_sec is not None and end_sec <= start_sec):
            raise ValueError("0 <= start_sec < end_sec 조건을 만족해야 합니다")
        if max_frames is not None and max_frames <= 0:
            raise ValueError("max_frames는 양수여야 합니다")
        emitted = 0
        if self._container is not None:
            initial_pts_time = None
            for index, raw_frame in enumerate(self._container.decode(self._stream)):
                if raw_frame.time is not None:
                    timestamp = float(raw_frame.time)
                    source = "pts"
                else:
                    estimated_fps = self.info.fps_estimate
                    if not estimated_fps:
                        raise ValueError("프레임에 PTS가 없고 미디어에 유효한 FPS도 없습니다")
                    timestamp = index / estimated_fps
                    source = "fps_estimate"
                if initial_pts_time is None:
                    initial_pts_time = timestamp
                relative = max(0.0, timestamp - initial_pts_time)
                if relative + 1e-9 < start_sec:
                    continue
                if end_sec is not None and relative >= end_sec:
                    break
                yield VideoFrame(
                    index=index, bgr=raw_frame.to_ndarray(format="bgr24"),
                    timestamp_sec=timestamp, relative_sec=relative,
                    pts=raw_frame.pts,
                    time_base=str(raw_frame.time_base) if raw_frame.time_base else None,
                    timestamp_source=source,
                )
                emitted += 1
                if max_frames is not None and emitted >= max_frames:
                    break
        else:
            assert self._capture is not None
            fps = self.info.fps_estimate
            if not fps:
                raise ValueError("OpenCV 백엔드에는 유효한 FPS 추정값이 필요합니다")
            index = 0
            while True:
                ok, bgr = self._capture.read()
                if not ok:
                    break
                relative = index / fps
                if end_sec is not None and relative >= end_sec:
                    break
                if relative + 1e-9 >= start_sec:
                    yield VideoFrame(
                        index=index, bgr=bgr, timestamp_sec=relative,
                        relative_sec=relative, pts=None, time_base=None,
                        timestamp_source="fps_estimate",
                    )
                    emitted += 1
                    if max_frames is not None and emitted >= max_frames:
                        break
                index += 1

    def close(self) -> None:
        if self._container is not None:
            self._container.close()
            self._container = None
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
