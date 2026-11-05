"""명시적으로 허용한 경우에만 열리고 기본 일시정지되는 무음 OpenCV 미리보기.

주의: 이 미리보기는 PSE 안전 필터가 아니다. 현재 단계에서는 위험을 분석하지 않는다.
처리되지 않은 위험 영상은 광과민성 시청자에게 해로울 수 있으므로
사용자가 명시적으로 허용한 경우에만 창을 연다.
"""
from __future__ import annotations

import os
import sys
import time

import cv2

from lumi.frame_buffer import FrameHistory
from lumi.video_io import VideoReader, VideoFrame


HELP = "SPACE play/pause | N next frame | P previous (buffered) | Q or ESC close"


def _fit(image, max_width: int = 960, max_height: int = 540):
    h, w = image.shape[:2]
    scale = min(1.0, max_width / w, max_height / h)
    if scale < 1.0:
        return cv2.resize(image, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
    return image.copy()


def _overlay(frame: VideoFrame, playing: bool):
    # 화면에 표시하는 사본만 축소한다. 이후 위험 탐지는 원본 해상도를 사용한다.
    canvas = _fit(frame.bgr)
    h, w = canvas.shape[:2]
    cv2.rectangle(canvas, (0, 0), (w, 65), (0, 0, 0), -1)
    text = f"LUMI / {'PLAY' if playing else 'PAUSED'} | frame {frame.index} | {frame.relative_sec:.3f}s"
    cv2.putText(canvas, text, (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(canvas, "NO PSE CORRECTION ACTIVE | " + HELP,
                (10, 47), cv2.FONT_HERSHEY_SIMPLEX,
                min(0.43, 800 / max(w, 1)), (210, 210, 210), 1, cv2.LINE_AA)
    return canvas


def preview(path: str, backend: str = "auto", allow_preview: bool = False,
            history_size: int = 12) -> None:
    if not allow_preview:
        raise PermissionError(
            "미리보기는 기본적으로 비활성화되어 있으며 위험 영상을 필터링하지 않습니다. "
            "위험 요소가 없음을 확인한 영상에서만 --allow-preview를 사용하세요."
        )
    if sys.platform.startswith("linux") and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        raise RuntimeError("그래픽 화면을 찾지 못했습니다. 화면이 없는 환경에서는 `lumi scan`을 사용하세요.")
    with VideoReader(path, backend) as reader:
        assert reader.info is not None
        print("주의: PSE 탐지·보정 기능이 없습니다. 오디오는 비활성화되어 있습니다.")
        print(HELP)
        iterator = iter(reader.frames())
        try:
            first = next(iterator)
        except StopIteration as ex:
            raise ValueError("디코딩할 수 있는 영상 프레임이 없습니다") from ex
        history = FrameHistory(history_size)
        current = history.push(first)
        playing = False  # 보정되지 않은 영상은 절대로 자동 재생하지 않는다.
        reached_eof = False
        pending: VideoFrame | None = None
        wall_reference = time.monotonic()
        video_reference = current.relative_sec
        window = "LUMI Step 1 - Unfiltered preview"
        cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
        try:
            while True:
                cv2.imshow(window, _overlay(current, playing))
                key = cv2.waitKey(20) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord(" "):
                    playing = not playing
                    wall_reference = time.monotonic()
                    video_reference = current.relative_sec
                elif key == ord("p"):
                    playing = False
                    previous = history.previous()
                    if previous is not None:
                        current = previous
                elif key == ord("n"):
                    playing = False
                    cached = history.forward_cached()
                    if cached is not None:
                        current = cached
                    elif pending is not None:
                        current = history.push(pending)
                        pending = None
                    elif not reached_eof:
                        try:
                            current = history.push(next(iterator))
                        except StopIteration:
                            reached_eof = True
                if playing:
                    # 다음 프레임을 미리 읽되 상대 표시 시각에 도달했을 때만 보여준다.
                    # GUI 재생 시각은 근사값이며 오디오 동기화는 지원하지 않는다.
                    following = history.peek_forward_cached()
                    if following is None and pending is None and not reached_eof:
                        try:
                            pending = next(iterator)
                        except StopIteration:
                            reached_eof = True
                    if following is None:
                        following = pending
                    now_video = video_reference + (time.monotonic() - wall_reference)
                    if following is None:
                        playing = False
                    elif now_video + 0.001 >= following.relative_sec:
                        cached = history.forward_cached()
                        if cached is not None:
                            current = cached
                        else:
                            assert pending is not None
                            current = history.push(pending)
                            pending = None
        finally:
            cv2.destroyWindow(window)
