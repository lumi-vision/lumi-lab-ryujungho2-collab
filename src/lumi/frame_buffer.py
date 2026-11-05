"""기계학습 모델과 무관하게 최근 프레임을 제한된 개수만큼 저장하는 버퍼."""
from __future__ import annotations

from collections import deque
from lumi.video_io import VideoFrame


class FrameHistory:
    def __init__(self, capacity: int = 12) -> None:
        if capacity <= 0:
            raise ValueError("버퍼 용량은 양수여야 합니다")
        self._items: deque[VideoFrame] = deque(maxlen=capacity)
        self._cursor = -1

    def push(self, frame: VideoFrame) -> VideoFrame:
        if self._cursor < len(self._items) - 1:
            # 일반적으로 발생하지 않지만 커서 뒤쪽에 캐시된 프레임이 있다면 제거한다.
            while len(self._items) - 1 > self._cursor:
                self._items.pop()
        self._items.append(frame)
        self._cursor = len(self._items) - 1
        return frame

    def current(self) -> VideoFrame | None:
        return self._items[self._cursor] if self._cursor >= 0 else None

    def previous(self) -> VideoFrame | None:
        if self._cursor > 0:
            self._cursor -= 1
        return self.current()

    def peek_forward_cached(self) -> VideoFrame | None:
        if 0 <= self._cursor < len(self._items) - 1:
            return self._items[self._cursor + 1]
        return None

    def forward_cached(self) -> VideoFrame | None:
        if 0 <= self._cursor < len(self._items) - 1:
            self._cursor += 1
            return self.current()
        return None

    def clear(self) -> None:
        self._items.clear()
        self._cursor = -1

    @property
    def size(self) -> int:
        return len(self._items)
