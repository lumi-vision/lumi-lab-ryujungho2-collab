"""IRIS 등 외부 검사기 입력용 무손실 영상 저장.

손실 압축(또는 RGB→YUV 4:2:0 변환)은 깜빡임 세기와 적색 포화도를 바꿀 수 있으므로,
FFV1/MKV로 저장한 뒤 다시 읽어 원본 배열과 바이트 단위로 같은지 확인합니다.
같지 않으면 예외를 던집니다. 이 파일은 재생·미리보기용이 아닙니다.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


class LosslessVideoError(RuntimeError):
    """저장한 영상이 원본 배열과 정확히 같지 않을 때 발생합니다."""


def read_video_rgb(path: Path) -> np.ndarray:
    """OpenCV로 모든 프레임을 RGB uint8 [T,H,W,3]로 읽습니다(IRIS와 같은 디코더 계열)."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise LosslessVideoError(f"영상을 열 수 없습니다: {path}")
    frames = []
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        cap.release()
    if not frames:
        raise LosslessVideoError(f"프레임을 읽지 못했습니다: {path}")
    return np.stack(frames)


def write_lossless_video(path: Path, rgb: np.ndarray, fps: float, *, verify: bool = True) -> Path:
    """RGB uint8 [T,H,W,3]을 FFV1/MKV로 저장하고 왕복 일치를 확인합니다."""
    path = Path(path)
    if rgb.dtype != np.uint8 or rgb.ndim != 4 or rgb.shape[-1] != 3:
        raise ValueError("입력은 [T,H,W,3] uint8 RGB여야 합니다.")
    if path.suffix.lower() != ".mkv":
        raise ValueError("무손실 저장은 .mkv 확장자만 지원합니다.")
    path.parent.mkdir(parents=True, exist_ok=True)
    t, h, w, _ = rgb.shape
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"FFV1"), float(fps), (w, h))
    if not writer.isOpened():
        raise LosslessVideoError("OpenCV에서 FFV1 인코더를 열 수 없습니다(ffmpeg 백엔드 확인).")
    try:
        for frame in rgb:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()
    if verify:
        decoded = read_video_rgb(path)
        if decoded.shape != rgb.shape or not np.array_equal(decoded, rgb):
            diff = (int(np.abs(decoded.astype(np.int16) - rgb.astype(np.int16)).max())
                    if decoded.shape == rgb.shape else None)
            path.unlink(missing_ok=True)
            raise LosslessVideoError(
                f"무손실 왕복 불일치: 저장 {rgb.shape}, 복원 {decoded.shape}, 최대 차이 {diff}")
    return path
