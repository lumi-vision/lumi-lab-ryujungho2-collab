"""범용 원본 기록과 분리된 기존 G 입력 형식 내보내기."""
from pathlib import Path
from fractions import Fraction
import numpy as np
import av
from .catalog import file_sha256


def export_backgrounds(paths: list[Path], fps: float, width: int, height: int, frames: int, output: Path) -> None:
    """같은 크기·고정 시간 간격의 정상 클립을 기존 NPZ 형식으로 저장한다."""
    if not paths or fps <= 0 or min(width, height, frames) < 1:
        raise ValueError('입력 클립과 양수 규격이 필요합니다')
    clips = []
    for path in paths:
        images, timestamps = [], []
        with av.open(str(path)) as container:
            for frame in container.decode(video=0):
                if frame.pts is None or frame.time_base is None:
                    raise ValueError('프레임 시간 누락')
                timestamps.append(float(frame.pts * frame.time_base))
                if frame.width != width or frame.height != height:
                    raise ValueError('G 입력 해상도 불일치')
                images.append(frame.to_ndarray(format='bgr24'))
        if len(images) != frames or any(abs((b - a) - 1 / fps) > 1e-6 for a, b in zip(timestamps, timestamps[1:])):
            raise ValueError('G 입력 길이 또는 일정 fps 불일치')
        clips.append(np.stack(images))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('xb') as stream:
        np.savez_compressed(stream, clips=np.stack(clips), fps=np.float64(fps), width=width, height=height)


def export_virtual_backgrounds(clips: list[dict], fps: str, width: int, height: int, frames: int, output: Path) -> None:
    """원본의 정확한 CFR 프레임을 읽어 모델 입력으로 내보낸다. 재표본화하지 않는다."""
    rate = Fraction(fps)
    if not clips or rate <= 0 or min(width, height, frames) < 1:
        raise ValueError('입력 클립과 양수 규격이 필요합니다')
    images = []
    for clip in clips:
        source = Path(clip['source'])
        if file_sha256(source) != clip['source_sha256']:
            raise ValueError('원본 해시 불일치')
        mapping = clip['frames']
        indices = [m['source_index'] for m in mapping]
        times = [Fraction(m['source_time']) for m in mapping]
        if len(indices) != frames or len(set(indices)) != frames or indices != sorted(indices):
            raise ValueError('프레임 수 또는 순서 오류')
        if any(b - a != 1 / rate for a, b in zip(times, times[1:])):
            raise ValueError('원본 프레임이 지정된 일정 fps와 다릅니다')
        wanted = set(indices)
        decoded = []
        with av.open(str(source)) as container:
            for i, frame in enumerate(container.decode(video=0)):
                if i in wanted:
                    item = mapping[len(decoded)]
                    if frame.pts != item['source_pts'] or str(frame.time_base) != item['source_time_base']:
                        raise ValueError('원본 프레임 시간 불일치')
                    if frame.width != width or frame.height != height:
                        raise ValueError('G 입력 해상도 불일치')
                    decoded.append(frame.to_ndarray(format='bgr24'))
        if len(decoded) != frames:
            raise ValueError('원본 프레임 누락')
        images.append(np.stack(decoded))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('xb') as stream:
        np.savez_compressed(stream, clips=np.stack(images), fps=np.float64(float(rate)), width=width, height=height)
