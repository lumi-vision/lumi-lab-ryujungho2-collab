"""검증된 정상 가상 클립을 한 클립씩 G 입력으로 변환한다."""
from __future__ import annotations
from fractions import Fraction
from pathlib import Path
import av
import numpy as np
from .catalog import file_sha256


def export_one(record: dict, output: Path, fps: str = '24', width: int = 768, height: int = 432) -> dict:
    """프레임 재표본화 없이 한 클립만 변환하고 변환 이력을 돌려준다."""
    if record.get('origin') != 'virtual_clip' or record.get('verification_status') != 'verified' or record.get('risk_label') != 'safe' or not record.get('label_evidence'):
        raise ValueError('검증 근거가 있는 정상 가상 클립만 변환할 수 있습니다')
    clip = record['canonical_clip']
    source = Path(clip['source'])
    if file_sha256(source) != clip['source_sha256']:
        raise ValueError('원본 해시 불일치')
    mapping = clip['frames']
    rate = Fraction(fps)
    if rate <= 0 or len(mapping) < 2 or width < 1 or height < 1:
        raise ValueError('유효한 fps·크기와 프레임이 필요합니다')
    times = [Fraction(item['source_time']) for item in mapping]
    if any(b - a != 1 / rate for a, b in zip(times, times[1:])):
        raise ValueError('원본 fps 불일치: 이 단계는 시간 재표본화를 수행하지 않습니다')
    indices = [item['source_index'] for item in mapping]
    if indices != list(range(indices[0], indices[0] + len(indices))):
        raise ValueError('원본 프레임이 연속적이지 않습니다')
    frames = []
    with av.open(str(source)) as container:
        stream = container.streams.video[0]
        ctx = stream.codec_context
        pix = ctx.format.name if ctx.format else None
        transfer = int(ctx.color_trc)
        primaries = int(ctx.color_primaries)
        if pix not in ('yuv420p', 'yuvj420p', 'bgr0', 'rgb24') or transfer in (16, 18) or primaries == 9:
            raise ValueError(f'지원하지 않는 HDR 또는 픽셀 형식: {pix}, trc={transfer}, primaries={primaries}')
        for i, frame in enumerate(container.decode(stream)):
            if i > indices[-1]:
                break
            if i < indices[0]:
                continue
            item = mapping[len(frames)]
            if frame.pts != item['source_pts'] or str(frame.time_base) != item['source_time_base']:
                raise ValueError('원본 PTS 또는 time_base 불일치')
            converted = frame.reformat(width=width, height=height, format='bgr24')
            frames.append(converted.to_ndarray())
        meta = {'source_width': ctx.width, 'source_height': ctx.height, 'source_pixel_format': pix,
                'source_color_range': int(ctx.color_range), 'source_color_primaries': primaries,
                'source_color_transfer': transfer, 'source_colorspace': int(ctx.colorspace)}
    if len(frames) != len(mapping):
        raise ValueError('원본 프레임 누락')
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('xb') as f:
        np.savez_compressed(f, clips=np.stack([frames]), fps=np.float64(float(rate)), width=width, height=height)
    return {'type': 'model_input', 'output': str(output), 'sha256': file_sha256(output),
            'fps': str(rate), 'width': width, 'height': height, 'pixel_format': 'bgr24',
            'resize': 'pyav_reformat', 'frame_mapping': mapping, 'resampling': 'none',
            'timestamp_error': '0', 'source_metadata': meta, 'risk_label': None,
            'label_source': None, 'observations': []}
