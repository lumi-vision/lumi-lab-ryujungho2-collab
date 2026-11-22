"""PyAV 기반 메타데이터와 원본 시간에 대응하는 무손실 클립 생성."""
from __future__ import annotations

from fractions import Fraction
from pathlib import Path
import hashlib
import av
from .catalog import file_sha256


def probe_video(path: Path, verify_decode: bool = True) -> dict:
    """미리보기 없이 영상의 메타데이터와 디코딩·PTS 상태를 검사한다."""
    path = path.resolve(strict=True)
    with av.open(str(path)) as container:
        if not container.streams.video:
            raise ValueError('비디오 스트림이 없습니다')
        stream = container.streams.video[0]
        info = {'codec': stream.codec_context.name, 'width': stream.codec_context.width,
                'height': stream.codec_context.height, 'time_base': str(stream.time_base),
                'average_rate': str(stream.average_rate) if stream.average_rate else None,
                'base_rate': str(stream.base_rate) if stream.base_rate else None,
                'duration_ticks': stream.duration, 'container_format': container.format.name,
                'container_duration_microseconds': container.duration,
                'pixel_format': stream.codec_context.format.name if stream.codec_context.format else None,
                'decode_status': 'not_checked', 'pts_status': 'not_checked', 'decoded_frames': None}
        if verify_decode:
            count, last, valid = 0, None, True
            for frame in container.decode(stream):
                count += 1
                if frame.pts is None or frame.time_base is None:
                    valid = False
                else:
                    now = frame.pts * frame.time_base
                    if last is not None and now <= last:
                        valid = False
                    last = now
            if not count:
                raise ValueError('디코딩된 프레임이 없습니다')
            info.update(decode_status='ok', pts_status='valid' if valid else 'missing_or_nonmonotonic', decoded_frames=count)
        return info


def register_source(path: Path, work_id: str, dataset_version: str, license_name: str, role='unassigned') -> dict:
    """원본을 읽기 전용으로 등록하며 위험 여부는 미확인으로 둔다."""
    path = path.resolve(strict=True)
    digest = file_sha256(path)
    sid = 'src:' + digest
    return {'schema_version': 1, 'dataset_version': dataset_version, 'sample_id': sid,
            'work_id': work_id, 'source_id': sid, 'parent_ids': [], 'root_source_ids': [sid],
            'origin': 'external', 'pairing': 'uncertain', 'pair_id': None, 'risk_label': None,
            'verification_status': 'unverified', 'label_source': None, 'label_evidence': None,
            'dataset_role': role, 'transform_history': [], 'license': license_name,
            'sha256': digest, 'path': str(path), 'metadata': probe_video(path), 'observations': []}


def clip_record(parent: dict, result: dict, sample_id: str) -> dict:
    """클립에 부모의 작품·원본·분할과 변환 이력을 연결한다."""
    return {**parent, 'sample_id': sample_id, 'parent_ids': [parent['sample_id']],
            'root_source_ids': list(parent['root_source_ids']), 'source_id': parent['source_id'],
            'origin': 'derived', 'pairing': 'uncertain', 'pair_id': None, 'risk_label': None,
            'verification_status': 'unverified', 'label_source': None, 'label_evidence': None,
            'path': str(Path(result['output']).resolve()), 'sha256': result['sha256'],
            'metadata': probe_video(Path(result['output'])), 'observations': [],
            'transform_history': [*parent['transform_history'], {'type': 'clip', **result}]}


def select_frames(frames, start: Fraction, end: Fraction):
    """전체 입력의 PTS 단조성을 검사하고 반열린 시간 구간을 선택한다."""
    selected, last = [], None
    for index, frame in enumerate(frames):
        if frame.pts is None or frame.time_base is None:
            raise ValueError('원본 PTS 누락')
        timestamp = frame.pts * frame.time_base
        if last is not None and timestamp <= last:
            raise ValueError('원본 PTS 비단조 또는 중복')
        last = timestamp
        if start <= timestamp < end:
            selected.append((index, frame, timestamp))
    if not selected:
        raise ValueError('구간에 프레임이 없습니다')
    return selected


def virtual_clip(source: Path, start: str, end: str) -> dict:
    """원본 해시와 정확한 프레임 시간으로 클립 참조를 작성한다."""
    begin, stop = Fraction(start), Fraction(end)
    if begin < 0 or stop <= begin:
        raise ValueError('구간 오류')
    selected, last, first = [], None, None
    with av.open(str(source)) as container:
        for i, frame in enumerate(container.decode(video=0)):
            if frame.pts is None or frame.time_base is None:
                raise ValueError('원본 PTS 누락')
            ts = frame.pts * frame.time_base
            if last is not None and ts <= last:
                raise ValueError('원본 PTS 비단조 또는 중복')
            last = ts
            if ts >= stop:
                break
            if ts >= begin:
                if first is None:
                    first = ts
                selected.append({'source_index': i, 'source_pts': frame.pts,
                                 'source_time_base': str(frame.time_base), 'source_time': str(ts),
                                 'relative_time': str(ts - first)})
    if not selected:
        raise ValueError('구간에 프레임이 없습니다')
    return {'kind': 'virtual_clip', 'source': str(source.resolve()), 'source_sha256': file_sha256(source),
            'start': str(begin), 'end': str(stop), 'end_exclusive': True,
            'last_frame_duration': None, 'last_frame_duration_status': 'unknown',
            'frames': selected}


def extract_clip(source: Path, output: Path, start: str, end: str, width: int | None = None, height: int | None = None) -> dict:
    """원본 시간과 MKV 양자화 시간을 분리 기록하고 디코딩 결과를 검증한다."""
    start_time, end_time = Fraction(start), Fraction(end)
    if start_time < 0 or end_time <= start_time or (width is None) != (height is None):
        raise ValueError('구간 또는 출력 크기 오류')
    if output.exists() or output.resolve() == source.resolve():
        raise FileExistsError('출력 파일이 이미 존재하거나 원본과 같습니다')
    if width is not None and (width < 1 or height < 1):
        raise ValueError('크기는 양수여야 합니다')
    with av.open(str(source)) as src:
        chosen = select_frames(src.decode(video=0), start_time, end_time)
        selected = [(i, ts, frame.to_ndarray(format='bgr24'), frame.pts, str(frame.time_base)) for i, frame, ts in chosen]
    output.parent.mkdir(parents=True, exist_ok=True)
    times = [x[1] - selected[0][1] for x in selected]
    ticks = [round(t * 1000) for t in times]
    if any(b <= a for a, b in zip(ticks, ticks[1:])):
        raise ValueError('MKV 시간 양자화로 프레임 시간이 중복됩니다')
    hashes = []
    try:
        with av.open(str(output), mode='w', format='matroska') as dst:
            out = dst.add_stream('ffv1', rate=1000)
            out.width = width or selected[0][2].shape[1]
            out.height = height or selected[0][2].shape[0]
            out.pix_fmt = 'bgr0'
            out.time_base = Fraction(1, 1000)
            for (_, _, array, _, _), tick in zip(selected, ticks):
                frame = av.VideoFrame.from_ndarray(array, format='bgr24').reformat(width=out.width, height=out.height, format='bgr0')
                hashes.append(hashlib.sha256(frame.to_ndarray(format='bgr24').tobytes()).hexdigest())
                frame.pts, frame.time_base = tick, Fraction(1, 1000)
                for packet in out.encode(frame):
                    dst.mux(packet)
            for packet in out.encode():
                dst.mux(packet)
        actual, actual_hashes = [], []
        with av.open(str(output)) as check:
            for frame in check.decode(video=0):
                if frame.pts is None:
                    raise ValueError('출력 PTS 누락')
                actual.append((frame.pts, frame.time_base))
                actual_hashes.append(hashlib.sha256(frame.to_ndarray(format='bgr24').tobytes()).hexdigest())
        if [pts * tb for pts, tb in actual] != [Fraction(tick, 1000) for tick in ticks] or actual_hashes != hashes:
            raise ValueError('출력 프레임 수·순서·내용 또는 시간 검증 실패')
    except Exception:
        output.unlink(missing_ok=True)
        raise
    return {'output': str(output), 'sha256': file_sha256(output), 'width': width or selected[0][2].shape[1],
            'height': height or selected[0][2].shape[0], 'source_start_time': str(selected[0][1]),
            'last_frame_duration': None, 'last_frame_duration_status': 'unknown',
            'mapping': [{'source_index': i, 'source_pts': p, 'source_time_base': tb, 'source_time': str(ts),
                         'relative_time': str(t), 'output_pts': op, 'output_time_base': str(otb),
                         'output_time': str(op * otb), 'quantization_error': str(op * otb - t)}
                        for (i, ts, _, p, tb), t, (op, otb) in zip(selected, times, actual)]}
