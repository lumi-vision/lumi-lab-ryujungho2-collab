"""원본과 가상 클립을 manifest 계보에 연결하고 실제 시간 정보를 검증한다."""
from __future__ import annotations

import hashlib
from fractions import Fraction
from pathlib import Path
from .catalog import file_sha256, validate
from .video import virtual_clip


def add_virtual_clip(rows: list[dict], parent_id: str, start: str, end: str) -> list[dict]:
    """동일 원본·구간의 레코드를 중복 생성하지 않고 새 레코드를 돌려준다."""
    errors = validate(rows, check_files=True)
    if errors:
        raise ValueError('\n'.join(errors))
    parents = [row for row in rows if row['sample_id'] == parent_id]
    if len(parents) != 1 or parents[0]['parent_ids']:
        raise ValueError('등록된 원본 레코드가 필요합니다')
    parent = parents[0]
    begin, stop = str(Fraction(start)), str(Fraction(end))
    key = hashlib.sha256(f"{parent['sample_id']}|{begin}|{stop}".encode()).hexdigest()
    clip_id = 'clip:' + key
    if any(row['sample_id'] == clip_id for row in rows):
        errors = validate_canonical(rows)
        if errors:
            raise ValueError('\n'.join(errors))
        return list(rows)
    clip = virtual_clip(Path(parent['path']), begin, stop)
    record = {**parent, 'sample_id': clip_id, 'clip_id': clip_id,
              'parent_ids': [parent_id], 'root_source_ids': list(parent['root_source_ids']),
              'origin': 'virtual_clip', 'pairing': 'uncertain', 'pair_id': None,
              'risk_label': None, 'verification_status': 'unverified', 'label_source': None,
              'label_evidence': None, 'observations': [], 'canonical_clip': clip,
              'transform_history': [*parent['transform_history'], {'type': 'virtual_clip', 'start': begin, 'end': stop}]}
    result = [*rows, record]
    errors = validate(result, check_files=True)
    if errors:
        raise ValueError('\n'.join(errors))
    return result


def validate_canonical(rows: list[dict]) -> list[str]:
    """가상 클립 계보·파일 해시와 PTS 목록을 원본과 다시 대조한다."""
    errors = validate(rows, check_files=True)
    by_id = {r.get('sample_id'): r for r in rows}
    for row in rows:
        if row.get('origin') != 'virtual_clip':
            continue
        sid = row.get('sample_id', '?')
        try:
            clip = row['canonical_clip']
            parent = by_id[row['parent_ids'][0]]
            if len(row['parent_ids']) != 1 or row.get('clip_id') != sid:
                raise ValueError('클립 식별자 또는 부모 오류')
            if row['work_id'] != parent['work_id'] or row['source_id'] != parent['source_id'] or row['root_source_ids'] != parent['root_source_ids']:
                raise ValueError('작품 또는 원본 계보 불일치')
            if row['dataset_role'] != parent['dataset_role']:
                raise ValueError('분할 불일치')
            if Path(clip['source']).resolve() != Path(parent['path']).resolve() or clip['source_sha256'] != file_sha256(Path(parent['path'])):
                raise ValueError('원본 경로 또는 해시 불일치')
            if clip != virtual_clip(Path(parent['path']), clip['start'], clip['end']):
                raise ValueError('원본 프레임 또는 시간 정보 불일치')
            if not row['transform_history'] or row['transform_history'][-1] != {'type': 'virtual_clip', 'start': clip['start'], 'end': clip['end']}:
                raise ValueError('변환 이력 불일치')
        except (KeyError, IndexError, ValueError, OSError) as exc:
            errors.append(f'{sid}: {exc}')
    return errors
