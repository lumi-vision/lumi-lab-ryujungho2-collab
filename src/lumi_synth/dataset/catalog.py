"""버전 있는 레코드, 작품 단위 분할, 계보와 무결성 검사."""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

REQUIRED = ('schema_version', 'dataset_version', 'sample_id', 'work_id', 'source_id',
            'parent_ids', 'root_source_ids', 'origin', 'pairing', 'pair_id', 'risk_label',
            'verification_status', 'label_source', 'label_evidence', 'dataset_role',
            'transform_history', 'license', 'sha256')
PAIRINGS = {'clean', 'paired', 'identity', 'unpaired', 'uncertain'}
ROLES = {'train', 'val', 'test', 'external', 'unassigned'}


def file_sha256(path: Path) -> str:
    """파일을 변경하지 않고 SHA-256을 계산한다."""
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read_manifest(path: Path) -> list[dict]:
    """JSONL 레코드를 읽고 파싱 오류를 전파한다."""
    with path.open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_manifest(path: Path, rows: list[dict]) -> None:
    """기존 manifest를 덮어쓰지 않고 새 파일을 만든다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + '\n')


def split_works(rows: list[dict], seed: int, ratios: dict[str, float]) -> list[dict]:
    """미지정 작품만 안정적 해시로 분할하며, 지정 역할은 보존한다."""
    if set(ratios) != {'train', 'val', 'test'} or any(v < 0 for v in ratios.values()) or abs(sum(ratios.values()) - 1) > 1e-9:
        raise ValueError('분할 비율은 train/val/test의 합 1이어야 합니다')
    fixed = defaultdict(set)
    for row in rows:
        if row.get('dataset_role') != 'unassigned':
            fixed[row['work_id']].add(row['dataset_role'])
    if any(len(values) > 1 for values in fixed.values()):
        raise ValueError('같은 작품에 서로 다른 역할이 지정되었습니다')
    assigned = {key: next(iter(value)) for key, value in fixed.items()}
    for work in sorted({r['work_id'] for r in rows} - set(assigned)):
        value = int(hashlib.sha256(f'{seed}:{work}'.encode()).hexdigest(), 16) / 2**256
        assigned[work] = 'train' if value < ratios['train'] else ('val' if value < ratios['train'] + ratios['val'] else 'test')
    return [{**r, 'dataset_role': assigned[r['work_id']]} for r in rows]


def validate(rows: list[dict], check_files: bool = False) -> list[str]:
    """필드, 정답 근거, 해시, 다중 부모 계보와 누출을 검사한다."""
    errors = []
    by_id = {}
    for row in rows:
        missing = [k for k in REQUIRED if k not in row]
        if missing:
            errors.append(f"필수 필드 누락: {row.get('sample_id', '?')}: {missing}")
            continue
        sid = row['sample_id']
        if not isinstance(sid, str) or not sid or sid in by_id:
            errors.append(f'잘못되거나 중복된 sample_id: {sid}')
            continue
        by_id[sid] = row
        if row['schema_version'] != 1 or not all(isinstance(row[k], str) and row[k] for k in ('dataset_version', 'work_id', 'source_id', 'origin', 'verification_status', 'license')):
            errors.append(f'{sid}: 스키마 버전 또는 식별자 오류')
        if row['pairing'] not in PAIRINGS or row['dataset_role'] not in ROLES:
            errors.append(f'{sid}: 데이터 유형 또는 역할 오류')
        if not isinstance(row['parent_ids'], list) or not isinstance(row['root_source_ids'], list) or not isinstance(row['transform_history'], list) or not isinstance(row.get('observations', []), list):
            errors.append(f'{sid}: 계보 또는 관측 필드 형식 오류')
            continue
        for observation in row.get('observations', []):
            if not isinstance(observation, dict) or not {'tool', 'version', 'settings', 'run', 'result'} <= observation.keys():
                errors.append(f'{sid}: 판정 관측의 도구·버전·설정·실행·결과가 필요합니다')
        if row['pairing'] in ('clean', 'identity') and (row['verification_status'] != 'verified' or not row['label_evidence']):
            errors.append(f'{sid}: 정상 영상은 검증 근거가 필요합니다')
        if row['pairing'] == 'paired' and not row['pair_id']:
            errors.append(f'{sid}: paired 정답 참조가 없습니다')
        if row['sha256'] is not None and (not isinstance(row['sha256'], str) or not re.fullmatch(r'[0-9a-f]{64}', row['sha256'])):
            errors.append(f'{sid}: sha256 형식 오류')
        if check_files and row.get('path'):
            path = Path(row['path'])
            if not path.is_file():
                errors.append(f'{sid}: 파일 없음')
            elif row['sha256'] != file_sha256(path):
                errors.append(f'{sid}: 파일 해시 불일치')
    works, hashes = defaultdict(set), defaultdict(set)
    for row in by_id.values():
        works[row['work_id']].add(row['dataset_role'])
        if row.get('sha256'):
            hashes[row['sha256']].add(row['dataset_role'])
        if row['pairing'] == 'identity' and row.get('pair_id') not in (None, row['sample_id']):
            errors.append(f"{row['sample_id']}: identity는 자기 자신을 참조해야 합니다")
        if row['pairing'] == 'paired':
            pair = by_id.get(row['pair_id'])
            if pair is None or row['pair_id'] == row['sample_id'] or pair['dataset_role'] != row['dataset_role']:
                errors.append(f"{row['sample_id']}: 정답 참조 누락 또는 분할 불일치")
    for key, values in works.items():
        if len(values) > 1:
            errors.append(f'작품 분할 충돌: {key}')
    for key, values in hashes.items():
        if len(values) > 1:
            errors.append(f'동일 파일 분할 충돌: {key}')
    state, roots = {}, {}

    def visit(sid):
        if state.get(sid) == 1:
            errors.append(f'순환 참조: {sid}')
            return set()
        if state.get(sid) == 2:
            return roots[sid]
        state[sid] = 1
        row = by_id[sid]
        parents = row['parent_ids'] if isinstance(row['parent_ids'], list) else []
        found = set()
        for parent in parents:
            if parent not in by_id:
                errors.append(f'{sid}: 부모 없음 {parent}')
                continue
            if by_id[parent]['dataset_role'] != row['dataset_role']:
                errors.append(f'{sid}: 부모와 분할 불일치 {parent}')
            found.update(visit(parent))
        if not parents:
            found.add(row['source_id'])
        if set(row['root_source_ids']) != found:
            errors.append(f'{sid}: root_source_ids 불일치')
        state[sid], roots[sid] = 2, found
        return found

    for sid in by_id:
        visit(sid)
    return errors


def statistics(rows: list[dict]) -> dict:
    """유형과 역할, 검증 상태 및 작품 수를 집계한다."""
    return {'samples': len(rows), 'works': len({r.get('work_id') for r in rows}),
            **{key: dict(Counter(str(r.get(key)) for r in rows)) for key in
               ('origin', 'pairing', 'dataset_role', 'risk_label', 'verification_status')}}
