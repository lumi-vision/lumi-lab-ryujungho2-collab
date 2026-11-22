"""평가 데이터의 실험 사용 이력을 별도 레코드로 관리한다."""
from pathlib import Path
import json

FIELDS = ('experiment_id', 'sample_id', 'usage_role', 'model_selection_used',
          'threshold_selection_used', 'first_accessed_at', 'evidence')


def validate_usage(row: dict) -> None:
    """불명확한 사용 이력을 독립 평가로 판정하지 않는다."""
    if any(field not in row for field in FIELDS):
        raise ValueError('사용 이력 필수 필드 누락')
    if not all(isinstance(row[k], str) and row[k] for k in ('experiment_id', 'sample_id', 'usage_role')):
        raise ValueError('실험·표본·역할은 비어 있을 수 없습니다')
    if row['model_selection_used'] not in (True, False, None) or row['threshold_selection_used'] not in (True, False, None):
        raise ValueError('선택 사용 여부는 true, false 또는 null이어야 합니다')
    if not row['evidence']:
        raise ValueError('이력 출처가 필요합니다')


def append_usage(path: Path, row: dict) -> None:
    """기존 이력을 삭제하거나 수정하지 않고 한 행을 추가한다."""
    validate_usage(row)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + '\n')


def eligible_independent(rows: list[dict], sample_id: str) -> bool:
    """검증 가능한 미사용 증거가 있을 때만 독립 평가 후보로 반환한다."""
    relevant = [r for r in rows if r.get('sample_id') == sample_id]
    for row in relevant:
        validate_usage(row)
    return bool(relevant) and all(r['model_selection_used'] is False and r['threshold_selection_used'] is False
                                  and r['usage_role'] == 'not_accessed' for r in relevant)
