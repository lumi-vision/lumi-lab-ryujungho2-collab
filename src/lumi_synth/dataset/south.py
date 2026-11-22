"""South 자료의 로컬 등록 인터페이스. 다운로드와 재배포는 수행하지 않는다."""
import csv
from pathlib import Path


def read_south_labels(csv_path: Path, *, terms_verified: bool) -> list[dict]:
    """이용 조건 확인 이후 원본 CSV 라벨을 관측값으로 읽는다."""
    if not terms_verified:
        raise PermissionError('South 접근 권한 및 이용 조건 확인이 필요합니다')
    with csv_path.open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    required = {'label', 'source', 'dangerous', 'flash', 'red', 'pattern'}
    if not rows or not required.issubset(rows[0]):
        raise ValueError('South CSV 열이 부족합니다')
    return [{'id': r['label'], 'source': r['source'], 'risk_label': None,
             'verification_status': 'unverified', 'observation': {'tool': 'South dataset label',
             'version': 'unknown', 'settings': None, 'run': 'dataset publication',
             'result': {k: r[k] for k in ('dangerous', 'flash', 'red', 'pattern')}}} for r in rows]
