"""로컬 원본 세 작품의 읽기 전용 조사 및 신규 manifest를 만든다."""
from pathlib import Path
import json
import time
import psutil
from lumi_synth.dataset.canonical import add_virtual_clip, validate_canonical
from lumi_synth.dataset.catalog import statistics, write_manifest
from lumi_synth.dataset.video import register_source

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'data' / 'datasets' / 'phase1_2_v1'
SOURCES = [
    ('bbb', 'data/sources/big_buck_bunny_1080p/big_buck_bunny_1080p_h264.mov', 'CC BY 3.0', 'train'),
    ('sintel', 'data/raw/sintel/Sintel.2010.1080p.mkv', 'CC BY 3.0', 'val'),
    ('tears_of_steel', 'data/phase1_2_sources/tears_of_steel/tears_of_steel_720p.mov', 'CC BY 3.0', 'test'),
]


def main():
    """기존 파일을 덮어쓰지 않고 새 데이터셋을 작성한다."""
    manifest = OUT / 'manifest.jsonl'
    report = OUT / 'statistics.json'
    if manifest.exists() or report.exists():
        raise FileExistsError('新規出力先が必要です')
    rows = []
    start = time.perf_counter()
    for work, relative, license_name, role in SOURCES:
        print(f'원본 검사 {work}', flush=True)
        row = register_source(ROOT / relative, work, 'phase1_2_v1', license_name, role)
        rows.append(row)
        rows = add_virtual_clip(rows, row['sample_id'], '30', '32')
        print(f'완료 {work}', flush=True)
    errors = validate_canonical(rows)
    if errors:
        raise ValueError('\n'.join(errors))
    memory = psutil.Process().memory_info()
    result = {**statistics(rows), 'errors': errors, 'elapsed_seconds': round(time.perf_counter() - start, 2),
              'peak_working_set_bytes': getattr(memory, 'peak_wset', None),
              'note': '원본 영상은 검증되지 않았으며 재생하지 않았다.'}
    OUT.mkdir(parents=True, exist_ok=True)
    write_manifest(manifest, rows)
    with report.open('x', encoding='utf8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
