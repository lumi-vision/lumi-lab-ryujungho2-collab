"""BBB 배경 전수의 계보 및 저장된 원본 배열을 재생 없이 대조한다."""
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
from lumi_synth.data import ClipSpec, file_sha256, read_clip

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data' / 'oracle_768x432_24fps'
SOURCE = ROOT / 'data' / 'sources' / 'big_buck_bunny_1080p' / 'big_buck_bunny_1080p_h264.mov'
OUTPUT = ROOT / 'data' / 'datasets' / 'research_data_v1' / 'bbb_audit.json'


def main():
    """기존 데이터와 판단 이력을 읽기 전용으로 검사한다."""
    if OUTPUT.exists():
        raise FileExistsError('기존 보고서를 덮어쓰지 않습니다')
    rows = [json.loads(line) for line in (DATA / 'manifest.jsonl').read_text(encoding='utf8').splitlines() if line.strip()]
    source_hash = file_sha256(SOURCE)
    results = []
    for split in ('train', 'val'):
        groups = defaultdict(list)
        for row in rows:
            if row['split'] == split:
                groups[int(row['id'].split('_')[1])].append(row)
        with np.load(DATA / f'backgrounds_{split}.npz') as archive:
            backgrounds = archive['clips']
        if set(groups) != set(range(len(backgrounds))):
            raise ValueError('배경 인덱스와 매니페스트의 불일치')
        for idx in range(len(backgrounds)):
            group = groups[idx]
            first = group[0]
            consistent = all(r['source'] == first['source'] and r['start_frame'] == first['start_frame'] and r['background_check'] == first['background_check'] for r in group)
            actual = read_clip(SOURCE, int(first['start_frame']), ClipSpec(768, 432, 48, 24), 24)
            original_equal = actual is not None and np.array_equal(actual, backgrounds[idx])
            checks = []
            for row in group:
                with np.load(DATA / row['file']) as sample:
                    checks.append({'id': row['id'], 'clean_matches_background': bool(np.array_equal(sample['clean'], backgrounds[idx]))})
            results.append({'split': split, 'bg': idx, 'source': first['source'], 'start_frame': first['start_frame'],
                            'metadata_consistent': consistent, 'source_clip_matches': bool(original_equal),
                            'background_check': first['background_check'], 'samples': checks})
            print(split, idx + 1, '/', len(backgrounds), flush=True)
    if file_sha256(SOURCE) != source_hash:
        raise ValueError('검사 중 원본 해시 변경')
    report = {'source_sha256': source_hash, 'num_backgrounds': len(results),
              'original_matches': sum(r['source_clip_matches'] for r in results),
              'metadata_consistent': sum(r['metadata_consistent'] for r in results),
              'sample_count': sum(len(r['samples']) for r in results),
              'samples_match': sum(s['clean_matches_background'] for r in results for s in r['samples']),
              'judge_version': '당시 선별 코드 해시 미기록', 'results': results}
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open('x', encoding='utf8') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    print(json.dumps({k:v for k,v in report.items() if k != 'results'}, ensure_ascii=False))


if __name__ == '__main__':
    main()
