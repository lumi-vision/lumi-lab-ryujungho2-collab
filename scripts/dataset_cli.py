"""영상 미리보기 없이 데이터셋 등록, 분할, 검사 및 통계를 수행한다."""
import argparse
import json
from pathlib import Path
from lumi_synth.dataset.catalog import read_manifest, split_works, statistics, validate, write_manifest
from lumi_synth.dataset.canonical import add_virtual_clip, validate_canonical
from lumi_synth.dataset.model_input import export_one
from lumi_synth.dataset.video import clip_record, extract_clip, register_source, virtual_clip


def main():
    """설정 파일의 명령을 실행한다."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', type=Path)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text(encoding='utf8'))
    action = cfg['action']
    if action == 'register':
        rows = [register_source(Path(x['path']), x['work_id'], cfg['dataset_version'], x['license'], x.get('role', 'unassigned')) for x in cfg['sources']]
        errors = validate(rows, check_files=True)
        if errors:
            raise ValueError('\n'.join(errors))
        write_manifest(Path(cfg['output']), rows)
    elif action == 'split':
        rows = split_works(read_manifest(Path(cfg['input'])), cfg['seed'], cfg['ratios'])
        errors = validate(rows)
        if errors:
            raise ValueError('\n'.join(errors))
        write_manifest(Path(cfg['output']), rows)
    elif action == 'validate':
        errors = validate(read_manifest(Path(cfg['input'])), check_files=cfg.get('check_files', True))
        print(json.dumps({'errors': errors}, ensure_ascii=False, indent=2))
        if errors:
            raise SystemExit(1)
    elif action == 'statistics':
        print(json.dumps(statistics(read_manifest(Path(cfg['input']))), ensure_ascii=False, indent=2))
    elif action == 'virtual_clip':
        result = virtual_clip(Path(cfg['source']), cfg['start'], cfg['end'])
        output = Path(cfg['output'])
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open('x', encoding='utf8') as stream:
            json.dump(result, stream, ensure_ascii=False, indent=2)
    elif action == 'register_virtual':
        rows = read_manifest(Path(cfg['input']))
        rows = add_virtual_clip(rows, cfg['parent_id'], cfg['start'], cfg['end'])
        errors = validate_canonical(rows)
        if errors:
            raise ValueError('\n'.join(errors))
        write_manifest(Path(cfg['output']), rows)
    elif action == 'export_model':
        rows = read_manifest(Path(cfg['input']))
        errors = validate_canonical(rows)
        matches = [row for row in rows if row['sample_id'] == cfg['clip_id']]
        if errors or len(matches) != 1:
            raise ValueError('가상 클립 또는 manifest 검증 실패: ' + '; '.join(errors))
        sidecar = Path(cfg['metadata_output'])
        if sidecar.exists():
            raise FileExistsError('변환 이력 파일이 이미 있습니다')
        result = export_one(matches[0], Path(cfg['output']), cfg.get('fps', '24'), cfg.get('width', 768), cfg.get('height', 432))
        result.update(parent_id=cfg['clip_id'], work_id=matches[0]['work_id'], dataset_role=matches[0]['dataset_role'])
        sidecar.parent.mkdir(parents=True, exist_ok=True)
        with sidecar.open('x', encoding='utf8') as stream:
            json.dump(result, stream, ensure_ascii=False, indent=2)
    elif action == 'clip':
        rows = read_manifest(Path(cfg['input']))
        parents = [r for r in rows if r['sample_id'] == cfg['parent_id']]
        if len(parents) != 1 or validate(rows, check_files=True):
            raise ValueError('부모 또는 입력 manifest 검증 실패')
        result = extract_clip(Path(parents[0]['path']), Path(cfg['output']), cfg['start'], cfg['end'], cfg.get('width'), cfg.get('height'))
        rows.append(clip_record(parents[0], result, cfg['sample_id']))
        errors = validate(rows, check_files=True)
        if errors:
            raise ValueError('\n'.join(errors))
        write_manifest(Path(cfg['manifest_output']), rows)
    else:
        raise ValueError(f'알 수 없는 작업: {action}')


if __name__ == '__main__':
    main()
