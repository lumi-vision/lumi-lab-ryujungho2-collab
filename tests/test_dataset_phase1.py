"""정상 단색 영상만 사용하는 데이터셋 인프라 검증."""
from fractions import Fraction
import importlib.util
from pathlib import Path
import av
import numpy as np
import pytest
from lumi_synth.dataset.catalog import file_sha256, read_manifest, split_works, statistics, validate, write_manifest
from lumi_synth.dataset.export import export_backgrounds
from lumi_synth.dataset.south import read_south_labels
from lumi_synth.dataset.video import clip_record, extract_clip, probe_video, register_source


@pytest.fixture
def normal_video(tmp_path):
    """동일한 단색 프레임으로 구성된 정상 시험 파일을 만든다."""
    path = tmp_path / 'normal.mkv'
    with av.open(str(path), 'w') as container:
        stream = container.add_stream('ffv1', rate=25)
        stream.width, stream.height, stream.pix_fmt = 32, 24, 'bgr0'
        for i in range(10):
            frame = av.VideoFrame.from_ndarray(np.full((24, 32, 3), 90, np.uint8), format='bgr24')
            frame.pts, frame.time_base = i, Fraction(1, 25)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return path


def test_source_manifest_split_and_stats(normal_video, tmp_path):
    row = register_source(normal_video, 'film1', 'v1', 'test')
    assert row['metadata']['pts_status'] == 'valid'
    assert row['risk_label'] is None and row['pairing'] == 'uncertain'
    result = split_works([row], 7, {'train': .7, 'val': .2, 'test': .1})
    assert result == split_works([row], 7, {'train': .7, 'val': .2, 'test': .1})
    assert not validate(result, check_files=True)
    path = tmp_path / 'manifest.jsonl'
    write_manifest(path, result)
    assert read_manifest(path) == result and statistics(result)['samples'] == 1
    with pytest.raises(FileExistsError):
        write_manifest(path, result)


def test_lineage_leak_cycle_missing_and_pair(normal_video):
    root = register_source(normal_video, 'film', 'v1', 'test', 'train')
    child = {**root, 'sample_id': 'child', 'parent_ids': [root['sample_id']], 'dataset_role': 'val'}
    assert any('불일치' in x or '충돌' in x for x in validate([root, child]))
    child['parent_ids'] = ['missing']
    assert any('부모 없음' in x for x in validate([root, child]))
    child['parent_ids'] = ['child']
    assert any('순환' in x for x in validate([root, child]))
    child.update(parent_ids=[root['sample_id']], dataset_role='train', pairing='paired', pair_id='missing')
    assert any('정답 참조' in x for x in validate([root, child]))


def test_clip_timing_hash_export_and_existing_loader(normal_video, tmp_path):
    digest = file_sha256(normal_video)
    parent = register_source(normal_video, 'film', 'v1', 'test', 'train')
    result = extract_clip(normal_video, tmp_path / 'clip.mkv', '0.08', '0.24', 32, 24)
    child = clip_record(parent, result, 'clip1')
    assert len(result['mapping']) == 4 and file_sha256(normal_video) == digest
    assert not validate([parent, child], check_files=True)
    assert probe_video(tmp_path / 'clip.mkv')['pts_status'] == 'valid'
    target = tmp_path / 'backgrounds_train.npz'
    export_backgrounds([tmp_path / 'clip.mkv'], 25, 32, 24, 4, target)
    script = Path(__file__).resolve().parents[1] / 'scripts' / 'train_gan.py'
    spec = importlib.util.spec_from_file_location('existing_train_gan', script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    clips, fps = module.load_backgrounds(tmp_path, 'train')
    assert clips.shape == (1, 4, 24, 32, 3) and fps == 25


def test_errors_and_south_guard(tmp_path):
    bad = tmp_path / 'broken.mkv'
    bad.write_bytes(b'invalid')
    with pytest.raises(Exception):
        probe_video(bad)
    with pytest.raises(PermissionError):
        read_south_labels(tmp_path / 'none.csv', terms_verified=False)


def test_multiple_parents_and_observations(normal_video):
    a = register_source(normal_video, 'one', 'v1', 'test', 'train')
    b = {**a, 'sample_id': 'src:other', 'source_id': 'src:other', 'work_id': 'two',
         'root_source_ids': ['src:other'], 'sha256': None, 'dataset_role': 'val'}
    child = {**a, 'sample_id': 'mixed', 'parent_ids': [a['sample_id'], b['sample_id']],
             'root_source_ids': [a['source_id'], b['source_id']],
             'observations': [{'tool': 'a', 'version': '1', 'settings': {}, 'run': 'run1', 'result': 'unknown'},
                              {'tool': 'b', 'version': '2', 'settings': {}, 'run': 'run2', 'result': 'unknown'}]}
    assert any('부모와 분할 불일치' in e for e in validate([a, b, child]))
    b['dataset_role'] = 'train'
    assert not validate([a, b, child])
    child['observations'].append({'tool': 'missing fields'})
    assert any('판정 관측' in e for e in validate([a, b, child]))
