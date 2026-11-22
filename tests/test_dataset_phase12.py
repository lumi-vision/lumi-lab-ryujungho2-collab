"""가상 계보와 모델 변환의 최소 동작 및 실패 조건을 검증한다."""
import importlib.util
from fractions import Fraction
from pathlib import Path
import av
import numpy as np
import pytest
from lumi_synth.dataset.canonical import add_virtual_clip, validate_canonical
from lumi_synth.dataset.model_input import export_one
from lumi_synth.dataset.video import register_source


@pytest.fixture
def video(tmp_path):
    """시간과 내용이 안정적인 정상 시험 영상을 만든다."""
    path = tmp_path / 'v.nut'
    with av.open(str(path), 'w', format='nut') as c:
        stream = c.add_stream('ffv1', rate=24)
        stream.width, stream.height, stream.pix_fmt = 32, 24, 'bgr0'
        stream.time_base = stream.codec_context.time_base = Fraction(1, 24)
        for i in range(8):
            frame = av.VideoFrame.from_ndarray(np.full((24, 32, 3), 80, np.uint8), format='bgr24')
            frame.pts, frame.time_base = i, Fraction(1, 24)
            for packet in stream.encode(frame): c.mux(packet)
        for packet in stream.encode(): c.mux(packet)
    return path


def test_lineage_and_duplicate(video):
    root = register_source(video, 'work', 'v1', 'test', 'train')
    rows = add_virtual_clip([root], root['sample_id'], '0', '1/4')
    assert len(rows) == 2 and len(add_virtual_clip(rows, root['sample_id'], '0/1', '0.25')) == 2
    assert not validate_canonical(rows)
    bad = [rows[0], {**rows[1], 'dataset_role': 'test'}]
    assert validate_canonical(bad)
    bad = [rows[0], {**rows[1], 'canonical_clip': {**rows[1]['canonical_clip'], 'frames': []}}]
    assert validate_canonical(bad)
    with pytest.raises(ValueError): add_virtual_clip([root], root['sample_id'], '2', '1')


def test_model_input_provenance_and_loader(video, tmp_path):
    root = register_source(video, 'work', 'v1', 'test', 'train')
    row = add_virtual_clip([root], root['sample_id'], '0', '1/4')[1]
    with pytest.raises(ValueError): export_one(row, tmp_path / 'x.npz', width=16, height=12)
    row.update(verification_status='verified', risk_label='safe', label_evidence='정상 단색 시험 영상')
    result = export_one(row, tmp_path / 'backgrounds_train.npz', width=16, height=12)
    assert result['risk_label'] is None and result['resampling'] == 'none'
    assert len(result['frame_mapping']) == 6 and result['source_metadata']['source_width'] == 32
    script = Path(__file__).resolve().parents[1] / 'scripts' / 'train_gan.py'
    spec = importlib.util.spec_from_file_location('test_existing_g', script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    arr, fps = module.load_backgrounds(tmp_path, 'train')
    assert arr.shape == (1, 6, 12, 16, 3) and fps == 24


def test_source_change(video):
    root = register_source(video, 'work', 'v1', 'test', 'train')
    rows = add_virtual_clip([root], root['sample_id'], '0', '1/4')
    with video.open('ab') as f: f.write(b'changed')
    assert validate_canonical(rows)
