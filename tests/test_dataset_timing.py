"""정상 단색·저변화 영상으로 시간축과 디코딩 왕복을 검증한다."""
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
import av
import importlib.util
import numpy as np
import pytest
from lumi_synth.dataset.catalog import file_sha256
from lumi_synth.dataset.export import export_virtual_backgrounds
from lumi_synth.dataset.usage import append_usage, eligible_independent
from lumi_synth.dataset.video import extract_clip, select_frames, virtual_clip


def make_video(path, times):
    """프레임 구분이 가능한 어두운 영상의 시간축을 저장한다."""
    with av.open(str(path), 'w', format='nut') as container:
        stream = container.add_stream('ffv1', rate=25)
        stream.width, stream.height, stream.pix_fmt = 32, 24, 'bgr0'
        stream.time_base = Fraction(1, 120000)
        stream.codec_context.time_base = Fraction(1, 120000)
        for i, time in enumerate(times):
            frame = av.VideoFrame.from_ndarray(np.full((24, 32, 3), 70 + i, np.uint8), format='bgr24')
            frame.time_base = Fraction(1, 120000)
            frame.pts = int(time / frame.time_base)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


@pytest.mark.parametrize('fps', [Fraction(24), Fraction(25), Fraction(30), Fraction(24000,1001), Fraction(30000,1001), Fraction(60)])
def test_rates_roundtrip(tmp_path, fps):
    source, out = tmp_path / 'source.nut', tmp_path / 'out.mkv'
    make_video(source, [Fraction(i, 1) / fps for i in range(8)])
    before = file_sha256(source)
    ref = virtual_clip(source, '0', '1')
    result = extract_clip(source, out, '0', '1')
    assert file_sha256(source) == before
    assert len(ref['frames']) == len(result['mapping']) == 8
    for original, row in zip(ref['frames'], result['mapping']):
        assert row['source_pts'] == original['source_pts'] and row['source_time_base'] == original['source_time_base']
        assert Fraction(row['source_time']) == Fraction(row['source_pts']) * Fraction(row['source_time_base'])
        assert abs(Fraction(row['quantization_error'])) <= Fraction(1, 2000)
    assert result['last_frame_duration_status'] == 'unknown'


def test_vfr_nonzero_boundary_and_missing(tmp_path):
    source = tmp_path / 'source.nut'
    times = [Fraction(2), Fraction(51,25), Fraction(21,10), Fraction(54,25)]
    make_video(source, times)
    ref = virtual_clip(source, '2.04', '2.16')
    assert len(ref['frames']) == 2 and ref['frames'][0]['source_index'] == 1
    result = extract_clip(source, tmp_path / 'out.mkv', '2.04', '2.16')
    assert [r['source_index'] for r in result['mapping']] == [1, 2]
    assert Fraction(result['mapping'][0]['source_time']) == Fraction(51, 25)
    def f(pts):
        return SimpleNamespace(pts=pts, time_base=Fraction(1, 25))
    for frames in ([f(None)], [f(1), f(1)], [f(2), f(1)]):
        with pytest.raises(ValueError):
            select_frames(iter(frames), Fraction(0), Fraction(1))


def test_usage_history(tmp_path):
    record = {'experiment_id': 'a', 'sample_id': 'kaya:1', 'usage_role': 'val',
              'model_selection_used': True, 'threshold_selection_used': True,
              'first_accessed_at': None, 'evidence': 'checks/hazcls/a.md'}
    out = tmp_path / 'usage.jsonl'
    append_usage(out, record)
    append_usage(out, {**record, 'experiment_id': 'b'})
    assert len(out.read_text(encoding='utf8').splitlines()) == 2
    assert not eligible_independent([record], 'kaya:1')
    assert not eligible_independent([], 'unknown')


def test_g_24fps_virtual_export(tmp_path):
    source = tmp_path / 'source.nut'
    make_video(source, [Fraction(i, 24) for i in range(8)])
    clip = virtual_clip(source, '0', '1')
    export_virtual_backgrounds([clip], '24', 32, 24, 8, tmp_path / 'backgrounds_train.npz')
    script = Path(__file__).resolve().parents[1] / 'scripts' / 'train_gan.py'
    spec = importlib.util.spec_from_file_location('existing_train_gan', script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    data, fps = module.load_backgrounds(tmp_path, 'train')
    assert fps == 24 and data.shape == (1, 8, 24, 32, 3)
