"""비영상 배열과 정상 단색 영상으로 기준선 인터페이스를 확인한다."""
import numpy as np
import pytest
import av
from lumi_synth.baselines.flashguard import analyze, evaluate_binary, features, fit_logistic, mitigate, preservation


def test_normal_video_and_local_correction():
    frames = np.full((4, 12, 16, 3), 80, dtype=np.uint8)
    assert np.all(features(frames, 24) == 0)
    result = analyze(frames, 24, 1, -1, grid=(3, 4))
    assert result['scores'].shape == (3, 3, 4)
    mask = np.zeros((4, 12, 16), bool)
    mask[:, 2:5, 3:8] = True
    corrected = mitigate(frames, mask, 0.2, 0.5)
    assert np.array_equal(frames[~mask], corrected[~mask])
    assert preservation(frames, corrected, mask)['outside_mae'] == 0


def test_training_and_binary_metrics():
    model = fit_logistic(np.array([0, 0.1, 1, 1.1]), np.array([0, 0, 1, 1]))
    assert np.isfinite(model['coefficient'])
    assert evaluate_binary([1, 0, 1], [1, 1, 0]) == {'tp': 1, 'fn': 1, 'tn': 0, 'fp': 1, 'recall': 0.5, 'specificity': 0.0}
    with pytest.raises(ValueError): fit_logistic(np.zeros(3), np.zeros(3))


def test_normal_video_decode(tmp_path):
    path = tmp_path / 'normal.mkv'
    with av.open(str(path), 'w') as container:
        stream = container.add_stream('ffv1', rate=25)
        stream.width, stream.height, stream.pix_fmt = 16, 12, 'bgr0'
        for _ in range(4):
            frame = av.VideoFrame.from_ndarray(np.full((12, 16, 3), 80, dtype=np.uint8), format='bgr24')
            for packet in stream.encode(frame): container.mux(packet)
        for packet in stream.encode(): container.mux(packet)
    with av.open(str(path)) as container:
        frames = np.stack([frame.to_ndarray(format='bgr24') for frame in container.decode(video=0)])
    assert np.all(features(frames, 25) == 0)
