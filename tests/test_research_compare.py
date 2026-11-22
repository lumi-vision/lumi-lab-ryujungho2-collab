"""정상 영상과 비영상 배열로만 공통 평가 인터페이스를 확인한다."""
import numpy as np
import pytest
import torch
from lumi_synth.hazcls import HazNet
from lumi_synth.baselines.comparison import evaluate_records, infer, metrics


def test_metrics_and_undefined():
    r = metrics([0, 1, 1, 0], [0, 1, 0, 1], [.1, .8, .3, .6])
    assert r['precision'] == r['recall'] == r['f1'] == r['fnr'] == 0.5
    assert r['auroc'] == 0.75
    assert metrics([0], [0])['auroc'] is None
    with pytest.raises(ValueError):
        evaluate_records([{'role': 'final_test'}])


def test_normal_array_b0_b1():
    frames = np.full((4, 32, 48, 3), 80, dtype=np.uint8)
    b0 = infer(frames, 24, 'B0')
    b1 = infer(frames, 24, 'B1', coefficient=1, intercept=-1, threshold=.5)
    assert b0['model'] == 'B0' and b1['model'] == 'B1'
    assert b1['decisions']['pattern'] is None
    assert b0['any_score'] is None
    with pytest.raises(ValueError): infer(frames, 24, 'B1')
    with pytest.raises(ValueError): infer(frames, 24, 'B2')


def test_b2_interface_normal_only(tmp_path):
    path = tmp_path / 'test.pt'
    net = HazNet(width=4)
    torch.save({'config': {'width': 4}, 'model': net.state_dict()}, path)
    result = infer(np.full((48, 32, 48, 3), 80, np.uint8), 24, 'B2', checkpoint=path)
    assert set(result['scores']) == {'lum', 'red', 'pattern', 'any'}
    assert all(0 <= value <= 1 for value in result['scores'].values())
