import numpy as np
import pytest
from lumi_synth.dynamic import render


def test_moving_support_preserves_outside_and_changes_position():
    clean = np.full((12, 40, 60, 3), 70, dtype=np.uint8)
    out, mask = render(clean, 24, kind="flash", frequency=4, velocity=(0.5, 0))
    assert np.array_equal(out[~mask], clean[~mask])
    assert not np.array_equal(mask[0], mask[-1])
    assert np.any(out != clean)


def test_pattern_moves_and_bad_frequency_rejected():
    clean = np.full((8, 32, 32, 3), 128, dtype=np.uint8)
    out, mask = render(clean, 24, kind="pattern", frequency=0, pattern_speed=2)
    assert not np.array_equal(out[0][mask[0]], out[1][mask[1]])
    with pytest.raises(ValueError):
        render(clean, 24, kind="flash", frequency=12)
