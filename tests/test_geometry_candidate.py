"""G 데이터 생성 명령에서 기하학 대조 자료의 포맷과 라벨 의미를 점검한다."""

import json
from pathlib import Path
import subprocess
import sys

import numpy as np


def test_geometry_cli(tmp_path):
    script = Path(__file__).resolve().parents[1] / "scripts/make_gan_dataset.py"
    command = [sys.executable, str(script), "geometry", "--out", str(tmp_path), "--seed", "7"]
    subprocess.run(command, check=True)
    file = tmp_path / "geometry_000007.npz"
    with np.load(file, allow_pickle=False) as sample:
        frames, mask = sample["input"], sample["object_mask"]
        assert frames.shape == (192, 108, 192, 3)
        assert mask.shape == frames.shape[:3]
        assert not np.array_equal(mask[0], mask[-1])
        assert np.array_equal(frames, sample["clean"])
        assert np.max(np.ptp(frames, axis=(0, 1, 2))) <= 16
        np.testing.assert_allclose(np.diff(sample["timestamps"]), 1 / float(sample["fps"]))
        assert set(sample.files).isdisjoint({"mask", "target_mask", "risk_label"})
        assert str(sample["validation_status"]) == "pending_independent_review"
        assert json.loads(sample["metadata"].item())["object_mask_semantics"] == "implicit_field_contour_not_risk"
    before = file.read_bytes()
    assert subprocess.run(command, capture_output=True).returncode != 0
    assert file.read_bytes() == before
