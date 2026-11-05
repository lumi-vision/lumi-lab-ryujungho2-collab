"""연구용 영상 데이터 생성의 형식과 라벨 일치성 자동 검사."""
from pathlib import Path

import numpy as np
import pytest

from lumi.dataset_synthesis import DatasetConfig, STIMULUS_TYPES, build_dataset, synthesize
from lumi.dataset_audit import audit_dataset


def test_all_modes_preserve_outside_roi():
    cfg = DatasetConfig(frames=8, width=48, height=40, fps=24)
    clean = np.random.default_rng(50).integers(0, 256, (8, 40, 48, 3), dtype=np.uint8)
    for kind in STIMULUS_TYPES:
        affected, mask, activation, meta = synthesize(clean, cfg, kind, np.random.default_rng(42))
        assert affected.shape == clean.shape
        assert mask.shape == clean.shape[:3]
        assert activation.shape == (8,)
        assert np.array_equal(clean[mask == 0], affected[mask == 0])
        if kind == "normal":
            assert np.array_equal(clean, affected)
            assert not np.any(mask)
        else:
            assert mask.sum() > 0
            assert meta["intensity"] > 0


def test_deterministic_synthesis():
    cfg = DatasetConfig(frames=8, width=48, height=40)
    clean = np.zeros((8, 40, 48, 3), dtype=np.uint8)
    first = synthesize(clean, cfg, "luminance", np.random.default_rng(3))
    second = synthesize(clean, cfg, "luminance", np.random.default_rng(3))
    for a, b in zip(first[:3], second[:3]):
        assert np.array_equal(a, b)
    assert first[3] == second[3]


def test_dataset_audit_and_source_split(tmp_path: Path):
    cv2 = pytest.importorskip("cv2")
    source = tmp_path / "source.avi"
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"MJPG"), 24, (48, 40))
    assert writer.isOpened()
    for index in range(9):
        writer.write(np.full((40, 48, 3), 20 + index, dtype=np.uint8))
    writer.release()
    dataset = tmp_path / "oracle"
    entries = build_dataset([source], dataset, samples_per_source=8,
                            cfg=DatasetConfig(frames=8, width=48, height=40, fps=24))
    assert len(entries) == 8
    assert all(entry["split"] == "train" for entry in entries)
    assert set(entry["type"] for entry in entries) == set(STIMULUS_TYPES)
    report = audit_dataset(dataset)
    assert report["ok"], report["failures"]


def test_invalid_dimensions_rejected():
    with pytest.raises(ValueError):
        DatasetConfig(frames=1).validate()


def test_duplicate_sources_get_same_split(tmp_path: Path):
    from lumi.dataset_synthesis import _source_key
    a = tmp_path / "a.bin"
    b = tmp_path / "copied.bin"
    a.write_bytes(b"same source content")
    b.write_bytes(a.read_bytes())
    assert _source_key(a) == _source_key(b)
