import numpy as np
import pytest
import av

from lumi_synth import generation_workspace as workspace


def test_lossless_video_and_folder_scan(tmp_path):
    rng = np.random.default_rng(3)
    frames = rng.integers(0, 256, (12, 36, 64, 3), dtype=np.uint8)
    folder = tmp_path / "nested"
    folder.mkdir()
    movie = folder / "source.mkv"
    workspace.write_lossless_video(movie, frames, 24)
    with av.open(str(movie)) as c:
        decoded = np.stack([f.to_ndarray(format="bgr24") for f in c.decode(video=0)])
    assert np.array_equal(decoded, frames)
    assert workspace.list_sources(tmp_path) == []
    rows = workspace.list_sources(tmp_path, recursive=True)
    assert len(rows) == 1 and rows[0]["name"] == "nested/source.mkv"
    assert rows[0]["fps"] == 24 and rows[0]["width"] == 64
    assert rows[0]["height"] == 36 and rows[0]["seconds"] == pytest.approx(0.5)
    with pytest.raises(FileExistsError):
        workspace.write_lossless_video(movie, frames, 24)


def test_bad_file_does_not_hide_good_source(tmp_path):
    (tmp_path / "broken.mp4").write_bytes(b"not video")
    (tmp_path / "notes.txt").write_text("not video")
    rows = workspace.list_sources(tmp_path)
    assert len(rows) == 1 and "error" in rows[0]
    with pytest.raises(ValueError, match="폴더"):
        workspace.list_sources(tmp_path / "missing")
