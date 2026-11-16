"""생성 쌍을 플레이어에 다시 열 때 의도가 위험 정답으로 바뀌지 않아야 한다."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from lumi_synth import generate as gen


@pytest.fixture(scope="module")
def player():
    path = Path(__file__).resolve().parents[1] / "scripts/make_playlist.py"
    spec = importlib.util.spec_from_file_location("lumi_test_playlist", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_reopened_pair_preserves_original_and_intent(player, tmp_path):
    clean = np.full((24, 36, 64, 3), 80, np.uint8)
    recipe = gen.sample_recipe("lum", "hazard", 2, split="val", bg=0)
    recipe["id"] = "pair"
    res = gen.make(recipe, tmp_path, clean=clean, fps=24)
    items = player.npz_items(res["path"], "test", "pair", None)
    synth = next(it for it in items if it.sub.startswith("flicker"))
    original = next(it for it in items if it.sub.startswith("clean"))
    frames, fps, gt = synth.load()
    assert synth.label is True and synth.meta["label_source"] == "intent"
    assert original.label is None
    assert "hz_frames" not in gt and "hz_mask" not in gt
    assert np.array_equal(gt["clean"], clean)
    assert np.array_equal(gt["region"], (frames != clean).any(-1))
    assert fps == 24


def test_generated_item_does_not_invent_hazard_timeline(player, tmp_path, monkeypatch):
    clean = np.full((24, 36, 64, 3), 80, np.uint8)
    monkeypatch.setattr(gen, "load_backgrounds", lambda split: (clean[None], 24))
    recipe = gen.sample_recipe("lum", "near", 2, split="val", bg=0)
    recipe.update(id="generated", group="test")
    item = player.generated_item(recipe, tmp_path)
    _, _, gt = item.load()
    assert item.label is False and item.meta["label_source"] == "intent"
    assert "hz_frames" not in gt
    assert np.array_equal(gt["clean"], clean)


def test_generation_and_inspection_are_separate_http_jobs(tmp_path):
    """생성 전용 요청은 판정하지 않고 저장, 명시적 검사 요청에서만 판정한다."""
    import json
    import re
    import socket
    import subprocess
    import time
    import urllib.request
    import av
    from lumi_synth import generation_workspace as workspace

    source_dir = tmp_path / "originals"
    source_dir.mkdir()
    source = source_dir / "source.mkv"
    workspace.write_lossless_video(source, np.full((48, 36, 64, 3), 80, np.uint8), 24)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    root = Path(__file__).resolve().parents[1]
    out = tmp_path / "output"
    proc = subprocess.Popen([sys.executable, str(root / "scripts/make_playlist.py"), "--serve", "--no-browser",
                             "--port", str(port), "--out", str(out)], cwd=root,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    base = f"http://127.0.0.1:{port}"
    try:
        for _ in range(100):
            try:
                html = urllib.request.urlopen(base + "/generate", timeout=1).read().decode()
                break
            except OSError:
                time.sleep(0.1)
        else:
            pytest.fail("서버 시작 실패")
        token = re.search(r'const TOKEN = "([a-f0-9]+)"', html).group(1)

        def call(path, body=None):
            request = urllib.request.Request(base + path, data=json.dumps(body).encode() if body is not None else None,
                                             headers={"X-Lumi-Token": token, "Content-Type": "application/json"})
            return json.load(urllib.request.urlopen(request, timeout=30))

        def wait_job(job):
            for _ in range(200):
                state = call("/api/job/" + job["job"])
                if state["done"] == state["total"]:
                    assert not state["errors"], state
                    return state["records"]
                time.sleep(0.1)
            pytest.fail("작업 완료 대기 초과")

        assert len(call("/api/sources", {"folder": str(source_dir)})["files"]) == 1
        job = call("/api/generate", {"kind": "lum", "intent": "near", "source": str(source), "count": 1,
                                     "seed": 14, "generation_only": True, "freq_hz": 2, "seconds": 2})
        records = wait_job(job)
        rec = records[0]
        assert Path(rec["movie"]).is_file() and Path(rec["source"]).is_file()
        assert rec["outcome"] == "not_run" and not list((out / "cache").glob("*.json"))
        assert call("/api/generated")["results"][0]["id"] == rec["id"]
        with np.load(rec["source"]) as z, av.open(rec["movie"]) as c:
            decoded = np.stack([f.to_ndarray(format="bgr24") for f in c.decode(video=0)])
            assert np.array_equal(decoded, z["flicker"])
        inspection = wait_job(call("/api/inspect?fast=1", {"id": rec["id"]}))
        assert len(inspection) == 2  # 원본과 합성본 모두 검사
        assert all(r["det"] is not None for r in inspection)
        assert any(r["meta"]["label_source"] == "intent" for r in inspection)
    finally:
        if sys.platform == "win32":
            subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], capture_output=True)
        else:
            proc.terminate()
        proc.wait(timeout=10)
