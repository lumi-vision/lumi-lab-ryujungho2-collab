"""독립 노트북(notebooks/static_pattern_screening.ipynb)의 계산 셀이
저장소 모듈(lumi.static_features)과 같은 결과를 내는지 확인한다.

노트북은 모듈을 import하지 않으므로 한쪽만 고치면 결과가 갈라질 수 있다.
"""
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from lumi import static_features as module

NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks" / "static_pattern_screening.ipynb"


@pytest.fixture(scope="module")
def notebook_ns():
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    cells = [c for c in nb["cells"] if "lumi-core" in c.get("metadata", {}).get("tags", [])]
    assert len(cells) >= 6, "계산 셀(tag: lumi-core)을 찾지 못했습니다"
    source = "\n".join("".join(c["source"]) for c in cells)
    assert "import lumi" not in source and "from lumi" not in source, "노트북이 모듈에 의존하면 안 됩니다"
    # dataclass는 sys.modules에서 정의 모듈을 찾으므로 임시 모듈로 등록해 실행한다.
    mod = types.ModuleType("static_pattern_screening_nb")
    sys.modules[mod.__name__] = mod
    try:
        exec(compile(source, str(NOTEBOOK), "exec"), mod.__dict__)
        yield mod.__dict__
    finally:
        sys.modules.pop(mod.__name__, None)


def _images():
    h, w = 240, 360
    yy, xx = np.mgrid[:h, :w]
    gray = lambda g: np.repeat((g * 255).astype(np.uint8)[..., None], 3, axis=2)
    red = np.full((h, w, 3), 255, np.uint8)
    red[:, : w // 2] = (255, 0, 0)
    rng = np.random.default_rng(3)
    return {
        "vertical": gray((xx // 6) % 2),
        "rings": gray((np.hypot(yy - h / 2, xx - w / 2) // 9) % 2),
        "half_bw": gray(xx >= w // 2),
        "red": red,
        "noise": rng.integers(0, 256, (h, w, 3), dtype=np.uint8),
    }


@pytest.mark.parametrize("name", list(_images()))
def test_notebook_matches_module(notebook_ns, name):
    img = _images()[name]
    a = module.analyze_image(img)
    b = notebook_ns["analyze_image"](img)
    assert np.array_equal(a.pop("mask"), b.pop("mask"))
    assert a == b


def test_notebook_pair_counter_matches(notebook_ns):
    for pairs in (4, 5, 10):
        p = np.concatenate([np.full(10, 200.0 if i % 2 == 0 else 0.0) for i in range(2 * pairs)])
        assert notebook_ns["count_stripe_pairs"](p) == module.count_stripe_pairs(p) == pairs
