"""합성 데이터셋의 정합성, 원본 분할 누출, ROI 외 변형을 검사합니다."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def audit_dataset(root: Path) -> dict:
    """파일 존재, 배열 형상, 변형 경계, 원본 단위 분할을 확인합니다.

    이 검사는 PSE 안전성 또는 표준 적합성을 판단하지 않습니다.
    """
    root = Path(root)
    rows = [json.loads(line) for line in (root / "manifest.jsonl").read_text(
        encoding="utf-8").splitlines() if line.strip()]
    groups = defaultdict(set)
    failures = []
    type_counts = defaultdict(int)
    for row in rows:
        key = row["id"]
        type_counts[row["type"]] += 1
        groups[row["source_id"]].add(row["split"])
        try:
            with np.load(root / row["file"], allow_pickle=False) as sample:
                clean = sample["clean"]
                affected = sample["affected"]
                mask = sample["mask"]
                activation = sample["activation"]
            if clean.dtype != np.uint8 or affected.dtype != np.uint8 or clean.shape != affected.shape:
                failures.append(f"{key}: 영상 배열의 형식이 일치하지 않습니다.")
            if mask.shape != clean.shape[:3] or activation.shape != (clean.shape[0],):
                failures.append(f"{key}: ROI 또는 시간 라벨 형식이 잘못되었습니다.")
            if not np.array_equal(clean[mask == 0], affected[mask == 0]):
                failures.append(f"{key}: ROI 밖에 영상 변화가 발생했습니다.")
            if row["type"] == "normal" and not np.array_equal(clean, affected):
                failures.append(f"{key}: 정상 기준 영상이 수정되었습니다.")
        except (OSError, KeyError, ValueError) as exc:
            failures.append(f"{key}: {exc}")
    for source, splits in groups.items():
        if len(splits) > 1:
            failures.append(f"원본 분할 누출: {source}가 {sorted(splits)}에 존재합니다.")
    return {"total": len(rows), "sources": len(groups), "types": dict(type_counts),
            "failures": failures, "ok": not failures,
            "note": "데이터 정합성 검사일 뿐 PSE 안전성 판정이 아닙니다."}
