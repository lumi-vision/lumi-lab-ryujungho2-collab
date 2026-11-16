"""생성 결과 npz → 바로 재생되는 mp4 (판정 없이 변환만, 영상 하나에 1~2초).

    python scripts/npz_to_mp4.py data/playlist/review/20261116_2011

폴더 안 npz 마다 <이름>.mp4 를 옆에 쓴다: 왼쪽 원본(clean), 오른쪽 합성본(flicker). 원본이 없으면 합성본만.
G 별 하위 폴더마다 표본을 이어 붙인 _모음.mp4 도 쓴다(표본 사이 0.5초 검은 화면). Windows 기본 플레이어로 연다.
⚠️ 광과민성 위험 영상이다 — 이 PC 에서만, 밝기를 낮춰 짧게 본다.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).parent))
from make_playlist import write_mp4  # noqa: E402


def frames_of(p: Path) -> tuple[np.ndarray, float]:
    with np.load(p, allow_pickle=False) as z:
        if "flicker" not in z.files:
            raise ValueError("flicker 없음")
        x = z["flicker"]
        fps = float(z["fps"]) if "fps" in z.files else 24.0
        clean = z["clean"] if "clean" in z.files and z["clean"].shape == x.shape else None
    if clean is not None:
        gap = np.zeros((x.shape[0], x.shape[1], 8, 3), np.uint8)
        x = np.concatenate([clean, gap, x], axis=2)
    out = x.copy()
    for t in range(len(out)):
        for txt, x0 in (("original", 6), ("generated", out.shape[2] - (x.shape[2] - 8) // 2 + 6 if clean is not None else 6))[:2 if clean is not None else 1]:
            cv2.putText(out[t], txt if clean is not None else "generated", (x0, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(out[t], txt if clean is not None else "generated", (x0, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return out, fps


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", type=Path)
    args = ap.parse_args()
    files = sorted(p for p in args.folder.rglob("*.npz") if not p.name.endswith(".alpha.npz"))
    groups: dict[Path, list] = {}
    for p in files:
        try:
            fr, fps = frames_of(p)
        except ValueError:
            continue
        write_mp4(p.with_suffix(".mp4"), iter(fr), fps)
        groups.setdefault(p.parent, []).append((fr, fps))
        print(p.relative_to(args.folder).with_suffix(".mp4"), flush=True)
    for d, items in groups.items():
        fps = items[0][1]
        blank = np.zeros((int(fps // 2),) + items[0][0].shape[1:], np.uint8)
        def it():
            for fr, _ in items:
                yield from fr
                yield from blank
        write_mp4(d / "_모음.mp4", it(), fps)
        print(f"{d.relative_to(args.folder)}/_모음.mp4 ({len(items)}개)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
