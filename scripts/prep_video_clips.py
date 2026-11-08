"""긴 영상(예: Blender 오픈 영화)을 깜빡임 없는 G 배경 클립으로 자릅니다. 재생·미리보기 없음.

- 각 영상을 --stride초 간격으로 --frames장(@--fps) 구간으로 자르고 가운데 4:3을 W×H로 줄인다.
- 판정기(lumi.judge)로 1초 최대 전이가 --max-transitions 이하인 구간만 남긴다(기본 1: 장면 전환 1회 허용).
- 거의 검은 구간(평균 상대 휘도 < --min-luma, 예: 크레딧)은 뺀다.
- 분할: 영상마다 시간 순서로 앞 70% train, 다음 15% val, 끝 15% test(같은 장면이 학습·시험에 섞이지 않게).
- 출력: prep_real_clips.py와 같은 형식 — <out>/<split>.npy ([N,T,H,W,3] uint8), <out>/manifest.jsonl

사용: python scripts/prep_video_clips.py --video A.mp4 --video B.mkv --output ../_work/datasets/real_bg_movies_v1
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from lumi.judge import judge_luminance_flash, srgb_to_luminance_np


def read_resampled(path: Path, fps_out: float, w: int, h: int) -> tuple[np.ndarray, float]:
    """영상 전체를 fps_out으로 재샘플(가장 가까운 프레임)하면서 바로 W×H로 줄여 메모리를 아낀다."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"영상을 열 수 없습니다: {path}")
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 24.0
    frames, idx, next_t = [], 0, 0.0
    crop = None
    while True:
        ok, f = cap.read()
        if not ok:
            break
        t = idx / fps_in
        idx += 1
        if t + 1e-9 < next_t:
            continue
        if crop is None:
            h0, w0 = f.shape[:2]
            cw, ch = (h0 * 4 // 3, h0) if w0 * 3 > h0 * 4 else (w0, w0 * 3 // 4)
            crop = ((w0 - cw) // 2, (h0 - ch) // 2, cw, ch)
        x0, y0, cw, ch = crop
        small = cv2.resize(f[y0:y0 + ch, x0:x0 + cw], (w, h), interpolation=cv2.INTER_AREA)
        frames.append(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
        next_t += 1.0 / fps_out
    cap.release()
    return np.stack(frames), fps_in


def main() -> None:
    ap = argparse.ArgumentParser(description="긴 영상 → 깜빡임 없는 배경 클립")
    ap.add_argument("--video", type=Path, action="append", required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--fps", type=float, default=24.0)
    ap.add_argument("--frames", type=int, default=32)
    ap.add_argument("--width", type=int, default=128)
    ap.add_argument("--height", type=int, default=96)
    ap.add_argument("--stride", type=float, default=2.0, help="구간 시작 간격(초)")
    ap.add_argument("--max-transitions", type=int, default=1)
    ap.add_argument("--min-luma", type=float, default=0.02)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    buckets = {"train": [], "val": [], "test": []}
    manifest = []
    for path in args.video:
        movie, fps_in = read_resampled(path, args.fps, args.width, args.height)
        step = max(1, int(round(args.stride * args.fps)))
        starts = list(range(0, len(movie) - args.frames + 1, step))
        kept = {"flash": 0, "dark": 0, "ok": 0}
        for k, s in enumerate(starts):
            clip = movie[s:s + args.frames]
            if srgb_to_luminance_np(clip).mean() < args.min_luma:
                kept["dark"] += 1
                continue
            j = judge_luminance_flash(clip, args.fps)
            if j.max_transitions_1s > args.max_transitions:
                kept["flash"] += 1
                continue
            pos = k / max(1, len(starts) - 1)
            split = "train" if pos < 0.70 else "val" if pos < 0.85 else "test"
            manifest.append({"index": len(buckets[split]), "split": split, "source": path.stem,
                             "start_sec": round(s / args.fps, 2), "source_fps": fps_in,
                             "transitions_1s": j.max_transitions_1s})
            buckets[split].append(clip)
            kept["ok"] += 1
        print(f"{path.name}: {len(movie) / args.fps:.0f}초 · 구간 {len(starts)}개 → 사용 {kept['ok']}, "
              f"깜빡임 제외 {kept['flash']}, 어두움 제외 {kept['dark']}")
    for sp, clips in buckets.items():
        np.save(args.output / f"{sp}.npy", np.stack(clips) if clips else np.zeros((0,), np.uint8))
    with open(args.output / "manifest.jsonl", "w", encoding="utf-8") as f:
        for m in manifest:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")
    print("분할:", {sp: len(c) for sp, c in buckets.items()})


if __name__ == "__main__":
    main()
