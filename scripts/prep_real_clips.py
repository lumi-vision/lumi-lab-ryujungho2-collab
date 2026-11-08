"""실제 안전 GIF·영상을 G 학습용 배경 클립으로 정리합니다(재생·미리보기 없음).

- 입력: South 2021 OSF 자료(docs/DATASETS.md X1)의 미디어 폴더와 라벨 CSV
- 저자 라벨 'dangerous'는 시험 전용이므로 **제외**한다. 우리 판정기(lumi.judge)로 실패인 클립도 제외한다
  (배경에 이미 깜빡임이 있으면 G가 만든 위험과 구분되지 않기 때문).
- 각 원본에서 클립 1개: 실제 시각 기준으로 목표 FPS에 재샘플(가장 가까운 프레임), 짧으면 반복 재생,
  가운데를 4:3으로 잘라 W×H로 축소.
- 원본 이름의 해시로 train/val/test를 나눈다. test는 학습에 쓰지 않는다.
- 출력: <out>/<split>.npy ([N,T,H,W,3] uint8), <out>/manifest.jsonl
"""
import argparse
import csv
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from lumi.judge import judge_luminance_flash


def split_of(name: str) -> str:
    k = int(hashlib.sha256(name.encode()).hexdigest()[:8], 16) % 100
    return "test" if k < 15 else "val" if k < 25 else "train"


def load_clip(path: Path, fps_out: float, frames: int, width: int, height: int):
    cap = cv2.VideoCapture(str(path))
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 10.0
    src = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        src.append(f)
    cap.release()
    if len(src) < 2:
        return None, fps_in
    h0, w0 = src[0].shape[:2]
    # 가운데 4:3 자르기
    if w0 * 3 > h0 * 4:
        cw, ch = h0 * 4 // 3, h0
    else:
        cw, ch = w0, w0 * 3 // 4
    x0, y0 = (w0 - cw) // 2, (h0 - ch) // 2
    duration = len(src) / fps_in
    out = []
    for i in range(frames):
        t = (i / fps_out) % duration                       # 반복 재생
        f = src[min(len(src) - 1, int(t * fps_in))]
        f = cv2.resize(f[y0:y0 + ch, x0:x0 + cw], (width, height), interpolation=cv2.INTER_AREA)
        out.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    return np.stack(out), fps_in


def work(args):
    label, path, fps, frames, w, h = args
    clip, fps_in = load_clip(path, fps, frames, w, h)
    if clip is None:
        return label, None, "unreadable", fps_in
    j = judge_luminance_flash(clip, fps)
    if j.fail:
        return label, None, "judged_flash", fps_in
    return label, clip, "ok", fps_in


def main() -> None:
    ap = argparse.ArgumentParser(description="실제 안전 GIF → 배경 클립")
    ap.add_argument("--media", type=Path, required=True)
    ap.add_argument("--labels", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--fps", type=float, default=24.0)
    ap.add_argument("--frames", type=int, default=32)
    ap.add_argument("--width", type=int, default=128)
    ap.add_argument("--height", type=int, default=96)
    args = ap.parse_args()
    rows = [r for r in csv.DictReader(open(args.labels, encoding="utf-8")) if r["dangerous"] == "safe"]
    files = {p.stem: p for p in args.media.iterdir() if p.suffix.lower() in (".gif", ".mp4")}
    jobs = [(r["label"], files[r["label"]], args.fps, args.frames, args.width, args.height)
            for r in rows if r["label"] in files]
    args.output.mkdir(parents=True, exist_ok=True)
    buckets = {"train": [], "val": [], "test": []}
    manifest, skipped = [], {}
    with ProcessPoolExecutor(8) as ex:
        for label, clip, status, fps_in in ex.map(work, jobs, chunksize=8):
            if clip is None:
                skipped[status] = skipped.get(status, 0) + 1
                continue
            sp = split_of(label)
            manifest.append({"index": len(buckets[sp]), "split": sp, "source": label, "source_fps": fps_in})
            buckets[sp].append(clip)
    for sp, clips in buckets.items():
        np.save(args.output / f"{sp}.npy", np.stack(clips) if clips else np.zeros((0,), np.uint8))
    with open(args.output / "manifest.jsonl", "w", encoding="utf-8") as f:
        for m in manifest:
            f.write(json.dumps(m) + "\n")
    print({sp: len(c) for sp, c in buckets.items()}, "skipped:", skipped)
    print("주의: 저자 'dangerous' 163개는 넣지 않았다. test 분할은 학습에 쓰지 않는다.")


if __name__ == "__main__":
    main()
