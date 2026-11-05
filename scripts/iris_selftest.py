"""IRIS 라벨러 확인 시험(docs/IRIS_LABELING_PLAN.md 5절).

결과를 미리 아는 자극을 만들어 IRIS에만 입력하고, 기대 판정과 비교합니다.
주의: 시험 자극은 광과민성 유해 자극입니다. 생성 파일을 재생·미리보기하지 마세요.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from lumi.iris_labeling import IrisError, IrisRunner, standard_result, summarize
from lumi.lossless_video import write_lossless_video

FPS = 24
SECONDS = 6


def square_flash(hz: float, h: int, w: int, on=(255, 255, 255), off=(0, 0, 0),
                 area: float = 1.0) -> np.ndarray:
    t = np.arange(FPS * SECONDS) / FPS
    state = (np.sin(2 * np.pi * hz * t) >= 0)
    clip = np.zeros((len(t), h, w, 3), np.uint8)
    clip[:] = off
    side = np.sqrt(area)
    rh, rw = int(round(h * side)), int(round(w * side))
    y0, x0 = (h - rh) // 2, (w - rw) // 2
    for i, s in enumerate(state):
        if s:
            clip[i, y0:y0 + rh, x0:x0 + rw] = on
    return clip


def stripes(h: int, w: int, period: int = 8) -> np.ndarray:
    x = np.arange(w)
    row = np.where((x // (period // 2)) % 2 == 0, 255, 0).astype(np.uint8)
    frame = np.repeat(np.repeat(row[None, :, None], h, 0), 3, 2)
    return np.repeat(frame[None], FPS * SECONDS, 0)


def write_mp4(path: Path, rgb: np.ndarray) -> Path:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, rgb.shape[2:0:-1])
    if not writer.isOpened():
        raise RuntimeError("mp4v 인코더를 열 수 없습니다.")
    for frame in rgb:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()
    return path


# (번호, 이름, 자극 생성, 기대: {키: 값} — summarize 결과의 *_max와 비교)
H, W = 96, 128
CASES = [
    ("1", "lum_4hz", lambda: square_flash(4, H, W), {"luminance_max": 3}),
    ("2", "lum_1hz", lambda: square_flash(1, H, W), {"luminance_max": 0, "red_max": 0}),
    ("3", "lum_2p5hz", lambda: square_flash(2.5, H, W), {"luminance_max": 2}),
    ("4a", "stripes_128x96", lambda: stripes(96, 128), {"pattern_max": 1}),
    ("4b", "stripes_640x480", lambda: stripes(480, 640, period=16), {"pattern_max": 1}),
    ("5", "red_4hz", lambda: square_flash(4, H, W, on=(255, 0, 0)), {"red_max": 3}),
    ("6", "lum_4hz_area10", lambda: square_flash(4, H, W, area=0.10), {"luminance_max": 0}),
]


def main() -> None:
    parser = argparse.ArgumentParser(description="IRIS 라벨러 확인 시험(재생 금지)")
    parser.add_argument("--iris-exe", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("iris_checks"))
    args = parser.parse_args()
    runner = IrisRunner(args.iris_exe)
    args.output.mkdir(parents=True, exist_ok=True)
    report = {"iris_version": runner.version, "settings_sha256": runner.settings_sha256,
              "fps": FPS, "seconds": SECONDS, "cases": []}

    def check(num, name, path, expect):
        try:
            frames, _ = runner.run(path)
            s = summarize(frames)
            got = {k: s[k] for k in ("luminance_max", "red_max", "pattern_max")}
            ok = all(got[k] == v for k, v in expect.items())
            entry = {"case": num, "name": name, "file": path.name, "expect": expect,
                     "got": got, "result": standard_result(s), "frames": s["frames"], "ok": ok}
        except IrisError as exc:
            entry = {"case": num, "name": name, "file": path.name, "expect": expect,
                     "error": str(exc)[:500], "ok": False}
        report["cases"].append(entry)
        print(f"[{'OK' if entry['ok'] else '!!'}] {num:>3} {name:<18} 기대 {expect} → "
              f"{entry.get('got', entry.get('error'))} {entry.get('result', '')}")
        return entry

    for num, name, make, expect in CASES:
        check(num, name, write_lossless_video(args.output / f"{name}.mkv", make(), FPS), expect)

    # 7: 같은 자극(1번)의 무손실 MKV와 손실 MP4 판정 비교 → 저장 코덱 결정 근거
    mp4 = write_mp4(args.output / "lum_4hz.mp4", square_flash(4, H, W))
    e7 = check("7", "lum_4hz_mp4", mp4, {"luminance_max": 3})
    first = next(c for c in report["cases"] if c["case"] == "1")
    report["codec_same_result"] = first.get("got") == e7.get("got")
    print(f"MKV/MP4 판정 동일: {report['codec_same_result']}")

    (args.output / "selftest_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    passed = sum(c["ok"] for c in report["cases"])
    print(f"\n{passed}/{len(report['cases'])} 기대와 일치 · 보고서: {args.output / 'selftest_report.json'}")


if __name__ == "__main__":
    main()
