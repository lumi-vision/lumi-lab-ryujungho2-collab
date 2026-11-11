"""지속(연장) 플래시: 우리 판정(measure.extended_flash)과 EA IRIS 를 같은 시험 영상으로 비교한다.

⚠️ 만드는 영상은 깜빡임 자극이다. 재생·미리보기하지 않는다. 파일은 data/extended_test/ (Git 제외)와 IRIS 입력 폴더에만 둔다.

시험 영상: 화면 전체 회색 사각파 깜빡임 40 ↔ 100 cd/m² (전이 60 cd/m², 어두운 쪽 < 160), 320×180, 30 fps, 10초.
          빈도 {1.0, 1.5, 2.0, 2.5, 3.0, 3.5} 플래시/초 × 지속 {3, 4.5, 5.5, 7} 초, 0.5초 뒤 시작.
사용:
  python scripts/validate_extended_iris.py make          # 영상 만들기 + 우리 판정
  (IrisApp.exe 실행: tools/iris/TestVideos 에 복사된 영상을 분석)
  python scripts/validate_extended_iris.py report --iris-results <tools/iris/Results>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from lumi_synth import measure  # noqa: E402

OUT = ROOT / "data" / "extended_test"
FPS, W, H, TOTAL_S = 30.0, 320, 180, 10.0
HZ = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
DUR = [3.0, 4.5, 5.5, 7.0]


def gray_for_cd(cd: float) -> int:
    """휘도 cd (흰색 200 기준) 가 되는 sRGB 회색값."""
    lin = cd / measure.WHITE_CD
    s = 12.92 * lin if lin <= 0.0031308 else 1.055 * lin ** (1 / 2.4) - 0.055
    return int(round(np.clip(s, 0, 1) * 255))


def make_video(hz: float, dur: float) -> np.ndarray:
    n = int(TOTAL_S * FPS)
    lo, hi = gray_for_cd(40.0), gray_for_cd(100.0)
    state = np.zeros(n, bool)
    half = FPS / (2 * hz)                       # 반주기(프레임)
    t0, t1 = int(0.5 * FPS), int((0.5 + dur) * FPS)
    for t in range(t0, min(n, t1)):
        state[t] = int((t - t0) / half) % 2 == 0
    vals = np.where(state, hi, lo).astype(np.uint8)
    return np.broadcast_to(vals[:, None, None, None], (n, H, W, 3)).copy()


def ours(frames):
    r = measure.analyze(frames, FPS)
    e = r["transitions"]
    out = {"flash_dangerous": bool(r["dangerous"]), "max_flashes": float(r["max_flashes"])}
    for name, kw in (("iris", {}), ("iris_1to3hz", {"min_transitions": 2}), ("continuous", {"mode": "continuous"})):
        x = measure.extended_flash(e, FPS, **kw)
        out[f"ext_{name}"] = x["extended_dangerous"]
        out[f"ext_{name}_first_s"] = x["first_fail_s"]
    return out


def cmd_make():
    OUT.mkdir(parents=True, exist_ok=True)
    iris_in = ROOT.parent / "tools" / "iris" / "TestVideos"
    iris_in.mkdir(parents=True, exist_ok=True)
    rows = []
    for hz in HZ:
        for dur in DUR:
            name = f"ext_{hz:.1f}hz_{dur:.1f}s".replace(".", "p")
            frames = make_video(hz, dur)
            path = OUT / f"{name}.avi"
            wr = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"FFV1"), FPS, (W, H))
            for f in frames:
                wr.write(f)
            wr.release()
            (iris_in / path.name).write_bytes(path.read_bytes())
            rows.append({"name": name, "hz": hz, "dur_s": dur, **ours(frames)})
            print(rows[-1], flush=True)
    (OUT / "ours.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")


def cmd_report(iris_results: Path):
    import csv
    rows = json.loads((OUT / "ours.json").read_text(encoding="utf-8"))
    for r in rows:   # IRIS FlashResult: Pass=0, PassWithWarning=1, ExtendedFail=2, FlashFail=3 (TotalFlashIncidents.h)
        with open(iris_results / f"{r['name']}.avi" / "framedata.csv", newline="") as f:
            res = [int(x["LuminanceFrameResult"]) for x in csv.DictReader(f)]
        ext = [i for i, v in enumerate(res) if v == 2]
        r["iris_flash"] = any(v == 3 for v in res)
        r["iris_ext"] = bool(ext)
        r["iris_ext_first_s"] = ext[0] / FPS if ext else None
    yn = lambda b: "위험" if b else "·"
    t = lambda x: f"{x:.2f}" if x is not None else ""
    lines = ["# 지속(연장) 플래시: 우리 판정 대 EA IRIS (2026-11-12)", "",
             "> 시험 영상: 화면 전체 회색 사각파 40 ↔ 100 cd/m², 320×180, 30 fps, 10초, 0.5초부터 깜빡임 · "
             "`scripts/validate_extended_iris.py` · IRIS d96978a 빌드(`tools/iris`), 기본 설정",
             "> 우리 iris 모드 = 1초 전이 4~6(2~3 Hz) 프레임이 5초 창의 80% 이상. 1~3Hz 변형 = 전이 2~6. continuous = 2~3 Hz 깜빡임이 5초 '초과' 연속.", "",
             "| 빈도(Hz) | 지속(초) | 일반 플래시: 우리 / IRIS | 지속: 우리(iris) / IRIS | 처음 실패 시각(초): 우리 / IRIS | 1~3 Hz 변형 | 5초 초과 연속 |",
             "|---|---|---|---|---|---|---|"]
    agree_f = agree_e = 0
    for r in rows:
        agree_f += r["flash_dangerous"] == r["iris_flash"]
        agree_e += r["ext_iris"] == r["iris_ext"]
        lines.append(f"| {r['hz']:g} | {r['dur_s']:g} | {yn(r['flash_dangerous'])} / {yn(r['iris_flash'])} | "
                     f"{yn(r['ext_iris'])} / {yn(r['iris_ext'])} | {t(r['ext_iris_first_s'])} / {t(r['iris_ext_first_s'])} | "
                     f"{yn(r['ext_iris_1to3hz'])} | {yn(r['ext_continuous'])} |")
    lines += ["", f"**일치**: 일반 플래시 {agree_f}/{len(rows)}, 지속 플래시 {agree_e}/{len(rows)}"]
    out = ROOT / "checks" / "extended" / "README.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT / "compare.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["make", "report"])
    ap.add_argument("--iris-results", type=Path, default=ROOT.parent / "tools" / "iris" / "Results")
    a = ap.parse_args()
    cmd_make() if a.cmd == "make" else cmd_report(a.iris_results)
