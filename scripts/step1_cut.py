"""1단계: 원본 영상에서 깜빡임 없는 5초를 잘라 무손실로 저장하고, 확인용 그림과 표를 만든다.

하는 일
1. 원본에서 60초부터 10초 간격으로 5초 후보 구간을 훑는다.
2. 후보마다 화면 평균 밝기를 재서, 깜빡임이 0번이고 너무 어둡지 않은(평균 휘도 0.1 이상) 첫 구간을 고른다.
3. 고른 구간을 FFV1(.mkv, 무손실)로 저장한다. 세로 359px은 홀수라 맨 아래 1줄을 잘라 640×358로 맞춘다.
4. 저장한 파일을 다시 읽어서 확인한다: 프레임 수, 원본 프레임과 픽셀 일치, 밝기, 깜빡임 수.
5. checks/step1/에 contact_sheet.png, luma.png, README.md를 만든다.

사용: .venv\\Scripts\\python scripts\\step1_cut.py
"""
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from lumi_synth.measure import (  # noqa: E402
    DELTA, MAX_FLASHES_PER_SEC, count_transitions, max_flashes_per_second, relative_luminance_bgr,
)

ROOT = Path(__file__).resolve().parents[1]

SRC = ROOT / "data/sources/big_buck_bunny/BigBuckBunny_640x360.m4v"
OUT_VIDEO = ROOT / "data/clips/bbb_clip01.mkv"
CHECKS = ROOT / "checks/step1"
SECONDS = 5
SCAN_FROM, SCAN_STEP, SCAN_TO = 60, 10, 500
MIN_MEAN_LUMA = 0.1


def read_frames(path: Path, start: int, count: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    frames = []
    for _ in range(count):
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    return np.stack(frames)


def crop_even(frames: np.ndarray) -> np.ndarray:
    h, w = frames.shape[1:3]
    return frames[:, : h - h % 2, : w - w % 2]


def write_ffv1(path: Path, frames: np.ndarray, fps: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames.shape[1:3]
    wr = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"FFV1"), fps, (w, h))
    for f in frames:
        wr.write(f)
    wr.release()


def main() -> int:
    cap = cv2.VideoCapture(str(SRC))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    n = int(round(fps * SECONDS))

    # 1~2. 후보 구간 훑기
    scanned = []
    chosen = None
    for sec in range(SCAN_FROM, SCAN_TO, SCAN_STEP):
        start = int(round(sec * fps))
        frames = crop_even(read_frames(SRC, start, n))
        luma = relative_luminance_bgr(frames).mean(axis=(1, 2))
        flashes = max_flashes_per_second(luma, fps)
        scanned.append((sec, flashes, float(luma.mean())))
        if flashes == 0 and luma.mean() >= MIN_MEAN_LUMA:
            chosen = (sec, start, frames)
            break
    if chosen is None:
        print("조건에 맞는 구간을 찾지 못했습니다.")
        return 1
    sec, start, frames = chosen

    # 3. 저장
    write_ffv1(OUT_VIDEO, frames, fps)

    # 4. 저장한 파일을 다시 읽어서 확인
    back = read_frames(OUT_VIDEO, 0, n + 10)
    same_count = len(back) == n
    max_diff = int(np.abs(back[:n].astype(int) - frames.astype(int)).max()) if same_count else -1
    luma = relative_luminance_bgr(back).mean(axis=(1, 2))
    transitions = count_transitions(luma)
    flashes = max_flashes_per_second(luma, fps)
    steps = np.abs(np.diff(luma))

    # 5-1. 프레임 12장 모음
    CHECKS.mkdir(parents=True, exist_ok=True)
    idx = np.linspace(0, len(back) - 1, 12).astype(int)
    fig, axes = plt.subplots(3, 4, figsize=(12, 5.6))
    for ax, i in zip(axes.flat, idx):
        ax.imshow(cv2.cvtColor(back[i], cv2.COLOR_BGR2RGB))
        ax.set_title(f"{i}번 프레임 ({i / fps:.2f}초)", fontsize=9)
        ax.axis("off")
    fig.suptitle(f"1단계 클립: Big Buck Bunny {sec}초부터 {SECONDS}초 ({len(back)}프레임)")
    fig.tight_layout()
    fig.savefig(CHECKS / "contact_sheet.png", dpi=90)
    plt.close(fig)

    # 5-2. 밝기 그래프
    t = np.arange(len(luma)) / fps
    fig, ax = plt.subplots(figsize=(10, 3.8))
    ax.plot(t, luma, color="tab:blue", label="화면 평균 밝기(상대 휘도)")
    ax.fill_between(t, luma - DELTA / 2, luma + DELTA / 2, color="tab:blue", alpha=0.08,
                    label=f"±{DELTA / 2:.2f} 띠 (이 띠 폭 = 깜빡임으로 세는 최소 변화 {DELTA})")
    for e in transitions:
        ax.axvline(e / fps, color="red", lw=1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("시간(초)")
    ax.set_ylabel("밝기 (0=검정, 1=흰색)")
    ax.set_title(f"전이 {len(transitions)}번, 1초 최대 깜빡임 {flashes:g}번 (위험 기준: {MAX_FLASHES_PER_SEC}번 초과)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(CHECKS / "luma.png", dpi=90)
    plt.close(fig)

    # 5-3. 점검표
    checks = [
        ("저장 후 프레임 수", f"{len(back)} (기대 {n})", same_count),
        ("원본과 픽셀 차이", f"최대 {max_diff} (0이면 무손실)", max_diff == 0),
        ("1초 최대 깜빡임", f"{flashes:g}번 (기준 {MAX_FLASHES_PER_SEC}번 이하, 원본이므로 0이어야 함)", flashes == 0),
        ("평균 밝기", f"{luma.mean():.3f} (너무 어둡지 않게 {MIN_MEAN_LUMA} 이상)", luma.mean() >= MIN_MEAN_LUMA),
    ]
    all_ok = all(c[2] for c in checks)
    h, w = back.shape[1:3]
    lines = [
        "# 1단계 점검표: 원본 5초 클립",
        "",
        f"**결과: {'✅ 통과' if all_ok else '❌ 실패'}** ({sum(c[2] for c in checks)}/{len(checks)} 항목 통과)",
        "",
        "## 만든 것",
        "",
        f"- 클립: `data/clips/bbb_clip01.mkv` (Git에 안 올림). Big Buck Bunny **{sec}초~{sec + SECONDS}초**, "
        f"{w}×{h}, {fps:g} fps, {len(back)}프레임, FFV1 무손실",
        "- 원본 세로가 359px(홀수)이라 맨 아래 1줄을 잘라 358px로 맞췄습니다.",
        "",
        "## 점검 (저장한 파일을 다시 읽어서 측정)",
        "",
        "| 항목 | 결과 | 통과 |",
        "|---|---|---|",
        *[f"| {a} | {b} | {'✅' if c else '❌'} |" for a, b, c in checks],
        "",
        "## 직접 확인할 것",
        "",
        "1. **[contact_sheet.png](contact_sheet.png)**: 12장이 같은 장면이 자연스럽게 이어지는 그림인지",
        "2. **[luma.png](luma.png)**: 파란 선이 완만하고, 빨간 세로선(전이)이 없는지",
        "",
        "![프레임 모음](contact_sheet.png)",
        "",
        "![밝기 그래프](luma.png)",
        "",
        "## 참고: 훑어본 후보 구간",
        "",
        "| 시작(초) | 1초 최대 깜빡임 | 평균 밝기 | 선택 |",
        "|---|---|---|---|",
        *[f"| {s} | {f:g} | {m:.3f} | {'✅' if s == sec else ''} |" for s, f, m in scanned],
        "",
        "만든 명령: `.venv\\Scripts\\python scripts\\step1_cut.py`",
        "",
    ]
    (CHECKS / "README.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    return 0 if all_ok else 1


if __name__ == "__main__":
    plt.rcParams["font.family"] = ["Malgun Gothic", "sans-serif"]
    plt.rcParams["axes.unicode_minus"] = False
    raise SystemExit(main())
