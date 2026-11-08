"""1단계: 원본 영상에서 깜빡임 없는 5초를 잘라 무손실로 저장하고, 확인용 그림과 표를 만든다.

하는 일
1. 원본에서 60초부터 10초 간격으로 5초 후보 구간을 훑는다.
2. 후보마다 ITU 방식(lumi_synth.measure.analyze)으로 재서, 깜빡임이 0번이고
   너무 어둡지 않은(평균 20 cd/m² 이상) 첫 구간을 고른다.
3. 고른 구간을 FFV1(.mkv, 무손실)로 저장한다. 세로 359px은 홀수라 맨 아래 1줄을 잘라 640×358로 맞춘다.
4. 저장한 파일을 다시 읽어서 확인한다: 프레임 수, 원본 프레임과 픽셀 일치, 밝기, 깜빡임 수.
5. checks/step1/에 contact_sheet.png, luma.png, README.md를 만든다.

사용: python scripts\\step1_cut.py
"""
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from lumi_synth.measure import AREA_LIMIT, MAX_FLASHES_PER_SEC, analyze  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data/sources/big_buck_bunny/BigBuckBunny_640x360.m4v"
OUT_VIDEO = ROOT / "data/clips/bbb_clip01.mkv"
CHECKS = ROOT / "checks/step1"
SECONDS = 5
SCAN_FROM, SCAN_STEP, SCAN_TO = 60, 10, 500
MIN_MEAN_CD = 20.0


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
        r = analyze(frames, fps)
        mean_cd = float(r["mean_cd"].mean())
        scanned.append((sec, r["max_flashes"], r["peak_area"], mean_cd))
        if r["max_flashes"] == 0 and mean_cd >= MIN_MEAN_CD:
            chosen = (sec, frames)
            break
    if chosen is None:
        print("조건에 맞는 구간을 찾지 못했습니다.")
        return 1
    sec, frames = chosen

    # 3. 저장
    write_ffv1(OUT_VIDEO, frames, fps)

    # 4. 저장한 파일을 다시 읽어서 확인
    back = read_frames(OUT_VIDEO, 0, n + 10)
    same_count = len(back) == n
    max_diff = int(np.abs(back[:n].astype(int) - frames.astype(int)).max()) if same_count else -1
    r = analyze(back, fps)
    mean_cd = float(r["mean_cd"].mean())

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

    # 5-2. 밝기·전이 면적 그래프
    t = np.arange(len(back)) / fps
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5.6), sharex=True)
    ax1.plot(t, r["mean_cd"], color="tab:blue")
    ax1.set_ylim(0, 200)
    ax1.set_ylabel("밝기 (cd/m²)")
    ax1.set_title("위: 화면 평균 밝기 / 아래: 프레임마다 밝기가 20 cd/m² 이상 바뀐 픽셀 면적")
    ax1.grid(alpha=0.3)
    ax2.plot(t, r["area"] * 100, color="tab:orange", label="바뀐 픽셀 면적")
    ax2.axhline(AREA_LIMIT * 100, color="red", ls="--", label=f"기준 {AREA_LIMIT:.0%} (넘으면 화면 전이)")
    for e in r["events"]:
        ax2.axvline(e / fps, color="red", lw=1)
    ax2.set_ylim(0, 100)
    ax2.set_ylabel("화면 대비 %")
    ax2.set_xlabel("시간(초)")
    ax2.legend(loc="upper right", fontsize=8)
    ax2.grid(alpha=0.3)
    fig.suptitle(f"화면 전이 {len(r['events'])}번, 1초 최대 깜빡임 {r['max_flashes']:g}번 "
                 f"(위험 기준: {MAX_FLASHES_PER_SEC}번 초과)")
    fig.tight_layout()
    fig.savefig(CHECKS / "luma.png", dpi=90)
    plt.close(fig)

    # 5-3. 점검표
    checks = [
        ("저장 후 프레임 수", f"{len(back)} (기대 {n})", same_count),
        ("원본과 픽셀 차이", f"최대 {max_diff} (0이면 무손실)", max_diff == 0),
        ("1초 최대 깜빡임", f"{r['max_flashes']:g}번 (원본이므로 0이어야 함)", r["max_flashes"] == 0),
        ("가장 넓은 전이 면적", f"{r['peak_area']:.1%} (기준 {AREA_LIMIT:.0%} 이하)", r["peak_area"] <= AREA_LIMIT),
        ("평균 밝기", f"{mean_cd:.1f} cd/m² (너무 어둡지 않게 {MIN_MEAN_CD:g} 이상)", mean_cd >= MIN_MEAN_CD),
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
        f"{w}×{h}, {fps:.2f} fps, {len(back)}프레임, FFV1 무손실",
        "- 원본 세로가 359px(홀수)이라 맨 아래 1줄을 잘라 358px로 맞췄습니다.",
        "",
        "## 점검 (저장한 파일을 다시 읽어서 ITU-R BT.1702 방식으로 측정)",
        "",
        "| 항목 | 결과 | 통과 |",
        "|---|---|---|",
        *[f"| {a} | {b} | {'✅' if c else '❌'} |" for a, b, c in checks],
        "",
        "측정 기준: 픽셀 밝기가 20 cd/m² 이상 바뀌고(어두운 쪽 160 미만) 그런 픽셀이 화면의 25%를 넘으면 화면 전이 1번, "
        "전이 2번 = 깜빡임 1번, 1초에 3번 넘으면 위험. 자세한 설명은 `src/lumi_synth/measure.py` 맨 위.",
        "",
        "## 직접 확인할 것",
        "",
        "1. **[contact_sheet.png](contact_sheet.png)**: 12장이 같은 장면이 자연스럽게 이어지는 그림인지",
        "2. **[luma.png](luma.png)**: 위 그래프가 완만하고, 아래 주황 선이 빨간 점선(25%) 아래에 있는지",
        "",
        "![프레임 모음](contact_sheet.png)",
        "",
        "![밝기 그래프](luma.png)",
        "",
        "## 참고: 훑어본 후보 구간",
        "",
        "| 시작(초) | 1초 최대 깜빡임 | 가장 넓은 전이 면적 | 평균 밝기(cd/m²) | 선택 |",
        "|---|---|---|---|---|",
        *[f"| {s} | {f:g} | {a:.1%} | {m:.1f} | {'✅' if s == sec else ''} |" for s, f, a, m in scanned],
        "",
        "만든 명령: `python scripts\\step1_cut.py`",
        "",
    ]
    (CHECKS / "README.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    return 0 if all_ok else 1


if __name__ == "__main__":
    plt.rcParams["font.family"] = ["Malgun Gothic", "sans-serif"]
    plt.rcParams["axes.unicode_minus"] = False
    raise SystemExit(main())
