"""시연용 위험 영상(make_demo_hazards.py)을 우리 판정기와 팀원 판별기로 검사해 비교표를 만든다.

- 우리: lumi_synth.measure — 휘도 깜빡임(analyze, criteria_report), 적색 깜빡임(analyze_red), 줄무늬(analyze_pattern).
- 팀원: lumi-lab-rockrovks Fullscreen/A1/vdcoreops_v2.py 의 evaluate_video_pse_risk 를 **수정 없이** 불러서 실행
        (기본 설정 그대로: laptop 56 cm, 15인치, 긴 변 720으로 축소, 334 ms).
- 결과: checks/demo_hazards/README.md, team_results.csv, videos.png(영상별 정지 프레임 한 장 + 영역 표시)

사용: python scripts\\eval_demo_hazards.py --team-dir <lumi-lab-rockrovks 저장소 경로>
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from lumi_synth import measure  # noqa: E402
from lumi_synth.data import iter_frames_bgr  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "data/hazard_demo"
CHECKS = ROOT / "checks/demo_hazards"


def read_all(path: Path) -> np.ndarray:
    # 디코드는 영상의 색 행렬·range 표시대로 (2026-11-10, data.iter_frames_bgr)
    return np.stack(list(iter_frames_bgr(path)))


def ours(frames: np.ndarray, fps: float) -> dict:
    t0 = time.time()
    flash = measure.analyze(frames, fps)
    report = measure.criteria_report(frames, fps)
    pat = measure.analyze_pattern(frames, fps)
    red = measure.analyze_red(frames, fps)
    labels = [m for m in pat["motion"] if m]
    motion = max(set(labels), key=labels.count) if labels else "-"
    return {
        "휘도": f"{'위험' if flash['dangerous'] else '안전'} ({flash['max_flashes']:g}번/초, 면적 {flash['peak_area']:.0%})",
        "휘도_무너진조건": ", ".join(n for n, r in zip(["밝기", "횟수", "면적"], report[:3]) if not r["넘음"]) or "없음",
        "줄무늬": f"{'위험' if pat['dangerous'] else '안전'} (면적 {pat['peak_stripe_area']:.0%}, {motion})",
        "적색": f"{'위험' if red['dangerous'] else '안전'} ({red['max_flashes']:g}번/초, 면적 {red['peak_area']:.0%}, Δu′v′ {red['max_delta_uv']:.3f})",
        "dangerous": bool(flash["dangerous"] or pat["dangerous"] or red["dangerous"]),
        "초": time.time() - t0,
    }


def team(evaluate, path: Path) -> tuple[str, list[dict], float]:
    t0 = time.time()
    df = evaluate(video_path=str(path), device="laptop", display_area=15)
    rows = df.to_dict("records")
    if not rows:
        return "위험 구간 없음", rows, time.time() - t0
    text = "; ".join(f"{r['위험 구분']} {r['위험 구간 (시간)']} ({r['판정 근거 - 임계치 달성 정도']})" for r in rows)
    return text, rows, time.time() - t0


def contact_sheet(manifest: dict) -> None:
    vids = manifest["videos"]
    fig, axes = plt.subplots(3, 3, figsize=(10.5, 6.4))
    for ax, v in zip(axes.flat, vids):
        cap = cv2.VideoCapture(str(DEMO / v["file"]))
        cap.set(cv2.CAP_PROP_POS_FRAMES, 4)                  # 깜빡임이 '켜진' 프레임
        ok, f = cap.read()
        cap.release()
        img = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        mpath = DEMO / f"{v['name']}_mask.png"
        if mpath.exists():
            m = cv2.imread(str(mpath), cv2.IMREAD_GRAYSCALE)
            cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            img = cv2.drawContours(img.copy(), cnts, -1, (255, 0, 0), 4)
        ax.imshow(img)
        ax.set_title(v["name"], fontsize=8)
        ax.axis("off")
    fig.suptitle("시연 영상 9개: 정지 프레임 한 장씩 (빨간 테두리 = 합성 영역). 영상은 재생하지 마세요.", fontsize=10)
    fig.tight_layout()
    fig.savefig(CHECKS / "videos.png", dpi=70)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--team-dir", required=True, help="lumi-lab-rockrovks 저장소 경로 (읽기만 함)")
    args = ap.parse_args()
    team_dir = Path(args.team_dir) / "Fullscreen" / "A1"
    sys.path.insert(0, str(team_dir))
    import vdcoreops_v2 as team_mod  # noqa: E402  (팀원 코드, 수정하지 않음)
    import subprocess
    team_commit = subprocess.run(["git", "-C", str(args.team_dir), "rev-parse", "--short", "HEAD"],
                                 capture_output=True, text=True).stdout.strip()

    manifest = json.loads((DEMO / "manifest.json").read_text(encoding="utf-8"))
    fps = manifest["fps"]
    CHECKS.mkdir(parents=True, exist_ok=True)
    contact_sheet(manifest)

    rows, team_rows = [], []
    for v in manifest["videos"]:
        path = DEMO / v["file"]
        frames = read_all(path)
        o = ours(frames, fps)
        t_text, t_rows, t_sec = team(team_mod.evaluate_video_pse_risk, path)
        for r in t_rows:
            team_rows.append({"영상": v["name"], **r})
        rows.append((v, o, t_text, t_sec))
        print(v["name"], "| 우리:", o["휘도"], o["줄무늬"], "| 팀원:", t_text, flush=True)

    import pandas as pd
    pd.DataFrame(team_rows).to_csv(CHECKS / "team_results.csv", index=False, encoding="utf-8-sig")

    lines = [
        "# 시연 영상 9개: 우리 판정기 대 팀원 판별기 (만들기 전 / 후)",
        "",
        f"> 만든 날 {time.strftime('%Y-%m-%d')} · 영상: `scripts/make_demo_hazards.py` (Git 제외, `data/hazard_demo/`) · 평가: `scripts/eval_demo_hazards.py`",
        f"> 바탕: {manifest['source']}, {manifest['width']}×{manifest['height']}, {fps:g} fps, {manifest['frames']}프레임(2초), 1도당 {manifest['ppd']:.1f} px",
        f"> 팀원 판별기: lumi-lab-rockrovks `{team_commit}` `Fullscreen/A1/vdcoreops_v2.py` `evaluate_video_pse_risk` (수정 없이 기본 설정: laptop 56 cm, 15인치, 긴 변 720, 334 ms)",
        "> ⚠️ 영상은 광과민성 위험 자극이라 재생하지 않았습니다. 아래 그림은 정지 프레임 한 장씩입니다.",
        "",
        "![시연 영상](videos.png)",
        "",
        "| 영상 | 합성 내용 | 설계상 기대 | 우리: 휘도 | 우리: 휘도 무너진 조건 | 우리: 줄무늬 | 우리: 적색 | 팀원 판별기 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for v, o, t_text, _ in rows:
        lines.append(f"| `{v['name']}` | {v['설명']} | {v['설계상 기대']} | {o['휘도']} | {o['휘도_무너진조건']} | {o['줄무늬']} | {o['적색']} | {t_text} |")
    lines += [
        "",
        "팀원 결과 원본: [team_results.csv](team_results.csv)",
        "",
        "| 영상 | 우리 판정 시간(초) | 팀원 판정 시간(초) |",
        "|---|---|---|",
        *[f"| `{v['name']}` | {o['초']:.1f} | {t:.1f} |" for v, o, _, t in rows],
        "",
    ]
    (CHECKS / "README.md").write_text("\n".join(lines), encoding="utf-8")
    return 0


if __name__ == "__main__":
    plt.rcParams["font.family"] = ["Malgun Gothic", "sans-serif"]
    plt.rcParams["axes.unicode_minus"] = False
    raise SystemExit(main())
