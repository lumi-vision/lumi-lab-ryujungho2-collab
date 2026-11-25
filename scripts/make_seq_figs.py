"""시퀀스 탐지 비교 — 팀원 판(temporal.py) vs G(grid_flash = measure.py 규칙), 같은 합성 시험 영상.

두 엔진의 공통점: 픽셀 단위, 밝기 차 ≥ 20 cd/m²·어두운 < 160, 방향별 면적 > 25 %, 동시성 20 ms, 1 초 창.
차이(코드 근거):
  팀원 판  현재 프레임을 '66.8 ms 이내 과거 프레임'과 비교해 차이 ≥ 20 이면 전이 (PSEConfig.max_transition_ms = 66.8,
           근거 Jordan & Vanderheiden 2024 'qualifying duration 66 ms' 제안값)
  G        픽셀마다 '직전 극값'에서 반대 방향으로 ≥ 20 움직이면 전이(히스테리시스), 시간 제한 없음
           (ITU-R BT.1702-3 원문 "a pair of opposing changes in luminance of 20 cd/m² or more")
정답(ITU 원문 기준, 실행 전 고정): 아래 CASES 의 truth.
출력: seq_table.png · seq_trace.png · seq_numbers.json
⚠️ 깜빡이는 합성 영상 — 메모리에서만, 저장·재생 안 함.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from lumi_judge.cpu_limit import limit_cpu  # noqa: E402
limit_cpu()
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import lumi_judge as J  # noqa: E402
from lumi_judge.config import FPSWindowSpec, PSEConfig  # noqa: E402
from lumi_judge.grid_flash import judge_luminance, luminance_cd, pixel_transitions, window_counts  # noqa: E402
from lumi_judge.grid_judge import judge_video  # noqa: E402
from lumi_judge.photometry import PhotometricConverter  # noqa: E402
from lumi_judge.temporal import TemporalTransitionEngine  # noqa: E402

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data" / "before_after"
plt.rcParams.update({"font.family": ["Malgun Gothic", "DejaVu Sans"], "axes.unicode_minus": False, "font.size": 11})
C_V1, C_G, C_OK, C_BAD = "#C25E00", "#2F6FD6", "#1B9E5A", "#C0392B"
H, W, FPS, SEC, BG = 432, 768, 30.0, 2.5, 40.0


def to_u8(L):
    y = np.clip(L / 200.0, 0, 1)
    s = np.where(y <= 0.0031308, 12.92 * y, 1.055 * np.power(y, 1 / 2.4) - 0.055)
    return np.repeat(np.round(s * 255).astype(np.uint8)[..., None], 3, -1)


def region(area):
    yy, xx = np.mgrid[:H, :W]
    rw = math.sqrt(area * W * H * (W / H)); rh = area * W * H / rw
    return (np.abs(xx - W / 2) <= rw / 2) & (np.abs(yy - H / 2) <= rh / 2)


def wave(kind, f, lo, hi, ramp_s=0.0):
    t = np.arange(int(FPS * SEC)) / FPS
    if kind == "square":
        q = (np.floor(t * 2 * f) % 2 == 0).astype(float)
    elif kind == "sine":
        q = 0.5 + 0.5 * np.sin(2 * math.pi * f * t)
    elif kind == "trap":                                               # 사다리꼴: 한 번 변화에 ramp_s 초
        ph = (t * f) % 1.0; half = 0.5; r = ramp_s * f
        q = np.where(ph < r, ph / r, np.where(ph < half, 1.0, np.where(ph < half + r, 1 - (ph - half) / r, 0.0)))
    return lo + (hi - lo) * q


def clip_flash(kind, f, lo, hi, area=0.30, ramp_s=0.0):
    reg = region(area); lv = wave(kind, f, lo, hi, ramp_s)
    return np.stack([to_u8(np.where(reg, v, BG)) for v in lv]), lv


CASES = [  # (이름, truth(ITU 원문), builder)
    ("사각 깜빡임 4 Hz · ΔL 100 · 30 %", True, lambda: clip_flash("square", 4.0, 20, 120)),
    ("부드러운 맥동 3.5 Hz · 최고−최저 22", True, lambda: clip_flash("sine", 3.5, 49, 71)),
    ("부드러운 맥동 3.5 Hz · 최고−최저 60", True, lambda: clip_flash("sine", 3.5, 30, 90)),
    ("천천히 오르내림 3.5 Hz · 한 번 변화 120 ms · ΔL 100", True, lambda: clip_flash("trap", 3.5, 20, 120, ramp_s=0.12)),
    ("사각 3 Hz (정확히 3회 — 경계)", False, lambda: clip_flash("square", 3.0, 20, 120)),
    ("사각 4 Hz · 면적 24 %", False, lambda: clip_flash("square", 4.0, 20, 120, area=0.24)),
]


def teammate(frames, fps):
    r = J.analyze_frames(list(frames), fps, device="cuda", mask_config=J.PartialMaskConfig())
    return list(r.active_labels)


def teammate_events(frames, fps):
    pse = PSEConfig(); spec = FPSWindowSpec.from_fps(fps, pse)
    conv = PhotometricConverter(l_max_cd_m2=200.0, pse=pse, device="cuda")
    eng = TemporalTransitionEngine(spec, pse)
    for i, fr in enumerate(frames):
        _, lum, u, v, red = conv.convert_bgr(fr)
        eng.update(i, i * 1000.0 / fps, lum, u, v, red)
    return [(e.frame_idx, e.polarity) for e in eng.lum_flash_state.events]


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    rows = []
    traces = {}
    for name, truth, build in CASES:
        x, lv = build()
        tl = teammate(x, FPS)
        t_flash = "휘도 플래시" in tl
        jl = judge_luminance(x, FPS)
        g_flash = bool(jl["dangerous"])
        ev_t = teammate_events(x, FPS)
        cnt_g = window_counts(jl["transitions"], FPS)
        rows.append({"case": name, "truth": truth, "teammate": t_flash, "teammate_labels": tl, "G": g_flash,
                     "G_max_flashes": float(jl["max_flashes"]), "teammate_events": len(ev_t), "G_transitions": int((jl["transitions"] != 0).sum())})
        traces[name] = (lv, jl["transitions"], ev_t, cnt_g)
        print(rows[-1], flush=True)
        torch.cuda.empty_cache()
    # 대비 반전 줄무늬(24 fps, P2 비교) — 팀원 판 라벨과 G 전체 판정
    import pattern_motion_conformance as PM
    xr, _ = PM.clip(motion="reverse", area=0.30, freq=2.0)
    tl = teammate(xr, PM.FPS); gv = judge_video(xr, PM.FPS)
    rows.append({"case": "대비 반전 줄무늬 30 % · 초당 4번 반전 (전체 판정)", "truth": True, "teammate": bool(tl), "teammate_labels": tl,
                 "G": bool(gv["dangerous"]), "G_labels": gv["labels"],
                 "F1_teammate": "휘도 플래시" in tl, "F1_G": bool(judge_luminance(xr, PM.FPS)["dangerous"])})
    print(rows[-1], flush=True)
    (OUT / "seq_numbers.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")

    # ---------------- 그림 1: 표
    fig, ax = plt.subplots(figsize=(17, 4.6)); ax.axis("off")
    yn = lambda b: "위험" if b else "안전"
    cells, colors = [], []
    for r in rows:
        tcol = C_OK if r["teammate"] == r["truth"] else C_BAD
        gcol = C_OK if r["G"] == r["truth"] else C_BAD
        note = ""
        if r["case"].startswith("부드러운 맥동 3.5 Hz · 최고−최저 22"):
            note = "66 ms 안 변화 ≈ 15 < 20 → 팀원 판 전이 0 · JV24 제안(66 ms) 기준이면 안전 = 해석 차이"
        if r["case"].startswith("대비 반전"):
            note = "깜빡임 규칙은 둘 다 안전(방향별 15 %) · 팀원 판 '정적' 패턴(이유 틀림) · G 'P2 변화'"
        cells.append([r["case"], yn(r["truth"]), yn(r["teammate"]) + ("" if r["teammate"] == r["truth"] else " ✗"),
                      yn(r["G"]) + ("" if r["G"] == r["truth"] else " ✗"), note])
        colors.append(["white", "#F2F2F2", tcol + "33", gcol + "33", "white"])
    tb = ax.table(cellText=cells, cellColours=colors, colLabels=["시험 영상 (화면 30 %, 30 fps)", "정답 (ITU 원문)", "팀원 판 '휘도 플래시'", "G 휘도 깜빡임(F1)", "비고"],
                  colWidths=[0.27, 0.08, 0.11, 0.11, 0.43], loc="center", cellLoc="left")
    tb.auto_set_font_size(False); tb.set_fontsize(10.5); tb.scale(1, 1.75)
    ax.set_title("시퀀스 탐지 — 같은 시험 영상, 두 엔진 (정답은 실행 전 고정)", fontsize=14, pad=12)
    fig.tight_layout(); fig.savefig(OUT / "seq_table.png", dpi=130); plt.close(fig)

    # ---------------- 그림 2: 차이가 나는 사례의 시간축
    pick = [CASES[1][0], CASES[2][0], CASES[0][0]]
    fig, axs = plt.subplots(len(pick), 1, figsize=(15, 3.0 * len(pick)), sharex=True)
    for ax, name in zip(axs, pick):
        lv, e, ev, cnt = traces[name]
        t = np.arange(len(lv)) / FPS
        ax.plot(t, lv, color="#444", lw=1.6, label="깜빡이는 영역 한 픽셀의 밝기 (cd/m²)")
        for i in np.flatnonzero(e):
            ax.axvline(i / FPS, color=C_G, lw=2.2, alpha=0.75, ymin=0.5, ymax=1.0)
        for fi, pol in ev:
            ax.axvline(fi / FPS, color=C_V1, lw=2.2, alpha=0.85, ymin=0.0, ymax=0.5)
        r = next(q for q in rows if q["case"] == name)
        ax.set_title(f"{name} — G 전이 {r['G_transitions']}회(위 파랑) → {'위험' if r['G'] else '안전'} · 팀원 판 전이 {r['teammate_events']}회(아래 주황) → {'위험' if r['teammate'] else '안전'}", fontsize=11.5)
        ax.set_ylabel("cd/m²")
    axs[-1].set_xlabel("초")
    fig.suptitle("파랑 = G 가 센 전이(직전 극값에서 20 이상 반대로) · 주황 = 팀원 판이 센 전이(66.8 ms 안에 20 이상 변함)", fontsize=13)
    fig.tight_layout(); fig.savefig(OUT / "seq_trace.png", dpi=120); plt.close(fig)


if __name__ == "__main__":
    main()
