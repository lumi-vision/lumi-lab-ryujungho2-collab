"""South 2021 라벨의 정체 확인 — PhotosensitivityPal(South, Saffo, Borkin, CHI '21 §4.2.1–4.2.2)의 깜빡임·적색 규칙 재구현 → CSV 라벨과 일치도.

원문(§6.2): "Of the 2000 GIFs tested ..., 163 were labelled dangerous by PhotosensitivityPal" — 라벨 = 저자 검출기 출력(사람 라벨 아님).
원문 규칙(§4.2.1, WCAG 2.0 2.3.1):
  - 픽셀 상대휘도 Y(XYZ). 인접 두 프레임에서 Dark = (Y1 < 0.8) OR (Y2 < 0.8), BrightDiff = |Y1 − Y2| > 0.1 × (두 프레임 최대 Y)
  - Dark AND BrightDiff 픽셀이 전체의 25 % 초과 → 밝→어/어→밝 '전환' 한 번
  - 같은 1 초 구간(**끊어진** 1 초 블록) 안의 반대 방향 전환 두 개 = 깜빡임 1 회. 1 초에 > 3 회면 위험
  - 1 초보다 짧은 GIF: 허용 횟수를 비례(0.5 s → 1 회, 0.7 s → 2 회 = floor(3 × 길이))
적색(§4.2.2): RedRatio = R/(R+G+B) ≥ 0.8 (어느 한 프레임) AND ΔPureRed > 20, PureRed = max(R−G−B, 0) × 320 (원문 그대로; RGB 0–1 가정),
  25 % 초과 → 전환, 1 초 블록 안 반대 전환 둘 = 적색 깜빡임, **반복 재생(마지막→처음 프레임) 포함**.
구현 선택(원문에 없는 부분 — 변형으로 둘 다 잼): 전환 방향 = 깜빡임 픽셀의 평균 ΔY 부호 · 깜빡임 수 = 방향이 바뀐 전환 쌍 수 ·
  깜빡임 반복 재생 포함 여부(원문은 적색에만 명시) · RGB 를 감마 그대로(0–1) 쓰는지.
출력: checks/pspal_reimpl.md (CSV 라벨과 일치표·κ)
"""
from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from lumi_judge.cpu_limit import limit_cpu  # noqa: E402
limit_cpu()
from eval_real_hazards import decode as dec_any  # noqa: E402


def rel_lum(x):
    c = x[..., ::-1].astype(np.float32) / 255.0                     # BGR → RGB
    lin = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    return 0.2126 * lin[..., 0] + 0.7152 * lin[..., 1] + 0.0722 * lin[..., 2]


def shifts_flash(Y):
    """인접 프레임 쌍마다 전환 방향(+1 밝아짐, −1 어두워짐, 0 없음)."""
    out = np.zeros(len(Y) - 1, np.int8)
    for t in range(len(Y) - 1):
        a, b = Y[t], Y[t + 1]
        dark = (a < 0.8) | (b < 0.8)
        diff = np.abs(b - a) > 0.1 * max(float(a.max()), float(b.max()))
        m = dark & diff
        if m.mean() > 0.25:
            out[t] = 1 if float((b - a)[m].mean()) > 0 else -1
    return out


def shifts_red(x):
    c = x[..., ::-1].astype(np.float32) / 255.0
    s = c.sum(-1)
    rr = np.where(s > 0, c[..., 0] / np.maximum(s, 1e-6), 0.0) >= 0.8
    pr = np.maximum(c[..., 0] - c[..., 1] - c[..., 2], 0) * 320.0
    out = np.zeros(len(x) - 1, np.int8)
    for t in range(len(x) - 1):
        m = (rr[t] | rr[t + 1]) & (np.abs(pr[t + 1] - pr[t]) > 20)
        if m.mean() > 0.25:
            out[t] = 1 if float((pr[t + 1] - pr[t])[m].mean()) > 0 else -1
    return out


def count_blocks(sh, fps, dur):
    """끊어진 1 초 블록마다 깜빡임 수(반대 방향 전환 쌍) — 최대값과 허용치."""
    n = max(1, int(round(fps)))
    best = 0
    for s0 in range(0, len(sh), n):
        seq = [v for v in sh[s0:s0 + n] if v != 0]
        flips, last, pend = 0, 0, 0
        for v in seq:                                               # 방향이 바뀐 전환을 짝지어 센다
            if last == 0 or v != last:
                pend += 1
                if pend == 2:
                    flips += 1; pend = 0
            last = v
        best = max(best, flips)
    allowed = 3 if dur >= 1.0 else int(math.floor(3 * dur + 1e-9))
    return best, allowed


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    src = ROOT / "data/real/south2021"
    lab = {r["label"]: r for r in csv.DictReader(open(src / "social-media-study.csv", encoding="utf-8", errors="ignore"))}
    files = {p.stem: p for p in (src / "social-media-study-gifs").iterdir() if p.suffix.lower() in (".mp4", ".gif")}
    cache = ROOT / "data" / "fftml" / "pspal.jsonl"
    done = {json.loads(l)["id"] for l in cache.read_text(encoding="utf-8").splitlines() if l.strip()} if cache.exists() else set()
    with cache.open("a", encoding="utf-8") as f:
        for i, k in enumerate(sorted(lab), 1):
            if k in done or k not in files:
                continue
            try:
                x, fps = dec_any(files[k])
                if x.shape[2] > 480:                                    # 속도: 비율 기반 규칙이라 축소해도 같은 정의
                    import cv2
                    x = np.stack([cv2.resize(q, (480, int(round(q.shape[0] * 480 / q.shape[1]))), interpolation=cv2.INTER_AREA) for q in x])
                dur = len(x) / fps
                Y = rel_lum(x)
                xl = np.concatenate([x, x[:1]])                         # 반복 재생: 마지막 → 처음
                Yl = np.concatenate([Y, Y[:1]])
                fl, al = count_blocks(shifts_flash(Y), fps, dur)
                fl_l, _ = count_blocks(shifts_flash(Yl), fps, dur)
                rd_l, _ = count_blocks(shifts_red(xl), fps, dur)
                row = {"id": k, "fps": fps, "dur": dur, "flash_n": fl, "flash_n_loop": fl_l, "red_n_loop": rd_l, "allowed": al,
                       "y": {q: lab[k][q] for q in ("dangerous", "flash", "red", "pattern")}}
            except Exception as e:  # noqa: BLE001
                row = {"id": k, "error": f"{type(e).__name__}: {e}"}
            f.write(json.dumps(row, ensure_ascii=False) + "\n"); f.flush()
            if i % 200 == 0:
                print(f"  {i}/{len(lab)}", flush=True)
    report(cache)


def report(cache):
    from sklearn.metrics import cohen_kappa_score, roc_auc_score
    R = [json.loads(l) for l in cache.read_text(encoding="utf-8").splitlines() if l.strip()]
    R = [r for r in R if "error" not in r]
    L = ["# South 2021 라벨 = PhotosensitivityPal 출력인가 — 원문 규칙 재구현과의 일치 (scripts/pspal_reimpl.py)", "",
         f"> n = {len(R)} · 라벨 = CSV `flash`/`red` · 재구현 = 원문 §4.2.1–4.2.2 (구현 선택은 스크립트 머리말)", "",
         "| 라벨 | 재구현 변형 | 일치 | 재현율(라벨 양성 중) | 정밀도 | Cohen κ | AUC(깜빡임 수) |", "|---|---|---|---|---|---|---|"]
    for name, key, ykey, pos in (("flash", "flash_n", "flash", "flash"), ("flash", "flash_n_loop", "flash", "flash"), ("red", "red_n_loop", "red", "red")):
        y = np.array([r["y"][ykey] == pos for r in R], int)
        s = np.array([r[key] for r in R], float)
        p = np.array([r[key] > r["allowed"] for r in R], int)
        tp = int(((p == 1) & (y == 1)).sum())
        L.append(f"| {name} ({y.sum()}) | {key} | {(p == y).mean():.3f} | {tp}/{y.sum()} | {tp}/{p.sum()} | {cohen_kappa_score(y, p):.3f} | "
                 f"{roc_auc_score(y, s) if 0 < y.sum() < len(y) else float('nan'):.3f} |")
    (ROOT / "checks" / "pspal_reimpl.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    if "--report" in sys.argv:
        report(ROOT / "data" / "fftml" / "pspal.jsonl")
    else:
        main()
