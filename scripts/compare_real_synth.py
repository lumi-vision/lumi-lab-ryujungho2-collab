"""실제 위험 영상과 합성(hazard7)의 '생김새'를 같은 지표로 비교한다 (영상은 보지 않고 수치로).

같은 조건: 160×90, 24 fps, 2 초 창 (실제 = data/hazcls/real 캐시, 합성 = data/hazard7/<run>/val mp4 를 줄여서).
영상마다 '가장 위험해 보이는 창'(바뀐 면적이 가장 큰 창)에서 잰다.

지표
  change_area   프레임 간 |ΔL| ≥ 20 cd/m² 인 픽셀 비율 (창 안 상위 10% 프레임의 평균) — 깜빡임이 화면을 얼마나 덮나
  global_frac   바뀐 픽셀이 화면 전체에 고르게 퍼진 정도 (3×3 칸 중 바뀐 칸 비율)
  trans_rate    화면 평균 휘도가 방향을 바꾼 횟수/초 (|Δ| ≥ 2 cd/m²)
  rhythm_cv     방향 전환 간격의 변동계수 (0 = 완전 규칙, 클수록 불규칙)
  dL_p90        바뀐 픽셀의 |ΔL| 90% 값 (cd/m²) — 세기
  sat_change    바뀐 픽셀의 채도 변화 평균 (0~1) — 색이 바뀌는 깜빡임인가
  hue_spread    바뀐 픽셀 색상(hue)의 퍼짐 (0 = 한 색, 1 = 여러 색)
  edge_density  프레임의 윤곽선 비율 (그래픽·만화일수록 큼)
  mean_L        화면 평균 휘도 (cd/m²)
  dark_frac     L < 5 cd/m² 픽셀 비율 (검은 배경·검은 프레임)
결과: checks/hazard7/real_vs_synth.md
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from make_hazard7_dataset import decode  # noqa: E402

LUT = None


def lum_cd(bgr):
    global LUT
    if LUT is None:
        c = np.arange(256) / 255.0
        LUT = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4).astype(np.float32)
    x = LUT[bgr]
    return 200 * (0.0722 * x[..., 0] + 0.7152 * x[..., 1] + 0.2126 * x[..., 2])


def feats(win: np.ndarray) -> dict:
    """win: T×90×160×3 BGR uint8."""
    L = lum_cd(win)
    d = np.abs(np.diff(L, axis=0))
    ch = d >= 20
    area = ch.mean((1, 2))
    top = np.argsort(area)[-max(1, len(area) // 10):]
    h, w = ch.shape[1:]
    cells = [ch[top][:, i * h // 3:(i + 1) * h // 3, j * w // 3:(j + 1) * w // 3].mean() > 0.05 for i in range(3) for j in range(3)]
    m = L.mean((1, 2))
    dm = np.diff(m)
    sgn = np.sign(np.where(np.abs(dm) >= 2, dm, 0))
    nz = np.nonzero(sgn)[0]
    turns = nz[1:][np.diff(sgn[nz]) != 0] if len(nz) > 1 else np.array([])
    iv = np.diff(turns)
    hsv = cv2.cvtColor(win.reshape(-1, win.shape[2], 3), cv2.COLOR_BGR2HSV).reshape(win.shape).astype(np.float32)
    s = hsv[..., 1] / 255
    ds = np.abs(np.diff(s, axis=0))
    sel = ch & (np.maximum(s[1:], s[:-1]) > 0.2)
    hue = hsv[1:, ..., 0][sel] * (2 * np.pi / 180)
    hs = 1 - float(np.hypot(np.cos(hue).mean(), np.sin(hue).mean())) if hue.size > 50 else 0.0
    g = cv2.cvtColor(win[len(win) // 2], cv2.COLOR_BGR2GRAY)
    return {"change_area": float(area[top].mean()), "global_frac": float(np.mean(cells)),
            "trans_rate": float(len(turns) / (len(win) / 24)), "rhythm_cv": float(iv.std() / iv.mean()) if len(iv) >= 3 else np.nan,
            "dL_p90": float(np.percentile(d[ch], 90)) if ch.any() else 0.0, "sat_change": float(ds[ch].mean()) if ch.any() else 0.0,
            "hue_spread": hs, "edge_density": float((cv2.Canny(g, 80, 160) > 0).mean()), "mean_L": float(L.mean()),
            "dark_frac": float((L < 5).mean())}


def best(wins) -> dict:
    fs = [feats(w) for w in wins]
    return max(fs, key=lambda f: f["change_area"])


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=["d3", "d4red"], help="합성 run (뒤 run 이 같은 종류를 덮음)")
    ap.add_argument("--out", default="real_vs_synth")
    a = ap.parse_args()
    out = {}
    real = ROOT / "data" / "hazcls" / "real"
    tem = np.load(real / "tem.npy", mmap_mode="r")
    rows = [json.loads(l) for l in (real / "rows.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    items = [json.loads(l) for l in (real / "items.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    by = {}
    for i, r in enumerate(rows):
        by.setdefault(r["item"], []).append(i)
    groups = {"실제 Kaya (22)": [k for k, it in enumerate(items) if it["source"] == "kaya"],
              "실제 South 위험-깜빡임": [k for k, it in enumerate(items) if it["source"] == "south" and it["any"] and it["lum"]],
              "실제 South 위험-적색": [k for k, it in enumerate(items) if it["source"] == "south" and it["any"] and it["red"]],
              "실제 South 위험-패턴": [k for k, it in enumerate(items) if it["source"] == "south" and it["any"] and it["pattern"]],
              "실제 South 안전 (100)": [k for k, it in enumerate(items) if it["source"] == "south" and not it["any"]][:100]}
    for name, ks in groups.items():
        out[name] = [best([tem[i] for i in by[k][:20]]) for k in ks if k in by]
        print(name, len(out[name]), flush=True)
    later = {}
    for run in a.runs:
        d = ROOT / "data" / "hazard7" / run / "val"
        man = [json.loads(l) for l in (d / "manifest.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
        for r in man:
            later[r["kind"]] = run
    for run in a.runs:
        d = ROOT / "data" / "hazard7" / run / "val"
        man = [json.loads(l) for l in (d / "manifest.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
        for r in man:
            if r["intent"] != "hazard" or not r.get("hit_added") or later[r["kind"]] != run:
                continue
            fr = decode(d / f"{r['id']}.mp4")
            small = np.stack([cv2.resize(f, (160, 90), interpolation=cv2.INTER_AREA) for f in fr])
            wins = [small[s:s + 48] for s in range(0, len(small) - 47, 48)]
            out.setdefault(f"합성 {r['kind']} ({run})", []).append(best(wins))
    keys = list(next(iter(out.values()))[0].keys())
    L = ["# 실제 위험 영상 대 합성(hazard7) — 생김새 지표 비교", "",
         f"> `scripts/compare_real_synth.py --runs {' '.join(a.runs)}` · 160×90 · 24 fps · 2 s 창 · 영상마다 바뀐 면적이 가장 큰 창 · 값 = 중앙값 [25%–75%]", "",
         "| 묶음 | n | " + " | ".join(keys) + " |", "|" + "---|" * (len(keys) + 2)]
    for name, fs in out.items():
        cells = []
        for k in keys:
            v = np.array([f[k] for f in fs], float); v = v[~np.isnan(v)]
            cells.append(f"{np.median(v):.2f} [{np.percentile(v, 25):.2f}–{np.percentile(v, 75):.2f}]" if len(v) else "–")
        L.append(f"| {name} | {len(fs)} | " + " | ".join(cells) + " |")
    L += ["", __doc__.split("지표", 1)[1].split("결과:")[0].strip().replace("\n  ", "\n- ")]
    p = ROOT / "checks" / "hazard7" / f"{a.out}.md"
    p.write_text("\n".join(L) + "\n", encoding="utf-8")
    json.dump(out, open(ROOT / "data" / "hazard7" / f"{a.out}.json", "w"), ensure_ascii=False)
    print(p)


if __name__ == "__main__":
    main()
