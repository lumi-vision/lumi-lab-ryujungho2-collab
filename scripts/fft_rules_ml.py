"""Temporal FFT · 격자 · 규칙 margin · 학습 가중치 비교 — 사전 등록 docs/PREREG_FFT_RULES_ML_2026-11-24.md 그대로.

    python scripts/fft_rules_ml.py            # 특징 추출(캐시 이어서) + 평가
    python scripts/fft_rules_ml.py --eval-only
    python scripts/fft_rules_ml.py --limit 5  # 시험용

⚠️ 영상은 메모리에서만(재생·저장 없음). 특징 캐시 data/fftml/feats.jsonl (커밋하지 않음).
"""
from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from lumi_judge.cpu_limit import limit_cpu  # noqa: E402
limit_cpu()
from eval_real_hazards import decode as dec_any  # noqa: E402
from lumi_judge.grid_flash import (grid_shape, judge_luminance, judge_red, luminance_cd,  # noqa: E402
                                   pixel_transitions, red_transitions, window_frames)
from lumi_synth.measure import extended_flash  # noqa: E402

OUT = ROOT / "data" / "fftml"
HAZ_HZ, LOW_HZ = 3.0, 1.0
AMP_TH = (5.0, 10.0)          # cd/m² (규격 ΔL 20 의 1/4·1/2 — 문턱 아래 여유를 보려는 구현 선택, 사전 등록)
RED_AMP_TH = 0.1


# ---------------------------------------------------------------- FFT
def stft_amp(s: torch.Tensor, fps: float):
    """s [T,N] → (haz [Nw,N], low [Nw,N], dom_freq [Nw,N]) 단측 진폭(입력 단위). 1 초 창, 반 창 이동, Hann."""
    T = s.shape[0]
    w = min(window_frames(fps), T)
    if w < 4:
        z = torch.zeros(1, s.shape[1], device=s.device)
        return z, z, z
    hop = max(1, w // 2)
    segs = s.unfold(0, w, hop)                                   # [Nw,N,w]
    segs = segs - segs.mean(-1, keepdim=True)
    win = torch.hann_window(w, periodic=False, device=s.device)
    X = torch.fft.rfft(segs * win, dim=-1)
    amp = 2.0 * X.abs() / win.sum()
    f = torch.fft.rfftfreq(w, 1.0 / fps).to(s.device)
    haz_b, low_b = f >= HAZ_HZ, (f >= LOW_HZ) & (f < HAZ_HZ)
    neg = torch.zeros(1, device=s.device)
    haz = amp[..., haz_b].amax(-1) if haz_b.any() else neg.expand(amp.shape[:2])
    low = amp[..., low_b].amax(-1) if low_b.any() else neg.expand(amp.shape[:2])
    dom = f[1:][amp[..., 1:].argmax(-1)] if amp.shape[-1] > 1 else neg.expand(amp.shape[:2])
    return haz, low, dom


def red_ratio(x: torch.Tensor) -> torch.Tensor:
    c = x.float() / 255.0
    lin = torch.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    return lin[..., 2] / (lin.sum(-1) + 1e-6)


# ---------------------------------------------------------------- 규칙 margin
def margins(signed: torch.Tensor, fps: float, prefix: str) -> dict:
    T = signed.shape[0]
    w = min(window_frames(fps), T)
    cs = torch.cat([torch.zeros_like(signed[:1], dtype=torch.int16), torch.cumsum((signed != 0).to(torch.int16), 0)])
    cnt = (cs[w:] - cs[:-w]).flatten(1).float()                  # [Nw, P] 1 초 창 픽셀별 전이 수
    d = {f"{prefix}_frac_ge4": float((cnt >= 4).float().mean(1).max()),
         f"{prefix}_frac_ge7": float((cnt >= 7).float().mean(1).max())}
    for a in (0.05, 0.10, 0.25):
        q = torch.quantile(cnt, 1.0 - a, dim=1) if cnt.shape[1] <= 16_000_000 else cnt.amax(1)
        d[f"{prefix}_flashes_at_area{int(a * 100)}"] = float(q.max()) / 2.0
    return d


@torch.no_grad()
def features(x: np.ndarray, fps: float) -> dict:
    xt = torch.from_numpy(np.ascontiguousarray(x)).cuda()
    lum = luminance_cd(xt)
    rr = red_ratio(xt)
    H, W = lum.shape[1:]
    gr, gc = grid_shape(H, W, 32)
    f = {}
    # M0 / M3: 규칙
    jl = judge_luminance(x, fps); jr = judge_red(x, fps)
    f["m0_score"] = max(jl["max_flashes"], jr["max_flashes"])
    f["m3_lum_flashes"] = jl["max_flashes"]; f["m3_red_flashes"] = jr["max_flashes"]
    f["m3_lum_ext"] = float(extended_flash(jl["transitions"], fps)["extended_dangerous"])
    f["m3_red_ext"] = float(extended_flash(jr["transitions"], fps)["extended_dangerous"])
    f["m3_red_max_duv"] = float(jr["evidence"].get("max_delta_uv", 0.0))
    f.update(margins(pixel_transitions(lum), fps, "m3_lum"))
    f.update(margins(red_transitions(xt)[0], fps, "m3_red"))
    # M1: 화면 평균(원문)
    gm = lum.mean((1, 2))[:, None]
    h, l_, dmn = stft_amp(gm, fps)
    f["m1_haz_amp"] = float(h.max()); f["m1_low_amp"] = float(l_.max()); f["m1_dom_hz"] = float(dmn[h.argmax() // h.shape[1]].max())
    dt = (lum[1:] - lum[:-1]).abs().mean((1, 2))
    f["m1_diff_mean"] = float(dt.mean()); f["m1_diff_max"] = float(dt.max())
    hr, _, _ = stft_amp(rr.mean((1, 2))[:, None], fps)
    f["m1_red_haz_amp"] = float(hr.max())
    # M2: 격자
    cells = torch.nn.functional.adaptive_avg_pool2d(lum[:, None], (gr, gc)).flatten(1)
    h, l_, dm = stft_amp(cells, fps)
    hmax_cell = h.amax(0)
    f["m2_haz_amp_max"] = float(h.max()); f["m2_haz_amp_p90"] = float(torch.quantile(hmax_cell, 0.9))
    for th in AMP_TH:
        f[f"m2_haz_area_ge{int(th)}"] = float((h >= th).float().mean(1).max())
        f[f"m2_low_area_ge{int(th)}"] = float((l_ >= th).float().mean(1).max())
    f["m2_low_amp_max"] = float(l_.max())
    i = int(h.flatten().argmax()); f["m2_dom_hz"] = float(dm.flatten()[i])
    rc = torch.nn.functional.adaptive_avg_pool2d(rr[:, None], (gr, gc)).flatten(1)
    hr, _, _ = stft_amp(rc, fps)
    f["m2_red_haz_amp_max"] = float(hr.max()); f["m2_red_haz_area"] = float((hr >= RED_AMP_TH).float().mean(1).max())
    return f


def extract(limit: int | None):
    OUT.mkdir(parents=True, exist_ok=True)
    src = ROOT / "data" / "real" / "south2021"
    lab = {r["label"]: r for r in csv.DictReader(open(src / "social-media-study.csv", encoding="utf-8", errors="ignore"))}
    files = {p.stem: p for p in (src / "social-media-study-gifs").iterdir() if p.suffix.lower() in (".mp4", ".gif")}
    keys = [k for k in sorted(lab) if k in files][:limit]
    cache = OUT / ("feats.jsonl" if limit is None else "feats_test.jsonl")
    done = {json.loads(l)["id"] for l in cache.read_text(encoding="utf-8").splitlines() if l.strip()} if cache.exists() else set()
    t0 = time.time()
    with cache.open("a", encoding="utf-8") as fo:
        for i, k in enumerate(keys, 1):
            if f"south:{k}" in done:
                continue
            try:
                x, fps = dec_any(files[k])
                row = {"id": f"south:{k}", "fps": fps, "T": len(x), "f": features(x, fps),
                       "y": {q: lab[k][q] for q in ("dangerous", "flash", "red", "pattern")}}
            except Exception as e:  # noqa: BLE001
                row = {"id": f"south:{k}", "error": f"{type(e).__name__}: {e}"}
            fo.write(json.dumps(row, ensure_ascii=False) + "\n"); fo.flush()
            torch.cuda.empty_cache()
            if i % 100 == 0 or limit:
                print(f"  {i}/{len(keys)} ({time.time() - t0:.0f}s)", flush=True)
    return cache


# ---------------------------------------------------------------- 평가 (사전 등록 §4)
def evaluate(cache: Path):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    R = [json.loads(l) for l in cache.read_text(encoding="utf-8").splitlines() if l.strip()]
    n_err = sum("error" in r for r in R)
    R = [r for r in R if "error" not in r]
    probe = {}
    pf = ROOT / "data" / "probe" / "feats.jsonl"
    for l in pf.read_text(encoding="utf-8").splitlines():
        if l.strip():
            q = json.loads(l)
            if "A" in q:
                probe[q["id"]] = q["A"]
    n_noprobe = sum(r["id"] not in probe for r in R)
    R = [r for r in R if r["id"] in probe]                       # M5 와 같은 표본에서 모두 비교(공정)
    names = sorted(R[0]["f"])
    sets = {"M1": [n for n in names if n.startswith("m1_")], "M2": [n for n in names if n.startswith("m2_")],
            "M3": [n for n in names if n.startswith("m3_")]}
    sets["M4"] = sets["M3"] + sets["M2"]
    F = {n: np.array([r["f"][n] for r in R], np.float64) for n in names}
    A = np.array([probe[r["id"]] for r in R], np.float64)

    def X_of(m):
        if m == "M5":
            return np.hstack([np.stack([F[n] for n in sets["M4"]], 1), A])
        return np.stack([F[n] for n in sets[m]], 1)

    def oof(X, y):
        P = np.zeros(len(y))
        for s in range(10):
            clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000))
            P += cross_val_predict(clf, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=s), method="predict_proba")[:, 1]
        return P / 10

    rng = np.random.default_rng(0)

    def boot(y, a, b, n=2000):
        pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
        d = np.empty(n)
        for i in range(n):
            ix = np.r_[rng.choice(pos, len(pos)), rng.choice(neg, len(neg))]
            d[i] = roc_auc_score(y[ix], a[ix]) - roc_auc_score(y[ix], b[ix])
        return np.percentile(d, [2.5, 97.5]), (np.sum(d <= 0) + 1) / (n + 1)

    L = ["# Temporal FFT · 격자 · 규칙 margin · 학습 가중치 비교 (사전 등록 docs/PREREG_FFT_RULES_ML_2026-11-24.md)", "",
         f"> South {len(R)} 영상(디코드 오류 {n_err}, CorrNet 특징 없음 {n_noprobe} 제외) · 로지스틱 회귀 5 겹 층화 CV × 10 반복 · 짝지은 층화 부트스트랩 2,000 회", ""]
    verdict_lines = []
    for lab_, pos in (("flash", "flash"), ("dangerous", "dangerous")):
        y = np.array([r["y"][lab_] == pos for r in R], int)
        P = {"M0": F["m0_score"]}
        for m in ("M1", "M2", "M3", "M4", "M5"):
            P[m] = oof(X_of(m), y)
        auc = {m: roc_auc_score(y, p) for m, p in P.items()}
        tag = "주 라벨" if lab_ == "flash" else "보조 라벨(탐색적)"
        L += [f"## {lab_} (양성 {y.sum()} / {len(y)}) — {tag}", "",
              "| 모델 | 내용 | AUC |", "|---|---|---|",
              f"| M0 | 규칙 판정 점수(G 1 초 최대 깜빡임, 학습 없음) | {auc['M0']:.3f} |",
              f"| M1 | 화면 평균 temporal FFT + 프레임 차 (원문 방식) | {auc['M1']:.3f} |",
              f"| M2 | 격자 칸 temporal FFT | {auc['M2']:.3f} |",
              f"| M3 | 규칙 margin (픽셀 전이) | {auc['M3']:.3f} |",
              f"| M4 | M3 + M2 | {auc['M4']:.3f} |",
              f"| M5 | M4 + CorrNet 고정 특징 | {auc['M5']:.3f} |", ""]
        tests = [("H1", "M2", "M1"), ("H2a", "M4", "M3"), ("H2b", "M4", "M0"), ("H3", "M5", "M4")]
        res = []
        for h, a, b in tests:
            ci, p = boot(y, P[a], P[b])
            res.append([h, a, b, auc[a] - auc[b], ci, p])
        order = sorted(range(len(res)), key=lambda i: res[i][5])
        holm = [None] * len(res); run = 0.0
        for k, i in enumerate(order):
            run = max(run, min(1.0, (len(res) - k) * res[i][5]))
            holm[i] = run
        L += ["| 가설 | 비교 | ΔAUC | 95 % CI | 한쪽 p | Holm p | 기준(Δ ≥ 0.02 · Holm p < 0.05) |", "|---|---|---|---|---|---|---|"]
        for (h, a, b, d, ci, p), hp in zip(res, holm):
            ok = d >= 0.02 and hp < 0.05
            L.append(f"| {h} | {a} vs {b} | {d:+.3f} | [{ci[0]:+.3f}, {ci[1]:+.3f}] | {p:.4f} | {hp:.4f} | {'**충족**' if ok else '미충족'} |")
            if lab_ == "flash":
                verdict_lines.append(f"- {h} ({a} vs {b}): ΔAUC {d:+.3f}, Holm p {hp:.4f} → {'개선 주장 가능' if ok else '개선이 보이지 않음'}")
        L.append("")
        if lab_ == "flash":
            clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000)).fit(X_of("M4"), y)
            co = clf[-1].coef_[0]
            top = sorted(zip(sets["M4"], co), key=lambda t: -abs(t[1]))[:8]
            L += ["M4 표준화 계수 상위(전체 데이터 학습, 해석용): " + ", ".join(f"`{n}` {c:+.2f}" for n, c in top), ""]
    yr = np.array([r["y"]["red"] == "red" for r in R], int)
    L += [f"red (양성 {yr.sum()}) — 사전 등록대로 검정 안 함. 규칙 점수 AUC(참고) {roc_auc_score(yr, F['m3_red_flashes']):.3f}" if yr.sum() else "", "",
          "## 판정 (주 라벨 flash, 사전 등록 기준)"] + verdict_lines
    out = ROOT / "checks" / "fft_rules_ml.md"
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    lim = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else None
    cache = OUT / ("feats.jsonl" if lim is None else "feats_test.jsonl")
    if "--eval-only" not in sys.argv:
        cache = extract(lim)
    if lim is None:
        evaluate(cache)
    else:
        print(cache.read_text(encoding="utf-8")[:3000])
