"""줄무늬 탐지 비교: 스캔선(measure.stripe_masks) 대 주파수(lumi_synth.stripe_freq: 규칙만 / + E-Means).

1. E-Means 학습: Big Buck Bunny(학습 영화) 배경 패치 + 합성 패턴(seed 0) → data/stripe_freq/emeans_model.npz
2. 시험: Elephants Dream(시험 영화) 프레임 또는 회색 바탕의 왼쪽 55% 에 패턴을 합성(영화·seed 가 학습과 다름).
   정답 = 합성 영역 ∩ 국소 cpd ∈ PATTERN_CPD (명암 조건 ΔL ≥ 20·어두운 쪽 < 160 을 만족하는 경우만).
   체커보드는 해석이 갈려(ITU 는 줄무늬 쪽, Harding·Wilkins 는 아님) 정답 없이 결과만 적는다.
3. 자연 영상만(ED 프레임 10장): 오탐 면적.
출력: checks/stripe_freq/README.md, results.csv, cases.png (정지 그림 — 깜빡임 없음)

사용: python scripts\\eval_stripe_freq.py [--width 384 --height 216]
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import av
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from lumi_synth import measure, stripe_freq as sf  # noqa: E402

OUT = ROOT / "checks" / "stripe_freq"
BBB = ROOT / "data/raw/BigBuckBunny_640x360.m4v"
ED = ROOT / "data/raw/elephantsdream-480-h264-st-aac.mov"
REGION = 0.55                     # 왼쪽 55% 에 패턴


def video_lum(path: Path, n: int, start: int, step: int, w: int, h: int) -> list[np.ndarray]:
    out = []
    with av.open(str(path)) as c:
        for i, fr in enumerate(c.decode(video=0)):
            if i >= start and (i - start) % step == 0:
                img = cv2.resize(fr.to_ndarray(format="bgr24"), (w, h), interpolation=cv2.INTER_AREA)
                out.append(measure.luminance_cd(img[None])[0])
            if len(out) >= n:
                break
    return out


CASES = [  # (이름, 종류, cpd, 어두운 cd, 밝은 cd, 추가 인자)
    ("직선 1.2 cpd", "straight", 1.2, 10, 110, {}),
    ("직선 1.5 cpd", "straight", 1.5, 10, 110, {}),
    ("직선 2 cpd", "straight", 2.0, 10, 110, {}),
    ("직선 5 cpd", "straight", 5.0, 10, 110, {}),
    ("직선 9 cpd", "straight", 9.0, 10, 110, {}),
    ("직선 12 cpd (범위 밖)", "straight", 12.0, 10, 110, {}),
    ("직선 45° 3 cpd", "straight", 3.0, 10, 110, {"angle": 45.0}),
    ("직선 3 cpd 명암 15 (미달)", "straight", 3.0, 40, 55, {}),
    ("동심원 3 cpd", "concentric", 3.0, 10, 110, {"center": (0.5, 0.25)}),
    ("방사형 40살", "radial", 3.0, 10, 110, {"spokes": 40, "center": (0.5, 0.27)}),
    ("나선 2 cpd", "spiral", 2.0, 10, 110, {"spokes": 12, "center": (0.5, 0.27)}),
    ("원근(처프) 2→6 cpd", "chirp", 2.0, 10, 110, {"chirp": 2.0}),
    ("체커보드 3 cpd", "checker", 3.0, 10, 110, {}),
]


def make_case(bg: np.ndarray, kind: str, cpd: float, lo: float, hi: float, kw: dict, ppd: float):
    h, w = bg.shape
    pat, loc = sf.render_pattern(kind, h, w, ppd, cpd, lo, hi, **kw)
    region = np.zeros((h, w), bool)
    region[:, : int(round(w * REGION))] = True
    lum = np.where(region, pat, bg).astype(np.float32)
    lo_cpd, hi_cpd = measure.PATTERN_CPD
    contrast_ok = (hi - lo) >= measure.DELTA_CD and lo < measure.DARK_LIMIT_CD
    gt = None if kind == "checker" else (region & (loc >= lo_cpd) & (loc <= hi_cpd) & contrast_ok)
    return lum, gt


def scores(pred: np.ndarray, gt: np.ndarray | None) -> dict:
    d = {"area": float(pred.mean()), "danger": bool(pred.mean() > measure.PATTERN_AREA_STATIC)}
    if gt is not None:
        tp = (pred & gt).sum()
        d["recall"] = float(tp / gt.sum()) if gt.sum() else None
        d["precision"] = float(tp / pred.sum()) if pred.sum() else None
        d["iou"] = float(tp / (pred | gt).sum()) if (pred | gt).sum() else None
    return d


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--width", type=int, default=384)
    ap.add_argument("--height", type=int, default=216)
    ap.add_argument("--train", type=int, default=40000, help="E-Means 학습 패치 수")
    ap.add_argument("--refit", action="store_true", help="저장된 E-Means 모델이 있어도 다시 학습")
    args = ap.parse_args()
    W, H = args.width, args.height
    ppd = measure.pixels_per_degree(W, H)
    OUT.mkdir(parents=True, exist_ok=True)

    # 1. E-Means 학습 (BBB + 합성 seed 0) — 저장된 모델이 있으면 그대로 씀
    t0 = time.time()
    model = None if args.refit else sf.load_model()
    if model is None:
        train_bg = video_lum(BBB, 30, 600, 211, W, H)
        feat, lab, kinds = sf.training_patches(args.train, W, H, train_bg, seed=0)
        model = sf.fit_model(feat, lab, path=sf.MODEL_PATH)
        t_fit = time.time() - t0
    else:
        t_fit = None
        lab = np.array([0.0])
    n_stripe_cl = int((model["stripe_ratio"] >= sf.CLUSTER_STRIPE_MIN).sum())
    print(f"E-Means {'저장된 모델' if t_fit is None else f'학습 {t_fit:.1f}s'}: 군집 {len(model['centroids'])}개 중 "
          f"줄무늬 군집 {n_stripe_cl}개, B={int(model['info_batch'])}, α0={float(model['info_alpha0']):.3f}", flush=True)
    # 학습과 다른 seed 의 패치로 군집 판별력 확인
    f2, l2, k2 = sf.training_patches(8000, W, H, video_lum(ED, 10, 300, 307, W, H), seed=1)
    _, ratio2 = sf.stripe_cluster(f2, model)
    pred2 = ratio2 >= sf.CLUSTER_STRIPE_MIN
    print(f"  시험 패치(ED·seed 1): 줄무늬 군집 정밀도 {(pred2 & l2).sum() / max(pred2.sum(), 1):.2f}, "
          f"재현율 {(pred2 & l2).sum() / max(l2.sum(), 1):.2f}", flush=True)

    # 2. 합성 시험
    test_bg = {"회색 60": np.full((H, W), 60.0, np.float32), "ED 장면": video_lum(ED, 1, 4200, 1, W, H)[0]}
    rows, sheet = [], []
    for bg_name, bg in test_bg.items():
        for name, kind, cpd, lo, hi, kw in CASES:
            lum, gt = make_case(bg, kind, cpd, lo, hi, kw, ppd)
            lum = measure.luminance_cd(sf.cd_to_bgr(lum)[None])[0]          # 8비트 양자화를 거친 휘도로 판정
            res = {}
            t = time.time(); res["스캔선"] = measure.stripe_masks(lum[None])[0]; ts = time.time() - t
            t = time.time(); res["주파수(규칙)"] = sf.frame_stripes(lum, None)["mask"]; tf = time.time() - t
            t = time.time(); res["주파수+E-Means"] = sf.frame_stripes(lum, model)["mask"]; te = time.time() - t
            t = time.time(); res["쿼드트리(규칙)"] = sf.frame_stripes_quadtree(lum, None)["mask"]; tq = time.time() - t
            t = time.time(); res["쿼드트리+E-Means"] = sf.frame_stripes_quadtree(lum, model)["mask"]; tqe = time.time() - t
            t = time.time(); res["쿼드트리+가버"] = sf.frame_stripes_quadtree(lum, None, refine=True)["mask"]; tqg = time.time() - t
            times = {"스캔선": ts, "주파수(규칙)": tf, "주파수+E-Means": te, "쿼드트리(규칙)": tq, "쿼드트리+E-Means": tqe,
                     "쿼드트리+가버": tqg}
            for meth, pred in res.items():
                rows.append({"바탕": bg_name, "경우": name, "방법": meth, "정답 면적": None if gt is None else float(gt.mean()),
                             "정답 위험": None if gt is None else bool(gt.mean() > measure.PATTERN_AREA_STATIC),
                             **scores(pred, gt), "시간(초)": times[meth]})
            if bg_name == "ED 장면":
                sheet.append((name, lum, gt, res))
            print(bg_name, name, {k: f"{v.mean():.2f}" for k, v in res.items()},
                  "정답", "-" if gt is None else f"{gt.mean():.2f}", flush=True)

    # 3. 자연 영상 오탐
    nat = video_lum(ED, 10, 1000, 997, W, H)
    fp = {"스캔선": [], "주파수(규칙)": [], "주파수+E-Means": [], "쿼드트리(규칙)": [], "쿼드트리+E-Means": [],
          "쿼드트리+가버": []}
    for lum in nat:
        lum = measure.luminance_cd(sf.cd_to_bgr(lum)[None])[0]
        fp["스캔선"].append(measure.stripe_masks(lum[None])[0].mean())
        fp["주파수(규칙)"].append(sf.frame_stripes(lum, None)["mask"].mean())
        fp["주파수+E-Means"].append(sf.frame_stripes(lum, model)["mask"].mean())
        fp["쿼드트리(규칙)"].append(sf.frame_stripes_quadtree(lum, None)["mask"].mean())
        fp["쿼드트리+E-Means"].append(sf.frame_stripes_quadtree(lum, model)["mask"].mean())
        fp["쿼드트리+가버"].append(sf.frame_stripes_quadtree(lum, None, refine=True)["mask"].mean())

    # 저장
    with open(OUT / "results.csv", "w", newline="", encoding="utf-8-sig") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()) + [k for k in ("recall", "precision", "iou") if k not in rows[0]])
        wr.writeheader()
        for r in rows:
            wr.writerow(r)
    figure(sheet, OUT / "cases.png")
    report(rows, fp, model, t_fit, (pred2, l2), W, H, ppd)
    print("완료:", OUT)
    return 0


def figure(sheet, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False
    cols = ["입력", "정답", "스캔선", "격자(규칙)", "쿼드트리(규칙)", "쿼드트리+가버"]
    fig, axes = plt.subplots(len(sheet), 6, figsize=(16, 1.45 * len(sheet)))
    for i, (name, lum, gt, res) in enumerate(sheet):
        imgs = [lum, gt, res["스캔선"], res["주파수(규칙)"], res["쿼드트리(규칙)"], res["쿼드트리+가버"]]
        for j, im in enumerate(imgs):
            ax = axes[i, j]
            if im is None:
                ax.imshow(np.zeros_like(lum), cmap="gray", vmin=0, vmax=1)
                ax.text(0.5, 0.5, "정답 없음", color="w", ha="center", va="center", transform=ax.transAxes, fontsize=8)
            elif j == 0:
                ax.imshow(lum, cmap="gray", vmin=0, vmax=200)
            else:
                ax.imshow(lum, cmap="gray", vmin=0, vmax=200, alpha=0.5)
                ax.imshow(np.ma.masked_where(~im, im), cmap="autumn", alpha=0.6, vmin=0, vmax=1)
                ax.text(0.98, 0.04, f"{im.mean():.0%}", color="w", ha="right", va="bottom", transform=ax.transAxes,
                        fontsize=7, bbox=dict(facecolor="black", alpha=0.5, pad=1, lw=0))
            ax.set_xticks([]); ax.set_yticks([])
            if i == 0:
                ax.set_title(cols[j], fontsize=9)
        axes[i, 0].set_ylabel(name, fontsize=8, rotation=0, ha="right", va="center")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def report(rows, fp, model, t_fit, cluster_eval, W, H, ppd) -> None:
    meths = ["스캔선", "주파수(규칙)", "주파수+E-Means", "쿼드트리(규칙)", "쿼드트리+E-Means", "쿼드트리+가버"]
    L = ["# 줄무늬 탐지 비교: 스캔선 대 주파수(FFT·DCT) + E-Means", "",
         f"> 만든 날 {time.strftime('%Y-%m-%d')} · 스크립트 `scripts/eval_stripe_freq.py` · 탐지기 `src/lumi_synth/stripe_freq.py`",
         f"> 해상도 {W}×{H}(1도 {ppd:.1f} px, 패치 {sf.patch_size(W, H)} px) · 규칙 상수는 measure.py 와 같음"
         f"(cpd {measure.PATTERN_CPD}, {measure.PATTERN_MIN_PAIRS}쌍, 면적 > {measure.PATTERN_AREA_STATIC:.0%})",
         "> 정지 그림 한 장씩 판정(움직임·지속 조건 제외). 시험 바탕(Elephants Dream)과 합성 seed 는 E-Means 학습(BBB, seed 0)과 다름.", "",
         "![경우별 결과](cases.png)", "",
         "## 1. 경우별 면적 (정답 / 방법별, 굵게 = 40% 판정이 정답과 다름)", ""]
    for bg in dict.fromkeys(r["바탕"] for r in rows):
        L += [f"### 바탕: {bg}", "", "| 경우 | 정답 | " + " | ".join(meths) + " |", "|---|---|" + "---|" * len(meths)]
        for case in dict.fromkeys(r["경우"] for r in rows if r["바탕"] == bg):
            rs = {r["방법"]: r for r in rows if r["바탕"] == bg and r["경우"] == case}
            g = rs[meths[0]]["정답 면적"]
            gtxt = "–" if g is None else f"{g:.0%}"
            cells = []
            for mth in meths:
                r = rs[mth]
                s = f"{r['area']:.0%}"
                if r.get("iou") is not None:
                    s += f" (IoU {r['iou']:.2f})"
                if r["정답 위험"] is not None and r["danger"] != r["정답 위험"]:
                    s = f"**{s}**"
                cells.append(s)
            L.append(f"| {case} | {gtxt} | " + " | ".join(cells) + " |")
        L.append("")
    L += ["## 2. 요약", "", "| 방법 | 40% 판정 일치 (정답 있는 경우) | 평균 IoU | 자연 영상 10장 오탐 면적 최대 / 평균 | 한 장 평균 시간 |",
          "|---|---|---|---|---|"]
    for mth in meths:
        rs = [r for r in rows if r["방법"] == mth and r["정답 위험"] is not None]
        agree = sum(r["danger"] == r["정답 위험"] for r in rs)
        ious = [r["iou"] for r in rs if r.get("iou") is not None]
        tm = np.mean([r["시간(초)"] for r in rows if r["방법"] == mth])
        L.append(f"| {mth} | {agree}/{len(rs)} | {np.mean(ious):.2f} | {max(fp[mth]):.1%} / {np.mean(fp[mth]):.1%} | {tm:.2f}초 |")
    pred2, l2 = cluster_eval
    L += ["", "## 3. E-Means 군집", "",
          f"- 구현: 논문 식 2·5·6·7·부록 H.1 을 numpy 로 직접 구현(공식 faiss 구현 아님). K = {len(model['centroids'])}, 5 epoch, "
          f"법칙이 고른 B = {int(model['info_batch'])}, α₀ = {float(model['info_alpha0']):.3f}, "
          + ("저장된 모델(data/stripe_freq/emeans_model.npz)을 다시 씀." if t_fit is None else f"학습 {t_fit:.1f}초."),
          f"- 학습 패치의 합성 줄무늬 비율이 {sf.CLUSTER_STRIPE_MIN:.0%} 이상인 '줄무늬 군집' {(model['stripe_ratio'] >= sf.CLUSTER_STRIPE_MIN).sum()}개.",
          f"- 학습과 다른 패치(ED 배경·seed 1)에서 '줄무늬 군집 = 줄무늬' 로 보면 정밀도 {(pred2 & l2).sum() / max(pred2.sum(), 1):.2f}, "
          f"재현율 {(pred2 & l2).sum() / max(l2.sum(), 1):.2f}.",
          f"- 쓰임: 줄무늬 군집의 패치는 봉우리 집중도 문턱을 {sf.RHO_MIN} → {sf.RHO_LOOSE} 로 낮춰 인정한다(곡선·방사·원근 줄무늬용).", "",
          "## 4. 한계", "",
          "- 정답 마스크는 합성 조건에서 계산한 것이다. 생성기와 탐지기가 같은 규칙 상수를 쓰므로 '규칙을 제대로 구현했는가'만 보여 준다.",
          "- 패치가 영역 경계에 걸치면 경계 밖까지 줄무늬로 칠한다(패치 절반 크기만큼 면적이 넓어질 수 있음).",
          "- 체커보드는 둘째 봉우리 조건으로 주파수 방식에서 빠진다. 스캔선 방식은 센다. 어느 쪽이 맞는지는 기준 해석 문제다.",
          "- 정지 그림만 비교했다. 움직임·지속 조건과 실제 줄무늬 영상(옷·블라인드·에스컬레이터)은 아직 시험하지 않았다.",
          "- 문턱(RHO_MIN·RHO_LOOSE·RHO2_MAX·패치 크기)은 이 생성기와 BBB·ED 에서 정한 구현 선택이다."]
    (OUT / "README.md").write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
