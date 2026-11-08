"""G v2 평가 (실험 E2) — 하드 판정기로 목표 달성률·다양성을 재고, 같은 배경·ROI의 무작위 Oracle과 비교.

평가는 G 학습에 쓴 소프트 판정과 같은 규칙의 **하드 판정**(lumi.judge.judge_luminance_flash)으로 한다.
val 분할만 쓴다. 지표:
- 목표 달성률: '문턱 넘기' 목표 → 하드 판정 실패, '문턱 근처' 목표 → 통과이면서 1초 전이 2~6회
- 다양성(같은 배경·ROI·목표에서 z만 바꾼 K개): 전이 수 표준편차, ROI 평균 휘도 진폭 표준편차,
  주된 깜빡임 주파수 표준편차, 출력 간 ROI 평균 절대차
- 공간 결함 대리 지표: ROI 안 라플라시안 에너지 비(출력/원본). 1에서 멀수록 얼룩·잡음 질감
Oracle: 같은 ROI에 휘도 혼합(목표색 0.95), 주파수·파형·위상·세기를 무작위로 뽑는다(dataset_synthesis와 같은 식).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from lumi.gan_v2 import TARGET_FAIL, TARGET_NEAR, GeneratorV2, random_roi_masks
from lumi.judge import judge_luminance_flash, srgb_to_luminance_np


def categorize(j):
    if j.fail:
        return "fail"
    return "near" if j.max_transitions_1s >= 2 else "quiet"


def roi_stats(clip: np.ndarray, clean: np.ndarray, mask: np.ndarray, fps: float):
    y = srgb_to_luminance_np(clip)
    m = mask.astype(bool)
    series = y[:, m].mean(axis=1)
    amp = float(series.max() - series.min())
    spec = np.abs(np.fft.rfft(series - series.mean()))
    freqs = np.fft.rfftfreq(len(series), 1 / fps)
    dom = float(freqs[1:][np.argmax(spec[1:])]) if len(spec) > 1 else 0.0
    def lap(v):
        g = srgb_to_luminance_np(v)
        l = np.abs(4 * g[:, 1:-1, 1:-1] - g[:, :-2, 1:-1] - g[:, 2:, 1:-1] - g[:, 1:-1, :-2] - g[:, 1:-1, 2:])
        return l[:, m[1:-1, 1:-1]].mean()
    return amp, dom, float(lap(clip) / max(lap(clean), 1e-6))


def oracle(clean: np.ndarray, mask: np.ndarray, fps: float, rng: np.random.Generator, broad: bool):
    t = np.arange(len(clean)) / fps
    fmax = 12.0 if broad else min(fps / 2 * 0.85, 10.0)
    freq = rng.uniform(0.5 if broad else 0.8, fmax)
    phase = rng.uniform(0, 2 * np.pi)
    wave = rng.choice(["sine", "square", "triangle"])
    pos = freq * t + phase / (2 * np.pi)
    act = {"square": (np.sin(2 * np.pi * pos) >= 0).astype(np.float32),
           "triangle": 1 - np.abs(2 * (pos - np.floor(pos)) - 1),
           "sine": (np.sin(2 * np.pi * pos) + 1) / 2}[wave]
    k = rng.uniform(0.05, 1.0) if broad else rng.uniform(0.15, 0.85)
    rgb = clean.astype(np.float32) / 255
    a = (k * act)[:, None, None, None]
    out = np.where(mask[None, :, :, None] > 0, np.clip(rgb * (1 - a) + a * 0.95, 0, 1), rgb)
    return np.rint(out * 255).astype(np.uint8)


def main() -> None:
    ap = argparse.ArgumentParser(description="G v2 평가")
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--samples", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = ck["config"]
    fps = cfg["fps"]
    G = GeneratorV2(cfg["width"], cfg["zdim"], cfg["frames"], fps, cfg.get("output", "residual")).to(device).eval()
    G.load_state_dict(ck["generator"])
    val = np.load(args.data / "val.npy")
    if args.limit:
        val = val[: args.limit]
    n, t, h, w, _ = val.shape
    rng = np.random.default_rng(args.seed)
    masks = random_roi_masks(n, h, w, rng)
    res = {"G": {"fail": [], "near": []}, "oracle": [], "oracle_broad": []}
    div = {"fail": [], "near": []}
    for i in range(n):
        clean = val[i]
        x = torch.from_numpy(clean).to(device).permute(3, 0, 1, 2).float()[None] / 255
        m = torch.from_numpy(masks[i]).to(device).float()[None, None, None].expand(1, 1, t, h, w)
        for name, tv in (("fail", TARGET_FAIL), ("near", TARGET_NEAR)):
            outs = []
            with torch.no_grad():
                for _ in range(args.samples):
                    z = torch.randn(1, cfg["zdim"], device=device)
                    o = G(x, m, torch.tensor([tv], device=device), z)
                    outs.append(np.rint(o[0].permute(1, 2, 3, 0).float().cpu().numpy() * 255).astype(np.uint8))
            js = [judge_luminance_flash(o, fps) for o in outs]
            st = [roi_stats(o, clean, masks[i], fps) for o in outs]
            for j, s in zip(js, st):
                res["G"][name].append({"cat": categorize(j), "count": j.max_transitions_1s, "amp": s[0], "freq": s[1], "lap": s[2]})
            pair = np.mean([np.abs(outs[a].astype(np.float32) - outs[b])[:, masks[i] > 0].mean() / 255
                            for a in range(len(outs)) for b in range(a + 1, len(outs))])
            div[name].append({"count_std": float(np.std([j.max_transitions_1s for j in js])),
                              "amp_std": float(np.std([s[0] for s in st])),
                              "freq_std": float(np.std([s[1] for s in st])), "pair_l1": float(pair)})
        for key, broad in (("oracle", False), ("oracle_broad", True)):
            for _ in range(args.samples):
                o = oracle(clean, masks[i], fps, rng, broad)
                j = judge_luminance_flash(o, fps); s = roi_stats(o, clean, masks[i], fps)
                res[key].append({"cat": categorize(j), "count": j.max_transitions_1s, "amp": s[0], "freq": s[1], "lap": s[2]})

    def share(lst):
        c = [r["cat"] for r in lst]
        return {k: round(c.count(k) / max(len(c), 1), 3) for k in ("fail", "near", "quiet")}

    def spread(lst, key):
        v = [r[key] for r in lst]
        return round(float(np.std(v)), 3)

    summary = {
        "clips": n, "samples": args.samples,
        "G_target_fail": {"share": share(res["G"]["fail"]), "lap_ratio": round(float(np.median([r["lap"] for r in res["G"]["fail"]])), 3)},
        "G_target_near": {"share": share(res["G"]["near"]), "lap_ratio": round(float(np.median([r["lap"] for r in res["G"]["near"]])), 3)},
        "G_within_clip_diversity": {k: {m: round(float(np.mean([d[m] for d in v])), 3) for m in v[0]} for k, v in div.items()},
        "oracle": {"share": share(res["oracle"]), "lap_ratio": round(float(np.median([r["lap"] for r in res["oracle"]])), 3)},
        "oracle_broad": {"share": share(res["oracle_broad"]), "lap_ratio": round(float(np.median([r["lap"] for r in res["oracle_broad"]])), 3)},
        "across_all_spread": {k: {m: spread(v if isinstance(v, list) else v["fail"] + v["near"], m) for m in ("count", "amp", "freq")}
                              for k, v in (("G", res["G"]), ("oracle", res["oracle"]), ("oracle_broad", res["oracle_broad"]))},
    }
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    if args.output:
        args.output.write_text(json.dumps({"summary": summary, "raw": res, "diversity": div}, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
