"""조건부 생성기 G 평가 — Oracle 재현도·시간 충실도·다양성·ROI 밖 보존.

학습에 쓰지 않은 원본(val/test 분할)에서 측정합니다. 위험도나 표준 적합성을 평가하지 않습니다.

지표(유형별, 클립 평균):
- roi_l1:        ROI 안에서 |G − Oracle| 평균 (0~1 스케일). 낮을수록 Oracle과 비슷
- roi_l1_clean:  ROI 안에서 |clean − Oracle| 평균. 아무것도 안 하는 G의 기준선
- temporal_corr: 프레임별 ROI 평균 변화량 |G−clean|과 |Oracle−clean|의 상관계수. 1에 가까울수록 시간 파형 재현
- diversity:     같은 조건에서 잡음만 바꾼 K개 출력의 ROI 픽셀 표준편차 평균. 0이면 잡음을 무시(Oracle 복사)
- diversity_rel: diversity / ROI 평균 변화량. 변형 크기 대비 다양성
- outside_max:   ROI 밖 최대 픽셀 변화(0이어야 함)
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from lumi.device import describe, resolve_device
from lumi.gan_synthesis import OracleDataset, load_generator

# dataset_synthesis.synthesize의 목표 색 (luminance: 0.95 회색, red: 순적에 가까운 색)
TARGET_COLOR = {"luminance": (0.95, 0.95, 0.95), "red": (1.0, 0.04, 0.08)}


def corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.std() < 1e-8 or b.std() < 1e-8:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def evaluate(checkpoint: Path, dataset: Path, split: str, samples: int, device: torch.device,
             seed: int = 0, limit: int | None = None) -> dict:
    model = load_generator(checkpoint, device)
    data = OracleDataset(dataset, split=split)
    gen = torch.Generator(device=device).manual_seed(seed)
    per_type = defaultdict(lambda: defaultdict(list))
    with torch.inference_mode():
        for idx in range(len(data) if limit is None else min(limit, len(data))):
            item = data[idx]
            clean = item["clean"].unsqueeze(0).to(device)
            real = item["affected"].unsqueeze(0).to(device)
            mask = item["mask"].unsqueeze(0).to(device)
            act = item["activation"].unsqueeze(0).to(device)
            mode = item["mode"].unsqueeze(0).to(device)
            outs = []
            for _ in range(samples):
                noise = torch.randn(mask.shape, generator=gen, device=device)
                outs.append(model(clean, mask, act, mode, noise=noise))
            outs = torch.cat(outs)                      # [K,3,T,H,W]
            m = mask[0, 0] > 0                          # [T,H,W]
            stats = per_type[item["type"]]
            stats["outside_max"].append(float((outs - clean).abs().amax(1)[:, ~m].max()) if (~m).any() else 0.0)
            if not m.any():
                continue
            g = outs[0]
            area = m.sum().item()
            roi = lambda x: (x.abs() * m).sum().item() / (area * 3)
            stats["roi_l1"].append(roi(g - real[0]))
            stats["roi_l1_clean"].append(roi(clean[0] - real[0]))
            # 프레임별 ROI 평균 변화량
            frame_area = m.sum((1, 2)).clamp(min=1)
            g_t = ((g - clean[0]).abs().mean(0) * m).sum((1, 2)) / frame_area
            o_t = ((real[0] - clean[0]).abs().mean(0) * m).sum((1, 2)) / frame_area
            stats["temporal_corr"].append(corr(g_t.float().cpu().numpy(), o_t.float().cpu().numpy()))
            if samples > 1:
                div = (outs.std(0).mean(0) * m).sum().item() / area
                stats["diversity"].append(div)
                stats["diversity_rel"].append(div / max(roi(g - clean[0]), 1e-6))
            if item["type"] in TARGET_COLOR:
                # Oracle 휘도·적색 변형: out − clean = k · act[t] · (target − clean).
                # 세기 k(0.15~0.85 균등)는 G 조건에 없으므로, 이상적인 G는 잡음으로 k를 고르게 뽑는다.
                target = torch.tensor(TARGET_COLOR[item["type"]], device=device).view(3, 1, 1, 1)
                basis = act[0] * (target - clean[0]) * m
                denom = (basis * basis).sum().clamp(min=1e-8)
                k_oracle = float(((real[0] - clean[0]) * basis).sum() / denom)
                k_g = [float(((o - clean[0]) * basis).sum() / denom) for o in outs]
                stats["k_oracle"].append(k_oracle)
                stats["k_g"].append(k_g[0])
                stats["k_abs_err"].append(abs(k_g[0] - k_oracle))
                if samples > 1:
                    stats["k_within_std"].append(float(np.std(k_g)))
    report = {"checkpoint": str(checkpoint), "dataset": str(dataset), "split": split,
              "clips": len(data) if limit is None else min(limit, len(data)), "samples": samples, "types": {}}
    for kind, stats in sorted(per_type.items()):
        report["types"][kind] = {k: float(np.nanmean(v)) if v else None for k, v in stats.items()}
        report["types"][kind]["n"] = len(stats["outside_max"])
        if stats.get("k_oracle"):
            # 클립 간 분포: Oracle k는 균등분포(표준편차 ≈ 0.20), 붕괴한 G는 한 값 근처에 몰린다.
            report["types"][kind]["k_oracle_std"] = float(np.std(stats["k_oracle"]))
            report["types"][kind]["k_g_std"] = float(np.std(stats["k_g"]))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="조건부 생성기 G 평가(영상 재생 없음)")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", default="val", choices=("train", "val", "test"))
    parser.add_argument("--samples", type=int, default=4, help="다양성 측정용 잡음 표본 수")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--output", type=Path, default=None, help="결과 JSON 저장 경로")
    args = parser.parse_args()
    device = resolve_device(args.device)
    report = evaluate(args.checkpoint, args.dataset, args.split, args.samples, device, limit=args.limit)
    print(f"장치: {describe(device)} · {args.split} 클립 {report['clips']}개 · 잡음 표본 {args.samples}")
    cols = ("roi_l1", "roi_l1_clean", "temporal_corr", "diversity", "diversity_rel", "outside_max")
    print(f"{'type':>10} " + " ".join(f"{c:>13}" for c in cols))
    for kind, s in report["types"].items():
        print(f"{kind:>10} " + " ".join(f"{s[c]:>13.4f}" if isinstance(s.get(c), float) else f"{'-':>13}" for c in cols))
    kcols = ("k_oracle", "k_g", "k_abs_err", "k_within_std", "k_oracle_std", "k_g_std")
    print("\n세기 역산 (이상적 G: k_within_std ≈ k_oracle_std ≈ 0.20, 붕괴: k_within_std ≈ 0)")
    print(f"{'type':>10} " + " ".join(f"{c:>13}" for c in kcols))
    for kind, s in report["types"].items():
        if s.get("k_oracle") is not None:
            print(f"{kind:>10} " + " ".join(f"{s[c]:>13.4f}" if isinstance(s.get(c), float) else f"{'-':>13}" for c in kcols))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
