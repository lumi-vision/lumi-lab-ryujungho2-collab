"""탐지 모델 '학습용' GAN 샘플 만들기 — train 바탕에서만 만든다(검증·시험 누수 방지).

eval_gan.py 가 저장하는 data/gan/val/ 은 G 평가용(purpose=evaluation_only)이라 학습에 쓰면 안 된다.
이 스크립트는 train 바탕(backgrounds_train.npz)으로만 샘플을 만들어 오라클 train 샘플과 같은 형식으로 저장한다.

실행 예
    python scripts/make_gan_dataset.py --checkpoint data/gan/gan_v1_div05.pt --z-per-clip 2

출력
    data/gan_train/train/<id>.npz   clean, flicker, mask_allow, hazard_frames, hazard_mask, light
    data/gan_train/manifest.jsonl   split=train, purpose=training, 목표, 판정 결과(라벨), 체크포인트 정보
    checks/gan/make_gan_dataset.md  목표별 라벨 분포 (목표와 판정이 다른 샘플도 판정대로 라벨)

라벨은 목표(c)가 아니라 하드 판정(measure.analyze) 결과다.
주의: flicker 배열은 광과민성 위험 자극이다. 재생·미리보기하지 않는다.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from lumi_synth import measure
from lumi_synth.data import save_sample
from lumi_synth.gan import LightGenerator

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_gan import rescale_clips, sample_masks, to_tensor  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


@torch.no_grad()
def main() -> int:
    ap = argparse.ArgumentParser(description="탐지 모델 학습용 GAN 샘플 (train 바탕만)")
    ap.add_argument("--data", type=Path, default=ROOT / "data" / "oracle")
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "gan_train")
    ap.add_argument("--z-per-clip", type=int, default=2)
    ap.add_argument("--scale", type=float, default=None, help="생략하면 학습 때 배율")
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    fps = float(cfg["fps"])
    scale = args.scale if args.scale is not None else float(cfg.get("scale", 1.0))
    bg = np.load(args.data / "backgrounds_train.npz")["clips"]
    if bg.ndim != 5 or len(bg) == 0:
        raise SystemExit(f"train 바탕이 없습니다: {args.data}/backgrounds_train.npz")
    bg = rescale_clips(bg, scale)
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    masks = sample_masks(len(bg), bg.shape[2], bg.shape[3], rng)
    G = LightGenerator(int(cfg["width"]), int(cfg["zdim"]), fps)
    G.load_state_dict(ck["generator"])
    G.eval()

    t0 = time.time()
    manifest, dist = [], Counter()
    for i in range(len(bg)):
        clean = to_tensor(bg[i:i + 1], "cpu")
        m = torch.from_numpy(masks[i:i + 1]).float()[:, None, None]
        for target, tv in (("hazard", 1.0), ("near", 0.0)):
            for k in range(args.z_per_clip):
                out, delta = G(clean, m, torch.tensor([tv]), torch.randn(1, int(cfg["zdim"])))
                x = np.clip(np.rint(out[0].permute(1, 2, 3, 0).numpy() * 255), 0, 255).astype(np.uint8)
                r = measure.analyze(x, fps)
                dist[(target, bool(r["dangerous"]))] += 1
                sid = f"train_{i:03d}_{target}_z{k}"
                manifest.append(save_sample(args.out / "train", sid, {
                    "clean": bg[i], "flicker": x, "mask_allow": masks[i],
                    "hazard_frames": r["hazard_frames"], "hazard_mask": r["hazard_mask"],
                    "light": (delta[0, 0].numpy() * masks[i]).astype(np.float16)},
                    {"split": "train", "purpose": "training", "generator": "LightGenerator",
                     "checkpoint": str(args.checkpoint.name),
                     "soft_judge_version": cfg.get("soft_judge_version", "v0"),
                     "target": target, "z_index": k, "fps": fps,
                     "label_dangerous": bool(r["dangerous"]), "max_flashes": r["max_flashes"],
                     "peak_area": r["peak_area"], "label_source": "lumi_synth.measure.analyze"}))
    args.out.mkdir(parents=True, exist_ok=True)
    with open(args.out / "manifest.jsonl", "w", encoding="utf-8") as f:
        for row in manifest:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    lines = ["# 학습용 GAN 샘플 (train 바탕만)", "",
             f"- 만든 날: {time.strftime('%Y-%m-%d %H:%M')} · {time.time() - t0:.0f}초",
             f"- 체크포인트: `{args.checkpoint.name}` (소프트 판정기 {cfg.get('soft_judge_version', 'v0')}), 배율 {scale}",
             f"- train 바탕 {len(bg)}개 × 목표 2종 × z {args.z_per_clip}개 = {len(manifest)}개", "",
             "| 목표 | 판정 위험 | 판정 안전 |", "|---|---|---|"]
    for target in ("hazard", "near"):
        lines.append(f"| {target} | {dist[(target, True)]} | {dist[(target, False)]} |")
    lines += ["", "라벨은 목표가 아니라 하드 판정 결과다. `data/` 는 Git 에 올리지 않는다."]
    (ROOT / "checks" / "gan").mkdir(parents=True, exist_ok=True)
    (ROOT / "checks" / "gan" / "make_gan_dataset.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
