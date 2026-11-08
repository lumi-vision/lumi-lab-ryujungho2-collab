"""G v2(또는 비교용 무작위 Oracle)로 판정 라벨이 붙은 국소 깜빡임 데이터셋을 만듭니다. 재생·미리보기 없음.

- 배경: scripts/prep_real_clips.py 결과(실제 안전 영상). --split train/val만 허용(test는 H1 평가용으로 보존).
- 라벨은 **목표(target)가 아니라 하드 판정 결과**다(lumi.judge.judge_luminance_flash, 화면 25% 기준).
  standard_result = "lumi_judge:fail" | "lumi_judge:near"(통과, 1초 전이 2~6) | "lumi_judge:quiet"(통과, 0~1)
- 파일: <out>/<split>/<id>.npz (clean, affected, mask[T,H,W], transition_area[T-1]) + <id>.json, <out>/manifest.jsonl
  transition_area = 프레임 간 같은 방향으로 전이한 픽셀 비율(판정 규칙의 2단계 값). 프레임별 시간 라벨로 쓸 수 있다.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_gan_v2 import oracle  # noqa: E402  같은 Oracle 정의를 공유

from lumi.gan_v2 import TARGET_FAIL, TARGET_NEAR, GeneratorV2, random_roi_masks  # noqa: E402
from lumi.judge import DARK_Y, DELTA_Y, judge_luminance_flash, srgb_to_luminance_np  # noqa: E402

RULE = "lumi.judge v1 — WCAG/ITU 일반 깜빡임, 픽셀 단위, 화면 25%, 1초 전이 > 6이면 실패"


def transition_area(clip: np.ndarray) -> np.ndarray:
    y = srgb_to_luminance_np(clip)
    d = np.diff(y, axis=0)
    dark = np.minimum(y[1:], y[:-1]) < DARK_Y
    pos = ((d >= DELTA_Y) & dark).mean(axis=(1, 2))
    neg = ((d <= -DELTA_Y) & dark).mean(axis=(1, 2))
    return np.where(pos >= neg, pos, -neg).astype(np.float32)   # 부호 = 방향


def category(j) -> str:
    return "fail" if j.fail else ("near" if j.max_transitions_1s >= 2 else "quiet")


def main() -> None:
    ap = argparse.ArgumentParser(description="판정 라벨 국소 깜빡임 데이터셋 생성")
    ap.add_argument("--data", type=Path, required=True, help="prep_real_clips.py 출력 폴더")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--generator", choices=("gan", "oracle"), default="gan")
    ap.add_argument("--checkpoint", type=Path, default=None)
    ap.add_argument("--split", choices=("train", "val"), default="train")
    ap.add_argument("--per-clip", type=int, default=2)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    bg = np.load(args.data / f"{args.split}.npy", mmap_mode="r")
    srcs = [json.loads(l) for l in open(args.data / "manifest.jsonl", encoding="utf-8")]
    srcs = sorted([s for s in srcs if s["split"] == args.split], key=lambda s: s["index"])
    n = len(bg) if args.limit is None else min(args.limit, len(bg))
    _, t, h, w, _ = bg.shape
    fps = 24.0
    gen_name, G, zdim = "oracle_random", None, 0
    if args.generator == "gan":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ck = torch.load(args.checkpoint, map_location=device, weights_only=False)
        cfg = ck["config"]
        fps, zdim = cfg["fps"], cfg["zdim"]
        G = GeneratorV2(cfg["width"], zdim, cfg["frames"], fps, cfg.get("output", "residual")).to(device).eval()
        G.load_state_dict(ck["generator"])
        gen_name = f"gan_v2:{args.checkpoint.stem}"
    out_dir = args.output / args.split
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest, counts = [], {"fail": 0, "near": 0, "quiet": 0}
    for i in range(n):
        clean = np.asarray(bg[i])
        for k in range(args.per_clip):
            mask = random_roi_masks(1, h, w, rng)[0]
            target = None
            if G is not None:
                target = TARGET_FAIL if (i * args.per_clip + k) % 2 == 0 else TARGET_NEAR
                x = torch.from_numpy(clean).to(G.head.weight.device).permute(3, 0, 1, 2).float()[None] / 255
                m = torch.from_numpy(mask).to(x.device).float()[None, None, None].expand(1, 1, t, h, w)
                with torch.no_grad():
                    o = G(x, m, torch.tensor([target], device=x.device), torch.randn(1, zdim, device=x.device))
                affected = np.rint(o[0].permute(1, 2, 3, 0).float().cpu().numpy() * 255).astype(np.uint8)
                affected = np.where(mask[None, :, :, None] > 0, affected, clean)   # ROI 밖 원본 보장
            else:
                affected = oracle(clean, mask, fps, rng, broad=True)
            j = judge_luminance_flash(affected, fps)
            cat = category(j)
            counts[cat] += 1
            cid = f"{srcs[i]['source']}_{k:02d}"
            np.savez_compressed(out_dir / f"{cid}.npz", clean=clean, affected=affected,
                                mask=np.repeat(mask[None], t, axis=0), transition_area=transition_area(affected))
            row = {"id": cid, "source_id": srcs[i]["source"], "split": args.split, "type": "luminance",
                   "generator": gen_name, "target": None if target is None else ("fail" if target > 0.5 else "near"),
                   "frames": t, "height": h, "width": w, "fps": fps, "roi_area": float(mask.mean()),
                   "file": f"{args.split}/{cid}.npz",
                   "judge": {"rule": RULE, "max_transitions_1s": j.max_transitions_1s, "fail": j.fail,
                             "transition_frames": list(j.transition_frames), "max_area": j.max_area},
                   "risk_verified": True, "standard_result": f"lumi_judge:{cat}", "preview_safe": False}
            (out_dir / f"{cid}.json").write_text(json.dumps(row, ensure_ascii=False, indent=1), encoding="utf-8")
            manifest.append(row)
    with open(args.output / "manifest.jsonl", "a", encoding="utf-8") as f:
        for r in manifest:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{gen_name} · {args.split} · 클립 {len(manifest)}개 · 판정 분포 {counts}")
    print("라벨은 하드 판정 결과(표준 문턱 판정)이며 임상 위험 판정이 아니다.")


if __name__ == "__main__":
    main()
