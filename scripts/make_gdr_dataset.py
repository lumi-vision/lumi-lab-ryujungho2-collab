"""2단계: G(또는 규칙 생성기)로 탐지·보정 학습용 쌍 데이터를 만든다. 정답은 하드 판정기가 붙인다.

    python scripts/make_gdr_dataset.py --split train --reps 2 --lum w_div2.0_2k_s0.pt --red r_div2.0_s0.pt --stripe s_div2.0_s0.pt
    python scripts/make_gdr_dataset.py --split val --reps 2 ...            # 같은 G, 다른 영화의 바탕
    python scripts/make_gdr_dataset.py --split val --generator rule --reps 1   # 다른 생성기(규칙) — 교차 확인용

바탕 하나마다 (종류 3 × 목표 2 × reps)개를 만들고, 바탕 자체(아무것도 넣지 않은 것)도 하나씩 판정해 넣는다.

저장: data/gdr/<이름>/<split>/<id>.npz
    flicker  T×H×W×3 uint8 BGR (바탕 자체인 표본에는 없음 — 원본을 그대로 쓴다)
    tubes    3×T×H×W bool, 순서 (lum, red, stripe) — roi.judge_tubes 의 A 라벨
    mask_allow H×W bool (G 에 준 영역. 정답이 아니다)
    meta     JSON: split, bg, kind, hazard, intent, recipe, 판정 요약
원본(clean)은 저장하지 않는다: data/oracle_768x432_24fps/backgrounds_<split>.npz 의 bg 번째 클립이다.
manifest.jsonl 에 표본마다 한 줄(meta)을 적는다.

G 추론은 이 프로세스에서 차례로(GPU), 판정·저장은 작업자 프로세스에서 한다(줄무늬 판정이 표본당 10~20초).
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import sys
import time
import zlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

KIND_OF = {"lum": "lum", "red": "red", "stripe": "stripe"}
LOSS_ALPHA = 0.95          # 이 이상이면 '심한 정보 감쇠'(M_loss) — 실험적 문턱, 수학적 완전 소실 기준이 아님


def save_alpha(path: str, alpha: np.ndarray) -> None:
    """합성 비중 α(T×H×W, 0~1)를 옆 파일 <id>.alpha.npz 에 uint8(×255)로 저장한다(M_loss 의 근거, docs/GDR_VERIFY.md)."""
    np.savez_compressed(path, alpha=np.round(np.clip(alpha, 0, 1) * 255).astype(np.uint8))


def judge_and_save(path: str, flicker, clean, mask_allow, meta: dict, fps: float, alpha=None) -> dict:
    """작업자: 판정 → npz 저장 → meta(판정 요약 포함). alpha 가 있으면 <id>.alpha.npz 도 쓴다."""
    from lumi_synth import roi
    frames = clean if flicker is None else flicker
    tubes, summary = roi.judge_tubes(frames, fps)
    meta = {**meta, "judge": summary, "tube_frac": {k: round(float(tubes[k].mean()), 5) for k in roi.HAZARDS}}
    if alpha is not None:
        save_alpha(path.replace(".npz", ".alpha.npz"), alpha)
        hz = roi.stack_tubes(tubes).any(axis=0)
        meta["loss_frac_in_tube"] = round(float((alpha[hz] >= LOSS_ALPHA).mean()), 4) if hz.any() else 0.0
    if flicker is not None:
        meta["changed_frac"] = round(float(np.any(flicker != clean, axis=-1).mean()), 5)
        meta["outside_changed"] = int(np.any(flicker != clean, axis=-1)[:, ~mask_allow].sum())
    arrays = {"tubes": roi.stack_tubes(tubes), "meta": np.array(json.dumps(meta, ensure_ascii=False))}
    if flicker is not None:
        arrays.update(flicker=flicker, mask_allow=mask_allow)
    np.savez_compressed(path, **arrays)
    return meta


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", required=True, choices=("train", "val", "test"))
    ap.add_argument("--generator", default="gan", choices=("gan", "rule"))
    ap.add_argument("--reps", type=int, default=2, help="바탕·종류·목표마다 만들 개수")
    ap.add_argument("--lum", default=None, help="휘도 G 체크포인트 (data/gan 안의 파일 이름)")
    ap.add_argument("--red", default=None)
    ap.add_argument("--stripe", default=None)
    ap.add_argument("--name", default="v1")
    ap.add_argument("--only", nargs="+", default=None, choices=("lum", "red", "stripe"),
                    help="이 종류만 만든다 (규칙 생성기에서 1차 = lum red)")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--limit-bg", type=int, default=0, help="바탕을 앞에서 N 개만 (동작 확인용)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from lumi_synth import generate
    ckpt = {"lum": args.lum, "red": args.red, "stripe": args.stripe}
    hazards = [h for h in ("lum", "red", "stripe") if (args.generator == "rule" or ckpt[h]) and (not args.only or h in args.only)]
    if not hazards:
        raise SystemExit("--generator gan 이면 --lum/--red/--stripe 중 하나 이상이 필요합니다")
    if args.generator == "gan":
        for h in hazards:
            got = generate.checkpoint_hazard(generate.GAN_DIR / ckpt[h])
            if got != h:
                raise SystemExit(f"--{h} {ckpt[h]} 는 {got} 체크포인트입니다")

    clips, fps = generate.load_backgrounds(args.split)
    n_bg = min(len(clips), args.limit_bg) if args.limit_bg else len(clips)
    out_dir = ROOT / "data" / "gdr" / args.name / (args.split if args.generator == "gan" else f"{args.split}_rule")
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = out_dir / "manifest.jsonl"
    done = set()
    if manifest.exists():
        done = {json.loads(line)["id"] for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()}

    jobs = []                                   # (id, bg, hazard|None, intent|None, rep)
    for bg in range(n_bg):
        jobs.append((f"{args.split}_{bg:03d}_clean", bg, None, None, 0))
        for h in hazards:
            for intent in generate.INTENTS:
                for rep in range(args.reps):
                    tag = "g" if args.generator == "gan" else "r"
                    jobs.append((f"{args.split}_{bg:03d}_{tag}{h}_{intent}_{rep}", bg, h, intent, rep))
    jobs = [j for j in jobs if j[0] not in done]
    print(f"{args.split} 바탕 {n_bg}개 · 만들 표본 {len(jobs)}개 (이미 있음 {len(done)}) → {out_dir}", flush=True)

    t0, n_ok = time.time(), 0
    with cf.ProcessPoolExecutor(max_workers=args.workers) as pool, manifest.open("a", encoding="utf-8") as mf:
        pending: set = set()

        def drain(block_until: int) -> None:
            nonlocal pending, n_ok
            while len(pending) > block_until:
                finished, pending = cf.wait(pending, return_when=cf.FIRST_COMPLETED)
                for fu in finished:
                    meta = fu.result()
                    mf.write(json.dumps(meta, ensure_ascii=False) + "\n")
                    mf.flush()
                    n_ok += 1
                    if n_ok % 20 == 0:
                        el = time.time() - t0
                        print(f"  {n_ok}/{len(jobs)}  {el:.0f}초  (남은 시간 약 {el / n_ok * (len(jobs) - n_ok) / 60:.0f}분)", flush=True)

        for sid, bg, h, intent, rep in jobs:
            clean = clips[bg]
            meta = {"id": sid, "split": args.split, "bg": bg, "generator": args.generator, "hazard": h, "intent": intent}
            path = str(out_dir / f"{sid}.npz")
            if h is None:
                pending.add(pool.submit(judge_and_save, path, None, clean, None, meta, fps))
            else:
                seed = zlib.crc32(f"{args.seed}/{sid}".encode())                 # 표본 id 로 정해지는 시드
                fixed = {"split": args.split, "bg": bg}
                if args.generator == "gan":
                    recipe = generate.sample_recipe("gan", intent, seed, checkpoint=ckpt[h], **fixed)
                else:
                    recipe = generate.sample_recipe(KIND_OF[h], intent, seed, **fixed)
                res = generate.render(recipe, clean, fps)
                meta["recipe"] = {**recipe, **res["detail"]}
                pending.add(pool.submit(judge_and_save, path, res["flicker"], clean, res["mask"], meta, fps, res["alpha"]))
            drain(args.workers * 2)
        drain(0)
    print(f"끝: {n_ok}개, {time.time() - t0:.0f}초", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
