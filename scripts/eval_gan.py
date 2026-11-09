"""학습한 G 를 검증 분할(val)에서 평가하고, GAN 데이터셋(정답 포함)을 저장한다.

실행 예:  python scripts/eval_gan.py --scale 0.5 --z-per-clip 4

무엇을 재나 (docs/GAN.md §5) — 모든 판정은 하드 판정 규칙(measure.analyze)으로 다시 잰다
    1. 조절력   목표 '위험' → 실제 위험 비율, 목표 '문턱 근처 안전' → 실제 안전이면서 깜빡임 1~3회/초 비율
    2. 대리 일치 소프트 판정기의 전이 수와 하드 판정의 전이 수가 얼마나 맞는지(상관, 위험/안전 일치율)
    3. 다양성   같은 바탕·마스크·목표에서 z 만 바꿨을 때 깜빡임 수·주 주파수의 표준편차
    4. 국소성   마스크 밖 픽셀이 원본과 다른 개수 (0 이어야 함)
    5. 자연스러움(코드 지표만 — 합성 결과는 위험 자극이라 사람이 보고 평가하지 않는다)
       - 질감 보존율: 마스크 안 윤곽선 에너지(라플라시안) 출력/원본. 1 = 장면이 그대로 남음
       - 빛 공간 변이계수: 마스크 안 빛 Δ 의 표준편차/평균. 오라클은 균일해서 0
       같은 바탕·마스크에 오라클(hazard 조건)로 만든 결과도 함께 재서 기준으로 삼는다.
    대조군: 학습하지 않은 G(같은 구조, 무작위 초기값). 학습이 효과가 있었는지 보는 기준선.

시험 분할(test, Elephants Dream)은 여기서 쓰지 않는다. 나중에 탐지 모델의 최종 평가에만 쓴다.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

import cv2

from lumi_synth import measure, oracle
from lumi_synth.data import save_sample
from lumi_synth.gan import LightGenerator, luminance_rel, soft_judge, srgb_to_linear

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_gan import check_track, load_backgrounds, rescale_clips, sample_masks, to_tensor  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def dominant_freq(series: np.ndarray, fps: float) -> float:
    """마스크 안 평균 빛 s(t) 의 가장 강한 주파수(Hz). 평균(직류)은 빼고 본다."""
    s = series - series.mean()
    spec = np.abs(np.fft.rfft(s))
    freqs = np.fft.rfftfreq(len(s), 1.0 / fps)
    return float(freqs[1:][np.argmax(spec[1:])]) if len(spec) > 1 else 0.0


def texture_ratio(x_bgr: np.ndarray, clean_bgr: np.ndarray, mask: np.ndarray) -> float:
    """질감 보존율 = 마스크 안 |라플라시안| 평균(출력) / 같은 값(원본). sRGB 밝기(Y') 기준.

    빛을 더하면 sRGB 곡선 때문에 밝은 쪽 대비가 눌려 값이 1보다 작아질 수 있다(오라클도 같음).
    """
    def energy(v):
        y = (0.114 * v[..., 0] + 0.587 * v[..., 1] + 0.299 * v[..., 2]).astype(np.float32)
        lap = np.stack([np.abs(cv2.Laplacian(f, cv2.CV_32F, ksize=3)) for f in y])
        return float(lap[:, mask].mean()) if mask.any() else 0.0
    base = energy(clean_bgr)
    return energy(x_bgr) / base if base > 1e-6 else float("nan")


def light_cv(delta: np.ndarray, mask: np.ndarray) -> float:
    """빛 공간 변이계수: 빛이 켜진 프레임마다 마스크 안 Δ 의 std/mean 을 구해 평균. 0 = 균일(오라클)."""
    vals = []
    for f in delta:
        inside = f[mask]
        if inside.size and inside.mean() > 1e-3:
            vals.append(float(inside.std() / inside.mean()))
    return float(np.mean(vals)) if vals else 0.0


def oracle_baseline(bg: np.ndarray, masks: np.ndarray, fps: float, seed: int) -> dict:
    """같은 바탕·마스크에 오라클 빛(hazard 조건의 파형·세기)을 넣었을 때의 질감 보존율과 위험 비율."""
    rng = np.random.default_rng(seed)
    tex, danger = [], []
    for i in range(len(bg)):
        p = oracle.sample_params("hazard", rng)
        w = oracle.waveform(p, bg.shape[1], fps)
        lin = oracle.srgb_to_linear(bg[i]) + (w[:, None, None] * p.delta_cd / measure.WHITE_CD)[..., None] * masks[i][None, :, :, None]
        x = np.where(masks[i][None, :, :, None], oracle.linear_to_srgb(lin), bg[i])
        tex.append(texture_ratio(x, bg[i], masks[i]))
        danger.append(bool(measure.analyze(x, fps)["dangerous"]))
    return {"texture_ratio": float(np.nanmean(tex)), "light_cv": 0.0, "hazard": (sum(danger), len(danger))}


@torch.no_grad()
def evaluate(G, bg, masks, fps, z_per_clip, zdim, seed, save_dir=None):
    torch.manual_seed(seed)
    rows, manifest = [], []
    for i in range(len(bg)):
        clean = to_tensor(bg[i:i + 1], "cpu")
        m = torch.from_numpy(masks[i:i + 1]).float()[:, None, None]
        for target_name, tv in (("hazard", 1.0), ("near", 0.0)):
            for k in range(z_per_clip):
                z = torch.randn(1, zdim)
                out, delta = G(clean, m, torch.tensor([tv]), z)
                soft = float(soft_judge(luminance_rel(srgb_to_linear(out)), fps)["count"][0])
                x = np.clip(np.rint(out[0].permute(1, 2, 3, 0).numpy() * 255), 0, 255).astype(np.uint8)
                r = measure.analyze(x, fps)
                outside = int((x != bg[i])[:, ~masks[i]].sum())
                s = (delta[0, 0] * m[0, 0, 0]).sum(dim=(1, 2)).numpy() / max(1.0, float(masks[i].sum()))
                row = {"clip": i, "target": target_name, "z": k, "dangerous": bool(r["dangerous"]),
                       "hard_transitions": 2 * r["max_flashes"], "max_flashes": r["max_flashes"],
                       "soft_count": soft, "peak_area": r["peak_area"], "outside_changed": outside,
                       "dominant_hz": dominant_freq(s, fps), "mean_light_cd": float(s.mean() * measure.WHITE_CD),
                       "texture_ratio": texture_ratio(x, bg[i], masks[i]),
                       "light_cv": light_cv(delta[0, 0].numpy(), masks[i])}
                rows.append(row)
                if save_dir is not None:
                    sid = f"val_{i:03d}_{target_name}_z{k}"
                    manifest.append(save_sample(save_dir, sid, {
                        "clean": bg[i], "flicker": x, "mask_allow": masks[i],
                        "hazard_frames": r["hazard_frames"], "hazard_mask": r["hazard_mask"],
                        "light": (delta[0, 0].numpy() * masks[i]).astype(np.float16)},
                        {"split": "val", "purpose": "evaluation_only", "generator": "LightGenerator", "target": target_name, "z_index": k,
                         "label_dangerous": row["dangerous"], "max_flashes": row["max_flashes"],
                         "peak_area": row["peak_area"], "label_source": "lumi_synth.measure.analyze"}))
    return rows, manifest


def summarize(rows):
    def frac(cond, sub):
        sub = list(sub)
        return (sum(cond(r) for r in sub), len(sub))
    haz = [r for r in rows if r["target"] == "hazard"]
    near = [r for r in rows if r["target"] == "near"]
    soft, hard = np.array([r["soft_count"] for r in rows]), np.array([r["hard_transitions"] for r in rows])
    corr = float(np.corrcoef(soft, hard)[0, 1]) if soft.std() > 0 and hard.std() > 0 else float("nan")
    agree = frac(lambda r: (r["soft_count"] >= 7) == r["dangerous"], rows)
    # 다양성: (클립, 목표) 묶음마다 z 에 따른 표준편차의 평균
    groups = {}
    for r in rows:
        groups.setdefault((r["clip"], r["target"]), []).append(r)
    within_flash = float(np.mean([np.std([r["max_flashes"] for r in g]) for g in groups.values()]))
    within_hz = float(np.mean([np.std([r["dominant_hz"] for r in g]) for g in groups.values()]))
    return {
        "hazard_ok": frac(lambda r: r["dangerous"], haz),
        "near_ok": frac(lambda r: (not r["dangerous"]) and 1.0 <= r["max_flashes"] <= 3.0, near),
        "near_safe": frac(lambda r: not r["dangerous"], near),
        "soft_hard_corr": corr, "soft_hard_agree": agree,
        "within_std_flashes": within_flash, "within_std_hz": within_hz,
        "overall_std_hz": float(np.std([r["dominant_hz"] for r in rows])),
        "outside_changed_max": int(max(r["outside_changed"] for r in rows)),
        "mean_light_cd": float(np.mean([r["mean_light_cd"] for r in rows])),
        "texture_ratio": float(np.nanmean([r["texture_ratio"] for r in rows])),
        "light_cv": float(np.mean([r["light_cv"] for r in rows])),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="G 평가와 GAN 데이터셋 저장")
    ap.add_argument("--data", type=Path, default=ROOT / "data" / "oracle_128x72_24fps")
    ap.add_argument("--allow-mismatch", action="store_true", help="G 학습 해상도·fps 와 달라도 진행(기록됨)")
    ap.add_argument("--checkpoint", type=Path, default=ROOT / "data" / "gan" / "gan.pt")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "gan")
    ap.add_argument("--scale", type=float, default=None, help="생략하면 학습 때 배율(체크포인트)을 그대로")
    ap.add_argument("--z-per-clip", type=int, default=4)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--report", type=Path, default=ROOT / "checks" / "gan" / "README.md",
                    help="점검표 경로(실험마다 다르게 두면 이전 결과를 덮어쓰지 않음)")
    args = ap.parse_args()

    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    scale = args.scale if args.scale is not None else float(cfg.get("scale", 1.0))
    bg, fps = load_backgrounds(args.data, "val")
    bg = rescale_clips(bg, scale)
    check_track(cfg, bg, fps, args.allow_mismatch)
    rng = np.random.default_rng(args.seed)
    masks = sample_masks(len(bg), bg.shape[2], bg.shape[3], rng)

    trained = LightGenerator(int(cfg["width"]), int(cfg["zdim"]), fps)
    trained.load_state_dict(ck["generator"])
    trained.eval()
    torch.manual_seed(0)
    untrained = LightGenerator(int(cfg["width"]), int(cfg["zdim"]), fps).eval()   # 대조군

    t0 = time.time()
    rows_t, manifest = evaluate(trained, bg, masks, fps, args.z_per_clip, int(cfg["zdim"]), args.seed,
                                save_dir=args.out / "val")
    rows_u, _ = evaluate(untrained, bg, masks, fps, args.z_per_clip, int(cfg["zdim"]), args.seed)
    st, su = summarize(rows_t), summarize(rows_u)
    so = oracle_baseline(bg, masks, fps, args.seed)
    with open(args.out / "manifest.jsonl", "w", encoding="utf-8") as f:
        for row in manifest:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    (ROOT / "checks" / "gan").mkdir(parents=True, exist_ok=True)
    args.report.with_name(args.report.stem + "_rows.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows_t) + "\n", encoding="utf-8")
    write_report(args, cfg, ck, bg, st, su, so, time.time() - t0)
    print(json.dumps({"trained": st, "untrained": su}, ensure_ascii=False, indent=1))
    return 0


def write_report(args, cfg, ck, bg, st, su, so, secs):
    f = lambda p: f"{p[0]}/{p[1]} ({100 * p[0] / max(1, p[1]):.0f}%)"
    # 학습 기록은 체크포인트 안의 것을 우선 쓴다(다른 실행의 train_log.jsonl 과 섞이지 않게)
    log = ck.get("train_log") or [json.loads(x) for x in
                                  (ROOT / "checks" / "gan" / "train_log.jsonl").read_text(encoding="utf-8").splitlines() if x]
    first, last = log[0], log[-1]
    lines = [
        "# GAN 점검표 (검증 분할, 하드 판정으로 측정)", "",
        f"- 만든 날: {time.strftime('%Y-%m-%d %H:%M')} · 평가 {secs:.0f}초",
        f"- 학습: {cfg['steps']}스텝, 배치 {cfg['batch_size']}, 폭 {cfg['width']}, z {cfg['zdim']}, "
        f"λ(판정/적대/다양성) = {cfg['lambda_judge']}/{cfg['lambda_adv']}/{cfg['lambda_div']}, "
        f"해상도 배율 {cfg.get('scale', 1.0)}, 장치 {cfg['device']}, 총 {last['elapsed_s']}초",
        f"- 평가: 검증 바탕 {len(bg)}개 [{bg.shape[1]}프레임 × {bg.shape[3]}×{bg.shape[2]}, {cfg['fps']} fps] × 목표 2종 × z {args.z_per_clip}개",
        f"- 데이터 트랙: `{args.data.name}` (G 학습 트랙: `{cfg.get('data_track', '기록 없음(옛 체크포인트)')}`)",
        f"- 대조군: 같은 구조의 학습하지 않은 G",
        f"- 학습에 쓴 소프트 판정기: {cfg.get('soft_judge_version', 'v0 (리뷰 전: 반대 위상 2번 셈, +0.8 과대)')} · "
        f"평가의 소프트↔하드 일치는 현재 코드의 소프트 판정기 {__import__('lumi_synth.gan', fromlist=['x']).SOFT_JUDGE_VERSION} 로 잼",
        f"- 저장한 GAN 샘플은 **검증(val) 바탕에서 만든 평가용**이다. 탐지 모델 학습에 넣으면 val 누수가 된다.", "",
        "| 지표 | 학습한 G | 학습 안 한 G | 뜻 |", "|---|---|---|---|",
        f"| 목표 '위험' → 실제 위험 | {f(st['hazard_ok'])} | {f(su['hazard_ok'])} | 조절력 |",
        f"| 목표 '근처 안전' → 안전 & 1~3회/초 | {f(st['near_ok'])} | {f(su['near_ok'])} | 조절력(경계 사례) |",
        f"| 목표 '근처 안전' → 안전 | {f(st['near_safe'])} | {f(su['near_safe'])} | |",
        f"| 소프트↔하드 전이 수 상관 | {st['soft_hard_corr']:.2f} | {su['soft_hard_corr']:.2f} | 대리 판정기 신뢰도 |",
        f"| 소프트↔하드 위험 판정 일치 | {f(st['soft_hard_agree'])} | {f(su['soft_hard_agree'])} | |",
        f"| z 만 바꿨을 때 깜빡임 수 표준편차 | {st['within_std_flashes']:.2f} | {su['within_std_flashes']:.2f} | 다양성 |",
        f"| z 만 바꿨을 때 주 주파수 표준편차(Hz) | {st['within_std_hz']:.2f} | {su['within_std_hz']:.2f} | 다양성 |",
        f"| 마스크 밖 바뀐 값 개수(최대) | {st['outside_changed_max']} | {su['outside_changed_max']} | 국소성(0이어야 함) |",
        f"| 마스크 안 평균 추가 빛(cd/m²) | {st['mean_light_cd']:.1f} | {su['mean_light_cd']:.1f} | |", "",
        "### 자연스러움 (코드 지표만 — 사람이 보고 평가하지 않음)", "",
        "| 지표 | 학습한 G | 학습 안 한 G | 오라클(같은 바탕·마스크) | 뜻 |", "|---|---|---|---|---|",
        f"| 질감 보존율 | {st['texture_ratio']:.3f} | {su['texture_ratio']:.3f} | {so['texture_ratio']:.3f} | 1 = 원본 윤곽선 그대로 |",
        f"| 빛 공간 변이계수 | {st['light_cv']:.3f} | {su['light_cv']:.3f} | 0 (균일) | 0 보다 크면 빛이 공간적으로 퍼짐 |",
        f"| (참고) 오라클 hazard 조건 → 위험 | | | {so['hazard'][0]}/{so['hazard'][1]} | 같은 마스크에서의 기준선 |", "",
        "## 학습 기록 (처음 → 마지막)", "",
        f"- 소프트 전이 수: 위험 목표 {first['soft_count_hazard']} → {last['soft_count_hazard']}, "
        f"근처 목표 {first['soft_count_near']} → {last['soft_count_near']} (목표: 위험 ≥ 8, 근처 2~5)",
        f"- 판정 손실 {first['judge']} → {last['judge']}, D 손실 {first['loss_d']} → {last['loss_d']}", "",
        "## 해석 시 주의", "",
        "- 노트북 CPU 동작 확인용 짧은 학습이다. 수치는 이 설정에서의 실측값이며 일반화된 성능 주장이 아니다.",
        "- 자연스러움은 코드 지표(질감 보존율, 빛 공간 변이계수)로만 잰다. 위험 자극이라 사람이 보고 평가하지 않는다. 실제 영상과의 분포 거리는 아직 없다.",
        "- 정답 라벨은 목표가 아니라 하드 판정 결과다. 목표와 다른 샘플도 판정 결과대로 라벨이 붙는다.",
    ]
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
