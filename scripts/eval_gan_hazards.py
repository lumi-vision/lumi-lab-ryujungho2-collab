"""종류별 GAN(휘도·적색·줄무늬) 평가: 검증 바탕에서 만들어 하드 판정으로 다시 잰다. 영상은 저장하지 않는다.

eval_gan.py(휘도 전용, 샘플 96개를 npz 로 저장)와 달리 판정 수치만 남긴다. 영상을 보려면 플레이어의 '생성…' 을 쓴다.

하드 판정 (학습에 쓴 소프트·대리 판정이 아니라 measure 의 판정기)
    lum     measure.analyze            1초 화면 전이 ≥ 7 이면 위험
    red     measure.analyze_red        적색 전이 ≥ 7 이면 위험
    stripe  stripe_freq 쿼드트리+가버   줄무늬 면적 > 40% 가 1초 창 안 0.5초 이상이면 위험 (규칙 상수는 measure 와 같음).
            --scanline N 을 주면 앞에서 N 개는 기준 판정기 measure.analyze_pattern(스캔선, 클립당 1~4분)으로도 잰다.
재는 것
    조절력    목표 '위험' → 그 종류로 위험 판정 / 목표 '근처 안전' → 그 종류로 안전이면서 '보이는' 것
              (휘도·적색: 초당 1~3회, 줄무늬: 최대 면적 ≥ 15%)
    대리 일치  학습 손실의 값(소프트 전이 수 ≥ 7, 줄무늬 지속 ≥ 0.5초)과 하드 판정의 위험 여부가 같은 비율
    다양성    같은 (바탕, 영역, 목표)에서 z 만 바꿨을 때의 표준편차 — 휘도·적색: 주 주파수(Hz)·깜빡임 수,
              줄무늬: cpd(로그), 방향(도), 속도(°/s)
    부수 효과  다른 종류의 위험이 함께 생긴 비율(적색·줄무늬에서 휘도 깜빡임 위험 등), 마스크 밖 바뀐 값(0 이어야 함)
사용: python scripts/eval_gan_hazards.py --checkpoint data/gan/<이름>.pt [--z-per-clip 4] [--scanline 0]
결과: checks/gan/<이름>.md, checks/gan/<이름>_rows.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from lumi_synth import gan, measure, stripe_freq as sf  # noqa: E402
from train_gan import check_track, load_backgrounds, sample_masks, to_tensor  # noqa: E402

KO = {"lum": "휘도 깜빡임", "red": "적색 깜빡임", "stripe": "줄무늬"}


def dominant_freq(series: np.ndarray, fps: float) -> float:
    s = series - series.mean()
    if np.abs(s).max() < 1e-6:
        return 0.0
    spec = np.abs(np.fft.rfft(s))
    return float(np.fft.rfftfreq(len(s), 1.0 / fps)[int(spec[1:].argmax()) + 1])


@torch.no_grad()
def evaluate(G, hazard: str, bg: np.ndarray, masks: np.ndarray, fps: float, z_per_clip: int, zdim: int, seed: int,
             scanline: int, device) -> list[dict]:
    torch.manual_seed(seed)
    rows = []
    for i in range(len(bg)):
        clean = to_tensor(bg[i:i + 1], device)
        m = torch.from_numpy(masks[i:i + 1]).to(device).float()[:, None, None]
        for target_name, tv in (("hazard", 1.0), ("near", 0.0)):
            for k in range(z_per_clip):
                z = torch.randn(1, zdim, device=device)
                outs = G(clean, m, torch.tensor([tv], device=device), z)
                out, field = outs[0], outs[1]
                x = (out[0].permute(1, 2, 3, 0) * 255.0).round().clamp(0, 255).byte().cpu().numpy()
                fl, rd = measure.analyze(x, fps), measure.analyze_red(x, fps)
                series = ((field[0, 0] * m[0, 0, 0]).sum(dim=(1, 2)) / max(1.0, float(masks[i].sum()))).cpu().numpy()
                row = {"clip": i, "target": target_name, "z": k, "mask_area": float(masks[i].mean()),
                       "outside_changed": int((x != bg[i])[:, ~masks[i]].sum()),
                       "lum_dangerous": bool(fl["dangerous"]), "lum_flashes": fl["max_flashes"], "lum_area": fl["peak_area"],
                       "red_dangerous": bool(rd["dangerous"]), "red_flashes": rd["max_flashes"], "red_area": rd["peak_area"],
                       "dominant_hz": dominant_freq(series, fps), "mean_field": float(series.mean())}
                if hazard == "lum":
                    soft = float(gan.soft_judge(gan.luminance_rel(gan.srgb_to_linear(out)), fps)["count"][0])
                    row.update(dangerous=row["lum_dangerous"], flashes=fl["max_flashes"], soft=soft, soft_dangerous=soft >= 7,
                               visible=1.0 <= fl["max_flashes"] <= 3.0, side=row["red_dangerous"])
                elif hazard == "red":
                    soft = float(gan.soft_judge_red(gan.srgb_to_linear(out), fps)["count"][0])
                    row.update(dangerous=row["red_dangerous"], flashes=rd["max_flashes"], soft=soft, soft_dangerous=soft >= 7,
                               visible=1.0 <= rd["max_flashes"] <= 3.0, side=row["lum_dangerous"])
                else:
                    cpd, theta, speed, hi = (float(v) for v in outs[2][0][:4])
                    pt = sf.analyze_pattern_freq(x, fps, model=None, method="quadtree", refine=True)
                    hi_t = outs[2][:, 3]
                    if outs[2].shape[1] > 4:
                        hi_t = gan.STRIPE_LO + (hi_t[:, None] - gan.STRIPE_LO) * gan.stripe_reversal(outs[2], field.shape[2], fps).abs()
                    soft = float(gan.soft_judge_stripe(m * field, hi_t, fps)["duration"][0])
                    row.update(dangerous=bool(pt["dangerous"]), stripe_area=pt["peak_stripe_area"], stripe_sec=pt["max_stripe_sec"],
                               soft=soft, soft_dangerous=soft >= measure.PATTERN_MIN_SEC, visible=pt["peak_stripe_area"] >= 0.15,
                               side=row["lum_dangerous"] or row["red_dangerous"],
                               cpd=cpd, angle=float(np.degrees(theta)), speed_dps=speed, hi_cd=hi * measure.WHITE_CD)
                    mb = m[0, 0, 0].bool()                                   # 덮는 정도가 시간에 따라 얼마나 바뀌나(정지 줄무늬 확인)
                    a = field[0, 0][:, mb]
                    row["cover_dt"] = float((a[1:] - a[:-1]).abs().mean())
                    if outs[2].shape[1] > 4:
                        row["motion"] = ("static", "reversing", "drifting")[int(outs[2][0][7])]
                    if outs[2].shape[1] >= 13:                       # v5: 무늬 종류
                        row["pattern"] = gan.PATTERN_SHAPES[int(outs[2][0][9])]
                    if len([r for r in rows if "scan_dangerous" in r]) < scanline:
                        sc = measure.analyze_pattern(x, fps)
                        row.update(scan_dangerous=bool(sc["dangerous"]), scan_area=float(sc["peak_stripe_area"]))
                rows.append(row)
        print(f"  바탕 {i + 1}/{len(bg)}", flush=True)
    return rows


def summarize(rows: list[dict], hazard: str) -> dict:
    haz = [r for r in rows if r["target"] == "hazard"]
    near = [r for r in rows if r["target"] == "near"]
    groups: dict = {}
    for r in rows:
        groups.setdefault((r["clip"], r["target"]), []).append(r)

    def within(key, f=lambda v: v):
        return float(np.mean([np.std([f(r[key]) for r in g]) for g in groups.values()]))
    s = {"n": len(rows),
         "hazard_ok": (sum(r["dangerous"] for r in haz), len(haz)),
         "near_ok": (sum((not r["dangerous"]) and r["visible"] for r in near), len(near)),
         "near_safe": (sum(not r["dangerous"] for r in near), len(near)),
         "agree": (sum(r["soft_dangerous"] == r["dangerous"] for r in rows), len(rows)),
         "side": (sum(r["side"] for r in rows), len(rows)),
         "outside_changed_max": max(r["outside_changed"] for r in rows)}
    if hazard == "stripe":
        ang = [np.radians(2 * r["angle"]) for r in rows]          # 방향은 180° 주기 → 2θ 의 원형 분산
        s.update(std_cpd_log=within("cpd", np.log), std_speed=within("speed_dps"), std_hi=within("hi_cd"),
                 cpd_range=(min(r["cpd"] for r in rows), max(r["cpd"] for r in rows)),
                 angle_spread=float(1 - np.hypot(np.mean(np.cos(ang)), np.mean(np.sin(ang)))),
                 static_share=float(np.mean([abs(r["speed_dps"]) < 1e-6 for r in rows])))
        if "motion" in rows[0]:                                   # v2+: 움직임별 판정과 덮는 정도의 프레임 간 변화
            s["by_motion"] = {mo: {"hazard_ok": (sum(r["dangerous"] for r in haz if r["motion"] == mo), sum(r["motion"] == mo for r in haz)),
                                   "near_ok": (sum((not r["dangerous"]) and r["visible"] for r in near if r["motion"] == mo),
                                               sum(r["motion"] == mo for r in near)),
                                   "cover_dt": float(np.median([r["cover_dt"] for r in rows if r["motion"] == mo] or [np.nan]))}
                              for mo in ("static", "reversing", "drifting")}
        if "pattern" in rows[0]:
            s["by_pattern"] = {pa: {"hazard_ok": (sum(r["dangerous"] for r in haz if r["pattern"] == pa), sum(r["pattern"] == pa for r in haz)),
                                    "near_ok": (sum((not r["dangerous"]) and r["visible"] for r in near if r["pattern"] == pa),
                                                sum(r["pattern"] == pa for r in near)),
                                    "side": (sum(r["side"] for r in rows if r["pattern"] == pa), sum(r["pattern"] == pa for r in rows))}
                               for pa in gan.PATTERN_SHAPES}
        sc = [r for r in rows if "scan_dangerous" in r]
        if sc:
            s["scan_agree"] = (sum(r["scan_dangerous"] == r["dangerous"] for r in sc), len(sc))
    else:
        s.update(std_hz=within("dominant_hz"), std_flashes=within("flashes"),
                 flashes_range=(min(r["flashes"] for r in haz), max(r["flashes"] for r in haz)))
    return s


def write_report(path: Path, name: str, cfg: dict, hazard: str, st: dict, su: dict, n_bg: int, zpc: int, secs: float) -> None:
    f = lambda c: f"{c[0]}/{c[1]} ({100 * c[0] / max(c[1], 1):.0f}%)"          # noqa: E731
    rule = cfg.get("judge_rule") or {}
    L = [f"# GAN 평가 — {KO[hazard]} · `{name}`", "",
         f"> {time.strftime('%Y-%m-%d %H:%M')} · `scripts/eval_gan_hazards.py` · {secs / 60:.1f}분 · 검증 바탕 {n_bg}개 × 목표 2 × z {zpc} = {st['n']}개 · **하드 판정**",
         f"> 학습: {cfg.get('steps')}스텝, 배치 {cfg.get('batch_size')}, 폭 {cfg.get('width')}, z {cfg.get('zdim')}, "
         f"λ(판정/적대/다양성) {cfg.get('lambda_judge')}/{cfg.get('lambda_adv')}/{cfg.get('lambda_div')}, 시드 {cfg.get('seed')}, "
         f"{cfg.get('train_hw', ['?', '?'])[1]}×{cfg.get('train_hw', ['?', '?'])[0]} {cfg.get('fps')} fps, 동시성 창 {rule.get('d3_sync_ms', '기록 없음')} ms",
         "> 학습 안 한 G = 같은 구조·무작위 가중치(대조군). 영상은 저장하지 않았다(보려면 플레이어의 '생성…').", "",
         "| 지표 | 학습한 G | 학습 안 한 G |", "|---|---|---|",
         f"| 목표 '위험' → {KO[hazard]} 위험 판정 | **{f(st['hazard_ok'])}** | {f(su['hazard_ok'])} |",
         f"| 목표 '근처 안전' → 안전이면서 보임 | **{f(st['near_ok'])}** | {f(su['near_ok'])} |",
         f"| 목표 '근처 안전' → 안전 | {f(st['near_safe'])} | {f(su['near_safe'])} |",
         f"| 대리 판정 ↔ 하드 판정 위험 여부 일치 | {f(st['agree'])} | {f(su['agree'])} |",
         f"| 다른 종류의 위험이 함께 생김 | {f(st['side'])} | {f(su['side'])} |",
         f"| 마스크 밖 바뀐 값(최대) | {st['outside_changed_max']} | {su['outside_changed_max']} |"]
    if hazard == "stripe":
        L += [f"| z 만 바꿨을 때 cpd 표준편차 (로그) | {st['std_cpd_log']:.3f} | {su['std_cpd_log']:.3f} |",
              f"| z 만 바꿨을 때 속도 표준편차 (°/s) | {st['std_speed']:.2f} | {su['std_speed']:.2f} |",
              f"| z 만 바꿨을 때 밝은 줄 휘도 표준편차 (cd/m²) | {st['std_hi']:.1f} | {su['std_hi']:.1f} |",
              f"| 만든 cpd 범위 | {st['cpd_range'][0]:.2f} ~ {st['cpd_range'][1]:.2f} | {su['cpd_range'][0]:.2f} ~ {su['cpd_range'][1]:.2f} |",
              f"| 방향 퍼짐 (0 = 한 방향, 1 = 고르게) | {st['angle_spread']:.2f} | {su['angle_spread']:.2f} |",
              f"| 정지 줄무늬 비율 | {st['static_share']:.0%} | {su['static_share']:.0%} |"]
        if "scan_agree" in st:
            L.append(f"| 스캔선 판정(measure.analyze_pattern)과 쿼드트리 판정 일치 | {f(st['scan_agree'])} | – |")
        if "by_motion" in st:
            ko = {"static": "정지", "reversing": "명암 반전", "drifting": "흐름"}
            L += ["", "움직임별 (학습한 G) — '덮는 정도 프레임 간 변화'는 마스크 안 |Δa| 중앙값(0 = 시간에 따라 일정)", "",
                  "| 움직임 | 목표 위험 → 위험 | 근처 → 안전이면서 보임 | 덮는 정도 프레임 간 변화 |", "|---|---|---|---|"]
            for mo, v in st["by_motion"].items():
                L.append(f"| {ko[mo]} | {f(v['hazard_ok'])} | {f(v['near_ok'])} | {v['cover_dt']:.3f} |")
        if "by_pattern" in st:
            L += ["", "무늬 종류별 (학습한 G) — 판정은 쿼드트리+가버. 체커는 이 판정기가 줄무늬로 보지 않는다(기준 해석 미결)", "",
                  "| 무늬 | 목표 위험 → 줄무늬 위험 | 근처 → 안전이면서 보임 | 휘도·적색 위험 동반 |", "|---|---|---|---|"]
            for pa, v in st["by_pattern"].items():
                L.append(f"| {gan.PATTERN_KO[pa]} | {f(v['hazard_ok'])} | {f(v['near_ok'])} | {f(v['side'])} |")
    else:
        L += [f"| z 만 바꿨을 때 주 주파수 표준편차 (Hz) | {st['std_hz']:.2f} | {su['std_hz']:.2f} |",
              f"| z 만 바꿨을 때 깜빡임 수 표준편차 | {st['std_flashes']:.2f} | {su['std_flashes']:.2f} |",
              f"| '위험' 목표의 깜빡임 수 범위 (회/초) | {st['flashes_range'][0]:g} ~ {st['flashes_range'][1]:g} | {su['flashes_range'][0]:g} ~ {su['flashes_range'][1]:g} |"]
    L += ["", "- '보임': 휘도·적색은 초당 1~3회, 줄무늬는 최대 면적 15% 이상. 근처 안전이 '아무것도 안 넣음'으로 풀리지 않았는지 보는 조건.",
          "- 줄무늬의 하드 판정은 쿼드트리+가버(`stripe_freq`)다. 기준 판정기는 스캔선(`measure.analyze_pattern`)이고 `--scanline` 으로 일부만 대조한다.",
          "- 시드 1개, 한 영화(BBB)의 검증 바탕이다. 수치는 이 설정의 동작 확인이고 일반화 주장이 아니다."]
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--data", type=Path, default=ROOT / "data" / "oracle_768x432_24fps")
    ap.add_argument("--z-per-clip", type=int, default=4)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--scanline", type=int, default=0, help="줄무늬: 앞에서 이 수만큼 스캔선 판정으로도 잰다(클립당 1~4분)")
    ap.add_argument("--report", type=Path, default=None, help="생략하면 checks/gan/<체크포인트 이름>.md")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    hazard = cfg.get("hazard", "lum")
    bg, fps = load_backgrounds(args.data, "val")
    check_track(cfg, bg, fps, False)
    masks = sample_masks(len(bg), bg.shape[2], bg.shape[3], np.random.default_rng(args.seed), hazard)
    dev = torch.device(args.device)
    trained = gan.make_generator(hazard, int(cfg["width"]), int(cfg["zdim"]), fps, red_palette=cfg.get("red_palette", False),
                                 lum_smooth=bool(cfg.get("lum_smooth", False)), stripe_cond=cfg.get("stripe_cond", "mlp"))
    trained.load_state_dict(ck["generator"])
    torch.manual_seed(0)
    untrained = gan.make_generator(hazard, int(cfg["width"]), int(cfg["zdim"]), fps, red_palette=cfg.get("red_palette", False),
                                   lum_smooth=bool(cfg.get("lum_smooth", False)), stripe_cond=cfg.get("stripe_cond", "mlp"))
    t0 = time.time()
    rows_t = evaluate(trained.eval().to(dev), hazard, bg, masks, fps, args.z_per_clip, int(cfg["zdim"]), args.seed, args.scanline, dev)
    rows_u = evaluate(untrained.eval().to(dev), hazard, bg, masks, fps, args.z_per_clip, int(cfg["zdim"]), args.seed, 0, dev)
    st, su = summarize(rows_t, hazard), summarize(rows_u, hazard)
    report = args.report or ROOT / "checks" / "gan" / f"{args.checkpoint.stem}.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.with_name(report.stem + "_rows.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows_t) + "\n", encoding="utf-8")
    write_report(report, args.checkpoint.name, cfg, hazard, st, su, len(bg), args.z_per_clip, time.time() - t0)
    print(json.dumps({"hazard": hazard, "trained": st, "untrained": su}, ensure_ascii=False, indent=1))
    print("점검표:", report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
