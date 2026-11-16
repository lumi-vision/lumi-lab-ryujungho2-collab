"""4단계 평가(2026-11-15 개정): 보정 방식 E0~E3 와 기준선을 같은 표본·같은 지표로 비교한다.

    python scripts/eval_pipeline.py --data data/gdr/v6/val data/gdr/v6/val_rule \\
        --roi data/gdr/v6/roi_c.pt --cor data/gdr/v6/cor_b.pt [--e0 data/gdr/v6/cor_e0.pt] \\
        [--e3-cor data/gdr/v6/cor_e3.pt --loss-net data/gdr/v6/loss_c.pt] --tag v6

방법 (사용자 결정 문서 2026-11-15 의 실험 구성)
    before   입력 그대로
    global   전체 감광 기준선: 위험 프레임에서 화면 전체를 시간 평균(휘도·적색) / 흐림(줄무늬) — '국소 보정'과 비교하는 대상
    rule     국소 규칙 보정: 판정기 튜브 안에서만 같은 처리
    E0       마스크 없는 보정기(End-to-End): 화면 전체를 입력으로, 어디를 고칠지 스스로 정함
    E1       보정기 + 판정기 튜브 (마스크를 완벽히 안다고 가정한 상한)
    E2       보정기 + 탐지기 예측 튜브 (실전 흐름)
    E3       보정기(손실 지도 입력) + 탐지기 튜브 + 손실 추정기의 M_loss
평가 체계
    Safety      보정 뒤 판정기 재검사: 위험 잔존(클립), 위험 프레임 비율, 새로 생긴 위험 종류. 판정기 = G 학습에 쓴 규칙과 같은
                구현이므로 독립 검증이 아니다(팀원 판별기 대조는 별도)
    Fidelity    원본과 PSNR·SSIM·LPIPS — 영역별: 위험 전체 / 복원 가능(α<0.95) / 시간 복원 가능 / 정보 없음 / 화면 전체
    Spatial     이음새(튜브 경계 기울기 비율), 위험 영역 밖 변경 픽셀 비율
    Temporal    워핑 오차 비율(원본 흐름 기준, 출력/원본), 프레임 간 변화량 비율, 질감
    Efficiency  탐지기·보정기 처리 시간(p50/p95), 초당 프레임, GPU 최대 메모리
판정기가 안전이라 한 표본: E0·E2·E3 가 영상을 얼마나 건드렸는지(바뀐 픽셀 비율, 입력 대비 PSNR).
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import time
from pathlib import Path

import numpy as np
import torch

from eval_roi import KO, load_roi
from gdr_common import ROOT, group_of, predict_tubes, prefetch, read_manifest, require_split, to_x, upsample_tubes
from lumi_synth import quality as q
from lumi_synth import roi

METHODS = ("before", "global", "rule", "E0", "E1", "E2", "E3")
METHOD_KO = {"before": "보정 전", "global": "전체 감광 (기준선)", "rule": "국소 규칙 보정", "E0": "E0 마스크 없는 보정기",
             "E1": "E1 보정기 + 판정기 튜브", "E2": "E2 보정기 + 탐지기 튜브", "E3": "E3 E2 + 손실 추정"}
REGIONS = ("hazard", "recoverable", "lost_temporal", "lost_never", "frame")
REGION_KO = {"hazard": "위험 전체", "recoverable": "복원 가능", "lost_temporal": "시간 복원 가능", "lost_never": "정보 없음",
             "frame": "화면 전체"}


def load_cor(path: Path, device):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    net = roi.Corrector(int(cfg["width"]), loss_in=bool(cfg.get("loss_input", False)))
    net.load_state_dict(ck["model"])
    return net.to(device).eval(), cfg


@torch.no_grad()
def run_cor(net, x_u8, tubes, device, lmap=None, no_mask=False):
    """→ (출력 uint8, 닿을 수 있는 영역, 걸린 초). no_mask: 튜브 = 화면 전체(E0)."""
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    if no_mask:
        m = torch.ones((1, 3) + x_u8.shape[:3], device=device)
    else:
        m = roi.soften(torch.from_numpy(tubes[None]).to(device))
    lm = None if lmap is None else torch.from_numpy(lmap[None, None]).to(device).float()
    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
        y = net(to_x([x_u8], device), m, lm)
    y8 = (y.float()[0].permute(1, 2, 3, 0) * 255.0).round().clamp(0, 255).byte().cpu().numpy()
    if device.type == "cuda":
        torch.cuda.synchronize()
    reach = (m.amax(dim=1)[0] > 0).cpu().numpy()
    y8[~reach] = x_u8[~reach]
    return y8, reach, time.perf_counter() - t0


def global_tubes(tubes: np.ndarray) -> np.ndarray:
    """전체 감광: 종류별로 위험 프레임이면 화면 전체를 튜브로."""
    frames = tubes.any(axis=(2, 3))                                       # 3×T
    return np.broadcast_to(frames[:, :, None, None], tubes.shape).copy()


def judge(frames: np.ndarray, fps: float) -> dict:
    tubes, su = roi.judge_tubes(frames, fps)
    return {"dangerous": bool(su["dangerous"]), "kinds": {k: bool(su[k]["dangerous"]) for k in roi.HAZARDS},
            "hazard_frames": float(roi.stack_tubes(tubes).any(axis=(0, 2, 3)).mean())}


def measure_quality(y, x, clean, reg, maps, device) -> dict:
    lp = q.lpips_map(y, clean, device.type)
    ss = q.ssim_map(y, clean)
    out = {}
    for name in REGIONS:
        w = np.ones(reg["hazard"].shape, bool) if name == "frame" else reg[name]
        out[name] = {"psnr": q.psnr(y, clean, w), "ssim": q.masked_mean(ss, w), "lpips": q.masked_mean(lp, w),
                     "frac": float(w.mean())}
    hz = reg["hazard"]
    we_y, we_c = q.warp_error(y, maps, hz), q.warp_error(clean, maps, hz)
    out["temporal"] = {"warp_ratio": we_y / max(we_c, 1e-6) if we_y == we_y else float("nan"),
                       "diff_ratio": roi.temporal_ratio(y, clean, hz), "texture": roi.texture_ratio(y, clean, hz),
                       "seam": roi.seam_ratio(y, clean, hz)}
    out["outside_changed"] = float(np.any(y != x, axis=-1)[~hz].mean()) if (~hz).any() else 0.0
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, type=Path, nargs="+")
    ap.add_argument("--roi", required=True, type=Path)
    ap.add_argument("--cor", required=True, type=Path)
    ap.add_argument("--e0", type=Path, default=None)
    ap.add_argument("--e3-cor", type=Path, default=None)
    ap.add_argument("--loss-net", type=Path, default=None)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    det = load_roi(args.roi, device)[0]
    cor, ccfg = load_cor(args.cor, device)
    e0 = load_cor(args.e0, device)[0] if args.e0 else None
    e3 = load_cor(args.e3_cor, device)[0] if args.e3_cor else None
    lnet = load_roi(args.loss_net, device)[0] if args.loss_net else None
    out_dir = ROOT / "checks" / "gdr"

    for folder in args.data:
        metas = read_manifest(folder)
        if args.limit:
            metas = metas[:args.limit]
        require_split(metas, "eval")
        t0, rows, futs, times = time.time(), [], [], {"det": [], "cor": [], "loss": []}
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        with cf.ProcessPoolExecutor(max_workers=args.workers) as pool:
            for i, s in enumerate(prefetch(metas)):
                x, clean, tubes, fps = s["x"], s["clean"], s["tubes"], s["fps"]
                row = {"id": s["meta"]["id"], "group": group_of(s["meta"]),
                       "true": {k: bool(tubes[j].any()) for j, k in enumerate(roi.HAZARDS)}, "m": {}}
                if device.type == "cuda":
                    torch.cuda.synchronize()
                td = time.perf_counter()
                pred = upsample_tubes(predict_tubes(det, x, device), x.shape[1:3])
                if device.type == "cuda":
                    torch.cuda.synchronize()
                times["det"].append(time.perf_counter() - td)
                lmap = None
                if lnet is not None:
                    tl = time.perf_counter()
                    lmap = upsample_tubes(predict_tubes(lnet, x, device), x.shape[1:3])[0].astype(np.float32)
                    times["loss"].append(time.perf_counter() - tl)
                outs = {}
                if tubes.any():
                    tdict = dict(zip(roi.HAZARDS, tubes))
                    outs["before"] = (x, None)
                    outs["global"] = (roi.rule_correct(x, dict(zip(roi.HAZARDS, global_tubes(tubes))), fps), None)
                    outs["rule"] = (roi.rule_correct(x, tdict, fps), None)
                    y, reach, dt = run_cor(cor, x, tubes, device)
                    outs["E1"] = (y, reach)
                    times["cor"].append(dt)
                outs["E2"] = run_cor(cor, x, pred, device)[:2] if pred.any() else (x, np.zeros(x.shape[:3], bool))
                if e0 is not None:
                    outs["E0"] = run_cor(e0, x, None, device, no_mask=True)[:2]
                if e3 is not None:
                    outs["E3"] = run_cor(e3, x, pred, device, lmap=lmap)[:2] if pred.any() else (x, np.zeros(x.shape[:3], bool))
                if tubes.any():
                    reg = q.regions(tubes, s["alpha"])
                    reg_frac = {k: float(v.mean()) for k, v in reg.items()}
                    row["regions"] = reg_frac
                    maps = q.clean_flow(clean)
                    for name, (y, _) in outs.items():
                        row["m"][name] = measure_quality(y, x, clean, reg, maps, device)
                        if name != "before":
                            futs.append((row, name, pool.submit(judge, y, fps)))
                else:                                            # 안전한 표본: 건드린 정도
                    row["safe"] = {name: {"changed_frac": float(np.any(y != x, axis=-1).mean()),
                                          "psnr_vs_input": q.psnr(y, x, np.ones(x.shape[:3], bool))}
                                   for name, (y, _) in outs.items() if name in ("E0", "E2", "E3")}
                rows.append(row)
                while sum(not f.done() for _, _, f in futs) > args.workers * 2:
                    time.sleep(0.5)
                if (i + 1) % 10 == 0:
                    print(f"  {folder.name} {i + 1}/{len(metas)}  {time.time() - t0:.0f}초", flush=True)
            for row, name, fu in futs:
                row["m"][name]["after"] = fu.result()
        eff = {k: {"p50": float(np.percentile(v, 50)), "p95": float(np.percentile(v, 95)), "n": len(v)} for k, v in times.items() if v}
        if device.type == "cuda":
            eff["gpu_peak_GB"] = round(torch.cuda.max_memory_allocated() / 2 ** 30, 2)
        name = f"{args.tag}_pipeline_{folder.name}"
        (out_dir / f"{name}_rows.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
        write_report(out_dir / f"{name}.md", args, folder, ccfg, rows, eff, time.time() - t0)
        print(f"저장: checks/gdr/{name}.md  ({time.time() - t0:.0f}초)", flush=True)
    return 0


def _m(vals) -> str:
    v = [x for x in vals if x == x and np.isfinite(x)]
    return f"{np.median(v):.3f}" if v else "–"


def write_report(path: Path, args, folder: Path, ccfg: dict, rows: list[dict], eff: dict, secs: float) -> None:
    haz = [r for r in rows if r["m"]]
    methods = [m for m in METHODS if any(m in r["m"] for r in haz)]
    L = [f"# 보정 파이프라인 평가 — `{folder.parent.name}/{folder.name}`", "",
         f"- 탐지기 `{args.roi.name}` · 보정기 `{args.cor.name}`" + (f" · E0 `{args.e0.name}`" if args.e0 else "")
         + (f" · E3 `{args.e3_cor.name}` + 손실 추정기 `{args.loss_net.name}`" if args.e3_cor else ""),
         f"- 표본 {len(rows)}개 (판정기 기준 위험 {len(haz)}개) · {secs:.0f}초 · 값은 위험 표본의 중앙값, 안전 칸은 개수",
         "- 정보 손실 영역: 합성 비중 α ≥ 0.95(실험적 문턱). 판정기는 G 학습 규칙과 같은 구현이라 독립 검증이 아니다.", "",
         "## 1. 안전 (최우선) — 위험 감소 대비 화질", "",
         "| 방법 | 보정 뒤 안전 | 위험 프레임 비율 | 새로 생긴 위험 | LPIPS 화면 전체 | PSNR 화면 전체 | LPIPS 위험 영역 | 위험 밖 바뀐 픽셀 |",
         "|---|---|---|---|---|---|---|---|"]
    for m in methods:
        rs = [r for r in haz if m in r["m"]]
        if m == "before":
            safe, hf, new = "0", "–", "–"
        else:
            safe = f"{sum(r['m'][m]['after']['dangerous'] is False for r in rs)} / {len(rs)}"
            hf = _m([r["m"][m]["after"]["hazard_frames"] for r in rs])
            new = str(sum(any(v and not r["true"][k] for k, v in r["m"][m]["after"]["kinds"].items()) for r in rs))
        L.append(f"| {METHOD_KO[m]} | {safe} | {hf} | {new} | {_m([r['m'][m]['frame']['lpips'] for r in rs])} | "
                 f"{_m([r['m'][m]['frame']['psnr'] for r in rs])} | {_m([r['m'][m]['hazard']['lpips'] for r in rs])} | "
                 f"{_m([r['m'][m]['outside_changed'] for r in rs])} |")
    L += ["", "## 2. 원본 충실도 — 영역별 (PSNR dB / SSIM / LPIPS)", ""]
    fr = {k: _m([r["regions"][k] for r in haz]) for k in ("hazard", "recoverable", "lost_temporal", "lost_never")}
    L += [f"- 영역 크기(화면·시간 대비 중앙값): 위험 {fr['hazard']} · 복원 가능 {fr['recoverable']} · 시간 복원 가능 {fr['lost_temporal']} · 정보 없음 {fr['lost_never']}", "",
          "| 방법 | " + " | ".join(REGION_KO[k] for k in REGIONS) + " |", "|---|" + "---|" * len(REGIONS)]
    for m in methods:
        rs = [r for r in haz if m in r["m"]]
        cells = [f"{_m([r['m'][m][k]['psnr'] for r in rs])} / {_m([r['m'][m][k]['ssim'] for r in rs])} / {_m([r['m'][m][k]['lpips'] for r in rs])}"
                 for k in REGIONS]
        L.append(f"| {METHOD_KO[m]} | " + " | ".join(cells) + " |")
    L += ["", "## 3. 공간·시간 품질 (위험 영역, 원본 = 1)", "",
          "| 방법 | 워핑 오차 비율 | 프레임 간 변화량 비율 | 질감 | 이음새 |", "|---|---|---|---|---|"]
    for m in methods:
        rs = [r for r in haz if m in r["m"]]
        L.append(f"| {METHOD_KO[m]} | " + " | ".join(_m([r["m"][m]["temporal"][k] for r in rs]) for k in ("warp_ratio", "diff_ratio", "texture", "seam")) + " |")
    L += ["", "## 4. 종류별 보정 뒤 안전", "", "| 방법 | " + " | ".join(f"{KO[k]} 위험 표본" for k in roi.HAZARDS) + " |", "|---|---|---|---|"]
    for m in methods:
        if m == "before":
            continue
        cells = []
        for k in roi.HAZARDS:
            rs = [r for r in haz if r["true"][k] and m in r["m"]]
            cells.append(f"{sum(not r['m'][m]['after']['kinds'][k] for r in rs)} / {len(rs)}" if rs else "–")
        L.append(f"| {METHOD_KO[m]} | " + " | ".join(cells) + " |")
    safe_rows = [r for r in rows if "safe" in r]
    if safe_rows:
        L += ["", f"## 5. 안전한 표본 {len(safe_rows)}개 — 건드린 정도", "", "| 방법 | 건드린 표본 | 바뀐 픽셀 비율(건드린 것 중앙값) | 입력 대비 PSNR |", "|---|---|---|---|"]
        for m in ("E0", "E2", "E3"):
            vs = [r["safe"][m] for r in safe_rows if m in r["safe"]]
            if vs:
                t = [v for v in vs if v["changed_frac"] > 0]
                L.append(f"| {METHOD_KO[m]} | {len(t)} / {len(vs)} | {_m([v['changed_frac'] for v in t])} | {_m([v['psnr_vs_input'] for v in t])} |")
    L += ["", "## 6. 효율", ""]
    for k, ko in (("det", "탐지기"), ("cor", "보정기"), ("loss", "손실 추정기")):
        if k in eff:
            e = eff[k]
            L.append(f"- {ko}: 48프레임(2초) 768×432 한 클립 p50 {e['p50'] * 1000:.0f} ms · p95 {e['p95'] * 1000:.0f} ms "
                     f"(≈ {48 / e['p50']:.0f} 프레임/초)")
    if "gpu_peak_GB" in eff:
        L.append(f"- GPU 최대 메모리 {eff['gpu_peak_GB']} GB (평가 전체, LPIPS 포함)")
    L += ["", "- LPIPS 는 원본과의 지각적 유사도이지 자연스러움 자체가 아니다. 정보 없음 영역에서는 그럴듯하게 채워도 나쁘게 나올 수 있다.",
          "- 워핑 오차가 낮아도 흐리거나 멈춘 결과일 수 있다 — 질감·변화량과 함께 본다."]
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
