"""4단계 평가: 보정 결과를 하드 판정기에 다시 넣고(안전해졌는가), 원본과 비교한다(자연스러운가).

    python scripts/eval_corrector.py --corrector data/gdr/v1/cor_a.pt --roi data/gdr/v1/roi_a.pt --data data/gdr/v1/val

판정기가 위험이라 한 표본마다 네 가지를 잰다.
    보정 전          입력 그대로 (기준선)
    규칙 보정        roi.rule_correct — 판정기 튜브 안에서 시간 평균(휘도·적색)·공간 흐림(줄무늬). 원본을 모른다
    R + 판정기 튜브   보정기에 판정기 튜브를 준 것 (보정기만의 성능)
    R + 탐지기 튜브   보정기에 ROI 탐지기의 예측을 준 것 (탐지 → 보정 전체. --roi 가 있을 때)
지표 (모두 코드로 잰다 — 결과 영상은 보지 않는다)
    안전            보정 결과를 판정기에 다시 넣었을 때 위험이 없는 비율
    PSNR           판정기 튜브 안에서 원본과의 차이 (dB, 클수록 원본에 가까움)
    질감            윤곽선 에너지 출력/원본 (1 = 그대로, < 1 = 뭉개짐, > 1 = 무늬가 남음)
    시간 변화        프레임 간 변화량 출력/원본 (1 = 원본과 같은 움직임, > 1 = 깜빡임이 남음, < 1 = 움직임이 뭉개짐)
    이음새          튜브 경계를 가로지르는 밝기 차이 출력/원본 (1 = 이음새 없음)
    튜브 밖 변화     보정이 튜브 밖 픽셀을 바꾼 수 (0 이어야 함)
판정기가 안전이라 한 표본(근처 안전·깨끗한 바탕)에서는 '탐지 → 보정'이 영상을 얼마나 건드렸는지만 잰다.
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
from lumi_synth import roi

VARIANTS = ("before", "rule", "r_judge", "r_roi")
VARIANT_KO = {"before": "보정 전", "rule": "규칙 보정 (판정기 튜브)", "r_judge": "R + 판정기 튜브", "r_roi": "R + 탐지기 튜브"}


def load_corrector(path: Path, device) -> tuple[roi.Corrector, dict]:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    net = roi.Corrector(int(ck["config"]["width"]))
    net.load_state_dict(ck["model"])
    return net.to(device).eval(), ck["config"]


@torch.no_grad()
def correct(net, x_u8: np.ndarray, tubes: np.ndarray, device) -> tuple[np.ndarray, np.ndarray]:
    """영상 + 튜브(3×T×H×W bool) → (보정 영상 uint8, 보정이 닿을 수 있는 영역 T×H×W bool)."""
    m = roi.soften(torch.from_numpy(tubes[None]).to(device))
    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
        y = net(to_x([x_u8], device), m)
    y8 = (y.float()[0].permute(1, 2, 3, 0) * 255.0).round().clamp(0, 255).byte().cpu().numpy()
    reach = (m.amax(dim=1)[0] > 0).cpu().numpy()
    y8[~reach] = x_u8[~reach]                    # 반올림 오차로도 튜브 밖이 바뀌지 않게 (구조상 이미 같다)
    return y8, reach


def quality(y: np.ndarray, x: np.ndarray, clean: np.ndarray, region: np.ndarray, reach: np.ndarray | None) -> dict:
    out = {"psnr_in": roi.psnr(y, clean, region), "texture": roi.texture_ratio(y, clean, region),
           "temporal": roi.temporal_ratio(y, clean, region), "seam": roi.seam_ratio(y, clean, region)}
    if reach is not None:
        out["outside_changed"] = int(np.any(y != x, axis=-1)[~reach].sum())
    return out


def judge_summary(frames: np.ndarray, fps: float) -> dict:
    return roi.judge_tubes(frames, fps)[1]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corrector", required=True, type=Path)
    ap.add_argument("--roi", type=Path, default=None)
    ap.add_argument("--data", required=True, type=Path, nargs="+")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    device = torch.device(args.device)
    cor, ccfg = load_corrector(args.corrector, device)
    det = load_roi(args.roi, device)[0] if args.roi else None
    out_dir = ROOT / "checks" / "gdr"
    out_dir.mkdir(parents=True, exist_ok=True)

    for folder in args.data:
        metas = read_manifest(folder)
        if args.limit:
            metas = metas[:args.limit]
        require_split(metas, "eval")
        t0, rows, futs = time.time(), [], []
        with cf.ProcessPoolExecutor(max_workers=args.workers) as pool:
            for i, s in enumerate(prefetch(metas)):
                x, clean, tubes, fps = s["x"], s["clean"], s["tubes"], s["fps"]
                row = {"id": s["meta"]["id"], "group": group_of(s["meta"]),
                       "true": {k: bool(tubes[j].any()) for j, k in enumerate(roi.HAZARDS)}, "v": {}}
                pred = upsample_tubes(predict_tubes(det, x, device), x.shape[1:3]) if det else None
                if pred is not None:
                    row["pred"] = {k: bool(pred[j].any()) for j, k in enumerate(roi.HAZARDS)}
                if tubes.any():
                    region = tubes.any(axis=0)
                    row["v"]["before"] = quality(x, x, clean, region, None)
                    outs = {"rule": (roi.rule_correct(x, dict(zip(roi.HAZARDS, tubes)), fps), region)}
                    outs["r_judge"] = correct(cor, x, tubes, device)
                    if pred is not None:
                        outs["r_roi"] = correct(cor, x, pred, device) if pred.any() else (x, np.zeros(region.shape, bool))
                    for name, (y, reach) in outs.items():
                        row["v"][name] = quality(y, x, clean, region, reach)
                        futs.append((row, name, pool.submit(judge_summary, y, fps)))
                elif pred is not None:                     # 안전한 표본: 탐지 → 보정이 건드린 정도
                    if pred.any():
                        y, _ = correct(cor, x, pred, device)
                        row["safe"] = {"changed_frac": float(np.any(y != x, axis=-1).mean()), "psnr_vs_input": roi.psnr(y, x)}
                    else:
                        row["safe"] = {"changed_frac": 0.0, "psnr_vs_input": float("inf")}
                rows.append(row)
                while sum(not f.done() for _, _, f in futs) > args.workers * 2:
                    time.sleep(0.5)
                if (i + 1) % 10 == 0:
                    print(f"  {folder.name} {i + 1}/{len(metas)}  {time.time() - t0:.0f}초", flush=True)
            for row, name, fu in futs:
                su = fu.result()
                row["v"][name]["after"] = {k: bool(su[k]["dangerous"]) for k in roi.HAZARDS}
                row["v"][name]["safe"] = not su["dangerous"]
        name = f"{folder.parent.name}_{args.corrector.stem}_{folder.name}"
        (out_dir / f"{name}_rows.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
                                                    encoding="utf-8")
        write_report(out_dir / f"{name}.md", f"`{args.corrector.name}`" + (f" + `{args.roi.name}`" if args.roi else "")
                     + f" → `{folder.parent.name}/{folder.name}`", ccfg, rows, time.time() - t0)
        print(f"저장: checks/gdr/{name}.md  ({time.time() - t0:.0f}초)", flush=True)
    return 0


def _med(vals: list[float]) -> str:
    v = [x for x in vals if x == x and np.isfinite(x)]
    return f"{np.median(v):.2f}" if v else "–"


def write_report(path: Path, title: str, cfg: dict, rows: list[dict], secs: float) -> None:
    L = [f"# 보정 평가 — {title}", "",
         f"- 보정기 학습: {cfg.get('data')} · 튜브가 있는 표본 {cfg.get('n_train')} · {cfg.get('epoch_done')} epoch · 목표 = 원본",
         f"- 표본 {len(rows)}개 · {secs:.0f}초 · 값은 표본 중앙값(안전 칸은 개수)", ""]
    haz = [r for r in rows if r["v"]]
    for k in roi.HAZARDS:
        rs = [r for r in haz if r["true"][k]]
        if not rs:
            continue
        L += [f"## {KO[k]} 위험 — 판정기가 {KO[k]} 위험이라 한 표본 {len(rs)}개", "",
              "| 방법 | 보정 뒤 안전 | 그 종류가 안전 | PSNR (튜브 안, dB) | 질감 | 시간 변화 | 이음새 | 튜브 밖 변화 |",
              "|---|---|---|---|---|---|---|---|"]
        for v in VARIANTS:
            vs = [r["v"][v] for r in rs if v in r["v"]]
            if not vs:
                continue
            if v == "before":
                safe = kind_safe = "0 (입력이 위험)"
                outside = "–"
            else:
                safe = f"{sum(x['safe'] for x in vs)} / {len(vs)}"
                kind_safe = f"{sum(not x['after'][k] for x in vs)} / {len(vs)}"
                outside = str(sum(x.get("outside_changed", 0) for x in vs))
            L.append(f"| {VARIANT_KO[v]} | {safe} | {kind_safe} | {_med([x['psnr_in'] for x in vs])} | {_med([x['texture'] for x in vs])} | "
                     f"{_med([x['temporal'] for x in vs])} | {_med([x['seam'] for x in vs])} | {outside} |")
        L.append("")
    safe_rows = [r for r in rows if "safe" in r]
    if safe_rows:
        touched = [r for r in safe_rows if r["safe"]["changed_frac"] > 0]
        L += [f"## 안전한 표본 {len(safe_rows)}개 (판정기가 위험 없다고 한 것) — 탐지 → 보정이 건드렸는가", "",
              f"- 한 픽셀이라도 바뀐 표본: {len(touched)} / {len(safe_rows)}",
              f"- 바뀐 표본의 바뀐 픽셀 비율 중앙값: {_med([r['safe']['changed_frac'] for r in touched])}",
              f"- 바뀐 표본의 입력 대비 PSNR 중앙값: {_med([r['safe']['psnr_vs_input'] for r in touched])} dB", ""]
        by = {}
        for r in safe_rows:
            by.setdefault(r["group"], []).append(r["safe"]["changed_frac"] > 0)
        L += ["| 묶음 | 건드린 표본 / 전체 |", "|---|---|"] + [f"| {g} | {sum(v)} / {len(v)} |" for g, v in sorted(by.items())] + [""]
    L += ["- '보정 뒤 안전' = 세 종류 모두 위험이 없음. '그 종류가 안전' = 해당 종류만 봄.",
          "- PSNR·질감·시간 변화·이음새는 판정기 튜브(세 종류 합집합) 안에서 원본과 비교한 값이다.",
          "- 줄무늬 판정은 초당 약 10장 표본이다(roi.judge_tubes)."]
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
