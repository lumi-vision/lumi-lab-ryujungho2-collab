"""3단계 평가: ROI 탐지기를 하드 판정기의 튜브와 비교한다.

    python scripts/eval_roi.py --checkpoint data/gdr/v1/roi_a.pt --data data/gdr/v1/val data/gdr/v1/val_rule

폴더마다 checks/gdr/<체크포인트>_<폴더>.md 와 _rows.jsonl 을 쓴다.
    클립 판정   종류별로 '이 클립에 그 위험이 있는가' — 판정기(정답) 대 탐지기(어느 프레임이든 예측 면적 ≥ 12.5%)
    튜브 IoU    판정기 튜브가 있는 표본에서, 1/4 해상도
    오탐 면적   판정기 튜브가 빈 표본에서 탐지기가 칠한 면적(시간·공간 평균)
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from gdr_common import ROOT, eval_roi, read_manifest, require_split, summarize_roi
from lumi_synth import roi

KO = {"lum": "휘도", "red": "적색", "stripe": "줄무늬"}


def load_roi(path: Path, device) -> tuple[roi.RoiNet, dict]:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    n_out = 1 if ck["config"].get("target") == "loss" else len(roi.HAZARDS)       # 손실 추정기는 출력 1채널
    net = roi.RoiNet(int(ck["config"]["width"]), ck["config"].get("features", "rgb"), n_out=n_out)
    net.load_state_dict(ck["model"])
    return net.to(device).eval(), ck["config"]


def write_report(path: Path, title: str, cfg: dict, rows: list[dict], secs: float) -> dict:
    su = summarize_roi(rows)
    f = lambda v: "–" if v != v else f"{v:.3f}"          # noqa: E731
    L = [f"# ROI 탐지기 평가 — {title}", "",
         f"- 학습: {cfg.get('data')} · 표본 {cfg.get('n_train')} · {cfg.get('epoch_done')} epoch · width {cfg.get('width')}",
         f"- 표본 {len(rows)}개 · 정답 = 하드 판정기 튜브 · 클립 판정 문턱 = 예측 면적 {roi.CLIP_AREA_MIN:.1%} · {secs:.0f}초", "",
         "## 종류별 (표본 전체)", "",
         "| 종류 | 정답 있음 | 맞게 찾음 | 놓침 | 정답 없음 | 잘못 찾음 | 정밀도 | 재현율 | 튜브 IoU | 오탐 면적 |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for k in roi.HAZARDS:
        s = su[k]
        L.append(f"| {KO[k]} | {s['tp'] + s['fn']} | {s['tp']} | {s['fn']} | {s['tn'] + s['fp']} | {s['fp']} | "
                 f"{f(s['precision'])} | {f(s['recall'])} | {f(s['iou'])} | {f(s['fp_area'])} |")
    L += ["", "## 묶음별 (생성 의도로 묶음 — 칸은 `탐지기가 위험이라 한 수 / 판정기가 위험이라 한 수 / 표본 수`)", "",
          "| 묶음 | " + " | ".join(KO[k] for k in roi.HAZARDS) + " |", "|---|---|---|---|"]
    groups = sorted({r["group"] for r in rows})
    for g in groups:
        rs = [r for r in rows if r["group"] == g]
        L.append(f"| {g} | " + " | ".join(f"{sum(r[k]['pred'] for r in rs)} / {sum(r[k]['true'] for r in rs)} / {len(rs)}"
                                         for k in roi.HAZARDS) + " |")
    L += ["", "- IoU 는 판정기 튜브가 있는 표본만 평균한다. 정밀도·재현율은 클립 단위다.",
          "- 묶음 이름은 생성 의도다. 정답은 의도가 아니라 판정기 결과다(의도가 위험이어도 판정이 안전이면 정답 없음)."]
    path.write_text("\n".join(L) + "\n", encoding="utf-8")
    return su


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--data", required=True, type=Path, nargs="+")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    device = torch.device(args.device)
    net, cfg = load_roi(args.checkpoint, device)
    out = ROOT / "checks" / "gdr"
    out.mkdir(parents=True, exist_ok=True)
    for folder in args.data:
        metas = read_manifest(folder)
        if args.limit:
            metas = metas[:args.limit]
        require_split(metas, "eval")
        t0 = time.time()
        rows = eval_roi(net, metas, device, log=lambda s: print(s, flush=True))
        name = f"{folder.parent.name}_{args.checkpoint.stem}_{folder.name}"     # 예: v1_roi_a_val
        su = write_report(out / f"{name}.md", f"`{args.checkpoint.name}` → `{folder.parent.name}/{folder.name}`", cfg, rows,
                          time.time() - t0)
        (out / f"{name}_rows.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
        print(name, json.dumps({k: {m: (round(v, 3) if isinstance(v, float) else v) for m, v in su[k].items()} for k in su}),
              flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
