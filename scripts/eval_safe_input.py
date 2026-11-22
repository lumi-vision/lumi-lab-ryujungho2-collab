"""A3 (모델 단독, 참고): 안전한 입력을 보정 모델에 넣으면 얼마나 바꾸나 (docs/CORRECTION_PROTOCOL.md §5.1, 결과 보기 전에 작성).

입력 = 합성 test 의 원본(`<id>.clean.mp4`, 위험을 덧씌우기 전). 모델 출력과 입력을 비교한다.
  - PSNR·LPIPS(입력 대비, 화면 전체)  — 클수록/작을수록 '안 바꿈'
  - 바뀐 픽셀 비율(|Δ| > 2/255 인 픽셀)
  - 보정 뒤 새로 생긴 Partial B2 라벨(입력에 없던 것, 무늬는 12 프레임 이상) — A2 와 같은 정의
같은 원본이 test 에 여러 번(레시피별) 나오므로 원본 파일(바탕 구간)이 겹치면 한 번만 쓴다.

    python scripts/eval_safe_input.py --team-dir <rockrovks> --models h_s0_e60 h_s1_e60 h_s2_e60
⚠️ 프레임은 메모리에서만 다룬다(재생·저장하지 않음).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from lumi_synth import hazard7, quality  # noqa: E402
from eval_correction import PAT, model_correct, team_judge  # noqa: E402
from make_hazard7_dataset import decode  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--team-dir", required=True)
    ap.add_argument("--data", default="big1")
    ap.add_argument("--split", default="test")
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    d = ROOT / "data" / "hazard7" / a.data / a.split
    ids, seen = [], set()
    for l in (d / "gen.jsonl").read_text(encoding="utf-8").splitlines():
        if not l.strip():
            continue
        r = json.loads(l)
        p = d / f"{r['id']}.clean.mp4"
        if "error" in r or not p.exists():
            continue
        h = hashlib.md5(p.read_bytes()).hexdigest()
        if h in seen:
            continue
        seen.add(h); ids.append(r["id"])
    ids = ids[: a.limit or None]
    out_rows = ROOT / "checks" / "correction" / f"safe_input_{a.split}_rows.jsonl"
    rows = []
    with out_rows.open("w", encoding="utf-8") as f:
        for i, sid in enumerate(ids, 1):
            x = decode(d / f"{sid}.clean.mp4")
            before = team_judge(x, a.team_dir)["masks"]
            for m in a.models:
                y = model_correct(m, x)
                after = team_judge(y, a.team_dir)["masks"]
                new = after & ~before
                labs = [hazard7.LABELS[k] for k in range(len(hazard7.LABELS)) if new[k].sum() >= (12 if k == PAT else 1)]
                diff = np.abs(y.astype(np.int16) - x.astype(np.int16)).max(-1)
                lp = quality.lpips_map(y[::4], x[::4])
                row = {"id": sid, "model": m, "psnr_vs_input": round(quality.psnr(y, x, np.ones(x.shape[:3], bool)), 3),
                       "lpips_vs_input": round(float(np.mean(lp)), 4), "changed_px_frac": round(float((diff > 2).mean()), 4),
                       "new_labels": labs}
                rows.append(row); f.write(json.dumps(row, ensure_ascii=False) + "\n"); f.flush()
            print(f"  {i}/{len(ids)} {sid}", flush=True)
    md = [f"# A3 안전 입력 — 모델 단독 ({a.split} 원본 {len(ids)} 개)", "",
          "| 모델 | PSNR 입력 대비 (중앙값) | LPIPS (중앙값) | 바뀐 픽셀 비율 (중앙값) | 새 라벨 생긴 클립 |", "|---|---|---|---|---|"]
    for m in a.models:
        r = [x for x in rows if x["model"] == m]
        md.append(f"| {m} | {np.median([x['psnr_vs_input'] for x in r]):.2f} | {np.median([x['lpips_vs_input'] for x in r]):.4f} | "
                  f"{np.median([x['changed_px_frac'] for x in r]):.4f} | {sum(bool(x['new_labels']) for x in r)}/{len(r)} |")
    (ROOT / "checks" / "correction" / f"safe_input_{a.split}.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
