"""α(합성 비중) 옆 파일이 없는 표본에 α 를 만들어 붙인다 — 레시피로 다시 합성해서 계산한다.

    python scripts/gdr_backfill_alpha.py --name v6

α 저장(2026-11-15) 이전에 만든 표본용. 다시 합성한 영상이 저장된 영상과 같은지(최대 차이)를 함께 확인해,
다르면(예: GPU/CPU 추론 차이) 그 값을 manifest 옆 alpha_backfill.jsonl 에 적는다. 저장된 표본 파일은 건드리지 않는다
(여러 버전 폴더가 하드링크로 같은 파일을 공유하므로).
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from gdr_common import ROOT, read_manifest
from lumi_synth import generate, roi
from make_gdr_dataset import LOSS_ALPHA, save_alpha


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", default="v6")
    args = ap.parse_args()
    base = ROOT / "data" / "gdr" / args.name
    for folder in sorted(p for p in base.iterdir() if (p / "manifest.jsonl").exists()):
        log = (folder / "alpha_backfill.jsonl").open("a", encoding="utf-8")
        todo = [m for m in read_manifest(folder) if m["hazard"] and not (folder / f"{m['id']}.alpha.npz").exists()]
        print(f"{folder.name}: {len(todo)}개", flush=True)
        for i, m in enumerate(todo):
            clips, fps = generate.load_backgrounds(m["split"])
            clean = clips[m["bg"]]
            res = generate.render(m["recipe"], clean, fps)
            with np.load(folder / f"{m['id']}.npz") as z:
                stored, tubes = z["flicker"], z["tubes"]
            diff = int(np.abs(res["flicker"].astype(np.int16) - stored).max())
            save_alpha(str(folder / f"{m['id']}.alpha.npz"), res["alpha"])
            hz = tubes.any(axis=0)
            rec = {"id": m["id"], "max_diff": diff,
                   "loss_frac_in_tube": round(float((res["alpha"][hz] >= LOSS_ALPHA).mean()), 4) if hz.any() else 0.0}
            log.write(json.dumps(rec) + "\n")
            log.flush()
            if (i + 1) % 25 == 0:
                print(f"  {i + 1}/{len(todo)}", flush=True)
        log.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
