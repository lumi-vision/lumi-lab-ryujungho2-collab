"""South 2021 전체(2,000)를 팀원 Partial B2 판정기로 판정만 하고 결과만 저장한다 (2026-11-23, 판정기 v2 의 기준선).

판정기: rockrovks Partial/B2 `vdcoreps.analyze_frames` (Update11, 495460d), GPU. 설정은 판정기 기본값 그대로.
입력: data/real/south2021/social-media-study-gifs/<label>.(mp4|gif) — 가로 640 이하로 줄여 원래 fps (11-17 기준선과 같은 전처리).
저장 (data/judge/south_partial/, Git 제외):
  rows.jsonl              파일마다 한 줄: 라벨·위험 여부·라벨별 구간·fps·크기·프레임 수·시간·사람 라벨(csv)
  frames/<label>.npz      프레임별 7 라벨 마스크(bool 7×T)
  metrics/<label>.csv.gz  판정기의 프레임별 지표 표(frame_metrics_df) — 문턱값 분석용
보정·마스크 영상은 만들지 않는다. 이미 저장된 파일은 건너뛴다(이어 하기).

    python scripts/judge_south_partial.py --team-dir <rockrovks>
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
LABELS = ["휘도 플래시", "교집합", "빠른 장면 전환", "적색 플래시", "휘도 연장", "적색 연장", "정적 공간 패턴"]


def jsonable(x):
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if hasattr(x, "__dict__") and not isinstance(x, type):
        return {k: jsonable(v) for k, v in vars(x).items() if not k.startswith("_")}
    if isinstance(x, np.ndarray):
        return np.round(x, 4).tolist() if x.dtype.kind == "f" else x.tolist()
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    return str(x)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--team-dir", required=True)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    sys.path.insert(0, str(Path(a.team_dir) / "Partial" / "B2"))
    import vdcoreps as vd
    from eval_real_hazards import decode as dec_any

    src = ROOT / "data" / "real" / "south2021"
    lab = {r["label"]: r for r in csv.DictReader(open(src / "social-media-study.csv", encoding="utf-8", errors="ignore"))}
    files = {p.stem: p for p in (src / "social-media-study-gifs").iterdir() if p.suffix.lower() in (".mp4", ".gif")}
    out = ROOT / "data" / "judge" / "south_partial"
    (out / "frames").mkdir(parents=True, exist_ok=True)
    (out / "metrics").mkdir(parents=True, exist_ok=True)
    rows_p = out / "rows.jsonl"
    done = {json.loads(l)["label"] for l in rows_p.read_text(encoding="utf-8").splitlines() if l.strip()} if rows_p.exists() else set()
    todo = [k for k in sorted(lab) if k in files and k not in done][: a.limit or None]
    missing = [k for k in lab if k not in files]
    print(f"South {len(lab)} · 파일 없음 {len(missing)} · 끝남 {len(done)} · 할 것 {len(todo)}", flush=True)
    t_all = time.time()
    with rows_p.open("a", encoding="utf-8") as f:
        for i, k in enumerate(todo, 1):
            t0 = time.time()
            try:
                x, fps = dec_any(files[k])
                res = vd.analyze_frames(list(x), fps, device="cuda")
                masks = np.stack([np.asarray(res.frame_masks.get(l, np.zeros(len(x), bool)), bool) for l in LABELS])
                np.savez_compressed(out / "frames" / f"{k}.npz", masks=masks, labels=np.array(LABELS))
                try:
                    res.frame_metrics_df.to_csv(out / "metrics" / f"{k}.csv.gz", index=False, compression="gzip")
                except Exception:  # noqa: BLE001
                    pass
                row = {"label": k, "file": files[k].name, "fps": round(float(fps), 3), "frames": int(len(x)),
                       "w": int(x.shape[2]), "h": int(x.shape[1]), "dangerous": bool(res.is_dangerous),
                       "labels": sorted(res.active_labels), "label_frames": {l: int(m.sum()) for l, m in zip(LABELS, masks)},
                       "intervals_by_label": jsonable(getattr(res, "intervals_by_label", {})),
                       "active_mask_area": jsonable(getattr(res, "active_mask_area", None)),
                       "human": {c: lab[k][c] for c in ("dangerous", "flash", "red", "pattern", "source")},
                       "judge": "Partial/B2 vdcoreps Update11 495460d", "sec": round(time.time() - t0, 2)}
            except Exception as e:  # noqa: BLE001
                import traceback
                row = {"label": k, "file": files[k].name, "error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()[-600:],
                       "human": {c: lab[k][c] for c in ("dangerous", "flash", "red", "pattern", "source")}}
            f.write(json.dumps(row, ensure_ascii=False) + "\n"); f.flush()
            if i % 25 == 0 or i == len(todo):
                print(f"  {i}/{len(todo)} ({time.time() - t_all:.0f}초) {k}", flush=True)
    print("끝 DONE", flush=True)


if __name__ == "__main__":
    main()
