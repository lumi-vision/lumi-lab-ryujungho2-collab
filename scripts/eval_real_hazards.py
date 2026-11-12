"""실제 위험 영상(South 2021, Kaya 2025)을 우리 판정기와 팀원 판별기(rockrovks B2 vdcorefs)로 판정한다.

⚠️ 입력은 실제 광과민성 위험 자극이다. 재생·미리보기하지 않는다. 프레임은 메모리에서만 다룬다.
- 영상은 PyAV 로 한 번만 디코드해 **같은 프레임**을 두 판정기에 넣는다(디코더 차이 제거).
- 가로 640 이하로 면적 평균 축소(팀원 판별기의 기본 분석 폭과 같음). fps = 프레임 수 ÷ 길이(GIF 의 가변 간격은 평균으로).
- 긴 영상은 60초 조각(6초 겹침)으로 나눠 판정하고 조각 결과를 OR 한다(판정 창은 최대 5초).
- 팀원 코드는 수정하지 않고 import 해서 analyze_frames 를 그대로 부른다(--team-dir).
사용: python scripts/eval_real_hazards.py --team-dir <lumi-lab-rockrovks> [--sources south kaya] [--workers 6]
결과: data/real/eval_rows.jsonl (이어 하기 가능)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REAL = ROOT / "data" / "real"
OUT = REAL / "eval_rows.jsonl"
MAX_W, CHUNK_S, OVERLAP_S = 640, 60.0, 6.0
VIDEO_EXT = {".gif", ".mp4", ".webm", ".mov", ".avi", ".mkv", ".m4v", ".mpg", ".mpeg", ".wmv", ".flv"}


def decode(path: Path):
    import av
    import cv2
    frames, times = [], []
    with av.open(str(path)) as c:
        s = c.streams.video[0]
        for f in c.decode(s):
            img = f.to_ndarray(format="bgr24")
            h, w = img.shape[:2]
            if w > MAX_W:
                img = cv2.resize(img, (MAX_W, max(2, int(round(h * MAX_W / w)))), interpolation=cv2.INTER_AREA)
            frames.append(img)
            times.append(float(f.time) if f.time is not None else None)
    if len(frames) < 2:
        raise ValueError("프레임 2장 미만")
    t = [x for x in times if x is not None]
    dur = (t[-1] - t[0]) if len(t) >= 2 else 0
    fps = (len(frames) - 1) / dur if dur > 0 else 30.0
    if not (1.0 <= fps <= 240.0):
        fps = 30.0
    return np.stack(frames), float(fps)


_TEAM = None


def team_mod(team_dir: str):
    global _TEAM
    if _TEAM is None:
        b2 = str(Path(team_dir) / "Fullscreen" / "B2")
        if b2 not in sys.path:
            sys.path.insert(0, b2)
        import vdcorefs
        _TEAM = vdcorefs
    return _TEAM


def judge_ours(frames, fps):
    from lumi_synth import measure
    r = measure.analyze_all(frames, fps)
    el = measure.extended_flash(r["flash"]["transitions"], fps)
    er = measure.extended_flash(r["red"]["transitions"], fps)
    return {"lum": bool(r["flash"]["dangerous"]), "red": bool(r["red"]["dangerous"]),
            "pattern": bool(r["pattern"]["dangerous"]), "ext_lum": el["extended_dangerous"],
            "ext_red": er["extended_dangerous"], "lum_max_flashes": float(r["flash"]["max_flashes"]),
            "lum_peak_area": float(r["flash"]["peak_area"]), "red_max_flashes": float(r["red"]["max_flashes"])}


def judge_team(frames, fps, team_dir):
    vd = team_mod(team_dir)
    res = vd.analyze_frames(list(frames), fps, device="cuda")
    return {"dangerous": bool(res.is_dangerous), "labels": sorted(set(res.active_labels))}


def one(path_s: str, source: str, team_dir: str):
    path = Path(path_s)
    t0 = time.time()
    try:
        frames, fps = decode(path)
        n = len(frames)
        step = int((CHUNK_S - OVERLAP_S) * fps)
        size = int(CHUNK_S * fps)
        starts = list(range(0, max(1, n - int(OVERLAP_S * fps)), step)) if n > size else [0]
        ours, team = {}, {"dangerous": False, "labels": set()}
        for s in starts:
            seg = frames[s:s + size]
            if len(seg) < 2:
                continue
            o = judge_ours(seg, fps)
            for k, v in o.items():
                ours[k] = (ours.get(k, False) or v) if isinstance(v, bool) else max(ours.get(k, 0.0), v)
            tm = judge_team(seg, fps, team_dir)
            team["dangerous"] |= tm["dangerous"]
            team["labels"] |= set(tm["labels"])
        ours["any"] = any(ours[k] for k in ("lum", "red", "pattern", "ext_lum", "ext_red"))
        ours["any_core"] = any(ours[k] for k in ("lum", "red", "pattern"))
        return {"source": source, "file": str(path.relative_to(REAL)), "frames": n, "fps": round(fps, 3),
                "w": int(frames.shape[2]), "h": int(frames.shape[1]), "chunks": len(starts),
                "ours": ours, "team": {"dangerous": team["dangerous"], "labels": sorted(team["labels"])},
                "sec": round(time.time() - t0, 1)}
    except Exception as e:  # noqa: BLE001
        return {"source": source, "file": str(path.relative_to(REAL)), "error": f"{type(e).__name__}: {e}",
                "trace": traceback.format_exc()[-800:], "sec": round(time.time() - t0, 1)}


def list_files(sources):
    out = []
    for src in sources:
        base = REAL / ("south2021" if src == "south" else "kaya2025")
        for p in sorted(base.rglob("*")):
            if p.is_file() and p.suffix.lower() in VIDEO_EXT and ".part" not in p.name:
                out.append((str(p), src))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--team-dir", required=True)
    ap.add_argument("--sources", nargs="+", default=["south", "kaya"])
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    done = set()
    if OUT.exists():
        done = {json.loads(l)["file"] for l in OUT.read_text(encoding="utf-8").splitlines() if l.strip()}
    jobs = [(p, s) for p, s in list_files(a.sources) if str(Path(p).relative_to(REAL)) not in done]
    print(f"남은 영상 {len(jobs)}개 (이미 {len(done)}개)", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(a.workers) as ex, open(OUT, "a", encoding="utf-8") as f:
        futs = [ex.submit(one, p, s, a.team_dir) for p, s in jobs]
        for i, fu in enumerate(as_completed(futs), 1):
            r = fu.result()
            f.write(json.dumps(r, ensure_ascii=False) + "\n"); f.flush()
            if i % 25 == 0 or "error" in r:
                print(f"  {i}/{len(jobs)} ({time.time() - t0:.0f}초) {r['file']} {r.get('error', '')}", flush=True)
    print("끝", flush=True)


if __name__ == "__main__":
    main()
