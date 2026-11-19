"""7종 위험 라벨 합성 데이터 (src/lumi_synth/hazard7.py) — 만들고, mp4 로 저장하고, 다시 디코드해서 판별기로 라벨을 붙인다.

    python scripts/make_hazard7_dataset.py --team-dir <lumi-lab-rockrovks> --split val --per 6 --name t1

- 바탕: Big Buck Bunny 1080p 의 8초 구간(24 fps, 768×432). train 30~420 s · val 450~560 s (기존 분할과 같은 구간),
  test = Sintel(data/raw/sintel) 앞부분 제외 구간. 장면 컷용 '다른 장면'도 같은 분할에서 뽑는다.
- 종류 7 × 의도 2(hazard / near) × --per 개.
- 정답 = **저장한 mp4 를 다시 디코드한 프레임**에 팀원 판별기(vdcorefs.analyze_frames)를 돌린 결과(7라벨, 프레임별).
  우리 판정기(휘도·적색·IRIS 연장)도 같이 기록한다(교차 확인). 팀원 코드는 import 만 한다.
저장: data/hazard7/<name>/<split>/<id>.mp4 · <id>.labels.npz (team 7×T bool, ours) · manifest.jsonl
보고서: checks/hazard7/<name>_<split>.md
⚠️ 결과는 광과민성 위험 자극이다. 재생·미리보기하지 않는다.
"""
from __future__ import annotations

import argparse
import collections
import concurrent.futures as cf
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from lumi_synth import hazard7, measure  # noqa: E402

FPS, N, W, H = 24.0, 192, 768, 432
SRC = {"train": ("data/sources/big_buck_bunny_1080p/big_buck_bunny_1080p_h264.mov", 30, 420),
       "val": ("data/sources/big_buck_bunny_1080p/big_buck_bunny_1080p_h264.mov", 450, 560),
       "test": ("data/raw/sintel/Sintel.2010.1080p.mkv", 60, 840)}


def bg_clip(split: str, rng) -> tuple[np.ndarray, float]:
    """분할 구간에서 무작위 8초 → N×432×768×3 BGR uint8 (24 fps 가 아니면 시간 기준으로 다시 뽑음)."""
    import av
    import cv2
    f, s0, s1 = SRC[split]
    start = float(rng.uniform(s0, s1 - N / FPS - 1))
    frames, times = [], []
    with av.open(str(ROOT / f)) as c:
        st = c.streams.video[0]
        c.seek(int(start / st.time_base), stream=st, any_frame=False, backward=True)
        for fr in c.decode(st):
            t = float(fr.time)
            if t < start:
                continue
            frames.append(cv2.resize(fr.to_ndarray(format="bgr24"), (W, H), interpolation=cv2.INTER_AREA))
            times.append(t)
            if t >= start + N / FPS + 0.1:
                break
    times = np.array(times) - times[0]
    idx = np.searchsorted(times, np.arange(N) / FPS + 1e-6, side="right") - 1
    return np.stack([frames[i] for i in np.clip(idx, 0, len(frames) - 1)]), start


def encode(frames: np.ndarray, path: Path, crf: int = 12) -> None:
    import av
    with av.open(str(path), "w") as c:
        st = c.add_stream("libx264", rate=int(FPS))
        st.width, st.height, st.pix_fmt = W, H, "yuv420p"
        st.options = {"crf": str(crf), "preset": "medium"}
        for f in frames:
            for p in st.encode(av.VideoFrame.from_ndarray(f, format="bgr24")):
                c.mux(p)
        for p in st.encode():
            c.mux(p)


def decode(path: Path) -> np.ndarray:
    import av
    with av.open(str(path)) as c:
        return np.stack([f.to_ndarray(format="bgr24") for f in c.decode(c.streams.video[0])])


_VD = None


VARIANT = "partial"


def team_judge(frames: np.ndarray, team_dir: str) -> dict:
    """팀원 판별기. partial = Partial/B2 vdcoreps (7라벨 + 픽셀 위험 마스크), fullscreen = Fullscreen/B2 vdcorefs."""
    global _VD
    if _VD is None:
        if VARIANT == "partial":
            sys.path.insert(0, str(Path(team_dir) / "Partial" / "B2"))
            import vdcoreps as mod
        else:
            sys.path.insert(0, str(Path(team_dir) / "Fullscreen" / "B2"))
            import vdcorefs as mod
        _VD = mod
    res = _VD.analyze_frames(list(frames), FPS, device="cuda")
    masks = np.stack([np.asarray(res.frame_masks[l], bool) for l in hazard7.LABELS])
    wm = getattr(res, "weight_masks_u8", None)
    if wm is not None:                                   # 픽셀 위험 마스크 (분석 해상도) → 1/4 로 줄여 저장
        import cv2
        wm = np.stack([cv2.resize(np.asarray(m), (W // 4, H // 4), interpolation=cv2.INTER_AREA) for m in wm])
    return {"masks": masks, "labels": sorted(set(res.active_labels)), "dangerous": bool(res.is_dangerous), "weight": wm}


def ours_judge(frames: np.ndarray) -> dict:
    r = measure.analyze(frames, FPS)
    rr = measure.analyze_red(frames, FPS)
    el = measure.extended_flash(r["transitions"], FPS)
    er = measure.extended_flash(rr["transitions"], FPS)
    return {"lum": bool(r["dangerous"]), "red": bool(rr["dangerous"]), "ext_lum": bool(el["extended_dangerous"]),
            "ext_red": bool(er["extended_dangerous"]), "lum_max_flashes": float(r["max_flashes"]),
            "lum_peak_area": float(r["peak_area"]), "red_max_flashes": float(rr["max_flashes"])}


def one(job):
    sid, kind, intent, seed, split, out_dir, team_dir, variant = job
    global VARIANT
    VARIANT = variant
    t0 = time.time()
    try:
        rng = np.random.default_rng(seed)
        clean, start = bg_clip(split, rng)
        other, start2 = bg_clip(split, rng) if kind in ("scene", "intersection") else (None, None)
        r = hazard7.sample_recipe(kind, intent, seed, N, FPS)
        out, a = hazard7.render(r, clean, other)
        mp4 = Path(out_dir) / f"{sid}.mp4"
        encode(out, mp4)
        dec = decode(mp4)
        tj = team_judge(dec, team_dir)
        oj = ours_judge(dec)
        # 바탕만(같은 인코딩)도 판별 — 팀원 판별기가 손대지 않은 바탕에서도 '정적 공간 패턴' 등을 내므로, 합성으로 '새로 생긴' 라벨을 따로 센다
        cmp4 = Path(out_dir) / f"{sid}.clean_tmp.mp4"
        encode(clean, cmp4)
        cj = team_judge(decode(cmp4), team_dir)
        cmp4.unlink()
        added = tj["masks"] & ~cj["masks"]
        extra = {"weight": tj["weight"]} if tj["weight"] is not None else {}
        np.savez_compressed(Path(out_dir) / f"{sid}.labels.npz", team=tj["masks"], clean=cj["masks"], added=added,
                            labels=np.array(hazard7.LABELS), **extra)
        target = hazard7.KIND_LABEL[kind]
        ti = hazard7.LABELS.index(target)
        min_added = 12 if kind == "pattern" else 1                # 무늬는 0.5 s 이상 새로 생겨야
        added_labels = [l for l, a_ in zip(hazard7.LABELS, added) if a_.sum() >= (12 if l == hazard7.LABELS[-1] else 1)]
        r = {k: (v if not isinstance(v, np.generic) else v.item()) for k, v in r.items()}
        return {"id": sid, "split": split, "kind": kind, "intent": intent, "target_label": target, "recipe": r,
                "bg": {"start": round(start, 3), "other_start": None if start2 is None else round(start2, 3)},
                "region_mean": round(float(a.mean()), 4),
                "team": {"labels": tj["labels"], "dangerous": tj["dangerous"],
                         "frames": {l: int(m.sum()) for l, m in zip(hazard7.LABELS, tj["masks"])}},
                "clean_labels": cj["labels"], "added_labels": added_labels,
                "judge": variant, "ours": oj, "hit": target in tj["labels"], "hit_added": bool(added[ti].sum() >= min_added), "sec": round(time.time() - t0, 1)}
    except Exception as e:  # noqa: BLE001
        import traceback
        return {"id": sid, "kind": kind, "intent": intent, "error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()[-1200:]}


def report(rows: list[dict], path: Path, title: str) -> None:
    ok = [r for r in rows if "error" not in r]
    L = [f"# 7종 위험 라벨 합성 — {title}", "",
         f"> {time.strftime('%Y-%m-%d %H:%M')} · `scripts/make_hazard7_dataset.py` · 8 s · 24 fps · 768×432 · mp4(crf 12) 로 저장 후 다시 디코드해 판별 · "
         f"정답 = 팀원 판별기({ok[0].get('judge', '?') if ok else '?'} B2) · 오류 {len(rows) - len(ok)}개", "",
         "## 목표 라벨 적중 (의도 hazard → 그 라벨이 나옴 / 의도 near → 그 라벨이 안 나옴)", "",
         "**적중 = 합성으로 새로 생긴 라벨**(같은 바탕만 판별한 결과를 프레임 단위로 뺌, 무늬는 0.5 s 이상). 괄호 = 바탕 빼기 전.", "",
         f"바탕만 판별했을 때 라벨이 나온 표본: {sum(bool(r.get('clean_labels')) for r in ok)}/{len(ok)} "
         f"({', '.join(f'{k} {v}' for k, v in collections.Counter(l for r in ok for l in r.get('clean_labels', [])).most_common()) or '없음'})", "",
         "| 종류 → 목표 라벨 | hazard: 적중 | near: 그 라벨 새로 안 생김 | hazard 에서 새로 생긴 다른 라벨 | 우리 판정기 (hazard 중 위험) |", "|---|---|---|---|---|"]
    for kind in hazard7.KINDS:
        lab = hazard7.KIND_LABEL[kind]
        hz = [r for r in ok if r["kind"] == kind and r["intent"] == "hazard"]
        nr = [r for r in ok if r["kind"] == kind and r["intent"] == "near"]
        co = collections.Counter(l for r in hz for l in r.get("added_labels", r["team"]["labels"]) if l != lab)
        key = {"lum_flash": "lum", "intersection": "lum", "scene": "lum", "red_flash": "red", "lum_ext": "ext_lum", "red_ext": "ext_red"}.get(kind)
        ours = f"{sum(r['ours'][key] for r in hz)}/{len(hz)} ({key})" if key else "– (줄무늬 판정은 느려 생략)"
        ha = lambda x: x.get("hit_added", x["hit"])
        L.append(f"| {kind} → {lab} | {sum(ha(r) for r in hz)}/{len(hz)} ({sum(r['hit'] for r in hz)}) | {sum(not ha(r) for r in nr)}/{len(nr)} ({sum(not r['hit'] for r in nr)}) | "
                 f"{', '.join(f'{k} {v}' for k, v in co.most_common()) or '–'} | {ours} |")
    L += ["", "## 마스크 방식별 (hazard, 적중)", "", "| 마스크 | 개수 | 적중 |", "|---|---|---|"]
    mm = collections.defaultdict(list)
    for r in ok:
        if r["intent"] == "hazard":
            mm[r["recipe"].get("mask_mode", "static")].append(r.get("hit_added", r["hit"]))
    for k, v in sorted(mm.items()):
        L.append(f"| {k} | {len(v)} | {sum(v)}/{len(v)} |")
    L += ["", "## near 의 방식별 (새로 생긴 라벨)", "", "| 종류 | near 방식 | 개수 | 나온 라벨 |", "|---|---|---|---|"]
    grp = collections.defaultdict(list)
    for r in ok:
        if r["intent"] == "near":
            grp[(r["kind"], r["recipe"].get("near"))].append(r)
    for (k, how), rs in sorted(grp.items(), key=lambda x: (hazard7.KINDS.index(x[0][0]), str(x[0][1]))):
        c = collections.Counter(l for r in rs for l in r.get("added_labels", r["team"]["labels"]))
        L.append(f"| {k} | {how} | {len(rs)} | {', '.join(f'{a} {b}' for a, b in c.most_common()) or '새로 생긴 라벨 없음'} |")
    L += ["", "## 무늬 종류별 (pattern, hazard)", "", "| 무늬 | 개수 | 정적 공간 패턴 적중 |", "|---|---|---|"]
    for p in hazard7.PATTERNS:
        rs = [r for r in ok if r["kind"] == "pattern" and r["intent"] == "hazard" and r["recipe"]["pattern"] == p]
        if rs:
            L.append(f"| {p} | {len(rs)} | {sum(r.get('hit_added', r['hit']) for r in rs)}/{len(rs)} |")
    L += ["", "## 놓친 hazard 표본 (레시피 요약)", ""]
    for r in ok:
        if r["intent"] == "hazard" and not r.get("hit_added", r["hit"]):
            rc = r["recipe"]
            keys = [k for k in ("mode", "mask_mode", "pattern", "area", "halves", "dur", "cpd", "lo", "hi", "delta_cd", "sat", "motion", "sigma") if k in rc]
            L.append(f"- `{r['id']}` 새로 생긴 라벨 {r.get('added_labels') or '없음'} · " + ", ".join(
                f"{k}={round(rc[k], 3) if isinstance(rc[k], float) else rc[k]}" for k in keys))
    errs = [r for r in rows if "error" in r]
    if errs:
        L += ["", "## 오류", ""] + [f"- `{r['id']}` {r['error']}" for r in errs]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--team-dir", required=True)
    ap.add_argument("--split", default="val", choices=tuple(SRC))
    ap.add_argument("--per", type=int, default=6)
    ap.add_argument("--kinds", nargs="+", default=list(hazard7.KINDS))
    ap.add_argument("--name", default="t1")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--variant", default="partial", choices=("partial", "fullscreen"),
                    help="팀원 판별기: Partial/B2 vdcoreps(기본, 7라벨 + 픽셀 위험 마스크) 또는 Fullscreen/B2 vdcorefs")
    a = ap.parse_args()
    out = ROOT / "data" / "hazard7" / a.name / a.split
    out.mkdir(parents=True, exist_ok=True)
    man = out / "manifest.jsonl"
    done = {json.loads(l)["id"] for l in man.read_text(encoding="utf-8").splitlines() if l.strip()} if man.exists() else set()
    base = {"train": 0, "val": 1_000_000, "test": 2_000_000}[a.split] + a.seed * 100_000
    jobs = []
    for ki, kind in enumerate(a.kinds):
        for intent in ("hazard", "near"):
            for i in range(a.per):
                sid = f"{a.split}_{kind}_{intent}_{i:03d}"
                seed = base + hazard7.KINDS.index(kind) * 10_000 + (0 if intent == "hazard" else 5_000) + i
                if sid not in done:
                    jobs.append((sid, kind, intent, seed, a.split, str(out), a.team_dir, a.variant))
    print(f"만들 표본 {len(jobs)}개 (이미 {len(done)}개) → {out}", flush=True)
    t0 = time.time()
    with cf.ProcessPoolExecutor(a.workers) as ex, man.open("a", encoding="utf-8") as f:
        for i, r in enumerate(ex.map(one, jobs), 1):
            f.write(json.dumps(r, ensure_ascii=False) + "\n"); f.flush()
            if "error" in r or i % 10 == 0:
                print(f"  {i}/{len(jobs)} ({time.time() - t0:.0f}초) {r['id']} {r.get('error', '적중' if r.get('hit') else '')}", flush=True)
    rows = [json.loads(l) for l in man.read_text(encoding="utf-8").splitlines() if l.strip()]
    rep = ROOT / "checks" / "hazard7" / f"{a.name}_{a.split}.md"
    report(rows, rep, f"{a.name} / {a.split}")
    print("보고서:", rep, flush=True)


if __name__ == "__main__":
    main()
