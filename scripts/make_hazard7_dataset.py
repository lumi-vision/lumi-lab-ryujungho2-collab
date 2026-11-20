"""7종 위험 라벨 합성 데이터 (src/lumi_synth/hazard7.py) — 만들고, mp4 로 저장하고, 다시 디코드해서 판별기로 라벨을 붙인다.

    python scripts/make_hazard7_dataset.py --team-dir <lumi-lab-rockrovks> --split val --per 6 --name t1
    python scripts/make_hazard7_dataset.py ... --phase gen      # 생성만 (GPU = 섞기)
    python scripts/make_hazard7_dataset.py ... --phase judge    # 판정만 (GPU = 팀원 판별기 한 프로세스, CPU = 우리 판정기)

두 단계로 나눈다: 생성 단계에서는 GPU 로 섞고, 판정 단계에서는 GPU 를 판별기가 쓴다(서로 GPU 를 다투지 않음).
판정 기준이 바뀌면 --phase judge 만 다시 돌려 라벨을 새로 붙인다(영상은 그대로). 각 단계는 이어 하기가 된다.

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


# 바탕 소스 비율 (분할별). movie = BBB(train/val 구간) 또는 Sintel(test), south = South 2021 안전 영상(사용자 결정 2026-11-20),
# graphic = 합성 그래픽(단색·그라데이션·도형·글자, 천천히 움직임)
BG_MIX = {"train": {"movie": 0.35, "south": 0.45, "graphic": 0.20},
          "val": {"movie": 0.35, "south": 0.45, "graphic": 0.20},
          "test": {"movie": 1.0}}
_SOUTH = None


def south_bg_split() -> dict:
    """South 안전 영상을 파일 SHA-256 으로 80:20 (train:val) 나눈 목록. data/hazard7/south_bg_split.json 에 고정 저장."""
    global _SOUTH
    if _SOUTH is not None:
        return _SOUTH
    import csv
    import hashlib
    p = ROOT / "data" / "hazard7" / "south_bg_split.json"
    if p.exists():
        _SOUTH = json.loads(p.read_text(encoding="utf-8"))
        return _SOUTH
    base = ROOT / "data" / "real" / "south2021"
    files = {f.stem: f for f in (base / "social-media-study-gifs").rglob("*") if f.suffix.lower() in (".gif", ".mp4")}
    with open(base / "social-media-study.csv", newline="", encoding="utf-8-sig") as f:
        safe = [r["label"] for r in csv.DictReader(f) if r["dangerous"] != "dangerous" and r["label"] in files]
    out = {"train": [], "val": [], "note": "South 안전 영상 → hazard7 바탕. 해시 앞 8자리 mod 5 == 0 → val. 같은 해시는 같은 분할"}
    for lab in sorted(safe):
        h = hashlib.sha256(files[lab].read_bytes()).hexdigest()
        out["val" if int(h[:8], 16) % 5 == 0 else "train"].append(str(files[lab].relative_to(ROOT)))
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, ensure_ascii=False, indent=0), encoding="utf-8")
    _SOUTH = out
    return out


def fit_frame(img: np.ndarray, how: str) -> np.ndarray:
    """임의 크기 → 768×432. crop = 16:9 로 가운데 자르기, pad = 흐린 확대 바탕 위에 비율 유지(소셜 영상에서 흔한 꼴)."""
    import cv2
    h, w = img.shape[:2]
    if how == "crop":
        if w / h > W / H:
            nw = int(h * W / H); x0 = (w - nw) // 2; img = img[:, x0:x0 + nw]
        else:
            nh = int(w * H / W); y0 = (h - nh) // 2; img = img[y0:y0 + nh]
        return cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA if img.shape[1] > W else cv2.INTER_LINEAR)
    bg = cv2.GaussianBlur(cv2.resize(img, (W, H)), (0, 0), 12)
    s = min(W / w, H / h)
    nw, nh = max(2, int(w * s)), max(2, int(h * s))
    fg = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
    y0, x0 = (H - nh) // 2, (W - nw) // 2
    bg[y0:y0 + nh, x0:x0 + nw] = fg
    return bg


def south_clip(split: str, rng) -> tuple[np.ndarray, dict]:
    """South 안전 영상 하나 → 24 fps 로 시간 기준 재표본, 8 초가 안 되면 반복(GIF 는 반복 재생된다)."""
    sys.path.insert(0, str(ROOT / "scripts"))
    from eval_real_hazards import decode as dec_any
    lst = south_bg_split()[split]
    for _ in range(10):
        f = lst[int(rng.integers(len(lst)))]
        try:
            frames, fps = dec_any(ROOT / f)
            break
        except Exception:  # noqa: BLE001 — 한 장짜리 GIF 등은 건너뜀
            continue
    n = len(frames)
    t = np.arange(N) / FPS + float(rng.uniform(0, max(0.0, n / fps - N / FPS)) if n / fps > N / FPS else 0)
    idx = np.floor(t * fps).astype(int) % n
    how = str(rng.choice(["crop", "pad"]))
    cache = {}
    out = np.stack([cache.setdefault(i, fit_frame(frames[i], how)) for i in idx])
    return out, {"source": "south", "file": f, "fit": how, "src_fps": round(fps, 2), "src_frames": n}


def bg_clip(split: str, rng) -> tuple[np.ndarray, dict]:
    """바탕 8 초 → (N×432×768×3 BGR uint8, 정보). 소스는 BG_MIX 비율로 고른다."""
    mix = BG_MIX[split]
    src = str(rng.choice(list(mix), p=list(mix.values())))
    if src == "south":
        return south_clip(split, rng)
    if src == "graphic":
        return hazard7.graphic_bg(N, H, W, rng), {"source": "graphic"}
    clip, start = movie_clip(split, rng)
    return clip, {"source": "movie", "file": SRC[split][0].split("/")[-1], "start": round(start, 3)}


def movie_clip(split: str, rng) -> tuple[np.ndarray, float]:
    """영화 분할 구간에서 무작위 8초 → N×432×768×3 BGR uint8 (24 fps 가 아니면 시간 기준으로 다시 뽑음)."""
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


def gen_one(job):
    """1단계(생성): 바탕 디코드(CPU) → 마스크(CPU) → 섞기(GPU) → mp4 인코딩(CPU). 바탕만 인코딩한 것도 저장(판정 단계에서 뺄셈용)."""
    sid, kind, intent, seed, split, out_dir, device = job
    t0 = time.time()
    try:
        rng = np.random.default_rng(seed)
        clean, start = bg_clip(split, rng)
        other, start2 = bg_clip(split, rng) if kind in ("scene", "intersection") else (None, None)
        r = hazard7.sample_recipe(kind, intent, seed, N, FPS)
        clean = hazard7.prepare_bg(r, clean)              # 어두운 장면 흉내(일부 레시피) — 바탕 mp4 도 이것으로 저장
        dev = None
        if device != "cpu":
            import torch
            dev = torch.device(device)
        out, a = hazard7.render(r, clean, other, device=dev)
        encode(out, Path(out_dir) / f"{sid}.mp4")
        encode(clean, Path(out_dir) / f"{sid}.clean.mp4")
        r = {k: (v if not isinstance(v, np.generic) else v.item()) for k, v in r.items()}
        return {"id": sid, "split": split, "kind": kind, "intent": intent, "target_label": hazard7.KIND_LABEL[kind], "recipe": r,
                "bg": start, "bg_other": start2,
                "region_mean": round(float(a.mean()), 4), "gen_device": device, "gen_sec": round(time.time() - t0, 1)}
    except Exception as e:  # noqa: BLE001
        import traceback
        return {"id": sid, "kind": kind, "intent": intent, "error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()[-1200:]}


def ours_one(path_s: str) -> dict:
    """2단계의 CPU 몫: 우리 판정기(휘도·적색·IRIS 연장)."""
    return ours_judge(decode(Path(path_s)))


def judge_rows(g: dict, out_dir: Path, team_dir: str, variant: str, ours: dict) -> dict:
    """2단계의 GPU 몫: 팀원 판별기를 합성본·바탕에 → 라벨, 새로 생긴 라벨, 픽셀 위험 마스크."""
    global VARIANT
    VARIANT = variant
    sid, kind = g["id"], g["kind"]
    tj = team_judge(decode(out_dir / f"{sid}.mp4"), team_dir)
    # 바탕만도 판별 — 팀원 판별기가 손대지 않은 바탕에서도 '정적 공간 패턴' 등을 내므로, 합성으로 '새로 생긴' 라벨을 따로 센다
    cj = team_judge(decode(out_dir / f"{sid}.clean.mp4"), team_dir)
    added = tj["masks"] & ~cj["masks"]
    extra = {"weight": tj["weight"]} if tj["weight"] is not None else {}
    np.savez_compressed(out_dir / f"{sid}.labels.npz", team=tj["masks"], clean=cj["masks"], added=added,
                        labels=np.array(hazard7.LABELS), **extra)
    target = hazard7.KIND_LABEL[kind]
    ti = hazard7.LABELS.index(target)
    min_added = 12 if kind == "pattern" else 1                    # 무늬는 0.5 s 이상 새로 생겨야
    added_labels = [l for l, a_ in zip(hazard7.LABELS, added) if a_.sum() >= (12 if l == hazard7.LABELS[-1] else 1)]
    return {**g, "team": {"labels": tj["labels"], "dangerous": tj["dangerous"],
                          "frames": {l: int(m.sum()) for l, m in zip(hazard7.LABELS, tj["masks"])}},
            "clean_labels": cj["labels"], "added_labels": added_labels, "judge": variant, "ours": ours,
            "hit": target in tj["labels"], "hit_added": bool(added[ti].sum() >= min_added)}


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
    ap.add_argument("--phase", default="all", choices=("gen", "judge", "all"),
                    help="gen: 합성(GPU)·인코딩만 · judge: 판정만(팀원 판별기 GPU 한 프로세스 + 우리 판정기 CPU 병렬) · all: 차례로")
    ap.add_argument("--gen-device", default="cuda", help="합성(섞기) 장치. cpu 면 numpy")
    a = ap.parse_args()
    out = ROOT / "data" / "hazard7" / a.name / a.split
    out.mkdir(parents=True, exist_ok=True)
    gen_man, man = out / "gen.jsonl", out / "manifest.jsonl"
    read = lambda p: [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()] if p.exists() else []
    t0 = time.time()

    if a.phase in ("gen", "all"):                     # ---- 1단계: 생성 (GPU 는 섞기에)
        done = {r["id"] for r in read(gen_man) if "error" not in r}
        base = {"train": 0, "val": 1_000_000, "test": 2_000_000}[a.split] + a.seed * 100_000
        jobs = []
        for kind in a.kinds:
            for intent in ("hazard", "near"):
                for i in range(a.per):
                    sid = f"{a.split}_{kind}_{intent}_{i:03d}"
                    seed = base + hazard7.KINDS.index(kind) * 10_000 + (0 if intent == "hazard" else 5_000) + i
                    if sid not in done:
                        jobs.append((sid, kind, intent, seed, a.split, str(out), a.gen_device))
        print(f"[생성] {len(jobs)}개 (이미 {len(done)}개) · 섞기 {a.gen_device} · 작업자 {a.workers} → {out}", flush=True)
        with cf.ProcessPoolExecutor(a.workers) as ex, gen_man.open("a", encoding="utf-8") as f:
            for i, r in enumerate(ex.map(gen_one, jobs), 1):
                f.write(json.dumps(r, ensure_ascii=False) + "\n"); f.flush()
                if "error" in r or i % 10 == 0:
                    print(f"  생성 {i}/{len(jobs)} ({time.time() - t0:.0f}초) {r['id']} {r.get('error', '')}", flush=True)

    if a.phase in ("judge", "all"):                   # ---- 2단계: 판정 (GPU 는 팀원 판별기에)
        gens = {r["id"]: r for r in read(gen_man) if "error" not in r}
        done = {r["id"] for r in read(man) if "error" not in r}
        todo = [g for sid, g in sorted(gens.items()) if sid not in done]
        print(f"[판정] {len(todo)}개 (이미 {len(done)}개) · 팀원 {a.variant} B2 GPU 1 프로세스 · 우리 판정기 CPU {a.workers}", flush=True)
        t1 = time.time()
        with cf.ProcessPoolExecutor(a.workers) as ex, man.open("a", encoding="utf-8") as f:
            futs = {g["id"]: ex.submit(ours_one, str(out / f"{g['id']}.mp4")) for g in todo}   # CPU: 우리 판정기
            for i, g in enumerate(todo, 1):
                try:
                    row = judge_rows(g, out, a.team_dir, a.variant, futs[g["id"]].result())     # GPU: 팀원 판별기
                except Exception as e:  # noqa: BLE001
                    import traceback
                    row = {**g, "error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()[-1200:]}
                f.write(json.dumps(row, ensure_ascii=False) + "\n"); f.flush()
                if "error" in row or i % 10 == 0:
                    print(f"  판정 {i}/{len(todo)} ({time.time() - t1:.0f}초) {row['id']} {row.get('error', '적중' if row.get('hit_added') else '')}", flush=True)
    rows = read(man)
    if not rows:
        return
    rep = ROOT / "checks" / "hazard7" / f"{a.name}_{a.split}.md"
    report(rows, rep, f"{a.name} / {a.split}")
    print("보고서:", rep, flush=True)


if __name__ == "__main__":
    main()
