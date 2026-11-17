"""HazNet 학습·평가용 캐시를 만든다 (data/hazcls/<이름>/, Git 제외).

  synth : G 데이터(data/gdr/<버전>/<폴더>) → 클립 하나 = 표본 하나. 정답 = 하드 판정기(manifest judge), 튜브 = 지도 보조 정답
  real  : 실제 위험 영상(Kaya 2025 원본 22개, South 2021) → 24 fps 로 다시 뽑아 2초 창으로 자름. 정답은 영상 단위(데이터셋 라벨)

⚠️ 실제 영상·합성 영상 모두 광과민성 위험 자극이다. 프레임은 메모리·캐시 배열로만 다루고 재생·미리보기하지 않는다.

실제 영상 분할 (시드 고정, docs: checks/hazcls/README.md):
  - South 2021 위험 163개 → 전부 test (학습·검증 금지 — 프로젝트 규칙)
  - Kaya 2025 원본 22개 → test 6개 · val 16개 (val 은 조기 종료 판단에만, 학습에는 안 씀)
  - South 2021 안전 1,837개 → val 300 · test 600 · 나머지는 쓰지 않음
  Kaya 'out/' (보정본과 원본을 이어 붙인 영상)은 정답이 없어 쓰지 않는다.

배열 (np.load(mmap_mode='r')):
  tem.npy  N×48×90×160×3 uint8 BGR (2초·24 fps, 시간 갈래)
  pat.npy  N×2×360×640×3 uint8 BGR (프레임 12·36, 공간 갈래)
  y.npy    N×3 uint8 (lum, red, pattern) — synth 만
  tub_t.npy N×2×48×45×80 uint8 (휘도·적색 튜브 면적 비율 ×255), tub_p.npy N×2×45×80 uint8 (줄무늬) — synth 만
  rows.jsonl 표본(창)마다 한 줄: synth = id·폴더·의도, real = item·창 번호
  items.jsonl (real) 영상마다 한 줄: source, file, split, any, lum, red, pattern

사용:
  python scripts/hazcls_cache.py synth --name train --dirs data/gdr/v8/train data/gdr/v9s/train
  python scripts/hazcls_cache.py synth --name val   --dirs data/gdr/v8/val data/gdr/v9s/val
  python scripts/hazcls_cache.py synth --name val_rule --dirs data/gdr/v8/val_rule
  python scripts/hazcls_cache.py real  --name real
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import csv
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from lumi_synth import generate  # noqa: E402
from lumi_synth.hazcls import MH, MW, P, PAT_IDX, PH, PW, T, TH, TW  # noqa: E402

OUT = ROOT / "data" / "hazcls"
REAL = ROOT / "data" / "real"
FPS = 24.0


def shrink(frames: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """T×H×W×3 (T=48) → tem 48×90×160×3, pat 2×360×640×3."""
    tem = np.stack([cv2.resize(f, (TW, TH), interpolation=cv2.INTER_AREA) for f in frames])
    pat = np.stack([cv2.resize(frames[i], (PW, PH), interpolation=cv2.INTER_AREA if frames.shape[2] >= PW else cv2.INTER_LINEAR)
                    for i in PAT_IDX])
    return tem, pat


def pool_tube(t: np.ndarray) -> np.ndarray:
    """bool …×H×W → …×45×80 uint8 면적 비율×255."""
    sh = t.shape
    flat = t.reshape(-1, sh[-2], sh[-1]).astype(np.float32)
    out = np.stack([cv2.resize(x, (MW, MH), interpolation=cv2.INTER_AREA) for x in flat])
    return (out * 255 + 0.5).astype(np.uint8).reshape(*sh[:-2], MH, MW)


# ---------------- synth ----------------

def synth(name: str, dirs: list[Path], threads: int) -> None:
    metas = []
    for d in dirs:
        rows = [json.loads(x) for x in (d / "manifest.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
        first = d == dirs[0]
        for r in rows:
            if r["hazard"] is None and not first:          # 바탕 그대로는 첫 폴더 것만 (같은 바탕 중복 방지)
                continue
            r["folder"] = str(d)
            metas.append(r)
    for sp in {m["split"] for m in metas}:
        generate.load_backgrounds(sp)
    n = len(metas)
    out = OUT / name
    out.mkdir(parents=True, exist_ok=True)
    mm = lambda f, shape: np.lib.format.open_memmap(out / f, mode="w+", dtype=np.uint8, shape=shape)
    tem, pat = mm("tem.npy", (n, T, TH, TW, 3)), mm("pat.npy", (n, P, PH, PW, 3))
    tub_t, tub_p = mm("tub_t.npy", (n, 2, T, MH, MW)), mm("tub_p.npy", (n, P, MH, MW))
    y = np.zeros((n, 3), np.uint8)

    def one(i: int):
        m = metas[i]
        clips, _ = generate.load_backgrounds(m["split"])
        with np.load(Path(m["folder"]) / f"{m['id']}.npz") as z:
            x = z["flicker"] if "flicker" in z else clips[m["bg"]]
            tubes = z["tubes"]
        a, b = shrink(x)
        return i, a, b, pool_tube(tubes[:2]), pool_tube(tubes[2][list(PAT_IDX)])

    t0 = time.time()
    with cf.ThreadPoolExecutor(threads) as ex:
        for k, (i, a, b, tt, tp) in enumerate(ex.map(one, range(n)), 1):
            tem[i], pat[i], tub_t[i], tub_p[i] = a, b, tt, tp
            j = metas[i]["judge"]
            y[i] = [bool(j["lum"]["dangerous"]), bool(j["red"]["dangerous"]), bool(j["stripe"]["dangerous"])]
            if k % 200 == 0:
                print(f"  {k}/{n} ({time.time() - t0:.0f}초)", flush=True)
    for a in (tem, pat, tub_t, tub_p):
        a.flush()
    np.save(out / "y.npy", y)
    with (out / "rows.jsonl").open("w", encoding="utf-8") as f:
        for m in metas:
            f.write(json.dumps({"id": m["id"], "folder": m["folder"], "split": m["split"], "hazard": m["hazard"],
                                "intent": m["intent"], "generator": m["generator"],
                                "ckpt": (m.get("recipe") or {}).get("checkpoint")}, ensure_ascii=False) + "\n")
    print(f"끝: {out} · {n}개 · 위험 휘도 {y[:, 0].sum()} · 적색 {y[:, 1].sum()} · 무늬 {y[:, 2].sum()} · 안전 {(y.sum(1) == 0).sum()}", flush=True)


# ---------------- real ----------------

def real_items(seed: int = 0) -> list[dict]:
    rng = np.random.default_rng(seed)
    items = []
    kaya = sorted((REAL / "kaya2025" / "extracted").glob("*.*"), key=lambda p: int(p.stem) if p.stem.isdigit() else 999)
    kaya = [p for p in kaya if p.suffix.lower() in (".mp4", ".avi", ".mov", ".mkv")]
    test_k = set(rng.choice(len(kaya), 6, replace=False).tolist())
    for i, p in enumerate(kaya):
        items.append({"source": "kaya", "file": str(p.relative_to(REAL)), "split": "test" if i in test_k else "val",
                      "any": 1, "lum": None, "red": None, "pattern": None})
    base = REAL / "south2021"
    files = {p.stem: p for p in (base / "social-media-study-gifs").rglob("*") if p.is_file() and p.suffix.lower() in (".gif", ".mp4")}
    with open(base / "social-media-study.csv", newline="", encoding="utf-8-sig") as f:
        rows = [r for r in csv.DictReader(f) if r["label"] in files]
    safe = [r for r in rows if r["dangerous"] != "dangerous"]
    perm = rng.permutation(len(safe))
    split_safe = {int(k): ("val" if j < 300 else "test" if j < 900 else None) for j, k in enumerate(perm)}
    for r in rows:
        if r["dangerous"] == "dangerous":
            sp = "test"
        else:
            sp = split_safe[safe.index(r)]
            if sp is None:
                continue
        items.append({"source": "south", "file": str(files[r["label"]].relative_to(REAL)), "split": sp,
                      "any": int(r["dangerous"] == "dangerous"), "lum": int(r["flash"] == "flash"),
                      "red": int(r["red"] == "red"), "pattern": int(r["pattern"] == "pattern")})
    return items


def pair_items() -> list[dict]:
    """Kaya out/<n>/concatenated_video.mp4 = 원본과 보정본을 가로로 이어 붙인 영상 → 보정본 쪽만(짝 비교용, 정답 없음)."""
    out = []
    for d in sorted((REAL / "kaya2025" / "extracted" / "out").iterdir(), key=lambda p: int(p.name)):
        f = d / "concatenated_video.mp4"
        if f.exists():                                    # 같은 파일의 두 쪽을 같은 처리로 (해상도 차이가 점수에 섞이지 않게)
            for side in ("active", "calm"):
                out.append({"source": f"kaya_pair_{side}", "file": str(f.relative_to(REAL)), "split": "pair", "any": None,
                            "pair": int(d.name), "half": side, "lum": None, "red": None, "pattern": None})
    return out


def pick_half(frames: np.ndarray, side: str) -> np.ndarray:
    """가로로 이어 붙인 영상에서 프레임 간 밝기 변화가 작은 쪽(calm = 보정본) 또는 큰 쪽(active = 원본). 눈으로 보지 않고 수치로."""
    w = frames.shape[2] // 2
    halves = {"left": frames[:, :, :w], "right": frames[:, :, w:2 * w]}
    act = {k: float(np.abs(np.diff(v.mean(-1, dtype=np.float32), axis=0)).mean()) for k, v in halves.items()}
    k = (min if side == "calm" else max)(act, key=act.get)
    return np.ascontiguousarray(halves[k])


def windows(path_s: str, max_win: int, half: str | None = None):
    """영상 → 2초 창 목록 [(tem, pat)] (24 fps 로 시간 기준 재표본, 짧으면 반복 — GIF 는 반복 재생된다)."""
    from eval_real_hazards import decode
    frames, fps = decode(Path(path_s))
    if half:
        frames = pick_half(frames, half)
    n = len(frames)
    dur = n / fps
    t = np.arange(0, max(dur, T / FPS), 1 / FPS)
    idx = np.floor(t * fps).astype(int) % n
    nw = max(1, len(idx) // T)
    starts = np.arange(nw) * T
    if nw > max_win:
        starts = starts[np.linspace(0, nw - 1, max_win).round().astype(int)]
    out = []
    for s in starts:
        seg = frames[idx[s:s + T]]
        if len(seg) < T:
            seg = frames[np.resize(idx, s + T)[s:s + T]]
        out.append(shrink(seg))
    return out, fps, n


def _job(args):
    k, path_s, max_win, half = args
    try:
        w, fps, n = windows(path_s, max_win, half)
        return k, w, {"fps": round(fps, 3), "frames": n}
    except Exception as e:  # noqa: BLE001
        return k, [], {"error": f"{type(e).__name__}: {e}"}


def real(name: str, workers: int, max_win_south: int, max_win_kaya: int, pairs: bool = False) -> None:
    items = pair_items() if pairs else real_items()
    out = OUT / name
    out.mkdir(parents=True, exist_ok=True)
    jobs = [(k, str(REAL / it["file"]), max_win_south if it["source"] == "south" else max_win_kaya, it.get("half"))
            for k, it in enumerate(items)]
    # 창 수를 미리 모르므로 조각 파일로 모은 뒤 합친다
    parts = out / "_parts"
    parts.mkdir(exist_ok=True)
    t0 = time.time()
    with cf.ProcessPoolExecutor(workers) as ex:
        for c, (k, w, info) in enumerate(ex.map(_job, jobs, chunksize=4), 1):
            items[k].update(info)
            items[k]["n_win"] = len(w)
            if w:
                np.save(parts / f"{k}_tem.npy", np.stack([a for a, _ in w]))
                np.save(parts / f"{k}_pat.npy", np.stack([b for _, b in w]))
            if c % 100 == 0:
                print(f"  {c}/{len(items)} ({time.time() - t0:.0f}초)", flush=True)
    total = sum(it["n_win"] for it in items)
    tem = np.lib.format.open_memmap(out / "tem.npy", mode="w+", dtype=np.uint8, shape=(total, T, TH, TW, 3))
    pat = np.lib.format.open_memmap(out / "pat.npy", mode="w+", dtype=np.uint8, shape=(total, P, PH, PW, 3))
    rows, o = [], 0
    for k, it in enumerate(items):
        if not it["n_win"]:
            continue
        a, b = np.load(parts / f"{k}_tem.npy"), np.load(parts / f"{k}_pat.npy")
        tem[o:o + len(a)], pat[o:o + len(a)] = a, b
        rows += [{"item": k, "win": j} for j in range(len(a))]
        o += len(a)
        (parts / f"{k}_tem.npy").unlink(); (parts / f"{k}_pat.npy").unlink()
    tem.flush(); pat.flush()
    parts.rmdir()
    with (out / "rows.jsonl").open("w", encoding="utf-8") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)
    with (out / "items.jsonl").open("w", encoding="utf-8") as f:
        f.writelines(json.dumps(it, ensure_ascii=False) + "\n" for it in items)
    bad = [it["file"] for it in items if "error" in it]
    print(f"끝: {out} · 영상 {len(items)}개 · 창 {total}개 · 실패 {len(bad)} {bad[:5]}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=("synth", "real"))
    ap.add_argument("--name", required=True)
    ap.add_argument("--dirs", nargs="+", type=Path)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-win-south", type=int, default=8)
    ap.add_argument("--max-win-kaya", type=int, default=60)
    ap.add_argument("--pairs", action="store_true", help="Kaya 보정본만 (짝 비교용 캐시)")
    a = ap.parse_args()
    if a.kind == "synth":
        synth(a.name, [d if d.is_absolute() else ROOT / d for d in a.dirs], a.threads)
    else:
        real(a.name, a.workers, a.max_win_south, a.max_win_kaya, a.pairs)


if __name__ == "__main__":
    main()
