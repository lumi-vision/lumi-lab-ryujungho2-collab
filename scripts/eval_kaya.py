"""Kaya et al. 2025 데이터셋(PSE_triggering_dataset)을 우리 판정기로 판정한다 — 영상은 재생하지 않고 수치만.

⚠️ 실제 위험 영상(섬광)이 들어 있다. 재생·미리보기하지 않는다. 결과에는 수치와 구간만 남긴다.
데이터: data/kaya/extracted/ (Git 제외). 출처·라벨 근거: docs/research/kaya_dataset.md

판정
  - 640×360 보다 큰 영상은 줄여서 판정한다(JUDGE_SIZE, 메모리 때문).
  - 휘도·적색: measure.analyze / analyze_red 를 96프레임 조각으로(1초 창보다 긴 겹침) 돌려 위험 프레임을 합친다.
    조각 경계에서 픽셀 상태가 새로 시작되므로 겹침이 1초 창보다 길어야 경계의 깜빡임을 놓치지 않는다.
  - 줄무늬: 주파수 방식(stripe_freq, E-Means 모델이 있으면 사용)을 약 5 fps 로 뽑은 프레임에 적용(지속 규칙도 5 fps 로).
    스캔선 방식(measure.stripe_masks)은 느려서 5초에 1장만 뽑아 면적만 잰다(판정은 하지 않음).
  - 같은 폴더에 `<영상 이름>_info.txt`(red:start-end / general:start-end, 프레임 구간)가 있으면 그 구간과 프레임 단위로 비교.
출력: data/kaya/results/<영상>.json, checks/kaya/README.md, checks/kaya/results.csv
사용: python scripts\\eval_kaya.py [--root data/kaya/extracted] [--limit N]
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path

import av
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from lumi_synth import measure, stripe_freq as sf  # noqa: E402

VIDEO_EXT = {".avi", ".mp4", ".mkv", ".mov", ".m4v", ".webm", ".mpg", ".mpeg", ".wmv", ".flv", ".gif"}
CHUNK_FRAMES = 96                 # 한 번에 판정하는 프레임 수 (+ 겹침). 메모리: 640×360 기준 수백 MB
PAT_FPS, SCAN_FPS = 5.0, 0.2      # 스캔선은 한 장에 수 초라 5초에 1장만 (참고용 면적)
JUDGE_SIZE = (640, 360)           # 이보다 큰 영상은 줄여서 판정 (2026-11-13: 1280×720 10초 조각의 휘도 배열이 3.7 GB 라
                                  # 메모리 7.6 GB PC 에서 MemoryError). Kaya 저자 쪽 PEAT 입력 규격도 640×360(Akyıldız 2025 6장)


def meta(path: Path) -> dict:
    with av.open(str(path)) as c:
        s = c.streams.video[0]
        cc = s.codec_context
        fps = float(s.average_rate or s.guessed_rate or 0) or 25.0
        dur = float(s.duration * s.time_base) if s.duration else (float(c.duration) / 1e6 if c.duration else None)
        return {"codec": cc.name, "width": cc.width, "height": cc.height, "fps": fps,
                "frames_reported": int(s.frames or 0), "duration_s": dur, "pix_fmt": cc.pix_fmt,
                "color_range": str(getattr(cc, "color_range", "")), "colorspace": str(getattr(cc, "colorspace", ""))}


def frames(path: Path, size: tuple[int, int] | None = None):
    """BGR 프레임을 하나씩. size(가로, 세로)를 주면 그보다 큰 영상은 면적 평균으로 줄인다."""
    import cv2
    with av.open(str(path)) as c:
        for fr in c.decode(video=0):
            img = fr.to_ndarray(format="bgr24")
            if size and (img.shape[1] > size[0] or img.shape[0] > size[1]):
                img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
            yield img


def parse_info(path: Path) -> dict[str, list[tuple[int, int]]] | None:
    """`red:120-180` / `general:300-420` 줄 → {'red': [(120,180)], 'general': [...]}. 없으면 None."""
    # Kaya 배포본: out/<번호>/correction_info.txt 의 "Original flashes : [('general', 7, 518), ('red', a, b)]"
    #   = 저자 판정기(W3C 모드 추정)가 원본에서 찾은 섬광 구간(프레임). 사람 정답이 아니라 '저자 판정'이다.
    ci = path.parent / "out" / path.stem / "correction_info.txt"
    if ci.exists():
        txt = ci.read_text(encoding="utf-8", errors="replace")
        m = re.search(r"Original flashes\s*:\s*\[(.*?)\]", txt)
        if m:
            out: dict[str, list[tuple[int, int]]] = {}
            for kind, a, b in re.findall(r"\(\s*'(\w+)'\s*,\s*(\d+)\s*,\s*(\d+)\s*\)", m.group(1)):
                out.setdefault(kind.lower(), []).append((int(a), int(b)))
            return out
    cands = [path.with_name(path.stem + "_info.txt"), path.with_name(path.stem.replace("_noised", "") + "_info.txt")]
    for c in cands:
        if c.exists():
            out: dict[str, list[tuple[int, int]]] = {}
            for line in c.read_text(encoding="utf-8", errors="replace").splitlines():
                m = re.match(r"\s*(\w+)\s*:\s*(\d+)\s*-\s*(\d+)", line)
                if m:
                    out.setdefault(m.group(1).lower(), []).append((int(m.group(2)), int(m.group(3))))
            return out
    return None


def run_video(path: Path, model) -> dict:
    md = meta(path)
    fps = md["fps"]
    over = measure.window_frames(fps) + 2              # 겹침: 1초 창보다 길게 (경계에 걸친 깜빡임용)
    chunk = max(CHUNK_FRAMES, over)
    lum_hz, red_hz = [], []
    lum_max = red_max = 0.0
    lum_area = red_area = 0.0
    pat_step, scan_step = max(1, int(round(fps / PAT_FPS))), max(1, int(round(fps / SCAN_FPS)))
    pat_area, scan_area = [], []
    buf: list[np.ndarray] = []
    start = 0                                         # buf[0] 의 전역 프레임 번호
    n = 0
    t0 = time.time()

    def judge(block: np.ndarray, base: int, final: bool) -> None:
        nonlocal lum_max, red_max, lum_area, red_area
        lum = measure.luminance_cd(block)
        a = measure.analyze(block, fps, lum=lum)
        r = measure.analyze_red(block, fps)
        lum_max, red_max = max(lum_max, a["max_flashes"]), max(red_max, r["max_flashes"])
        lum_area, red_area = max(lum_area, a["peak_area"]), max(red_area, r["peak_area"])
        lum_hz.extend((base + np.flatnonzero(a["hazard_frames"])).tolist())
        red_hz.extend((base + np.flatnonzero(r["hazard_frames"])).tolist())

    judged = None
    for img in frames(path, JUDGE_SIZE):
        judged = judged or f"{img.shape[1]}x{img.shape[0]}"
        if n % pat_step == 0 or n % scan_step == 0:
            lum1 = measure.luminance_cd(img[None])[0]
            if n % pat_step == 0:
                pat_area.append(float(sf.frame_stripes(lum1, model)["mask"].mean()))
            if n % scan_step == 0:
                scan_area.append(float(measure.stripe_masks(lum1[None])[0].mean()))
        buf.append(img)
        n += 1
        if len(buf) >= chunk + over:
            judge(np.stack(buf), start, False)
            keep = over
            start += len(buf) - keep
            buf = buf[-keep:]
    if len(buf) >= 2:
        judge(np.stack(buf), start, True)

    # 줄무늬 지속 규칙 (PAT_FPS 기준)
    pf = fps / pat_step
    cand = np.array(pat_area) > measure.PATTERN_AREA_STATIC
    w = measure.window_frames(pf)
    need = max(1, int(np.ceil(measure.PATTERN_MIN_SEC * pf - 1e-9)))
    cs = np.concatenate([[0], np.cumsum(cand)])
    counts = cs[np.minimum(np.arange(len(cand)) + w, len(cand))] - cs[:len(cand)] if len(cand) else np.array([0])
    pat_danger = bool((counts >= need).any())

    def intervals(idx: list[int]) -> list[tuple[int, int]]:
        idx = sorted(set(idx))
        out = []
        for i in idx:
            if out and i <= out[-1][1] + 1:
                out[-1][1] = i
            else:
                out.append([i, i])
        return [tuple(x) for x in out]

    res = {"file": path.name, **md, "judged_size": judged, "frames_decoded": n, "sec": round(time.time() - t0, 1),
           "lum": {"dangerous": lum_max > measure.MAX_FLASHES_PER_SEC, "max_flashes": lum_max, "peak_area": lum_area,
                   "intervals": intervals(lum_hz)},
           "red": {"dangerous": red_max > measure.MAX_FLASHES_PER_SEC, "max_flashes": red_max, "peak_area": red_area,
                   "intervals": intervals(red_hz)},
           "pattern_freq": {"dangerous": pat_danger, "peak_area": max(pat_area, default=0.0), "fps": pf,
                            "frames_over_40": int(cand.sum()), "emeans": model is not None},
           "pattern_scan_area": {"peak_area": max(scan_area, default=0.0), "fps": fps / scan_step,
                                 "frames_over_40": int((np.array(scan_area) > measure.PATTERN_AREA_STATIC).sum())}}
    info = parse_info(path)
    if info is not None:
        res["info"] = info
        res["info_match"] = {}
        for kind, ours in (("general", lum_hz), ("red", red_hz)):
            truth = set()
            for a, b in info.get(kind, []) + info.get("both", []):     # 'both' = 휘도와 적색 모두
                truth.update(range(a, b + 1))
            o = set(ours)
            res["info_match"][kind] = {"truth_frames": len(truth), "ours_frames": len(o), "overlap": len(truth & o),
                                       "recall": len(truth & o) / len(truth) if truth else None,
                                       "precision": len(truth & o) / len(o) if o else None}
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", type=Path, default=ROOT / "data/kaya/extracted")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--skip", default="out", help="이 이름의 하위 폴더는 건너뜀(저자 보정 결과 out/, 쉼표로 여러 개)")
    args = ap.parse_args()
    skip = {s for s in args.skip.split(",") if s}
    vids = sorted((p for p in args.root.rglob("*")
                   if p.suffix.lower() in VIDEO_EXT and not (set(p.relative_to(args.root).parts[:-1]) & skip)),
                  key=lambda p: (len(p.parts), int(p.stem) if p.stem.isdigit() else 10 ** 9, p.name))
    if args.limit:
        vids = vids[: args.limit]
    out_dir = ROOT / "data/kaya/results"
    out_dir.mkdir(parents=True, exist_ok=True)
    model = sf.load_model()
    print(f"영상 {len(vids)}개, E-Means 모델 {'있음' if model is not None else '없음(규칙만)'}", flush=True)
    rows = []
    for i, p in enumerate(vids, 1):
        cache = out_dir / (re.sub(r"[^0-9A-Za-z가-힣_.-]+", "_", str(p.relative_to(args.root))) + ".json")
        if cache.exists():
            res = json.loads(cache.read_text(encoding="utf-8"))
        else:
            try:
                res = run_video(p, model)
            except Exception as e:
                print(f"[{i}/{len(vids)}] 실패 {p.name}: {type(e).__name__}: {e}", flush=True)
                continue
            res["path"] = str(p.relative_to(args.root))
            cache.write_text(json.dumps(res, ensure_ascii=False, default=float), encoding="utf-8")
        rows.append(res)
        print(f"[{i}/{len(vids)}] {res['path']} {res['width']}x{res['height']} {res['fps']:.2f}fps "
              f"{res['frames_decoded']}f | 휘도 {res['lum']['max_flashes']}/s 적색 {res['red']['max_flashes']}/s "
              f"줄무늬(주파수) {res['pattern_freq']['peak_area']:.0%} 스캔선 {res['pattern_scan_area']['peak_area']:.0%} "
              f"| {res['sec']}s", flush=True)
    report(rows)
    return 0


def report(rows: list[dict]) -> None:
    d = ROOT / "checks/kaya"
    d.mkdir(parents=True, exist_ok=True)
    with open(d / "results.csv", "w", newline="", encoding="utf-8-sig") as fh:
        wr = csv.writer(fh)
        wr.writerow(["영상", "해상도", "fps", "프레임", "코덱", "휘도 위험", "휘도 최대 회/초", "휘도 최대 면적", "적색 위험", "적색 최대 회/초",
                     "줄무늬(주파수) 위험", "줄무늬(주파수) 최대 면적", "줄무늬(스캔선) 최대 면적", "info 휘도 재현율", "info 적색 재현율"])
        for r in rows:
            im = r.get("info_match", {})
            wr.writerow([r["path"], f"{r['width']}x{r['height']}", round(r["fps"], 3), r["frames_decoded"], r["codec"],
                         r["lum"]["dangerous"], r["lum"]["max_flashes"], round(r["lum"]["peak_area"], 3),
                         r["red"]["dangerous"], r["red"]["max_flashes"], r["pattern_freq"]["dangerous"],
                         round(r["pattern_freq"]["peak_area"], 3), round(r["pattern_scan_area"]["peak_area"], 3),
                         (im.get("general") or {}).get("recall"), (im.get("red") or {}).get("recall")])
    n = len(rows)
    cnt = lambda f: sum(1 for r in rows if f(r))  # noqa: E731
    L = ["# Kaya et al. 2025 데이터셋 판정 결과", "",
         f"> 만든 날 {time.strftime('%Y-%m-%d')} · 스크립트 `scripts/eval_kaya.py` · 데이터 `data/kaya/extracted/`(Git 제외) · 근거 `docs/research/kaya_dataset.md`",
         "> ⚠️ 실제 위험 영상. 재생하지 않고 수치만 기록했다. 데이터셋 라벨은 사람이 아니라 PEAT 판정·합성 구간일 가능성이 높다(조사 문서 §2).", "",
         "## 요약", "",
         f"- 영상 {n}개 · 휘도 위험 {cnt(lambda r: r['lum']['dangerous'])} · 적색 위험 {cnt(lambda r: r['red']['dangerous'])} · "
         f"줄무늬 위험(주파수) {cnt(lambda r: r['pattern_freq']['dangerous'])} · 셋 다 안전 "
         f"{cnt(lambda r: not (r['lum']['dangerous'] or r['red']['dangerous'] or r['pattern_freq']['dangerous']))}",
         f"- 줄무늬: 주파수 방식 최대 면적 > 40% 프레임이 있는 영상 {cnt(lambda r: r['pattern_freq']['frames_over_40'] > 0)}개, "
         f"스캔선(5초에 1장 표본) {cnt(lambda r: r['pattern_scan_area']['frames_over_40'] > 0)}개", "",
         "## 영상별", "", "| 영상 | 해상도·fps | 휘도 (회/초, 면적) | 적색 (회/초) | 줄무늬 주파수 / 스캔선 최대 면적 | info 구간 재현율 (휘도·적색) |",
         "|---|---|---|---|---|---|"]
    for r in rows:
        im = r.get("info_match")
        it = "–" if not im else " · ".join(
            f"{k} {v['recall']:.2f}" if v.get("recall") is not None else f"{k} –" for k, v in im.items())
        L.append(f"| {r['path']} | {r['width']}×{r['height']} {r['fps']:.2f} | {'**위험**' if r['lum']['dangerous'] else '안전'} "
                 f"{r['lum']['max_flashes']}, {r['lum']['peak_area']:.0%} | {'**위험**' if r['red']['dangerous'] else '안전'} "
                 f"{r['red']['max_flashes']} | {'**위험** ' if r['pattern_freq']['dangerous'] else ''}{r['pattern_freq']['peak_area']:.0%} / "
                 f"{r['pattern_scan_area']['peak_area']:.0%} | {it} |")
    L += ["", "## 읽는 법·한계", "",
          "- 640×360 보다 큰 영상은 줄여서 판정했다(작은 영역의 면적이 조금 달라질 수 있음). "
          "휘도·적색은 96프레임 조각(1초 창보다 긴 겹침)으로 판정해 합쳤다. 조각 경계에서 픽셀 상태가 새로 시작되는 영향은 겹침으로 줄였지만 0 은 아니다.",
          "- 줄무늬는 정지 그림 판정을 약 5 fps 표본에 적용하고 지속 규칙만 그 표본률로 계산했다. 움직임 분류는 하지 않았다.",
          "- 스캔선 방식은 5초에 1장 표본의 면적만 잰 참고값이다(짧은 줄무늬 장면은 표본에 안 걸릴 수 있음).",
          "- 영상의 YUV range·행렬은 PyAV 기본 변환(bgr24)을 따랐다(docs/research/opencv_decode.md 참고)."]
    (d / "README.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("보고서:", d / "README.md")


if __name__ == "__main__":
    raise SystemExit(main())
