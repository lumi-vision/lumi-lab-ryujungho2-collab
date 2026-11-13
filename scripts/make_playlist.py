"""영상 플레이리스트(로컬 웹페이지) 만들기: npz·gif·mp4·avi·mkv → 브라우저용 mp4 + index.html.

클립마다 [영상 | 정답 | 우리 판정기] 세 칸을 나란히 붙인 mp4 를 만들고, 팀원 판별기(stage4 `_pse_graph.png`)처럼
위험 구간 타임라인 · 임계치 대비 위험도 곡선 · 1초 전이 수 그래프를 재생 위치와 맞춰 보여준다.

⚠️ 위험 자극(광과민성) 영상이 들어 있다. 이 PC 에서만 열고, 공유·발표 화면·녹화에 쓰지 않는다.
   결과 폴더(data/playlist/)는 data/ 아래라 Git 에 올라가지 않는다.

사용 (저장소 폴더에서)
    python scripts\\make_playlist.py                           # 기본 묶음 전체
    python scripts\\make_playlist.py --groups oracle --limit 5 # 묶음 하나, 5개만
    python scripts\\make_playlist.py --extra D:\\videos         # 폴더 안 gif/mp4/avi/mkv/mov/npz 추가
    python scripts\\make_playlist.py --no-detect               # 판정 없이 변환만 (빠름)
결과: data/playlist/index.html 을 더블클릭해서 연다. 이미 만든 클립은 건너뛴다(--force 로 다시 만듦).

묶음(--groups)
    oracle      data/oracle_128x72_24fps 의 합성 휘도 깜빡임 352개 (정답: manifest·npz)
    gan         data/gan/{val, v1_div005/val, v1_div05/val} GAN 출력 288개 (정답: manifest·npz)
    old         datasets/{oracle,gan} 초기화(11-08) 전 데이터 (2026-11-13 에 지움 — 폴더가 없으면 건너뜀)
    backgrounds 합성 전 배경 클립 88개 (정답 없음 — 오탐 확인용)
    demo        data/stripe_demo 줄무늬 시연 영상 8개 (scripts/make_stripe_demo.py, 정답 없음 — 줄무늬 탐지 눈으로 확인용)
    sweep       data/hazard_sweep 조건 스윕 114개 (회색 바탕, 768×432 — 줄무늬 판정이 클립당 수 분 걸림)
"""
from __future__ import annotations

import argparse
import functools
import hashlib
import json
import re
import sys
import time
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Callable

import av
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from lumi_synth import measure, oracle, stripe_freq as sf  # noqa: E402

VIDEO_EXT = {".gif", ".mp4", ".avi", ".mkv", ".mov", ".m4v", ".webm"}
DEFAULT_GROUPS = "oracle,gan,backgrounds,demo,sweep"   # 'old'(datasets/)는 11-13 에 지워서 기본에서 뺌
PANEL_H = 240                     # 한 칸 높이(px). 작은 영상은 정수배 최근접 확대, 큰 영상은 축소
GAP = 4                           # 칸 사이 간격(px)
FORMAT_VERSION = "2"              # 출력 형식이 바뀌면 올려서 캐시를 무효화 (2: 넷째 칸 줄무늬 주파수 탐지기)

# 겹쳐 칠할 색 (BGR)
COL_GT = np.array([80, 200, 60], np.float32)      # 정답: 초록
COL_LUM = np.array([0, 165, 255], np.float32)     # 휘도: 주황
COL_RED = np.array([255, 0, 255], np.float32)     # 적색: 자홍
COL_PAT = np.array([255, 220, 0], np.float32)     # 줄무늬: 하늘
COL_PAT2 = np.array([0, 225, 255], np.float32)    # 줄무늬(주파수 탐지기: 쿼드트리+가버): 노랑
COL_BLOCK_OK = (255, 255, 255)                     # 쿼드트리가 줄무늬로 인정한 블록 테두리: 흰색
COL_BLOCK_NO = (200, 80, 255)                      # 명암은 있지만 기각한 블록 테두리: 분홍


@dataclass
class Item:
    id: str
    group: str
    title: str
    source: Path
    load: Callable[[], tuple[np.ndarray, float, dict]]   # → (BGR 프레임 T×H×W×3, fps, 정답 dict)
    label: bool | None = None
    label_note: str = ""
    meta: dict = field(default_factory=dict)
    sub: str = ""                                        # 한 파일 안 여러 클립일 때 구분자


# ---- 읽기 ---------------------------------------------------------------------------------
@functools.lru_cache(maxsize=4)
def _npz_array(path: str, key: str) -> np.ndarray:
    with np.load(path, allow_pickle=False) as z:
        return z[key]


def _npz(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def decode_video(path: Path, max_seconds: float, max_side: int) -> tuple[np.ndarray, float, dict]:
    """gif·mp4·avi·mkv·mov → BGR 프레임. 앞에서 max_seconds 초까지, 긴 변이 max_side 를 넘으면 축소."""
    note = {}
    if path.suffix.lower() == ".gif":
        from PIL import Image, ImageSequence
        frames, durs = [], []
        with Image.open(path) as im:
            for fr in ImageSequence.Iterator(im):
                frames.append(cv2.cvtColor(np.asarray(fr.convert("RGB")), cv2.COLOR_RGB2BGR))
                d = fr.info.get("duration", 100) or 100
                durs.append(100 if d <= 10 else d)          # 브라우저처럼 10 ms 이하는 100 ms 로 보여 줌
        fps = 1000.0 / float(np.median(durs))
        frames = frames[: max(2, int(max_seconds * fps))]
        if len(set(durs)) > 1:
            note["gif"] = "프레임 간격이 일정하지 않아 중앙값으로 맞춤"
    else:
        with av.open(str(path)) as c:
            s = c.streams.video[0]
            rate = s.average_rate or s.guessed_rate or 30
            fps = float(rate)
            limit = max(2, int(max_seconds * fps))
            frames = []
            for fr in c.decode(s):
                frames.append(fr.to_ndarray(format="bgr24"))
                if len(frames) >= limit:
                    break
    arr = np.stack(frames)
    h, w = arr.shape[1:3]
    if max(h, w) > max_side:
        k = max_side / max(h, w)
        nw, nh = int(round(w * k)), int(round(h * k))
        arr = np.stack([cv2.resize(f, (nw, nh), interpolation=cv2.INTER_AREA) for f in arr])
        note["resize"] = f"{w}×{h} → {nw}×{nh} 로 줄여 판정"
    return arr, fps, note


# ---- 묶음별 목록 ----------------------------------------------------------------------------
def oracle_items() -> list[Item]:
    d = ROOT / "data/oracle_128x72_24fps"
    out = []
    if not (d / "manifest.jsonl").exists():
        return out
    for r in _read_jsonl(d / "manifest.jsonl"):
        p = d / r["file"]

        def load(p=p, r=r):
            z = _npz(p)
            return z["flicker"], float(r["fps"]), {"hz_frames": z["hazard_frames"], "hz_mask": z["hazard_mask"],
                                                   "region": z["mask_changed"]}
        prm = r["params"]
        out.append(Item(
            id=f"oracle__{r['id']}", group="oracle", title=r["id"], source=p, load=load,
            label=bool(r["label_dangerous"]),
            label_note=f"의도 {prm['intent']} · 라벨 출처 {r.get('label_source', '?')}",
            meta={"영역": f"{prm['shape']} {prm['area']:.0%}", "주파수": f"{prm['freq_hz']:.2f} Hz {prm['waveform']}",
                  "밝기 변화": f"{prm['delta_cd']:.1f} cd/m²", "원본": f"{r['source']} #{r['start_frame']}",
                  "기록된 최대 깜빡임": f"{r['max_flashes']}/초", "기록된 면적": f"{r['peak_area']:.1%}"}))
    return out


def gan_items() -> list[Item]:
    out = []
    d = ROOT / "data/gan"
    for tag, base in [("base", d), ("v1_div005", d / "v1_div005"), ("v1_div05", d / "v1_div05")]:
        if not (base / "val").is_dir():
            continue
        mp = base / "manifest.jsonl"
        if mp.exists():
            rows = _read_jsonl(mp)
        else:                                     # manifest 가 없으면 npz 의 hazard_frames 로 라벨을 만든다
            rows = []
            for p in sorted((base / "val").glob("*.npz")):
                with np.load(p) as z:
                    rows.append({"id": p.stem, "file": f"val/{p.name}", "label_dangerous": bool(z["hazard_frames"].any()),
                                 "label_source": "npz hazard_frames (manifest 없음)",
                                 "target": "hazard" if "hazard" in p.stem else "safe"})
        for r in rows:
            p = base / r["file"]

            def load(p=p):
                z = _npz(p)
                return z["flicker"], 24.0, {"hz_frames": z["hazard_frames"], "hz_mask": z["hazard_mask"],
                                            "region": z["mask_allow"]}
            out.append(Item(
                id=f"gan_{tag}__{r['id']}", group=f"gan_{tag}", title=r["id"], source=p, load=load,
                label=bool(r["label_dangerous"]),
                label_note=f"목표 {r.get('target')} · 라벨 출처 {r.get('label_source', '?')}",
                meta={"생성기": r.get("generator", ""), "z": r.get("z_index", ""), "fps": "24 (가정: 오라클과 같음)",
                      "기록된 최대 깜빡임": f"{r.get('max_flashes')}/초",
                      "기록된 면적": f"{r.get('peak_area', 0):.1%}"}))
    return out


def old_items() -> list[Item]:
    out = []
    for sub in ("oracle", "gan"):
        d = ROOT / "datasets" / sub
        if not (d / "manifest.jsonl").exists():
            continue
        for r in _read_jsonl(d / "manifest.jsonl"):
            p = d / r["file"]

            def load(p=p, r=r):
                z = _npz(p)
                return z["affected"], float(r.get("fps", 24)), {"region": z["mask"] > 0}
            out.append(Item(
                id=f"old_{sub}__{r['id']}", group=f"old_{sub}", title=f"{r['type']} · {r['id']}", source=p, load=load,
                label=r["type"] != "normal",
                label_note=f"종류 {r['type']} · 초기화 전 데이터, 위험 검증 안 됨(risk_verified={r.get('risk_verified')})",
                meta={k: v for k, v in r.items() if k in ("generator", "parameters", "source_name", "fps")}))
    return out


def background_items() -> list[Item]:
    d = ROOT / "data/oracle_128x72_24fps"
    out = []
    for split in ("train", "val", "test"):
        p = d / f"backgrounds_{split}.npz"
        if not p.exists():
            continue
        with np.load(p) as z:
            n, fps = z["clips"].shape[0], float(z["fps"])
        for i in range(n):
            def load(p=p, i=i, fps=fps):
                return _npz_array(str(p), "clips")[i], fps, {}
            out.append(Item(id=f"bg_{split}_{i:03d}", group="backgrounds", title=f"{split} {i:03d}", source=p,
                            load=load, label=None, label_note="합성 전 배경 · 정답 없음(오탐 확인용)", sub=str(i)))
    return out


def sweep_items(max_seconds: float, max_side: int) -> list[Item]:
    d = ROOT / "data/hazard_sweep"
    if not (d / "manifest.json").exists():
        return []
    m = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
    h, w = m["height"], m["width"]
    out = []
    for it in m["items"]:
        p = d / it["file"]
        if not p.exists():
            continue

        def load(p=p, it=it):
            fr, fps, _ = decode_video(p, max_seconds=1e9, max_side=10 ** 6)
            gt = {"hz_frames": np.full(len(fr), bool(it["expected"]))}
            if it["family"] != "control":
                gt["region"] = oracle.region_mask("rectangle", it["params"]["area"], h, w)
                if it["expected"]:
                    gt["hz_mask"] = gt["region"]
            return fr, fps, gt
        prm = {k: v for k, v in it.get("params", {}).items() if not isinstance(v, (dict, list))}
        out.append(Item(
            id=f"sweep_{it['background']}__{it['id']}", group=f"sweep_{it['background']}", title=it["id"], source=p,
            load=load, label=bool(it["expected"]),
            label_note=f"{it['family']} · {it['sweep']} = {it['value']} · 무너뜨린 조건: {it['broken']}",
            meta={"종류": it.get("expected_type"), "우리 기준 기대": it.get("expected_ours"), **prm}))
    return out


SYNTH_KEYS = ("flicker", "affected")                 # 합성 결과 영상: 같은 npz 의 정답 키를 붙인다
REGION_KEYS = ("mask_changed", "mask_allow", "mask")  # 합성 영역 (먼저 있는 것)


def npz_items(p: Path, group: str, prefix: str, fps: float | None) -> list[Item]:
    """npz 하나 → 그 안의 영상 전부. (T,H,W,3) uint8 키는 영상 1개, (N,T,H,W,3) 키는 N개.
    flicker·affected 에는 같은 파일의 hazard_frames·hazard_mask·mask_* 를 정답으로 붙인다(없으면 정답 없음)."""
    with np.load(p, allow_pickle=False) as z:
        files = list(z.files)
        info = {k: (z[k].shape, z[k].dtype) for k in files}
        npz_fps = float(z["fps"]) if "fps" in files and z["fps"].shape == () else None
        hz_any = bool(z["hazard_frames"].any()) if "hazard_frames" in files else None
    use_fps = fps or npz_fps or 24.0
    fps_note = {} if (fps or npz_fps) else {"fps": "24 (가정: 파일에 fps 없음)"}
    region_key = next((k for k in REGION_KEYS if k in files), None)
    vid = [k for k, (s, dt) in info.items() if dt == np.uint8 and len(s) in (4, 5) and s[-1] == 3]
    vid.sort(key=lambda k: (k not in SYNTH_KEYS, k))
    out = []
    for key in vid:
        shape = info[key][0]
        synth = key in SYNTH_KEYS
        for i in (range(shape[0]) if len(shape) == 5 else [None]):
            def load(p=p, key=key, i=i, synth=synth):
                a = _npz_array(str(p), key)
                frames = a[i] if i is not None else a
                gt = {"note": dict(fps_note)}
                if synth:
                    z = _npz(p)
                    for src, dst in (("hazard_frames", "hz_frames"), ("hazard_mask", "hz_mask")):
                        if src in z:
                            gt[dst] = z[src]
                    if region_key:
                        gt["region"] = z[region_key] > 0
                return frames, use_fps, gt
            tag = key if i is None else f"{key}[{i}]"
            label = hz_any if synth else None
            note = ("라벨: 같은 npz 의 hazard_frames" if label is not None
                    else "합성 전 원본 · 정답 없음" if key == "clean" else "정답 없음")
            out.append(Item(id=f"{prefix}__{key}{'' if i is None else f'_{i:03d}'}", group=group,
                            title=f"{p.name} : {tag}", source=p, load=load, label=label, label_note=note,
                            meta={"npz 키": tag, "모양": "×".join(map(str, shape))}, sub=f"{key}:{i}"))
    return out


def file_items(p: Path, group: str, prefix: str, max_seconds: float, max_side: int,
               fps: float | None = None) -> list[Item]:
    """파일 하나 → 영상 목록 (npz 는 안의 영상 전부, 나머지는 1개)."""
    ext = p.suffix.lower()
    if ext == ".npz":
        return npz_items(p, group, prefix, fps)
    if ext in VIDEO_EXT:
        def load(p=p):
            fr, f, note = decode_video(p, max_seconds, max_side)
            if fps:
                note["fps"] = f"{f:.2f} → {fps} 로 지정"
            return fr, (fps or f), {"note": note}
        return [Item(id=prefix, group=group, title=p.name, source=p, load=load, label=None, label_note="정답 없음")]
    return []


def _slug(s: str) -> str:
    return re.sub(r"[^0-9A-Za-z가-힣_.-]+", "_", s)


def extra_items(dirs: list[Path], max_seconds: float, max_side: int) -> list[Item]:
    out = []
    for base in dirs:
        files = [base] if base.is_file() else sorted(base.rglob("*"))
        for p in files:
            stem = _slug(str(p.relative_to(base.parent if base.is_file() else base)))
            out += file_items(p, f"extra_{base.name}", f"extra__{stem}", max_seconds, max_side)
    return out


# ---- 판정 → 그래프용 시계열 -------------------------------------------------------------------
def _trailing(e: np.ndarray, fps: float) -> np.ndarray:
    """프레임 t 까지의 최근 1초(t 포함 W = measure.window_frames(fps) 프레임) 화면 전이 수."""
    w = measure.window_frames(fps)
    cs = np.concatenate([[0], np.cumsum(np.asarray(e) != 0)])
    t = np.arange(len(e))
    return cs[t + 1] - cs[np.maximum(0, t + 1 - w)]


def detect(frames: np.ndarray, fps: float, fast: bool = False) -> tuple[dict, dict, dict]:
    """→ (요약, 그래프 시계열, 겹쳐 칠할 마스크).

    fast=True: 느린 스캔선 줄무늬 판정(measure.analyze_pattern, 768×432 2초에 1~4분)을 건너뛰고, 휘도·적색은 그대로,
    줄무늬는 주파수 탐지기(쿼드트리+가버)로만 판정한다. 이때 '판정' = 휘도 or 적색 or 줄무늬(주파수).
    """
    if fast:
        lum = measure.luminance_cd(frames)
        fl = measure.analyze(frames, fps, lum=lum)
        rd = measure.analyze_red(frames, fps)
        t_len = len(frames)
        pt = {"motion": np.full(t_len, "", dtype=object), "stripe_area": np.zeros(t_len), "stripes": np.zeros(lum.shape, bool),
              "hazard_frames": np.zeros(t_len, bool), "candidate": np.zeros(t_len, bool), "dangerous": False,
              "peak_stripe_area": 0.0, "max_stripe_sec": 0.0}
        r = {"flash": fl, "red": rd, "pattern": pt, "dangerous": bool(fl["dangerous"] or rd["dangerous"])}
    else:
        r = measure.analyze_all(frames, fps)
    fl, rd, pt = r["flash"], r["red"], r["pattern"]
    motion = pt["motion"]
    drift_lim = None if measure.PATTERN_DRIFT_EXEMPT else measure.PATTERN_AREA_DYNAMIC
    limit = [measure.PATTERN_AREA_STATIC if m == "static" else      # measure.analyze_pattern ⑦ 과 같은 규칙
             (drift_lim if m == "drifting" else measure.PATTERN_AREA_DYNAMIC) for m in motion]
    moving = [m for m in motion if m]
    summary = {
        "dangerous": r["dangerous"],
        "lum": {"dangerous": bool(fl["dangerous"]), "max_flashes": fl["max_flashes"], "peak_area": fl["peak_area"]},
        "red": {"dangerous": bool(rd["dangerous"]), "max_flashes": rd["max_flashes"], "peak_area": rd["peak_area"],
                "max_delta_uv": rd.get("max_delta_uv", 0.0)},
        "pat": {"dangerous": bool(pt["dangerous"]), "peak_area": pt["peak_stripe_area"],
                "max_sec": pt["max_stripe_sec"],
                "motion": max(set(moving), key=moving.count) if moving else ""},
    }
    series = {
        "lum_trans": _trailing(fl["transitions"], fps), "red_trans": _trailing(rd["transitions"], fps),
        "lum_area": fl["area"], "red_area": rd["area"], "pat_area": pt["stripe_area"], "pat_limit": limit,
        "motion": list(motion),
        "lum_hz": fl["hazard_frames"], "red_hz": rd["hazard_frames"], "pat_hz": pt["hazard_frames"],
        "pat_cand": pt["candidate"],
    }
    masks = {"lum": fl["hazard_mask"], "red": rd["hazard_mask"], "stripes": pt["stripes"]}

    # 줄무늬 주파수 탐지기(실험용 두 번째 탐지기: 쿼드트리 + 가버 다듬기, 규칙 상수는 measure 와 같음)
    if fast and fps > 10:
        # 빠른 판정: 줄무늬는 초당 약 10장만 판정하고 사이 프레임은 앞 결과를 이어 씀(지속 규칙도 그 표본률로)
        step = int(round(fps / 10))
        p2 = sf.analyze_pattern_freq(frames[::step], fps / step, model=None, method="quadtree", refine=True,
                                     keep_blocks=True)
        rep = lambda a: np.repeat(np.asarray(a), step, axis=0)[: len(frames)]          # noqa: E731
        p2.update(stripe_area=rep(p2["stripe_area"]), hazard_frames=rep(p2["hazard_frames"]),
                  candidate=rep(p2["candidate"]), stripes=rep(p2["stripes"]),
                  blocks=[b for b in p2["blocks"] for _ in range(step)][: len(frames)])
    else:
        p2 = sf.analyze_pattern_freq(frames, fps, model=None, method="quadtree", refine=True, keep_blocks=True)
    summary["pat2"] = {"dangerous": p2["dangerous"], "peak_area": p2["peak_stripe_area"], "max_sec": p2["max_stripe_sec"],
                       "method": "쿼드트리+가버"}
    if fast:                                   # 스캔선을 건너뛰었으므로 줄무늬 판정은 주파수 탐지기로
        summary["fast"] = True
        summary["dangerous"] = bool(summary["dangerous"] or p2["dangerous"])
    series.update(pat2_area=p2["stripe_area"], pat2_hz=p2["hazard_frames"], pat2_cand=p2["candidate"])
    masks.update(stripes2=p2["stripes"], blocks=p2["blocks"])
    return summary, series, masks


# ---- 세 칸 영상 ---------------------------------------------------------------------------------
def _at(mask, t):
    if mask is None:
        return None
    m = np.asarray(mask)
    return m[t] if m.ndim == 3 else m


def _tint(img: np.ndarray, mask, color: np.ndarray, a: float) -> None:
    if mask is None:
        return
    m = np.asarray(mask, bool)
    if m.any():
        img[m] = img[m] * (1 - a) + color * a


def _label(img: np.ndarray, text: str) -> np.ndarray:
    img = (img * 0.35).astype(np.float32)
    h, w = img.shape[:2]
    s = max(0.3, w / 400)
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, s, 1)
    cv2.putText(img, text, ((w - tw) // 2, (h + th) // 2), cv2.FONT_HERSHEY_SIMPLEX, s, (230, 230, 230), 1,
                cv2.LINE_AA)
    return img


def compose(frames: np.ndarray, gt: dict, label: bool | None, det: tuple | None):
    """프레임마다 [영상 | 정답 | 판정기 | 줄무늬 주파수 탐지기] 를 만들어 내보낸다 (BGR uint8).

    넷째 칸: 쿼드트리 블록 테두리(흰색 = 줄무늬로 인정, 분홍 = 명암은 있지만 기각, 명암 없는 블록은 그리지 않음)와
    가버로 다듬은 줄무늬 영역(노랑, 옅게 = 인식, 진하게 = 위험 구간)."""
    t_len, h, w = frames.shape[:3]
    k = PANEL_H / h
    pw = int(round(w * k))
    interp = cv2.INTER_NEAREST if k >= 1 else cv2.INTER_AREA
    total_w = 4 * pw + 3 * GAP
    total_w += total_w % 2
    has_gt = label is not None or gt.get("region") is not None or gt.get("hz_mask") is not None
    gt_frames = gt.get("hz_frames")
    for t in range(t_len):
        base = frames[t].astype(np.float32)
        p2 = base.copy()
        if has_gt:
            _tint(p2, _at(gt.get("region"), t), COL_GT, 0.22)
            if gt_frames is not None and bool(gt_frames[t]):
                _tint(p2, _at(gt.get("hz_mask"), t), COL_GT, 0.6)
        else:
            p2 = _label(p2, "NO LABEL")
        if det is None:
            p3 = _label(base.copy(), "NOT RUN")
        else:
            _, s, m = det
            p3 = base.copy()
            _tint(p3, m["stripes"][t], COL_PAT, 0.22)
            if s["lum_hz"][t]:
                _tint(p3, m["lum"], COL_LUM, 0.6)
            if s["red_hz"][t]:
                _tint(p3, m["red"], COL_RED, 0.6)
            if s["pat_hz"][t]:
                _tint(p3, m["stripes"][t], COL_PAT, 0.6)
        if det is None or "stripes2" not in det[2]:
            p4 = _label(base.copy(), "NOT RUN")
        else:
            _, s, m = det
            p4 = base.copy()
            _tint(p4, m["stripes2"][t], COL_PAT2, 0.6 if s["pat2_hz"][t] else 0.3)
        canvas = np.full((PANEL_H, total_w, 3), 40, np.uint8)
        for i, p in enumerate((base, p2, p3, p4)):
            x = i * (pw + GAP)
            canvas[:, x:x + pw] = cv2.resize(np.clip(p, 0, 255).astype(np.uint8), (pw, PANEL_H), interpolation=interp)
        if det is not None and "blocks" in det[2]:                    # 블록 테두리는 확대한 뒤 그려서 선이 또렷하게
            x4 = 3 * (pw + GAP)
            panel = canvas[:, x4:x4 + pw]
            for by, bx, bs, kind in det[2]["blocks"][t]:
                if kind == "flat":
                    continue
                y0, x0 = int(round(by * k)), int(round(bx * k))
                y1, x1 = int(round((by + bs) * k)) - 1, int(round((bx + bs) * k)) - 1
                cv2.rectangle(panel, (x0, y0), (x1, y1), COL_BLOCK_OK if kind == "stripe" else COL_BLOCK_NO, 1)
        yield canvas


def write_mp4(path: Path, frames_iter, fps: float) -> None:
    tmp = path.with_suffix(".tmp.mp4")
    c = av.open(str(tmp), mode="w", options={"movflags": "+faststart"})
    s = None
    try:
        for img in frames_iter:
            if s is None:
                s = c.add_stream("libx264", rate=Fraction(fps).limit_denominator(1001))
                s.height, s.width = img.shape[:2]
                s.pix_fmt = "yuv420p"
                s.options = {"crf": "18", "preset": "veryfast"}
            for pkt in s.encode(av.VideoFrame.from_ndarray(img, format="bgr24")):
                c.mux(pkt)
        for pkt in s.encode():
            c.mux(pkt)
    finally:
        c.close()
    tmp.replace(path)


# ---- 저장 ---------------------------------------------------------------------------------
def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        if x.dtype == bool:
            return x.astype(int).tolist()
        if x.dtype.kind == "f":
            return [None if not np.isfinite(v) else round(float(v), 4) for v in x]
        return x.tolist()
    if isinstance(x, (np.bool_, bool)):
        return bool(x)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, (np.floating, float)):
        return None if not np.isfinite(x) else round(float(x), 4)
    if isinstance(x, Path):
        return str(x)
    return x


def outcome(label: bool | None, det: dict | None) -> str:
    if det is None:
        return "not_run"
    if label is None:
        return "no_label_alarm" if det["dangerous"] else "no_label"
    if label:
        return "hit" if det["dangerous"] else "miss"
    return "false_alarm" if det["dangerous"] else "correct"


def cache_key(it: Item, detect_on: bool, det_hash: str, fast: bool = False) -> str:
    st = it.source.stat()
    raw = f"{it.source}|{st.st_mtime_ns}|{st.st_size}|{it.sub}|{detect_on}|{det_hash}|{PANEL_H}|{FORMAT_VERSION}"
    if fast:
        raw += "|fast"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def process_item(it: Item, out: Path, detect_on: bool, det_hash: str, force: bool,
                 fast: bool = False) -> tuple[dict, bool]:
    """클립 하나 → (기록, 캐시 사용 여부). 판정·세 칸 mp4·그래프 시계열을 만들고 cache/ 에 저장."""
    key = cache_key(it, detect_on, det_hash, fast)
    cpath, vpath = out / "cache" / f"{it.id}.json", out / "clips" / f"{it.id}.mp4"
    if not force and cpath.exists() and vpath.exists():
        rec = json.loads(cpath.read_text(encoding="utf-8"))
        if rec.get("key") == key:
            return rec, True
    t0 = time.time()
    frames, fps, gt = it.load()
    note = gt.pop("note", None) or {}
    det = detect(frames, fps, fast) if detect_on else None
    sec = time.time() - t0
    write_mp4(vpath, compose(frames, gt, it.label, det), fps)
    series = det[1] if det else {}
    if gt.get("hz_frames") is not None:
        series = {**series, "gt_hz": np.asarray(gt["hz_frames"], bool)}
    rec = _jsonable({
        "key": key, "id": it.id, "group": it.group, "title": it.title,
        "source": str(it.source.relative_to(ROOT)) if it.source.is_relative_to(ROOT) else str(it.source),
        "video": f"clips/{it.id}.mp4", "fps": fps, "n": len(frames), "w": frames.shape[2], "h": frames.shape[1],
        "label": it.label, "label_note": it.label_note, "meta": {**it.meta, **note},
        "det": det[0] if det else None, "series": series, "outcome": outcome(it.label, det[0] if det else None),
        "det_sec": round(sec, 2),
    })
    cpath.write_text(json.dumps(rec, ensure_ascii=False), encoding="utf-8")
    return rec, False


# ---- 서버 모드: 페이지에서 고른 파일을 그 자리에서 변환·판정 -----------------------------------------
MAX_UPLOAD = 4 * 1024 ** 3      # 업로드 상한 4 GB (더 큰 파일은 경로 칸으로 연다)
CTYPES = {".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8", ".mp4": "video/mp4",
          ".json": "application/json; charset=utf-8", ".png": "image/png"}


def serve(args, det_hash: str) -> int:
    import http.server
    import queue
    import secrets
    import threading
    import urllib.parse
    import uuid
    import webbrowser

    token = secrets.token_hex(16)                       # 페이지에만 심어 주는 값: POST 마다 확인(CSRF 방지)
    allowed_hosts = {f"127.0.0.1:{args.port}", f"localhost:{args.port}"}
    out = args.out.resolve()
    up_dir = out / "uploads"
    up_dir.mkdir(parents=True, exist_ok=True)
    jobs: dict[str, dict] = {}
    lock = threading.Lock()
    work: queue.Queue = queue.Queue()

    def worker() -> None:                      # 한 번에 한 클립씩 (판정이 CPU 를 많이 씀)
        while True:
            job_id, it, detect_on, fast = work.get()
            try:
                rec, _ = process_item(it, out, detect_on, det_hash, force=False, fast=fast)
                with lock:
                    jobs[job_id]["records"][it.id] = rec
                print(f"  {it.id}: {rec['outcome']} ({rec['det_sec']}s)", flush=True)
            except Exception as e:
                with lock:
                    jobs[job_id]["errors"][it.id] = f"{type(e).__name__}: {e}"
                print(f"  {it.id}: 실패 {type(e).__name__}: {e}", flush=True)

    threading.Thread(target=worker, daemon=True).start()

    def start_job(path: Path, detect_on: bool, fps: float | None, group: str | None = None, fast: bool = True) -> dict:
        if path.is_dir():                          # 폴더: 안의 지원 파일 전부 (하위 폴더 포함)
            files = sorted(p for p in path.rglob("*")
                           if p.is_file() and (p.suffix.lower() in VIDEO_EXT or p.suffix.lower() == ".npz"))
        elif path.is_file():
            files = [path]
        else:
            return {"error": f"파일·폴더가 없습니다: {path}"}
        items = []
        for p in files:
            h = hashlib.sha1(str(p.resolve()).encode("utf-8")).hexdigest()[:8]
            g = group or (f"폴더: {path.name}/{p.relative_to(path).parent}".rstrip("/.") if path.is_dir()
                          else f"파일: {p.name}")
            try:
                items += file_items(p, g, f"file_{h}_{_slug(p.stem)[:40]}", args.max_seconds, args.max_side, fps)
            except Exception as e:
                print(f"  건너뜀 {p}: {type(e).__name__}: {e}", flush=True)
        if not items:
            return {"error": f"영상을 찾지 못했습니다 (지원: npz 안의 (T,H,W,3)·(N,T,H,W,3) uint8, "
                             f"{', '.join(sorted(VIDEO_EXT))}): {path.name}"}
        job_id = uuid.uuid4().hex[:10]
        with lock:
            jobs[job_id] = {"ids": [it.id for it in items], "records": {}, "errors": {}}
        for it in items:
            work.put((job_id, it, detect_on, fast))
        print(f"열기 {path} → 영상 {len(items)}개", flush=True)
        return {"job": job_id, "name": path.name,
                "items": [{"id": it.id, "group": it.group, "title": it.title, "label": it.label,
                           "label_note": it.label_note, "source": str(path)} for it in items]}

    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a):            # 요청마다 찍히는 로그는 끔
            pass

        def _send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, code: int = 200) -> None:
            self._send(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"), CTYPES[".json"])

        def _file(self, fpath: Path) -> None:
            """정적 파일. 영상 탐색(seek)이 되도록 Range 요청(206)을 지원."""
            size = fpath.stat().st_size
            start, end, code = 0, size - 1, 200
            rng = self.headers.get("Range", "")
            if rng.startswith("bytes="):
                a, _, b = rng[6:].split(",")[0].partition("-")
                try:
                    if a:
                        start, end = int(a), (int(b) if b else size - 1)
                    else:
                        start = max(0, size - int(b))
                except ValueError:
                    return self._json({"error": "잘못된 Range"}, 400)
                end = min(end, size - 1)
                if start > end:
                    self.send_response(416)
                    self.send_header("Content-Range", f"bytes */{size}")
                    self.end_headers()
                    return
                code = 206
            self.send_response(code)
            self.send_header("Content-Type", CTYPES.get(fpath.suffix.lower(), "application/octet-stream"))
            self.send_header("Content-Length", str(end - start + 1))
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Cache-Control", "no-cache")
            if code == 206:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.end_headers()
            try:
                with open(fpath, "rb") as fh:
                    fh.seek(start)
                    left = end - start + 1
                    while left > 0:
                        chunk = fh.read(min(1 << 20, left))
                        if not chunk:
                            break
                        self.wfile.write(chunk)
                        left -= len(chunk)
            except (ConnectionResetError, ConnectionAbortedError, BrokenPipeError):
                pass

        def _host_ok(self) -> bool:
            """DNS rebinding 방지: 주소창이 127.0.0.1·localhost 일 때만 응답."""
            if self.headers.get("Host", "") in allowed_hosts:
                return True
            self._json({"error": "허용되지 않은 Host"}, 403)
            return False

        def do_GET(self):
            if not self._host_ok():
                return
            u = urllib.parse.urlparse(self.path)
            path = urllib.parse.unquote(u.path)
            if path in ("/", "/index.html"):
                return self._send(200, INDEX_HTML.replace("__LUMI_TOKEN__", token).encode("utf-8"), CTYPES[".html"])
            if path.startswith("/api/job/"):
                with lock:
                    j = jobs.get(path.rsplit("/", 1)[-1])
                    if j is None:
                        return self._json({"error": "없는 작업"}, 404)
                    recs = [j["records"][i] for i in j["ids"] if i in j["records"]]
                    return self._json({"total": len(j["ids"]), "done": len(recs) + len(j["errors"]),
                                       "records": recs, "errors": j["errors"]})
            fpath = (out / path.lstrip("/")).resolve()
            if fpath.is_relative_to(out) and fpath.is_file():
                return self._file(fpath)
            if path == "/clips.js":                      # 아직 묶음 생성을 안 했으면 빈 목록
                js = "window.CLIPS = []; window.META = " + json.dumps(meta_dict(), ensure_ascii=False) + ";"
                return self._send(200, js.encode("utf-8"), CTYPES[".js"])
            self._json({"error": "없음"}, 404)

        def do_POST(self):
            if not self._host_ok():
                return
            # CSRF 방지: 우리 페이지만 아는 토큰 헤더가 있어야 함(다른 사이트의 단순 POST 는 사용자 정의 헤더를 못 붙임)
            if not secrets.compare_digest(self.headers.get("X-Lumi-Token", ""), token):
                return self._json({"error": "토큰 없음 — 페이지를 새로고침하세요"}, 403)
            u = urllib.parse.urlparse(self.path)
            q = urllib.parse.parse_qs(u.query)
            detect_on = q.get("detect", ["1"])[0] == "1"
            fast = q.get("fast", ["1"])[0] == "1"
            try:
                fps = float(q.get("fps", [""])[0] or 0) or None
            except ValueError:
                fps = None
            length = int(self.headers.get("Content-Length", 0))
            if u.path == "/api/open":
                if not self.headers.get("Content-Type", "").startswith("application/json"):
                    return self._json({"error": "JSON 요청만 받습니다"}, 415)
                try:
                    body = json.loads(self.rfile.read(length) or b"{}")
                except json.JSONDecodeError:
                    return self._json({"error": "잘못된 JSON"}, 400)
                p = Path(str(body.get("path", "")).strip().strip('"').strip("'"))
                return self._json(start_job(p, detect_on, fps, fast=fast))
            if u.path == "/api/upload":
                group = urllib.parse.unquote(self.headers.get("X-Group", "")) or None
                name = Path(urllib.parse.unquote(self.headers.get("X-Filename", "upload.bin"))).name
                if length > MAX_UPLOAD:
                    return self._json({"error": f"파일이 너무 큽니다({length / 1e9:.1f} GB). 경로 칸에 붙여넣어 여세요"}, 413)
                tmp = up_dir / f"_{uuid.uuid4().hex}.part"
                sha = hashlib.sha1()
                with open(tmp, "wb") as fh:                               # 받으면서 내용 해시 계산
                    left = length
                    while left > 0:
                        chunk = self.rfile.read(min(1 << 20, left))
                        if not chunk:
                            break
                        sha.update(chunk)
                        fh.write(chunk)
                        left -= len(chunk)
                # 내용 해시 폴더에 원래 이름으로: 이름·크기가 같아도 내용이 다르면 다른 파일로 다룸
                dest = up_dir / sha.hexdigest()[:16] / name
                dest.parent.mkdir(parents=True, exist_ok=True)
                if dest.exists():
                    tmp.unlink()
                else:
                    tmp.replace(dest)
                return self._json(start_job(dest, detect_on, fps, group, fast=fast))
            self._json({"error": "없음"}, 404)

    url = f"http://127.0.0.1:{args.port}/"

    class Server(http.server.ThreadingHTTPServer):
        allow_reuse_address = False                   # Windows 는 재사용을 켜면 같은 포트를 두 번 열 수 있음

    import socket
    try:
        with socket.create_connection(("127.0.0.1", args.port), timeout=0.5):
            busy = True                               # 누군가 이미 듣고 있음
    except OSError:
        busy = False
    try:
        if busy:
            raise OSError("in use")
        srv = Server(("127.0.0.1", args.port), Handler)
    except OSError:                                   # 이미 켜져 있으면 그 페이지만 연다
        print(f"포트 {args.port} 가 이미 사용 중입니다. 이미 켜진 서버로 브라우저만 엽니다: {url}", flush=True)
        if not args.no_browser:
            webbrowser.open(url)
        return 0
    print(f"서버 시작: {url}  (이 PC 에서만 열림, 끄려면 Ctrl+C)\n결과 폴더: {out}", flush=True)
    if not args.no_browser:
        try:
            webbrowser.open(url)
        except Exception:
            pass
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


def meta_dict() -> dict:
    return {"made": time.strftime("%Y-%m-%d %H:%M"), "panel_h": PANEL_H,
            "rules": {"area": measure.AREA_LIMIT, "max_flashes": measure.MAX_FLASHES_PER_SEC,
                      "pat_static": measure.PATTERN_AREA_STATIC, "pat_dynamic": measure.PATTERN_AREA_DYNAMIC,
                      "pat_drift_exempt": measure.PATTERN_DRIFT_EXEMPT, "pat_cpd": list(measure.PATTERN_CPD),
                      "pat_min_pairs": measure.PATTERN_MIN_PAIRS}}


def write_index(out: Path, records: list[dict], args) -> None:
    js = ("window.CLIPS = " + json.dumps(records, ensure_ascii=False, separators=(",", ":")) + ";\n"
          + "window.META = " + json.dumps(meta_dict(), ensure_ascii=False) + ";\n")
    (out / "clips.js").write_text(js, encoding="utf-8")
    (out / "index.html").write_text(INDEX_HTML, encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--groups", default=DEFAULT_GROUPS, help=f"쉼표로 구분 (기본 {DEFAULT_GROUPS}, 빈 값이면 --extra 만)")
    ap.add_argument("--extra", type=Path, action="append", default=[], help="gif/mp4/avi/mkv/mov/npz 가 있는 폴더나 파일")
    ap.add_argument("--limit", type=int, default=0, help="묶음마다 앞에서 N개만 (0 = 전부)")
    ap.add_argument("--no-detect", action="store_true", help="판정하지 않고 변환만")
    ap.add_argument("--max-seconds", type=float, default=10.0, help="일반 영상은 앞에서 이 시간만 (기본 10초)")
    ap.add_argument("--max-side", type=int, default=640, help="일반 영상 긴 변이 이보다 크면 줄여서 판정 (기본 640)")
    ap.add_argument("--out", type=Path, default=ROOT / "data/playlist")
    ap.add_argument("--force", action="store_true", help="이미 만든 클립도 다시 만듦")
    ap.add_argument("--serve", action="store_true",
                    help="로컬 서버로 열기: 페이지에서 파일을 골라 그 안의 영상을 모두 변환·판정 (묶음 생성은 하지 않음)")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-browser", action="store_true", help="서버 모드에서 브라우저를 자동으로 열지 않음")
    args = ap.parse_args()
    if args.serve and args.groups == DEFAULT_GROUPS:
        args.groups = ""                                    # 서버 모드는 기존 결과(clips.js)를 그대로 보여주고 새로 만들지 않음

    out = args.out
    (out / "clips").mkdir(parents=True, exist_ok=True)
    (out / "cache").mkdir(parents=True, exist_ok=True)
    det_hash = hashlib.sha1((ROOT / "src/lumi_synth/measure.py").read_bytes()
                            + (ROOT / "src/lumi_synth/stripe_freq.py").read_bytes()).hexdigest()[:12]

    makers = {"oracle": oracle_items, "gan": gan_items, "old": old_items, "backgrounds": background_items,
              "demo": lambda: extra_items([ROOT / "data/stripe_demo"], args.max_seconds, 10 ** 6)
              if (ROOT / "data/stripe_demo").is_dir() else [],
              "sweep": lambda: sweep_items(args.max_seconds, args.max_side)}
    items: list[Item] = []
    for g in [x.strip() for x in args.groups.split(",") if x.strip()]:
        if g not in makers:
            ap.error(f"모르는 묶음: {g} (가능: {', '.join(makers)})")
        found = makers[g]()
        items += found[: args.limit] if args.limit else found
    found = extra_items(args.extra, args.max_seconds, args.max_side)
    items += found[: args.limit] if args.limit else found
    print(f"클립 {len(items)}개 → {out}", flush=True)

    if args.serve:
        return serve(args, det_hash)

    records: list[dict] = []
    t_all = time.time()
    for n, it in enumerate(items, 1):
        try:
            rec, cached = process_item(it, out, not args.no_detect, det_hash, args.force)
        except Exception as e:                                   # 한 클립 실패가 전체를 멈추지 않게
            print(f"[{n}/{len(items)}] 실패 {it.id}: {type(e).__name__}: {e}", flush=True)
            continue
        records.append(rec)
        if cached:
            continue
        d = rec["det"]
        print(f"[{n}/{len(items)}] {it.id}  정답 {it.label}  판정 {d['dangerous'] if d else '-'}  "
              f"{rec['outcome']}  {rec['det_sec']}s", flush=True)
        if n % 20 == 0:
            write_index(out, records, args)
    write_index(out, records, args)
    counts: dict[str, int] = {}
    for r in records:
        counts[r["outcome"]] = counts.get(r["outcome"], 0) + 1
    print(f"끝: {len(records)}개, {time.time() - t_all:.0f}초, {counts}")
    print(f"열기: {out / 'index.html'}")
    return 0


INDEX_HTML = r"""<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>LUMI 영상 플레이리스트</title>
<style>
:root {
  --bg: #f6f6f4; --panel: #ffffff; --ink: #1d1d1b; --muted: #6b6b66; --line: #deded8; --accent: #2a6fdb;
  --gt: #3cc850; --lum: #ff9f1a; --red: #e040e0; --pat: #12b5e6; --pat2: #e6b800; --warn-bg: #fff4d6; --warn-ink: #6b4a00;
  --hit: #2e9d4f; --miss: #d33c3c; --fa: #e08a00; --na: #8a8a85;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --bg: #171716; --panel: #212120; --ink: #ecece8; --muted: #a3a39c; --line: #353532; --accent: #7aa8ff;
    --warn-bg: #3a2f12; --warn-ink: #ffd98a;
  }
}
:root[data-theme="dark"] {
  --bg: #171716; --panel: #212120; --ink: #ecece8; --muted: #a3a39c; --line: #353532; --accent: #7aa8ff;
  --warn-bg: #3a2f12; --warn-ink: #ffd98a;
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--ink); font: 14px/1.45 system-ui, "Malgun Gothic", sans-serif; }
header { padding: 10px 16px; border-bottom: 1px solid var(--line); display: flex; gap: 16px; align-items: center; flex-wrap: wrap; }
header h1 { font-size: 17px; margin: 0; }
.warn { background: var(--warn-bg); color: var(--warn-ink); padding: 8px 16px; font-size: 13px; }
.layout { display: grid; grid-template-columns: 330px 1fr; min-height: calc(100vh - 90px); }
aside { border-right: 1px solid var(--line); display: flex; flex-direction: column; min-height: 0; max-height: calc(100vh - 90px); position: sticky; top: 0; }
.filters { padding: 10px; display: grid; gap: 6px; border-bottom: 1px solid var(--line); }
.filters select, .filters input { width: 100%; padding: 5px 6px; background: var(--panel); color: var(--ink); border: 1px solid var(--line); border-radius: 6px; font: inherit; }
#list { overflow-y: auto; flex: 1; }
.row { padding: 6px 10px; border-bottom: 1px solid var(--line); cursor: pointer; display: grid; grid-template-columns: 1fr auto; gap: 4px; }
.row:hover { background: var(--panel); }
.row.sel { background: var(--panel); box-shadow: inset 3px 0 0 var(--accent); }
.row .t { font-size: 13px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.row .g { font-size: 11px; color: var(--muted); }
.badge { font-size: 11px; padding: 1px 6px; border-radius: 10px; color: #fff; white-space: nowrap; align-self: center; }
.o-hit, .o-correct { background: var(--hit); } .o-miss { background: var(--miss); } .o-false_alarm { background: var(--fa); }
.o-no_label, .o-not_run { background: var(--na); } .o-no_label_alarm { background: var(--fa); }
.o-pending { background: var(--accent); } .o-error { background: var(--miss); }
.opener { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; font-size: 13px; }
.opener input { padding: 4px 6px; background: var(--panel); color: var(--ink); border: 1px solid var(--line); border-radius: 6px; font: inherit; }
button.primary { background: var(--accent); color: #fff; border-color: var(--accent); }
#status { color: var(--muted); font-size: 12px; }
.hint { color: var(--muted); font-size: 12px; } .hint code { background: var(--panel); padding: 1px 4px; border-radius: 4px; }
main { padding: 12px 16px; min-width: 0; }
.heads { display: grid; grid-template-columns: repeat(4, 1fr); gap: 4px; font-size: 12px; color: var(--muted); margin-bottom: 4px; }
.heads span b { color: var(--ink); }
video { width: 100%; max-width: 1200px; background: #000; display: block; image-rendering: pixelated; }
.controls { display: flex; flex-wrap: wrap; gap: 6px; align-items: center; margin: 8px 0; }
button { font: inherit; padding: 4px 10px; border-radius: 6px; border: 1px solid var(--line); background: var(--panel); color: var(--ink); cursor: pointer; }
button.on { border-color: var(--accent); color: var(--accent); }
.wrap { max-width: 1200px; }
canvas { width: 100%; display: block; background: var(--panel); border: 1px solid var(--line); border-radius: 6px; cursor: crosshair; }
.legend { display: flex; flex-wrap: wrap; gap: 12px; font-size: 12px; color: var(--muted); margin: 6px 0; }
.legend i { display: inline-block; width: 18px; height: 3px; vertical-align: middle; margin-right: 4px; }
.info { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 12px; margin-top: 12px; }
.card { background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 10px 12px; }
.card h3 { margin: 0 0 6px; font-size: 13px; color: var(--muted); font-weight: 600; }
.card table { border-collapse: collapse; width: 100%; font-size: 13px; }
.card td { padding: 2px 4px; vertical-align: top; overflow-wrap: anywhere; } .card td:first-child { color: var(--muted); white-space: nowrap; width: 1%; }
.big { font-size: 15px; font-weight: 600; }
#summary { font-size: 12px; color: var(--muted); }
#summary table { border-collapse: collapse; } #summary td, #summary th { padding: 1px 8px; text-align: right; } #summary td:first-child, #summary th:first-child { text-align: left; }
.empty { color: var(--muted); padding: 40px; }
@media (max-width: 800px) {
  .layout { grid-template-columns: 1fr; } aside { max-height: 40vh; position: static; }
}
</style>
</head>
<body>
<header>
  <h1>LUMI 영상 플레이리스트</h1>
  <span id="made" style="color:var(--muted);font-size:12px"></span>
  <span style="flex:1"></span>
  <div id="opener" class="opener" hidden>
    <button id="pickDir" class="primary">폴더 열기…</button>
    <button id="pick" class="primary">파일 열기…</button>
    <input type="file" id="file" multiple accept=".npz,.gif,.mp4,.avi,.mkv,.mov,.m4v,.webm" hidden>
    <input type="file" id="dir" webkitdirectory multiple hidden>
    <input id="path" placeholder="또는 파일·폴더 경로 붙여넣기 (Enter)" style="width:280px">
    <label title="비우면 파일의 fps(없으면 24)">fps <input id="fpsIn" placeholder="자동" style="width:52px"></label>
    <label><input type="checkbox" id="det" checked> 판정</label>
    <label title="느린 스캔선 줄무늬 판정을 건너뛰고 줄무늬는 쿼드트리+가버로만 판정 (2초 영상 1~3분 → 수 초)"><input type="checkbox" id="fast" checked> 빠른 판정</label>
    <span id="status"></span>
  </div>
  <span id="nosrv" class="hint" hidden>파일 열기는 서버 모드에서만 됩니다: <code>LUMI_플레이어.bat</code> 더블클릭</span>
  <button id="theme" title="밝게/어둡게">테마</button>
</header>
<div class="warn" id="warn">⚠️ 광과민성 위험 자극이 들어 있습니다. 이 PC에서만 보고, 다른 사람이 있는 곳·발표 화면·녹화에서는 재생하지 마세요. 느리게(0.25×) 재생하면 깜빡임이 초당 3회 아래로 내려갑니다(고대비 줄무늬는 느려도 그대로 남음).</div>
<div class="layout">
  <aside>
    <div class="filters">
      <select id="fGroup"></select>
      <select id="fLabel">
        <option value="">정답: 전체</option><option value="1">정답: 위험</option><option value="0">정답: 안전</option><option value="n">정답: 없음</option>
      </select>
      <select id="fOut">
        <option value="">결과: 전체</option>
        <option value="miss">놓침 (정답 위험 · 판정 안전)</option>
        <option value="false_alarm">오탐 (정답 안전 · 판정 위험)</option>
        <option value="hit">맞힘 (위험)</option>
        <option value="correct">맞힘 (안전)</option>
        <option value="no_label_alarm">정답 없음 · 판정 위험</option>
        <option value="no_label">정답 없음 · 판정 안전</option>
        <option value="not_run">판정 안 함</option>
      </select>
      <input id="fText" placeholder="이름·조건 검색">
      <div id="count" style="font-size:12px;color:var(--muted)"></div>
    </div>
    <div id="list"></div>
  </aside>
  <main>
    <div id="summary"></div>
    <div class="wrap" id="player" style="margin-top:10px">
      <div class="heads"><span><b>영상</b></span><span><b>정답</b> — <span style="color:var(--gt)">■</span> 합성 영역(옅게) · 위험 구간(진하게)</span><span><b>우리 판정기</b> — <span style="color:var(--lum)">■</span>휘도 <span style="color:var(--red)">■</span>적색 <span style="color:var(--pat)">■</span>줄무늬(스캔선, 옅게 = 인식, 면적·지속 기준 전)</span><span><b>줄무늬 주파수 탐지기</b> (쿼드트리+가버) — <span style="color:var(--pat2)">■</span>줄무늬 영역(진하게 = 위험 구간) · 테두리 흰색 = 인정한 블록, 분홍 = 기각한 블록</span></div>
      <video id="v" controls playsinline preload="metadata"></video>
      <div class="controls">
        <button id="prev">◀ 이전 (P)</button><button id="next">다음 (N) ▶</button>
        <span style="width:10px"></span>
        <span style="color:var(--muted);font-size:12px">속도</span>
        <button data-rate="0.25">0.25×</button><button data-rate="0.5">0.5×</button><button data-rate="1">1×</button>
        <span style="width:10px"></span>
        <button id="loop">반복</button><button id="auto">끝나면 다음</button>
        <span id="pos" style="color:var(--muted);font-size:12px;margin-left:8px"></span>
      </div>
      <canvas id="g" height="350"></canvas>
      <div class="legend" id="legend"></div>
      <div class="info" id="info"></div>
    </div>
  </main>
</div>
<script src="clips.js"></script>
<script>
(() => {
  const CLIPS = window.CLIPS || [], META = window.META || {};
  const $ = (id) => document.getElementById(id);
  const store = { get(k, d) { try { const v = localStorage.getItem("lumi_pl_" + k); return v === null ? d : JSON.parse(v); } catch (e) { return d; } },
                  set(k, v) { try { localStorage.setItem("lumi_pl_" + k, JSON.stringify(v)); } catch (e) {} } };
  const OUT_KO = { hit: "맞힘", correct: "맞힘", miss: "놓침", false_alarm: "오탐", no_label: "정답 없음", no_label_alarm: "정답 없음·판정 위험", not_run: "판정 안 함", pending: "처리 중", error: "실패" };
  const css = (n) => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
  $("made").textContent = META.made ? `만든 시각 ${META.made} · ${CLIPS.length}개` : "";

  // 테마
  const applyTheme = (t) => { if (t) document.documentElement.setAttribute("data-theme", t); else document.documentElement.removeAttribute("data-theme"); };
  applyTheme(store.get("theme", ""));
  $("theme").onclick = () => { const cur = document.documentElement.getAttribute("data-theme") || (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
    const nx = cur === "dark" ? "light" : "dark"; applyTheme(nx); store.set("theme", nx); draw(); };

  // 묶음 요약 (파일을 열면 다시 그림)
  function refreshGroups() {
    const keep = $("fGroup").value;
    const groups = [...new Set(CLIPS.map(c => c.group))];
    $("fGroup").innerHTML = `<option value="">묶음: 전체 (${CLIPS.length})</option>` + groups.map(g => `<option>${esc(g)}</option>`).join("");
    $("fGroup").value = groups.includes(keep) ? keep : "";
    const sumRows = groups.map(g => { const cs = CLIPS.filter(c => c.group === g); const n = (o) => cs.filter(c => c.outcome === o).length;
      const pend = n("pending"), err = n("error");
      return `<tr><td>${esc(g)}</td><td>${cs.length}</td><td>${n("hit")}</td><td>${n("correct")}</td><td style="color:var(--miss)">${n("miss")}</td><td style="color:var(--fa)">${n("false_alarm")}</td><td>${n("no_label_alarm")} / ${n("no_label") + n("no_label_alarm")}</td><td>${n("not_run")}${pend ? ` · 처리 중 ${pend}` : ""}${err ? ` · 실패 ${err}` : ""}</td></tr>`; });
    $("summary").innerHTML = groups.length ? `<table><tr><th>묶음</th><th>개수</th><th>맞힘(위험)</th><th>맞힘(안전)</th><th>놓침</th><th>오탐</th><th>정답 없음: 판정 위험/전체</th><th>판정 안 함</th></tr>${sumRows.join("")}</table>`
      : `<div class="empty">아직 클립이 없습니다. 위의 <b>파일 열기</b>로 npz·gif·mp4·avi 파일을 고르세요.</div>`;
  }

  // 목록
  let view = [], cur = -1, loadedId = null, loadedVideo = null;
  const filt = store.get("filters", {});
  for (const k of ["fGroup", "fLabel", "fOut", "fText"]) { if (filt[k] !== undefined) $(k).value = filt[k]; $(k).oninput = () => { store.set("filters", { fGroup: $("fGroup").value, fLabel: $("fLabel").value, fOut: $("fOut").value, fText: $("fText").value }); build(); }; }
  function build() {
    const g = $("fGroup").value, l = $("fLabel").value, o = $("fOut").value, q = $("fText").value.trim().toLowerCase();
    view = CLIPS.filter(c => (!g || c.group === g) && (!l || (l === "n" ? c.label === null : String(+c.label) === l && c.label !== null))
      && (!o || c.outcome === o) && (!q || (c.title + " " + c.label_note + " " + c.id).toLowerCase().includes(q)));
    $("count").textContent = `${view.length}개 표시`;
    $("list").innerHTML = view.map((c, i) => `<div class="row" data-i="${i}"><div><div class="t">${esc(c.title)}</div><div class="g">${esc(c.group)} · 정답 ${c.label === null ? "없음" : c.label ? "위험" : "안전"} · 판정 ${c.det ? (c.det.dangerous ? "위험" : "안전") : "-"}</div></div><span class="badge o-${c.outcome}">${OUT_KO[c.outcome]}</span></div>`).join("") || `<div class="empty">조건에 맞는 클립이 없습니다.</div>`;
    for (const r of $("list").children) if (r.dataset.i !== undefined) r.onclick = () => select(+r.dataset.i);
    const keep = view.findIndex(c => c.id === store.get("last", ""));
    select(keep >= 0 ? keep : 0, false);
  }
  const esc = (s) => String(s ?? "").replace(/[&<>"]/g, ch => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[ch]));

  // 재생
  const v = $("v");
  let rate = store.get("rate", 1), loop = store.get("loop", true), auto = store.get("auto", false);
  const syncButtons = () => { document.querySelectorAll("[data-rate]").forEach(b => b.classList.toggle("on", +b.dataset.rate === rate)); $("loop").classList.toggle("on", loop); $("auto").classList.toggle("on", auto); };
  document.querySelectorAll("[data-rate]").forEach(b => b.onclick = () => { rate = +b.dataset.rate; v.playbackRate = rate; store.set("rate", rate); syncButtons(); });
  $("loop").onclick = () => { loop = !loop; v.loop = loop; store.set("loop", loop); syncButtons(); };
  $("auto").onclick = () => { auto = !auto; store.set("auto", auto); syncButtons(); };
  $("prev").onclick = () => select(Math.max(0, cur - 1), true);
  $("next").onclick = () => select(Math.min(view.length - 1, cur + 1), true);
  v.onended = () => { if (auto && cur < view.length - 1) select(cur + 1, true); };
  document.addEventListener("keydown", (e) => { if (e.target.tagName === "INPUT") return; if (e.key === "n" || e.key === "N") $("next").click(); if (e.key === "p" || e.key === "P") $("prev").click(); });
  syncButtons();

  function select(i, play) {
    if (!view.length) { $("player").style.visibility = "hidden"; return; }
    $("player").style.visibility = "visible";
    cur = i; const c = view[i];
    for (const r of $("list").children) r.classList.toggle("sel", +r.dataset.i === i);
    const sel = $("list").children[i]; if (sel && play !== false) sel.scrollIntoView({ block: "nearest" });
    store.set("last", c.id);
    if (c.id === loadedId && c.video === loadedVideo && !play) { info(c); draw(); return; }   // 목록만 갱신: 재생 유지
    loadedId = c.id; loadedVideo = c.video;
    if (c.video) {
      v.src = c.video; v.loop = loop; v.playbackRate = rate;
      v.onloadedmetadata = () => { v.playbackRate = rate; };
      if (play) v.play().catch(() => {});
    } else { v.removeAttribute("src"); v.load(); }
    info(c); draw();
    if (!c.video) $("pos").textContent = c.outcome === "error" ? "실패: " + c.label_note : "처리 중… (판정이 끝나면 자동으로 나타남)";
  }

  // 파일 열기 (서버 모드에서만)
  const SERVED = location.protocol.startsWith("http");
  const TOKEN = "__LUMI_TOKEN__";                       // 서버가 페이지를 줄 때 채워 넣는 값
  $("opener").hidden = !SERVED; $("nosrv").hidden = SERVED;
  if (!SERVED) {                                          // 더블클릭으로 연 페이지: 파일 열기가 안 되는 이유를 크게 알림
    const b = document.createElement("div");
    b.className = "warn"; b.style.background = "var(--panel)"; b.style.color = "var(--ink)"; b.style.borderBottom = "1px solid var(--line)";
    b.innerHTML = "📂 이 페이지는 <b>파일로 직접 연 목록 보기</b>라서 <b>파일·폴더 열기를 할 수 없습니다</b>. 열려면 저장소 폴더의 <code>LUMI_플레이어.bat</code> 을 더블클릭하세요 (브라우저에 http://127.0.0.1:8765/ 가 열립니다).";
    $("warn").after(b);
  }
  // 서버 작업이 오래 걸릴 때 '처리 중' 이 멈춘 것처럼 보이지 않게 경과 시간을 함께 보여 줌
  const jobStart = {};
  const setStatus = (t) => { $("status").textContent = t; };
  const opts = () => `detect=${$("det").checked ? 1 : 0}&fast=${$("fast").checked ? 1 : 0}&fps=${encodeURIComponent($("fpsIn").value.trim())}`;
  const upsert = (rec) => { const i = CLIPS.findIndex(c => c.id === rec.id); if (i >= 0) CLIPS[i] = rec; else CLIPS.push(rec); };
  $("pick").onclick = () => $("file").click();
  $("file").onchange = async () => { for (const f of [...$("file").files]) await openUpload(f); $("file").value = ""; };
  $("pickDir").onclick = () => $("dir").click();
  $("dir").onchange = async () => {        // 폴더 안의 지원 파일만 골라 하나씩 올림 (하위 폴더 포함)
    const ok = /\.(npz|gif|mp4|avi|mkv|mov|m4v|webm)$/i;
    const files = [...$("dir").files].filter(f => ok.test(f.name)).sort((a, b) => a.webkitRelativePath.localeCompare(b.webkitRelativePath));
    $("dir").value = "";
    if (!files.length) { setStatus("이 폴더에는 npz·gif·mp4·avi·mkv·mov 파일이 없습니다."); return; }
    for (const f of files) {
      const parts = f.webkitRelativePath.split("/"); parts.pop();
      await openUpload(f, "폴더: " + parts.join("/"));
    }
  };
  $("path").onkeydown = (e) => { if (e.key === "Enter") openPath(); };
  async function openUpload(f, group) {
    setStatus(`올리는 중: ${f.name} (${(f.size / 1e6).toFixed(1)} MB)…`);
    const headers = { "X-Filename": encodeURIComponent(f.name), "X-Lumi-Token": TOKEN }; if (group) headers["X-Group"] = encodeURIComponent(group);
    try { const r = await fetch(`/api/upload?${opts()}`, { method: "POST", headers, body: f }); startJob(await r.json()); }
    catch (e) { setStatus("올리기 실패: " + e); }
  }
  async function openPath() {
    const p = $("path").value.trim(); if (!p) return;
    setStatus("여는 중…");
    try { const r = await fetch(`/api/open?${opts()}`, { method: "POST", headers: { "Content-Type": "application/json", "X-Lumi-Token": TOKEN }, body: JSON.stringify({ path: p }) }); startJob(await r.json()); }
    catch (e) { setStatus("열기 실패: " + e); }
  }
  function startJob(j) {
    if (j.error && j.error.startsWith("토큰")) {             // 서버를 다시 켜서 토큰이 바뀜 → 새 페이지를 받아야 함
      setStatus("서버가 다시 켜졌습니다. 페이지를 새로고침합니다…"); setTimeout(() => location.reload(), 800); return;
    }
    if (j.error) { setStatus("열기 실패: " + j.error); alert("열기 실패: " + j.error); return; }
    for (const it of j.items) upsert({ ...it, outcome: "pending", video: null, n: 1, fps: 24, w: 0, h: 0, meta: {}, det: null, series: {}, det_sec: 0 });
    refreshGroups(); $("fGroup").value = j.items[0].group; $("fLabel").value = ""; $("fOut").value = ""; $("fText").value = "";
    loadedId = null; build();
    setStatus(`${j.name}: 영상 ${j.items.length}개, 처리 중…`);
    poll(j.job, j.items.length, j.name);
  }
  async function poll(job, total, name) {
    let s; try { s = await (await fetch(`/api/job/${job}`)).json(); } catch (e) { setStatus("서버 연결 끊김: " + e); return; }
    let changed = false;
    for (const rec of s.records) { const o = CLIPS.find(c => c.id === rec.id); if (!o || o.outcome === "pending") { upsert(rec); changed = true; } }
    for (const [id, msg] of Object.entries(s.errors)) { const o = CLIPS.find(c => c.id === id); if (o && o.outcome === "pending") { upsert({ ...o, outcome: "error", label_note: msg }); changed = true; } }
    if (changed) { refreshGroups(); build(); }
    jobStart[job] = jobStart[job] || Date.now();
    const sec = Math.round((Date.now() - jobStart[job]) / 1000);
    setStatus(`${name}: ${s.done}/${total} 처리됨 · ${sec}초 경과` + (s.done < total ? " (큰 영상은 하나에 1~3분)" : "")
      + (Object.keys(s.errors).length ? ` (실패 ${Object.keys(s.errors).length})` : ""));
    if (s.done < total) setTimeout(() => poll(job, total, name), 1000);
  }

  function info(c) {
    const d = c.det, R = META.rules || {};
    const yn = (b) => b ? `<b style="color:var(--miss)">위험</b>` : "안전";
    const pct = (x) => x == null ? "-" : (100 * x).toFixed(1) + "%";
    const verdict = `<div class="big"><span class="badge o-${c.outcome}" style="font-size:13px">${OUT_KO[c.outcome]}</span> 정답 ${c.label === null ? "없음" : c.label ? "위험" : "안전"} · 판정 ${d ? (d.dangerous ? "위험" : "안전") : "-"}</div><div style="color:var(--muted);font-size:12px;margin-top:4px">${esc(c.label_note)}</div>`;
    const det = d ? `<table>
      <tr><td>휘도</td><td>${yn(d.lum.dangerous)} · 최대 ${d.lum.max_flashes}회/초 (기준 &gt; ${R.max_flashes}) · 동시 전이 면적 ${pct(d.lum.peak_area)} (기준 &gt; ${pct(R.area)})</td></tr>
      <tr><td>적색</td><td>${yn(d.red.dangerous)} · 최대 ${d.red.max_flashes}회/초 · 면적 ${pct(d.red.peak_area)} · Δu′v′ ${(+d.red.max_delta_uv).toFixed(3)} (기준 &gt; 0.2)</td></tr>
      <tr><td>줄무늬</td><td>${d.fast ? `<span style="color:var(--muted)">스캔선 판정 생략(빠른 판정) — 아래 주파수 결과가 판정에 들어감</span>`
        : `${yn(d.pat.dangerous)} · 최대 면적 ${pct(d.pat.peak_area)} (정지 &gt; ${pct(R.pat_static)}, 움직임 &gt; ${pct(R.pat_dynamic)}${R.pat_drift_exempt ? ", 흐름 제외" : ""}) · 1초 중 ${(+d.pat.max_sec).toFixed(2)}초 (기준 ≥ 0.5) · ${d.pat.motion || "-"}`}</td></tr>
      ${d.pat2 ? `<tr><td>줄무늬(주파수)</td><td>${yn(d.pat2.dangerous)} · 최대 면적 ${pct(d.pat2.peak_area)} (&gt; ${pct(R.pat_static)}) · 1초 중 ${(+d.pat2.max_sec).toFixed(2)}초 · ${esc(d.pat2.method)} <span style="color:var(--muted)">${d.fast ? "(빠른 판정: 이 결과가 '판정'에 들어감)" : "(실험용 탐지기, 위 '판정'에는 들어가지 않음)"}</span></td></tr>` : ""}
      <tr><td>판정 시간</td><td>${c.det_sec}초</td></tr></table>` : `<div style="color:var(--muted)">판정하지 않았습니다 (--no-detect).</div>`;
    const meta = Object.entries(c.meta || {}).map(([k, x]) => `<tr><td>${esc(k)}</td><td>${esc(typeof x === "object" ? JSON.stringify(x) : x)}</td></tr>`).join("");
    $("info").innerHTML = `<div class="card"><h3>결과</h3>${verdict}</div><div class="card"><h3>우리 판정기</h3>${det}</div>
      <div class="card"><h3>클립</h3><table><tr><td>파일</td><td>${esc(c.source)}</td></tr><tr><td>크기</td><td>${c.w}×${c.h}, ${c.n}프레임, ${(+c.fps).toFixed(2)} fps (${(c.n / c.fps).toFixed(2)}초)</td></tr>${meta}</table></div>`;
    $("legend").innerHTML = `<span><i style="background:var(--lum)"></i>휘도 면적</span><span><i style="background:repeating-linear-gradient(90deg,var(--lum) 0 4px,transparent 4px 7px)"></i>휘도 1초 전이 수</span>
      <span><i style="background:var(--red)"></i>적색 면적</span><span><i style="background:repeating-linear-gradient(90deg,var(--red) 0 4px,transparent 4px 7px)"></i>적색 1초 전이 수</span>
      <span><i style="background:var(--pat)"></i>줄무늬 면적(스캔선)</span><span><i style="background:var(--pat2)"></i>줄무늬 면적(주파수)</span><span style="margin-left:auto">가운데 그래프: 각 조건을 기준값으로 나눈 비율 (100% = 기준선, 넘으면 그 조건 충족)</span>`;
  }

  // 그래프: 위 = 위험 구간 타임라인, 가운데 = 임계치 대비 %, 아래 = 1초 전이 수
  const g = $("g"), ctx = g.getContext("2d");
  function draw() {
    const c = view[cur]; if (!c) return;
    const dpr = window.devicePixelRatio || 1, W = g.clientWidth, H = 350;
    g.width = W * dpr; g.height = H * dpr; ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, W, H);
    const s = c.series || {}, n = c.n, L = 96, Rm = 12, PW = W - L - Rm;
    const X = (t) => L + (t / n) * PW;
    const ink = css("--ink"), muted = css("--muted"), line = css("--line");
    ctx.font = "11px system-ui, 'Malgun Gothic', sans-serif"; ctx.textBaseline = "middle";
    // (1) 타임라인
    const rows = [["정답", s.gt_hz, css("--gt"), null], ["휘도", s.lum_hz, css("--lum"), null], ["적색", s.red_hz, css("--red"), null], ["줄무늬", s.pat_hz, css("--pat"), s.pat_cand], ["줄무늬(주파수)", s.pat2_hz, css("--pat2"), s.pat2_cand]];
    let y = 8; const rh = 14;
    rows.forEach(([name, arr, col, cand], k) => {
      const yy = y + k * (rh + 4);
      ctx.fillStyle = muted; ctx.textAlign = "right"; ctx.fillText(name, L - 8, yy + rh / 2);
      ctx.fillStyle = line; ctx.fillRect(L, yy, PW, rh);
      if (cand) { ctx.globalAlpha = 0.35; ctx.fillStyle = col; cand.forEach((b, t) => { if (b) ctx.fillRect(X(t), yy, Math.max(1, PW / n), rh); }); ctx.globalAlpha = 1; }
      if (arr) { ctx.fillStyle = col; arr.forEach((b, t) => { if (b) ctx.fillRect(X(t), yy, Math.max(1, PW / n), rh); }); }
      else if (!c.det && k > 0) { ctx.fillStyle = muted; ctx.textAlign = "left"; ctx.fillText("판정 안 함", L + 4, yy + rh / 2); }
    });
    // (2) 임계치 대비 %
    const top2 = 108, h2 = 140, ymax = 300;
    const Y2 = (p) => top2 + h2 - Math.min(p, ymax) / ymax * h2;
    axes(top2, h2, [0, 100, 200, 300], (p) => Y2(p), (p) => p + "%", "임계치 대비");
    ctx.strokeStyle = css("--miss"); ctx.setLineDash([5, 4]); ctx.beginPath(); ctx.moveTo(L, Y2(100)); ctx.lineTo(L + PW, Y2(100)); ctx.stroke(); ctx.setLineDash([]);
    const R = META.rules || { area: 0.25, max_flashes: 3 }, need = 2 * R.max_flashes + 1;
    const curve = (arr, f, col, dash, yfun) => { if (!arr) return; ctx.strokeStyle = col; ctx.lineWidth = 1.6; ctx.setLineDash(dash ? [4, 3] : []); ctx.beginPath();
      let on = false; arr.forEach((x, t) => { const p = f(x, t); if (p == null) { on = false; return; } const px = X(t + 0.5), py = yfun(p); on ? ctx.lineTo(px, py) : ctx.moveTo(px, py); on = true; }); ctx.stroke(); ctx.setLineDash([]); ctx.lineWidth = 1; };
    curve(s.lum_area, (x) => 100 * x / R.area, css("--lum"), false, Y2);
    curve(s.lum_trans, (x) => 100 * x / need, css("--lum"), true, Y2);
    curve(s.red_area, (x) => 100 * x / R.area, css("--red"), false, Y2);
    curve(s.red_trans, (x) => 100 * x / need, css("--red"), true, Y2);
    curve(s.pat_area, (x, t) => { const lim = s.pat_limit ? s.pat_limit[t] : R.pat_static; return lim == null ? null : 100 * x / lim; }, css("--pat"), false, Y2);
    curve(s.pat2_area, (x) => 100 * x / R.pat_static, css("--pat2"), false, Y2);
    // (3) 1초 전이 수
    const top3 = 270, h3 = 64;
    const mx = Math.max(need + 3, ...(s.lum_trans || [0]), ...(s.red_trans || [0]));
    const Y3 = (k) => top3 + h3 - k / mx * h3;
    axes(top3, h3, [0, need, mx], Y3, (k) => k, "1초 전이 수");
    ctx.strokeStyle = css("--miss"); ctx.setLineDash([5, 4]); ctx.beginPath(); ctx.moveTo(L, Y3(need)); ctx.lineTo(L + PW, Y3(need)); ctx.stroke(); ctx.setLineDash([]);
    ctx.fillStyle = css("--miss"); ctx.textAlign = "right"; ctx.fillText(`${need}번 = 초당 ${R.max_flashes}회 초과`, L + PW - 2, Y3(need) - 7);
    curve(s.lum_trans, (x) => x, css("--lum"), false, Y3);
    curve(s.red_trans, (x) => x, css("--red"), false, Y3);
    // 시간 눈금
    ctx.fillStyle = muted; ctx.textAlign = "center";
    const dur = n / c.fps, step = dur > 20 ? 5 : dur > 6 ? 1 : 0.5;
    for (let tt = 0; tt <= dur + 1e-9; tt += step) ctx.fillText(tt.toFixed(step < 1 ? 1 : 0) + "s", X(tt * c.fps), H - 6);
    // 재생 위치
    const f = Math.min(n - 1, Math.floor(v.currentTime * c.fps + 1e-6));
    ctx.strokeStyle = ink; ctx.lineWidth = 1.5; ctx.beginPath(); ctx.moveTo(X(f + 0.5), 4); ctx.lineTo(X(f + 0.5), H - 16); ctx.stroke(); ctx.lineWidth = 1;
    const mo = s.motion ? s.motion[f] : "";
    $("pos").textContent = `프레임 ${f + 1}/${n} · ${v.currentTime.toFixed(2)}s` + (s.lum_trans ? ` · 휘도 1초 전이 ${s.lum_trans[f]} · 면적 ${(100 * s.lum_area[f]).toFixed(1)}% · 줄무늬 ${(100 * s.pat_area[f]).toFixed(1)}%${mo ? " " + mo : ""}` : "");
    function axes(top, h, ticks, Yf, fmt, title) {
      ctx.strokeStyle = line; ctx.fillStyle = muted; ctx.textAlign = "right";
      ticks.forEach(k => { const yy = Yf(k); ctx.beginPath(); ctx.moveTo(L, yy); ctx.lineTo(L + PW, yy); ctx.stroke(); ctx.fillText(fmt(k), L - 6, yy); });
      ctx.save(); ctx.translate(12, top + h / 2); ctx.rotate(-Math.PI / 2); ctx.textAlign = "center"; ctx.fillText(title, 0, 0); ctx.restore();
    }
  }
  const tick = () => { draw(); if (!v.paused) requestAnimationFrame(tick); };
  v.addEventListener("play", tick); v.addEventListener("seeked", draw); v.addEventListener("timeupdate", () => { if (v.paused) draw(); });
  window.addEventListener("resize", draw);
  g.addEventListener("click", (e) => { const c = view[cur]; if (!c) return; const r = g.getBoundingClientRect(); const L = 96, PW = r.width - L - 12;
    const t = (e.clientX - r.left - L) / PW; if (t >= 0 && t <= 1) { v.currentTime = t * c.n / c.fps; draw(); } });
  refreshGroups(); if (filt.fGroup !== undefined) $("fGroup").value = filt.fGroup;
  build();
})();
</script>
</body>
</html>
"""

if __name__ == "__main__":
    raise SystemExit(main())
