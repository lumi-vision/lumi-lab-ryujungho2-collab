"""7종 합성 결과를 사람이 눈으로 확인할 모음 영상 하나로 만든다 (원본 | 합성본 나란히 + 설명 자막).

    python scripts/make_hazard7_montage.py --runs d3 d4red --per 1 [--near]

종류마다 판별기가 목표 라벨을 붙인(적중) 'hazard' 표본을 마스크 방식이 겹치지 않게 고른다. --near 면 경계 표본도 하나씩.
각 표본 앞에 1.5 초 검은 설명 화면(종류·마스크·판별기 라벨)을 넣는다. 맨 앞에 경고 화면.
저장: data/hazard7/review/hazard7_montage_<시각>.mp4 (+ 같은 이름 .md 에 목록) — data/ 는 Git 제외.

⚠️ 결과는 광과민성 위험 자극이다. 이 PC 에서만, 화면 밝기를 낮추고 작은 창으로, 짧게 본다. 공유·녹화하지 않는다.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from lumi_synth import hazard7  # noqa: E402
from make_hazard7_dataset import FPS, H, W, decode  # noqa: E402

FONT = "C:/Windows/Fonts/malgun.ttf"
KIND_KO = {"lum_flash": "휘도 플래시", "intersection": "교집합 (휘도 플래시 + 빠른 장면 전환)", "scene": "빠른 장면 전환",
           "red_flash": "적색 플래시", "lum_ext": "휘도 연장 (5초 창)", "red_ext": "적색 연장 (5초 창)", "pattern": "정적 공간 패턴"}
MASK_KO = {"static": "고정 영역", "drift": "움직이는 영역(이동·확대·회전)", "anchored": "장면 속 덩어리 + 광학 흐름 추적",
           "spot": "점광원(섬광)", "multi": "광원 여러 개", "full": "화면 전체", "perspective": "평면 원근 무늬"}


def card(lines: list[str], w: int, h: int, sizes=None) -> np.ndarray:
    img = Image.new("RGB", (w, h), (0, 0, 0))
    d = ImageDraw.Draw(img)
    y = h // 2 - 30 * len(lines)
    for i, t in enumerate(lines):
        f = ImageFont.truetype(FONT, (sizes or [34] + [24] * 10)[i])
        tw = d.textlength(t, font=f)
        d.text(((w - tw) / 2, y), t, font=f, fill=(230, 230, 230))
        y += int(f.size * 1.7)
    return np.array(img)[..., ::-1].copy()


def caption(frame: np.ndarray, text: str) -> np.ndarray:
    img = Image.fromarray(frame[..., ::-1].copy())
    d = ImageDraw.Draw(img)
    f = ImageFont.truetype(FONT, 20)
    d.rectangle([0, 0, img.width, 30], fill=(0, 0, 0))
    d.text((8, 3), text, font=f, fill=(255, 255, 255))
    return np.array(img)[..., ::-1]


def pick(rows: list[dict], kind: str, intent: str, k: int) -> list[dict]:
    rs = [r for r in rows if r["kind"] == kind and r["intent"] == intent and "error" not in r
          and (r.get("hit_added") if intent == "hazard" else True)]
    out, seen = [], set()
    for r in rs:                                        # 마스크 방식·무늬가 겹치지 않게 먼저
        key = (r["recipe"].get("mask_mode"), r["recipe"].get("pattern"), r["recipe"].get("mode"))
        if key not in seen:
            out.append(r); seen.add(key)
        if len(out) >= k:
            return out
    return (out + [r for r in rs if r not in out])[:k]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", default=["d3"], help="data/hazard7/<run>/val — 뒤의 run 이 같은 종류를 덮어씀(예: 적색은 d4red)")
    ap.add_argument("--split", default="val")
    ap.add_argument("--per", type=int, default=1)
    ap.add_argument("--near", action="store_true")
    a = ap.parse_args()
    import av
    rows_by_kind: dict[str, list[dict]] = {}
    for run in a.runs:
        d = ROOT / "data" / "hazard7" / run / a.split
        rows = [json.loads(l) for l in (d / "manifest.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
        for r in rows:
            r["_dir"] = str(d)
        for kind in {r["kind"] for r in rows}:
            rows_by_kind[kind] = [r for r in rows if r["kind"] == kind]
    sel = []
    for kind in hazard7.KINDS:
        rs = rows_by_kind.get(kind, [])
        sel += [("hazard", r) for r in pick(rs, kind, "hazard", a.per)]
        if a.near:
            sel += [("near", r) for r in pick(rs, kind, "near", 1)]
    out_dir = ROOT / "data" / "hazard7" / "review"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M")
    path = out_dir / f"hazard7_montage_{stamp}.mp4"
    W2 = 2 * W
    lines_md = [f"# 7종 합성 모음 영상 {stamp}", "", f"`{path.name}` · 왼쪽 원본 | 오른쪽 합성본 · 24 fps", "",
                "| # | 종류 | 의도 | 마스크 | 판별기 라벨(새로 생긴 것) | 표본 |", "|---|---|---|---|---|---|"]
    with av.open(str(path), "w") as c:
        st = c.add_stream("libx264", rate=int(FPS))
        st.width, st.height, st.pix_fmt = W2, H, "yuv420p"
        st.options = {"crf": "16"}

        def put(fr):
            for p in st.encode(av.VideoFrame.from_ndarray(np.ascontiguousarray(fr), format="bgr24")):
                c.mux(p)

        warn = card(["⚠ 광과민성 위험 자극", "깜빡임·적색 점멸·줄무늬가 들어 있습니다",
                     "화면 밝기를 낮추고, 작은 창으로, 짧게 보세요. 불편하면 바로 멈추세요.",
                     "왼쪽 = 원본 바탕   |   오른쪽 = 합성본"], W2, H, [40, 26, 22, 22])
        for _ in range(int(3 * FPS)):
            put(warn)
        for i, (intent, r) in enumerate(sel, 1):
            rc = r["recipe"]
            mm = rc.get("mask_mode", "static")
            extra = f" · 무늬 {rc['pattern']} · {rc['cpd']:.1f} cpd" if r["kind"] == "pattern" else f" · 섞기 {rc.get('mode')}"
            labs = ", ".join(r.get("added_labels") or []) or "없음"
            t = card([f"{i}. {KIND_KO[r['kind']]}  ({'위험' if intent == 'hazard' else '경계(near: ' + str(rc.get('near')) + ')'})",
                      f"마스크: {MASK_KO.get(mm, mm)}{extra}",
                      f"면적 {rc['area']:.0%} · 지속 {rc['dur']:.1f} s · 판별기 라벨: {labs}"], W2, H, [32, 22, 22])
            for _ in range(int(1.5 * FPS)):
                put(t)
            d = Path(r["_dir"])
            syn, cln = decode(d / f"{r['id']}.mp4"), decode(d / f"{r['id']}.clean.mp4")
            for k in range(min(len(syn), len(cln))):
                fr = np.concatenate([cln[k], syn[k]], 1)
                put(caption(fr, f"{i}. {KIND_KO[r['kind']]} | {MASK_KO.get(mm, mm)} | {k / FPS:4.1f} s   (원본 | 합성)"))
            lines_md.append(f"| {i} | {KIND_KO[r['kind']]} | {intent} | {MASK_KO.get(mm, mm)} | {labs} | `{r['id']}` ({Path(r['_dir']).parent.name}) |")
        for p in st.encode():
            c.mux(p)
    path.with_suffix(".md").write_text("\n".join(lines_md) + "\n", encoding="utf-8")
    print(path)
    print(f"표본 {len(sel)}개 · 길이 약 {3 + len(sel) * 9.5:.0f} 초")


if __name__ == "__main__":
    main()
