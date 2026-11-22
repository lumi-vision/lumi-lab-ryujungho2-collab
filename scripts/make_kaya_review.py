"""Kaya 실영상 보정 결과 육안 확인용 비교 영상 — 사람이 본다(작성자 Claude 는 재생하지 않는다).

data/correction/kaya_out/<방법>/<번호>.mp4 (eval_correction.py --real kaya --save-models 가 저장)를
원본과 나란히 붙인다:  [원본 | H(시드 0, 60 epoch) | B2 팀원 감광]  + 위에 이름·보정 전후 판정 라벨.
⚠️ 원본 칸은 광과민성 위험 자극 그대로다. 첫 화면에 경고 카드(3 초)를 넣는다. 밝기를 낮추고 작은 창으로 볼 것.

    python scripts/make_kaya_review.py                       # → data/correction/kaya_review/<번호>_review.mp4 + index.md
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from eval_real_hazards import decode as dec_any  # noqa: E402

PANEL_H = 270


def fit(img: np.ndarray, h: int = PANEL_H) -> np.ndarray:
    s = h / img.shape[0]
    return cv2.resize(img, (max(2, int(round(img.shape[1] * s)) // 2 * 2), h), interpolation=cv2.INTER_AREA)


def text_img(lines: list[str], w: int, h: int, size: int = 18) -> np.ndarray:
    from PIL import Image, ImageDraw, ImageFont
    im = Image.new("RGB", (w, h), (18, 18, 18))
    d = ImageDraw.Draw(im)
    font = None
    for f in ("C:/Windows/Fonts/malgun.ttf", "C:/Windows/Fonts/gulim.ttc"):
        try:
            font = ImageFont.truetype(f, size); break
        except OSError:
            pass
    y = 6
    for ln in lines:
        d.text((8, y), ln, fill=(235, 235, 235), font=font); y += size + 6
    return np.array(im)[..., ::-1]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--methods", nargs="+", default=["h_s0_e60", "B2"])
    ap.add_argument("--rows", default=str(ROOT / "checks" / "correction" / "eval_kaya_all_rows.merged.jsonl"))
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    src = ROOT / "data" / "real" / "kaya2025" / "extracted"
    outd = ROOT / "data" / "correction" / "kaya_review"
    outd.mkdir(parents=True, exist_ok=True)
    rows = {}
    rp = Path(a.rows)
    if rp.exists():
        for l in rp.read_text(encoding="utf-8").splitlines():
            if l.strip():
                r = json.loads(l)
                if "error" not in r:
                    rows[(Path(r["file"]).stem, r["method"])] = r
    files = sorted((p for p in src.glob("*.*") if p.suffix.lower() in (".mp4", ".avi")), key=lambda p: int(p.stem) if p.stem.isdigit() else 999)
    files = files[: a.limit or None]
    idx = ["# Kaya 실영상 보정 육안 확인", "", "⚠️ 원본 칸은 광과민성 위험 자극 그대로 — 화면 밝기를 낮추고 작은 창으로 짧게.", "",
           f"칸 순서: 원본 | {' | '.join(a.methods)}", "", "| 파일 | 원본 라벨 | " + " | ".join(f"{m} 보정 뒤 라벨" for m in a.methods) + " |",
           "|---|---|" + "---|" * len(a.methods)]
    for p in files:
        outs = [ROOT / "data" / "correction" / "kaya_out" / m / f"{p.stem}.mp4" for m in a.methods]
        if not all(o.exists() for o in outs):
            print("건너뜀(보정 영상 없음):", p.name, flush=True); continue
        x, fps = dec_any(p)
        ys = [dec_any(o)[0] for o in outs]
        n = min([len(x)] + [len(y) for y in ys])
        r0 = rows.get((p.stem, a.methods[0]), {})
        lab0 = ", ".join(r0.get("labels_before", [])) or "-"
        heads = [f"원본 · 라벨: {lab0}"] + [f"{m} · 뒤: {', '.join(rows.get((p.stem, m), {}).get('labels_after', [])) or '없음'}" for m in a.methods]
        panels0 = [fit(x[0])] + [fit(y[0]) for y in ys]
        widths = [q.shape[1] for q in panels0]
        W = sum(widths)
        cap = np.concatenate([text_img([h], w, 34, 15) for h, w in zip(heads, widths)], 1)
        warn = text_img(["⚠️ 광과민성 위험 자극 (왼쪽 원본 칸)", "화면 밝기를 낮추고, 불편하면 바로 멈추세요.", f"Kaya {p.name} · {fps:.1f} fps · {n} 프레임"], W, PANEL_H + 34, 22)
        vw = cv2.VideoWriter(str(outd / f"{p.stem}_review.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, PANEL_H + 34))
        for _ in range(int(round(3 * fps))):
            vw.write(warn)
        for t in range(n):
            row = np.concatenate([fit(x[t])] + [fit(y[t]) for y in ys], 1)[:, :W]
            vw.write(np.concatenate([cap, row], 0))
        vw.release()
        idx.append(f"| {p.name} | {lab0} | " + " | ".join(", ".join(rows.get((p.stem, m), {}).get("labels_after", [])) or "없음" for m in a.methods) + " |")
        print("만듦:", p.stem, flush=True)
    (outd / "index.md").write_text("\n".join(idx) + "\n", encoding="utf-8")
    print("완료:", outd, flush=True)


if __name__ == "__main__":
    main()
