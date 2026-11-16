"""생성기(G)가 제대로 만드는지 사람이 확인할 묶음: 플레이어로 여는 npz + 깜빡임 없이 보는 정지 그림(PNG).

    python scripts/make_review_set.py            # 기본: 지금 쓰는 G 들, G 마다 위험 2 + 근처 안전 2
    python scripts/make_review_set.py --per 3 --ckpt w2s_div2.0_1k_s0.pt s4dc_from_s3z2.pt

결과: data/playlist/review/<날짜시각>/
    <G 이름>/<번호>_<목표>.npz   플레이어(LUMI_플레이어.bat → '폴더 열기')로 연다. 원본·합성본·바뀐 영역이 들어 있어
                                 플레이어 둘째 칸에 원본과 비교가 나온다. 판정은 플레이어가 다시 한다
    <G 이름>.png                  깜빡임 없이 보는 정지 그림: 줄마다 표본 하나, 칸 = 원본 · 합성본 6프레임 · 바뀐 영역 · 판정
    index.md                      표본 목록과 판정 결과(판정기 기준)
바탕은 검증용(val) 영화 클립이다. 같은 --seed 면 같은 영상이 나온다.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from lumi_synth import generate, roi  # noqa: E402

DEFAULT = ("w_div2.0_2k_s0.pt", "w2s_div2.0_1k_s0.pt", "r_crisp2_div2.0_s0.pt", "r4z2_crisp2_div2.0_s0.pt",
           "s2z_div2.0_s0.pt", "s3z2_div2.0_s0.pt", "s4dc_from_s3z2.pt")
KO = {"lum": "휘도", "red": "적색", "stripe": "줄무늬"}


def judge(path: str) -> dict:
    with np.load(path) as z:
        x, fps = z["flicker"], float(z["fps"])
    _, su = roi.judge_tubes(x, fps)
    return su


def thumb(frame: np.ndarray, w: int = 192) -> np.ndarray:
    h = int(round(frame.shape[0] * w / frame.shape[1]))
    return cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)


def label(img: np.ndarray, text: str) -> np.ndarray:
    """그림 위에 글자(영문·숫자만 — OpenCV 기본 글꼴은 한글을 못 그림)."""
    out = img.copy()
    cv2.putText(out, text, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(out, text, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def sheet(rows: list[dict]) -> np.ndarray:
    lines = []
    for r in rows:
        with np.load(r["path"]) as z:
            clean, x, ch = z["clean"], z["flicker"], z["mask_changed"]
        t_idx = np.linspace(0, len(x) - 1, 6).round().astype(int)
        cells = [label(thumb(clean[0]), "clean")]
        cells += [label(thumb(x[t]), f"t={t}") for t in t_idx]
        m = (ch.any(axis=0) * 255).astype(np.uint8)
        cells.append(label(cv2.cvtColor(thumb(m), cv2.COLOR_GRAY2BGR), "changed"))
        j = r["judge"]
        txt = np.full_like(cells[0], 40)
        info = [f"intent: {r['intent']}", f"lum {'HAZ' if j['lum']['dangerous'] else 'ok'} {j['lum']['max_flashes']:.1f}/s",
                f"red {'HAZ' if j['red']['dangerous'] else 'ok'} {j['red']['max_flashes']:.1f}/s",
                f"stripe {'HAZ' if j['stripe']['dangerous'] else 'ok'} {j['stripe']['peak_area']:.0%}", r.get("extra", "")]
        for k, s in enumerate(info):
            cv2.putText(txt, s, (6, 18 + 18 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
        cells.append(txt)
        lines.append(np.concatenate([np.pad(c, ((2, 2), (2, 2), (0, 0))) for c in cells], axis=1))
    return np.concatenate(lines, axis=0)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", nargs="+", default=list(DEFAULT))
    ap.add_argument("--per", type=int, default=2, help="G·목표마다 표본 수")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--workers", type=int, default=10)
    args = ap.parse_args()
    out = ROOT / "data" / "playlist" / "review" / time.strftime("%Y%m%d_%H%M")
    clips, fps = generate.load_backgrounds("val")
    rng = np.random.default_rng(args.seed)
    made = []
    for ck in args.ckpt:
        hz = generate.checkpoint_hazard(generate.GAN_DIR / ck)
        d = out / Path(ck).stem
        for intent in ("hazard", "near"):
            for k in range(args.per):
                seed = int(rng.integers(0, 2 ** 31 - 1))
                bg = int(rng.integers(0, len(clips)))
                r = generate.sample_recipe("gan", intent, seed, split="val", bg=bg, checkpoint=ck)
                r["id"] = f"{k:02d}_{intent}"
                res = generate.make(r, d, clean=clips[bg], fps=fps)
                det = res["detail"]
                extra = (f"cpd {det.get('gan_cpd')} {det.get('gan_motion', '')}" if hz == "stripe"
                         else f"light {det.get('gan_mean_light_cd')} cd" if hz == "lum" else f"red {det.get('gan_mean_red')}")
                made.append({"ck": ck, "hazard": hz, "intent": intent, "path": str(res["path"]), "bg": bg, "seed": seed,
                             "extra": extra, "detail": {k2: v for k2, v in det.items() if k2.startswith("gan_")}})
        print(f"{ck}: {2 * args.per}개", flush=True)
    with cf.ProcessPoolExecutor(max_workers=args.workers) as pool:
        for r, su in zip(made, pool.map(judge, [m["path"] for m in made])):
            r["judge"] = su
    L = [f"# 생성기 확인 묶음 — {out.name}", "", "> `scripts/make_review_set.py` · 바탕 = 검증용(val) 클립 · 판정 = 하드 판정기(줄무늬는 초당 약 10장)",
         "> ⚠️ 광과민성 위험 영상이다. 먼저 PNG(정지 그림)로 보고, 영상은 이 PC 에서만 밝기를 낮춰 짧게 본다.", "",
         "| G | 파일 | 의도 | 휘도 (회/초) | 적색 (회/초) | 줄무늬 (최대 면적) | G 가 고른 조건 |", "|---|---|---|---|---|---|---|"]
    for ck in args.ckpt:
        rows = [m for m in made if m["ck"] == ck]
        cv2.imwrite(str(out / f"{Path(ck).stem}.png"), sheet(rows))
        for m in rows:
            j = m["judge"]
            f = lambda k, v: f"{'**위험**' if j[k]['dangerous'] else '안전'} {v}"      # noqa: E731
            L.append(f"| `{Path(ck).stem}` ({KO[m['hazard']]}) | `{Path(m['path']).name}` | {'위험' if m['intent'] == 'hazard' else '근처 안전'} | "
                     f"{f('lum', j['lum']['max_flashes'])} | {f('red', j['red']['max_flashes'])} | {f('stripe', format(j['stripe']['peak_area'], '.0%'))} | "
                     f"{json.dumps(m['detail'], ensure_ascii=False)} |")
    (out / "index.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"저장: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
