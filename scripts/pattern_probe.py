"""공간 패턴 진단: 같은 정지 프레임을 우리 줄무늬 판정(measure.stripe_masks)과 팀원 탐지기
(rockrovks B2 vdcorefs.StaticPatternDetector.inspect_frame)에 넣어 어디서 갈리는지 잰다.

⚠️ 합성 패턴은 메모리에서만 만들고 저장·표시하지 않는다. 실제 위험 영상(Kaya)도 숫자로만 다룬다.
팀원 코드는 수정하지 않고 import 만 한다.

  A. 합성 패턴 (640×360, 회색 30 ↔ 200 = 2.6 ↔ 115 cd/m²): 줄무늬·체커보드·동심원·방사·나선·물방울·물결 + 음성 대조군
  B. 깨끗한 자연 프레임: 오라클 바탕(BBB val 12, Sintel test 16 클립에서 프레임 3장씩) → 오탐율
  C. Kaya 실제 위험 영상 중 팀원이 '정적 공간 패턴'으로 판정한 영상: 고르게 뽑은 20프레임
사용: python scripts/pattern_probe.py --team-dir <lumi-lab-rockrovks>
결과: checks/pattern/probe.md
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from lumi_synth import measure  # noqa: E402

W, H = 640, 360
LO, HI = 30, 200


def to_lum(gray_u8: np.ndarray) -> np.ndarray:
    return measure.luminance_cd(np.repeat(gray_u8[..., None], 3, axis=-1))


def synth(ppd: float) -> dict[str, tuple[np.ndarray, str]]:
    """이름 → (회색 uint8 H×W, 설계상 기대). 주기는 3 cpd 근처."""
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    per = ppd / 3.0                                             # 3 cpd 의 주기(px)
    sq = lambda phase: np.where(np.floor(phase * 2) % 2 == 0, HI, LO).astype(np.uint8)
    cx, cy = W / 2, H / 2
    r = np.hypot(xx - cx, yy - cy)
    th = np.arctan2(yy - cy, xx - cx)
    rng = np.random.default_rng(0)
    out = {}
    out["세로 줄무늬 (사각파)"] = (sq(xx / per), "위험")
    out["세로 줄무늬 (사인파)"] = ((115 + 85 * np.sin(2 * np.pi * xx / per)).astype(np.uint8), "위험")
    a = np.deg2rad(30)
    out["비스듬한 줄무늬 30°"] = (sq((xx * np.cos(a) + yy * np.sin(a)) / per), "위험")
    out["체커보드"] = (np.where((np.floor(xx / (per / 2)) + np.floor(yy / (per / 2))) % 2 == 0, HI, LO).astype(np.uint8), "위험(문헌: 줄무늬보다 약함)")
    out["동심원 (가운데)"] = (sq(r / per), "위험")
    out["동심원 (중심 치우침)"] = (sq(np.hypot(xx - 0.25 * W, yy - 0.3 * H) / per), "위험")
    n_spoke = int(round(2 * np.pi * (0.3 * H) / per))          # 반지름 0.3H 에서 3 cpd
    out["방사 (바큇살)"] = (sq(th / (2 * np.pi) * n_spoke), "위험")
    out["나선"] = (sq(r / per + th / (2 * np.pi) * 3), "위험")
    out["물결 줄무늬"] = (sq((xx + 0.8 * per * np.sin(2 * np.pi * yy / (6 * per))) / per), "위험")
    dots = ((xx % per - per / 2) ** 2 + (yy % per - per / 2) ** 2) < (0.3 * per) ** 2
    out["물방울 격자"] = (np.where(dots, HI, LO).astype(np.uint8), "불명(문헌 근거 약함)")
    half = sq(xx / per); half[:, int(0.5 * W):] = 90
    out["줄무늬 왼쪽 50%"] = (half, "위험 (> 40%)")
    part = sq(xx / per); part[:, int(0.3 * W):] = 90
    out["줄무늬 왼쪽 30%"] = (part, "안전 (면적 미달)")
    out["줄무늬 0.5 cpd"] = (sq(xx / (ppd / 0.5)), "안전 (대역 밖)")
    low = np.where(np.floor(xx / per * 2) % 2 == 0, 100, 80).astype(np.uint8)
    out["줄무늬 저대비 (ΔL < 20)"] = (low, "안전 (대비 미달)")
    edges = np.cumsum(rng.uniform(0.3, 3.0, 400) * per / 2)
    irr = np.searchsorted(edges, xx[0]) % 2
    out["불규칙 줄무늬 (폭 제각각)"] = (np.broadcast_to(np.where(irr == 0, HI, LO).astype(np.uint8), (H, W)).copy(), "불명(규칙 패턴 아님)")
    noise = cv2.resize(rng.integers(0, 2, (int(H / (per / 2)) + 1, int(W / (per / 2)) + 1)).astype(np.uint8),
                       (W, H), interpolation=cv2.INTER_NEAREST)
    out["무작위 블록 잡음"] = (np.where(noise == 1, HI, LO).astype(np.uint8), "안전 (규칙 패턴 아님)")
    out["균일 회색"] = (np.full((H, W), 128, np.uint8), "안전")
    return out


class Team:
    def __init__(self, team_dir: str):
        sys.path.insert(0, str(Path(team_dir) / "Fullscreen" / "B2"))
        import vdcorefs
        self.vd = vdcorefs
        self.dev = vdcorefs.resolve_device("cuda")
        self.cache = {}

    def frame(self, lum: np.ndarray):
        h, w = lum.shape
        if (h, w) not in self.cache:
            geom = self.vd.ViewingConfig().resolve_geometry(w, h)
            self.cache[(h, w)] = (self.vd.StaticPatternDetector(geometry=geom, pse=self.vd.PSEConfig(), device=self.dev), geom)
        det, geom = self.cache[(h, w)]
        det.reset()
        is_pat, area, pairs, cpd = det.inspect_frame(torch.from_numpy(lum.astype(np.float32)).to(self.dev))
        hit = bool(is_pat) and area > det.pse.pattern_area_min
        return hit, float(area), int(pairs), float(cpd), geom


def ours_frame(lum: np.ndarray):
    t0 = time.perf_counter()
    m = measure.stripe_masks(lum[None].astype(np.float32))
    return float(m.mean()), time.perf_counter() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--team-dir", required=True)
    ap.add_argument("--kaya-frames", type=int, default=20)
    a = ap.parse_args()
    team = Team(a.team_dir)
    ppd = measure.pixels_per_degree(W, H, 0.0)
    L = ["# 공간 패턴 진단: 우리 줄무늬 판정 대 팀원 탐지기 (B2 vdcorefs)", "",
         f"> {time.strftime('%Y-%m-%d %H:%M')} · `scripts/pattern_probe.py` · 정지 프레임 한 장씩 판정 · 640×360 · 우리 ppd {ppd:.1f} px/° "
         "· 우리: 줄무늬 픽셀 면적 > 40% 이면 후보 · 팀원: is_pattern 이고 area > 40% 이면 후보", ""]

    # ---- A
    L += ["## A. 합성 패턴 (3 cpd 근처, 2.6 ↔ 115 cd/m²)", "",
          "| 패턴 | 설계상 기대 | 우리: 줄무늬 면적 | 우리 판정 | 팀원: 대비 면적 / 쌍 수 / cpd | 팀원 판정 |", "|---|---|---|---|---|---|"]
    secs = []
    for name, (img, exp) in synth(ppd).items():
        lum = to_lum(img)
        area, sec = ours_frame(lum); secs.append(sec)
        hit, t_area, pairs, cpd, geom = team.frame(lum)
        L.append(f"| {name} | {exp} | {area:.1%} | {'후보' if area > measure.PATTERN_AREA_STATIC else '·'} | "
                 f"{t_area:.1%} / {pairs} / {cpd:.1f} | {'후보' if hit else '·'} |")
        print(L[-1], flush=True)
    L += ["", f"우리 판정 시간: 프레임당 평균 {np.mean(secs):.2f}초 (CPU). 팀원 ppd x {geom.px_per_deg_x:.1f} / y {geom.px_per_deg_y:.1f} px/°.", ""]

    # ---- B
    L += ["## B. 깨끗한 자연 프레임 (오탐 확인)", "", "| 출처 | 프레임 수 | 우리 후보 | 우리 줄무늬 면적 최대 | 팀원 후보 | 팀원 is_pattern (면적 무관) | 팀원 대비 면적 중앙값 |", "|---|---|---|---|---|---|---|"]
    for split, label in (("val", "BBB (애니메이션)"), ("test", "Sintel (애니메이션)")):
        clips = np.load(ROOT / "data" / "oracle_768x432_24fps" / f"backgrounds_{split}.npz")["clips"]
        o_hit = t_hit = t_pat = n = 0; o_max = 0.0; t_areas = []
        for c in clips:
            for f in c[[0, len(c) // 2, len(c) - 1]]:
                f = cv2.resize(f, (W, H), interpolation=cv2.INTER_AREA)
                lum = measure.luminance_cd(f)
                area, _ = ours_frame(lum)
                hit, t_area, pairs, cpd, _ = team.frame(lum)
                n += 1; o_hit += area > measure.PATTERN_AREA_STATIC; o_max = max(o_max, area)
                t_hit += hit; t_pat += pairs >= 6; t_areas.append(t_area)
        L.append(f"| {label} | {n} | {o_hit} | {o_max:.1%} | {t_hit} | {t_pat} | {np.median(t_areas):.1%} |")
        print(L[-1], flush=True)

    # ---- C
    rows_path = ROOT / "data" / "real" / "eval_rows.jsonl"
    if rows_path.exists():
        sys.path.insert(0, str(ROOT / "scripts"))
        from eval_real_hazards import decode
        rows = [json.loads(x) for x in rows_path.read_text(encoding="utf-8").splitlines() if x.strip()]
        sel = [r for r in rows if "error" not in r and "정적 공간 패턴" in r["team"]["labels"]]
        L += ["", f"## C. Kaya 실제 위험 영상 중 팀원이 '정적 공간 패턴'으로 판정한 {len(sel)}개 (영상마다 고르게 {a.kaya_frames}프레임)", "",
              "| 영상 | 팀원 후보 프레임 | 팀원 대비 면적 중앙값 | 팀원 쌍 수 중앙값 (후보 프레임) | 우리 후보 프레임 | 우리 줄무늬 면적 중앙값 / 최대 |", "|---|---|---|---|---|---|"]
        for r in sel:
            frames, _ = decode(ROOT / "data" / "real" / r["file"])
            idx = np.linspace(0, len(frames) - 1, a.kaya_frames).astype(int)
            t_hits, t_areas, t_pairs, o_areas = 0, [], [], []
            for i in idx:
                lum = measure.luminance_cd(frames[i])
                hit, t_area, pairs, cpd, _ = team.frame(lum)
                area, _ = ours_frame(lum)
                t_hits += hit; t_areas.append(t_area); o_areas.append(area)
                if hit:
                    t_pairs.append(pairs)
            name = "/".join(Path(r["file"]).parts[-2:])
            L.append(f"| {name} | {t_hits}/{len(idx)} | {np.median(t_areas):.1%} | {np.median(t_pairs) if t_pairs else 0:.0f} | "
                     f"{int(np.sum(np.array(o_areas) > measure.PATTERN_AREA_STATIC))}/{len(idx)} | {np.median(o_areas):.1%} / {np.max(o_areas):.1%} |")
            print(L[-1], flush=True)
    out = ROOT / "checks" / "pattern" / "probe.md"
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("저장:", out)


if __name__ == "__main__":
    main()
