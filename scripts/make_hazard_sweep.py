"""위험 3종(휘도 깜빡임·적색 깜빡임·줄무늬) 생성 스윕: 조건 하나씩 판정 경계를 가로질러 바꾼 영상들을 만든다.

⚠️ 결과 영상(data/hazard_sweep/**/*.mkv)은 광과민성 발작을 일으킬 수 있는 자극입니다. 재생·미리보기하지 마세요.

목적
    생성기(lumi_synth.oracle 의 render_*)가 판정 규칙의 각 조건을 '의도한 대로' 넘기고 무너뜨리는지,
    우리 판정기와 팀원 판별기가 각각 어떻게 반응하는지 그래프로 보려는 것 (평가: scripts/eval_hazard_sweep.py).

바탕 2개 (같은 합성을 두 바탕에 똑같이 넣음)
    gray  : 균일 회색 (sRGB 128, 약 43 cd/m²) — 판정기 동작만 분리해서 보기 위한 대조군
    bbb80 : Big Buck Bunny 1080p 80초부터 2초 (시연 영상 9개와 같은 장면), 768×432로 축소, 24→30 fps(프레임 복제)
해상도·길이: 768×432 (판정·학습 해상도 통일, 2026-11-09), 30 fps, 60프레임(2초)

기본값 (스윕하지 않는 조건은 아래 값, 모두 '위험' 쪽)
    휘도  : 가운데 16:9 사각형 면적 40%, 사각파 5 Hz, ΔL +50 cd/m² (선형 빛 더하기)
    적색  : 가운데 사각형 40%, 5 Hz, A 의 R/(R+G+B) = 0.85, Δu′v′ 0.25, 휘도 12 cd/m² (A·B 같음)
    줄무늬: 가운데 사각형 50%, 정지, 3 cpd, 세로 줄(0°), 어두운 줄 10 ↔ 밝은 줄 110 cd/m² (ΔL 100)

설계상 기대(expected)는 ITU-R BT.1702 원문 + 우리 구현 선택(docs/STANDARDS.md)으로 '조건값에서' 계산한다.
판정기 출력을 쓰지 않는다. 줄무늬 cpd 범위는 1~8 (EF 2026), 팀원 판별기는 1~10 이라 9 cpd 에서 다를 수 있음.

사용: python scripts\\make_hazard_sweep.py [--backgrounds gray,bbb80]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from lumi_synth import measure, oracle
from lumi_synth.data import iter_frames_bgr

ROOT = Path(__file__).resolve().parents[1]
BBB = ROOT / "data/sources/big_buck_bunny_1080p/big_buck_bunny_1080p_h264.mov"
W, H, FPS, N = 768, 432, 30.0, 60

LUM = dict(area=0.40, hz=5.0, delta=50.0, wave="square")
RED = dict(area=0.40, hz=5.0, rho=0.85, duv=0.25)
STR = dict(area=0.50, cpd=3.0, angle=0.0, lo=10.0, hi=110.0, motion="static", hz=5.0, speed=4.0)


def background(name: str) -> np.ndarray:
    if name == "gray":
        return np.full((N, H, W, 3), 128, np.uint8)
    if name == "bbb80":
        cap = cv2.VideoCapture(str(BBB))
        src_fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        need = int(np.ceil(N * src_fps / FPS)) + 1
        frames = [cv2.resize(f, (W, H), interpolation=cv2.INTER_AREA)
                  for f in iter_frames_bgr(BBB, int(round(80 * src_fps)), need)]
        return np.stack([frames[int(i * src_fps / FPS)] for i in range(N)])
    raise ValueError(name)


def flashes_per_sec(hz: float) -> float:
    """사각파 hz 를 30 fps 로 그렸을 때 아무 1초 창의 최대 깜빡임 수 (전이 수 ÷ 2)."""
    w = oracle.wave("square", hz, 4 * int(FPS), FPS)
    tr = np.flatnonzero(np.diff(w) != 0) + 1
    win = int(round(FPS))
    return max(((tr >= s) & (tr < s + win)).sum() for s in range(len(w) - win + 1)) / 2.0


# ---- 설계상 기대 ------------------------------------------------------------------------------
def expect_lum(area, hz, delta_cd, dark_cd) -> tuple[bool, str]:
    bad = []
    if not delta_cd >= measure.DELTA_CD:
        bad.append("밝기")
    if not dark_cd < measure.DARK_LIMIT_CD:
        bad.append("어두운 쪽")
    if not area > measure.AREA_LIMIT:
        bad.append("면적")
    if not flashes_per_sec(hz) > measure.MAX_FLASHES_PER_SEC:
        bad.append("횟수")
    return not bad, ", ".join(bad) or "모두 넘음"


def expect_red(area, hz, pair) -> tuple[bool, str]:
    bad = []
    if not (pair["sat_a"] and not pair["sat_b"]):
        bad.append("포화 빨강")
    if not pair["delta_uv"] > measure.RED_DELTA_UV:
        bad.append("Δu′v′")
    if not area > measure.AREA_LIMIT:
        bad.append("면적")
    if not flashes_per_sec(hz) > measure.MAX_FLASHES_PER_SEC:
        bad.append("횟수")
    return not bad, ", ".join(bad) or "모두 넘음"


def expect_stripe(area, cpd, lo, hi, motion, pairs) -> tuple[bool, str]:
    bad = []
    if not (measure.PATTERN_CPD[0] <= cpd <= measure.PATTERN_CPD[1]):
        bad.append("cpd")
    if not (hi - lo >= measure.DELTA_CD and lo < measure.DARK_LIMIT_CD):
        bad.append("명암")
    if not pairs >= measure.PATTERN_MIN_PAIRS:
        bad.append("쌍 수")
    if motion == "drifting":
        bad.append("흐름 제외")
    else:
        limit = measure.PATTERN_AREA_STATIC if motion == "static" else measure.PATTERN_AREA_DYNAMIC
        if not area > limit:
            bad.append("면적")
    return not bad, ", ".join(bad) or "모두 넘음"


# ---- 스윕 목록 ---------------------------------------------------------------------------------
def specs() -> list[dict]:
    out = [{"family": "control", "sweep": "none", "value": 0, "id": "c00_clean"}]
    for v in [5, 10, 15, 18, 22, 25, 35, 60]:
        out.append({"family": "lum", "sweep": "delta_cd", "value": v})
    for v in [0.05, 0.10, 0.15, 0.20, 0.23, 0.27, 0.30, 0.40, 0.60]:
        out.append({"family": "lum", "sweep": "area", "value": v})
    for v in [1, 2, 2.5, 3, 3.5, 4, 5, 7, 10]:
        out.append({"family": "lum", "sweep": "hz", "value": v})
    for v in ["square", "sine", "triangle"]:
        out.append({"family": "lum", "sweep": "wave", "value": v})
    for v in [120, 140, 150, 155, 165, 170]:
        out.append({"family": "lum", "sweep": "dark_cd", "value": v})
    for v in [0.10, 0.15, 0.18, 0.19, 0.21, 0.23, 0.27, 0.32]:
        out.append({"family": "red", "sweep": "delta_uv", "value": v})
    for v in [0.60, 0.70, 0.75, 0.78, 0.82, 0.85, 0.90, 0.95]:
        out.append({"family": "red", "sweep": "sat_ratio", "value": v})
    for v in [0.10, 0.20, 0.23, 0.27, 0.30, 0.40, 0.60]:
        out.append({"family": "red", "sweep": "area", "value": v})
    for v in [1, 2, 3, 3.5, 4, 5, 7]:
        out.append({"family": "red", "sweep": "hz", "value": v})
    for v in [0.5, 0.8, 1.2, 2, 3, 5, 7, 9, 10.5]:
        out.append({"family": "stripe", "sweep": "cpd", "value": v})
    for v in [0.20, 0.30, 0.36, 0.38, 0.42, 0.45, 0.50, 0.60]:
        out.append({"family": "stripe", "sweep": "area_static", "value": v})
    for v in [0.10, 0.18, 0.22, 0.28, 0.32, 0.40, 0.50]:
        out.append({"family": "stripe", "sweep": "area_reversing", "value": v})
    for v in [5, 10, 15, 18, 22, 30, 60, 100]:
        out.append({"family": "stripe", "sweep": "contrast_cd", "value": v})
    for v in [0, 30, 45, 90, 135]:
        out.append({"family": "stripe", "sweep": "angle", "value": v})
    for v in [2, 4, 8]:
        out.append({"family": "stripe", "sweep": "drift_dps", "value": v})
    for i, s in enumerate(out):
        s.setdefault("id", f"{s['family'][0]}_{s['sweep']}_{s['value']}")
    return out


def render(spec: dict, bg: np.ndarray) -> tuple[np.ndarray, np.ndarray | None, dict]:
    """조건 한 줄 → (프레임, 마스크, 기록할 조건·기대)."""
    fam, sw, v = spec["family"], spec["sweep"], spec["value"]
    if fam == "control":
        return bg, None, {"expected": False, "expected_type": "none", "broken": "합성 없음"}
    if fam == "lum":
        p = dict(LUM, dark=None)
        if sw == "delta_cd":
            p["delta"] = v
        elif sw == "area":
            p["area"] = v
        elif sw == "hz":
            p["hz"] = v
        elif sw == "wave":
            p["wave"] = v
        m = oracle.region_mask("rectangle", p["area"], H, W)
        w = oracle.wave(p["wave"], p["hz"], N, FPS)
        if sw == "dark_cd":                                   # 단색 회색 L ↔ L+30 (어두운 쪽 조건)
            lo, hi = float(v), float(v) + 30.0
            frames = oracle.render_flat_flash(bg, m, w, lo, hi)
            lo_m = measure.luminance_cd(np.array([[[oracle.gray_u8(lo)] * 3]], np.uint8))[0, 0]
            hi_m = measure.luminance_cd(np.array([[[oracle.gray_u8(hi)] * 3]], np.uint8))[0, 0]
            exp, why = expect_lum(m.mean(), p["hz"], hi_m - lo_m, lo_m)
            p.update(lo_cd=float(lo_m), hi_cd=float(hi_m), delta=float(hi_m - lo_m))
        else:
            frames = oracle.render_flicker(bg, m, w, p["delta"])
            # 기대: 사인·삼각파는 한 프레임 변화가 작아도 판정기는 극값 사이 변화를 보므로 ΔL 기준은 같다
            exp, why = expect_lum(m.mean(), p["hz"], p["delta"], 0.0)
        p["area_actual"] = float(m.mean())
        return frames, m, {"params": p, "expected": exp, "expected_type": "lum", "broken": why}
    if fam == "red":
        p = dict(RED)
        key = {"delta_uv": "duv", "sat_ratio": "rho", "area": "area", "hz": "hz"}[sw]
        p[key] = v
        pair = oracle.red_pair(p["rho"], p["duv"])
        m = oracle.region_mask("rectangle", p["area"], H, W)
        w = oracle.wave("square", p["hz"], N, FPS)
        frames = oracle.render_red(bg, m, w, pair["a_rgb"], pair["b_rgb"])
        exp, why = expect_red(m.mean(), p["hz"], pair)
        p.update(area_actual=float(m.mean()), pair=pair)
        return frames, m, {"params": p, "expected": exp, "expected_type": "red", "broken": why}
    if fam == "stripe":
        p = dict(STR)
        if sw == "cpd":
            p["cpd"] = v
        elif sw == "area_static":
            p["area"] = v
        elif sw == "area_reversing":
            p.update(area=v, motion="reversing")
        elif sw == "contrast_cd":
            p.update(lo=60.0 - v / 2, hi=60.0 + v / 2)
        elif sw == "angle":
            p["angle"] = v
        elif sw == "drift_dps":
            p.update(motion="drifting", speed=v)
        m = oracle.region_mask("rectangle", p["area"], H, W)
        frames = oracle.render_stripes(bg, m, FPS, p["cpd"], p["angle"], p["lo"], p["hi"], p["motion"],
                                       p["hz"], p["speed"])
        period = oracle.stripe_period_px(W, H, p["cpd"], p["angle"])
        pairs = oracle.stripe_pairs_in_mask(m, period, p["angle"])
        exp, why = expect_stripe(m.mean(), p["cpd"], p["lo"], p["hi"], p["motion"], pairs)
        p.update(area_actual=float(m.mean()), period_px=period, pairs=pairs)
        return frames, m, {"params": p, "expected": exp, "expected_type": "stripe", "broken": why}
    raise ValueError(fam)


def write(path: Path, frames: np.ndarray) -> None:
    wr = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"FFV1"), FPS, (W, H))
    for f in frames:
        wr.write(f)
    wr.release()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--backgrounds", default="gray,bbb80")
    ap.add_argument("--out", type=Path, default=ROOT / "data/hazard_sweep")
    args = ap.parse_args()
    items = []
    for bg_name in args.backgrounds.split(","):
        bg = background(bg_name)
        d = args.out / bg_name
        d.mkdir(parents=True, exist_ok=True)
        for s in specs():
            frames, m, info = render(s, bg)
            path = d / f"{s['id']}.mkv"
            write(path, frames)
            if m is not None:
                cv2.imwrite(str(d / f"{s['id']}_mask.png"), m.astype(np.uint8) * 255)
            items.append({**s, "background": bg_name, "file": str(path.relative_to(args.out)).replace("\\", "/"),
                          **info})
            print(bg_name, s["id"], info["expected"], info["broken"], flush=True)
    meta = {"width": W, "height": H, "fps": FPS, "frames": N,
            "ppd": measure.pixels_per_degree(W, H, 0.0), "defaults": {"lum": LUM, "red": RED, "stripe": STR},
            "items": items}
    (args.out / "manifest.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1, default=float),
                                            encoding="utf-8")
    print(len(items), "개")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
