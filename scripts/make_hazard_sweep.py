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

설계상 기대는 '조건값에서' 계산한다(판정기 출력을 쓰지 않는다). 2026-11-12 부터 기준이 둘이다.
    expected       팀원 보고서 기준 (stage3 정량 기준 명세서 = stage4 vdcore 가 구현한 것) — 기본 정답
    expected_ours  우리 판정기 기준 (ITU-R BT.1702-3 원문 + 우리 구현 선택, docs/STANDARDS.md)
두 기준이 갈리는 곳 (나머지는 같다)
    줄무늬 반전   팀원은 움직이는 패턴이 범위 밖이라 프레임마다 정지 패턴 규칙(> 40%)만 적용. 우리는 > 25%
    줄무늬 흐름   팀원은 흐름 제외가 없어 > 40% 면 위험. 우리는 한 방향 흐름을 제외(ITU 원문)
    적색 하한     팀원은 포화 빨강에 선형 R−G−B ≥ 0.0625(#470000 이상)도 요구. 우리는 하한 없음
    쌍 수         팀원 ≥ 6쌍(ITU 원문 '5쌍 초과'), 우리 ≥ 5쌍 — 이 스윕은 모두 7쌍 이상이라 갈리지 않음
    파형          팀원은 66.8 ms(30 fps 에서 2프레임) 안의 변화가 20 cd/m² 이상이어야 전이. 우리는 극값 사이 변화
팀원 기준에서 이 스윕이 다루지 않는 것: 빠른 장면 전환(면적 ≥ 80%, 최대 면적이 60%), 연장 점멸(5초 창, 클립이 2초).

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


# ---- 설계상 기대 ------------------------------------------------------------------------------
# 팀원 기준값 (stage3_pse_quantitative_criteria: §1.2 방법 A, §1.4 시간창, §2.1 휘도, §2.3 적색, §2.5 정적 패턴)
TEAM = dict(delta_cd=20.0, dark_cd=160.0, area=0.25, min_transitions=7, window_ms=1000.0, trans_ms=66.8,
            safe_ms=333.3, safe_ms_50hz=359.9, red_ratio=0.8, red_excess=0.0625, red_duv=0.2,
            pat_area=0.40, pat_cpd=(1.0, 10.0), pat_pairs=6, pat_sec=0.5)


def team_events(n: int, fps: float, step) -> list[int]:
    """팀원 교대 상태 기계(stage3 §2.1 C)를 영역의 대표값 하나에 적용 → 확정 전이 프레임 목록.

    step(ref, cur) 는 프레임 ref → cur 의 변화가 유효 전이면 +1/−1, 아니면 0. 기준 프레임은 직전 확정 전이 이후,
    66.8 ms 이내(30 fps 에서 2프레임 전까지). 직전과 같은 방향이면 횟수를 늘리지 않고 기준만 옮긴다.
    """
    k = max(1, int(TEAM["trans_ms"] / 1000.0 * fps + 1e-6))
    events, last, start = [], 0, 0
    for cur in range(1, n):
        pol = next((q for q in (step(cur - d, cur) for d in range(1, k + 1) if cur - d >= start) if q), 0)
        if pol:
            if pol != last:
                events.append(cur)
                last = pol
            start = cur
    return events


def team_flash(events: list[int], fps: float) -> bool:
    """전이 7개가 1000 ms 안에 있고, 그 안의 같은 방향 전이 간격이 모두 안전 간격 미만이면 플래시."""
    safe = TEAM["safe_ms_50hz"] if min(abs(fps - 25.0), abs(fps - 50.0)) < 0.5 else TEAM["safe_ms"]
    t = [e * 1000.0 / fps for e in events]
    m = TEAM["min_transitions"]
    return any(t[i + m - 1] - t[i] <= TEAM["window_ms"] and all(t[j + 2] - t[j] < safe for j in range(i, i + m - 2))
               for i in range(len(t) - m + 1))


def max_flashes(on: np.ndarray) -> float:
    """켜짐/꺼짐 열에서 아무 1초 창의 최대 깜빡임 수 (전이 수 ÷ 2) — 우리 기준."""
    tr = np.flatnonzero(np.diff(on.astype(np.int8)) != 0) + 1
    win = int(round(FPS))
    return max(((tr >= s) & (tr < s + win)).sum() for s in range(max(1, len(on) - win + 1))) / 2.0


def _both(team: list, ours: list) -> dict:
    return {"expected": not team, "broken": ", ".join(team) or "모두 넘음",
            "expected_ours": not ours, "broken_ours": ", ".join(ours) or "모두 넘음"}


def expect_lum(area, levels_cd, dark_cd) -> dict:
    """levels_cd = 영역의 프레임별 휘도(바탕에 더한 값이어도 됨), dark_cd = 가장 어두울 때의 절대 휘도."""
    levels_cd = np.asarray(levels_cd, np.float64)
    ours, team = [], []
    if not float(np.ptp(levels_cd)) >= measure.DELTA_CD:
        ours.append("밝기")
    if not dark_cd < measure.DARK_LIMIT_CD:
        ours.append("어두운 쪽")
        team.append("어두운 쪽")
    if not area > measure.AREA_LIMIT:
        ours.append("면적")
        team.append("면적")
    on = levels_cd >= (levels_cd.min() + levels_cd.max()) / 2
    if not max_flashes(on) > measure.MAX_FLASHES_PER_SEC:
        ours.append("횟수")

    def step(a, b):
        d = levels_cd[b] - levels_cd[a]
        return int(np.sign(d)) if abs(d) >= TEAM["delta_cd"] else 0

    ev = team_events(len(levels_cd), FPS, step)
    if not ev:
        team.append("밝기")
    elif not team_flash(ev, FPS):
        team.append("횟수")
    return _both(team, ours)


def expect_red(area, w, pair) -> dict:
    ours, team = [], []
    if not (pair["sat_a"] and not pair["sat_b"]):
        ours.append("포화 빨강")
    if not any(pair[f"sat_{k}"] and pair[f"excess_{k}"] >= TEAM["red_excess"] for k in "ab"):
        team.append("포화 빨강(하한 포함)")
    if not pair["delta_uv"] > measure.RED_DELTA_UV:
        ours.append("Δu′v′")
        team.append("Δu′v′")
    if not area > measure.AREA_LIMIT:
        ours.append("면적")
        team.append("면적")
    on = np.asarray(w) >= 0.5
    if not max_flashes(on) > measure.MAX_FLASHES_PER_SEC:
        ours.append("횟수")
    if not team_flash(team_events(len(on), FPS, lambda a, b: int(on[b]) - int(on[a])), FPS):
        team.append("횟수")
    return _both(team, ours)


def expect_stripe(area, cpd, lo, hi, motion, pairs) -> dict:
    ours, team = [], []
    if not (measure.PATTERN_CPD[0] <= cpd <= measure.PATTERN_CPD[1]):
        ours.append("cpd")
    if not (TEAM["pat_cpd"][0] <= cpd <= TEAM["pat_cpd"][1]):
        team.append("cpd")
    if not (hi - lo >= measure.DELTA_CD and lo < measure.DARK_LIMIT_CD):
        ours.append("명암")
        team.append("명암")
    if not pairs >= measure.PATTERN_MIN_PAIRS:
        ours.append("쌍 수")
    if not pairs >= TEAM["pat_pairs"]:
        team.append("쌍 수")
    if motion == "drifting":
        ours.append("흐름 제외")
    elif not area > (measure.PATTERN_AREA_STATIC if motion == "static" else measure.PATTERN_AREA_DYNAMIC):
        ours.append("면적")
    if not area > TEAM["pat_area"]:                     # 팀원: 움직임과 상관없이 프레임마다 정지 패턴 규칙
        team.append("면적")
    return _both(team, ours)


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
    for v in [40, 60, 68, 70, 71, 74, 90, 128]:      # 순수 빨강 (R,0,0) ↔ 검정. 팀원 하한 #470000 = R 71
        out.append({"family": "red", "sweep": "floor_r8", "value": v})
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
        return bg, None, {"expected": False, "expected_type": "none", "broken": "합성 없음",
                          "expected_ours": False, "broken_ours": "합성 없음"}
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
            exp = expect_lum(m.mean(), lo_m + (hi_m - lo_m) * w, lo_m)
            p.update(lo_cd=float(lo_m), hi_cd=float(hi_m), delta=float(hi_m - lo_m))
        else:
            frames = oracle.render_flicker(bg, m, w, p["delta"])
            # 기대: 바탕에 더한 빛 δ·w(t) 로 계산 (어두운 쪽은 바탕이 160 미만이라고 봄).
            # 사인·삼각파: 우리는 극값 사이 변화, 팀원은 66.8 ms 안의 변화로 본다 (expect_lum)
            exp = expect_lum(m.mean(), p["delta"] * w, 0.0)
        p["area_actual"] = float(m.mean())
        return frames, m, {"params": p, "expected_type": "lum", **exp}
    if fam == "red":
        p = dict(RED)
        if sw == "floor_r8":                                  # 거의 검정인 빨강: 팀원 하한(선형 R−G−B ≥ 0.0625)
            pair = oracle.red_pair_from_colors((int(v), 0, 0), (0, 0, 0))
        else:
            p[{"delta_uv": "duv", "sat_ratio": "rho", "area": "area", "hz": "hz"}[sw]] = v
            pair = oracle.red_pair(p["rho"], p["duv"])
        m = oracle.region_mask("rectangle", p["area"], H, W)
        w = oracle.wave("square", p["hz"], N, FPS)
        frames = oracle.render_red(bg, m, w, pair["a_rgb"], pair["b_rgb"])
        exp = expect_red(m.mean(), w, pair)
        p.update(area_actual=float(m.mean()), pair=pair)
        return frames, m, {"params": p, "expected_type": "red", **exp}
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
        exp = expect_stripe(m.mean(), p["cpd"], p["lo"], p["hi"], p["motion"], pairs)
        p.update(area_actual=float(m.mean()), period_px=period, pairs=pairs)
        return frames, m, {"params": p, "expected_type": "stripe", **exp}
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
            print(bg_name, s["id"], "팀원", info["expected"], info["broken"], "| 우리", info["expected_ours"],
                  info["broken_ours"], flush=True)
    meta = {"width": W, "height": H, "fps": FPS, "frames": N,
            "ppd": measure.pixels_per_degree(W, H, 0.0), "defaults": {"lum": LUM, "red": RED, "stripe": STR},
            "criteria": {"expected": "팀원 stage3 (vdcore)", "expected_ours": "ITU-R BT.1702-3 + docs/STANDARDS.md",
                         "team": TEAM},
            "items": items}
    (args.out / "manifest.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1, default=float),
                                            encoding="utf-8")
    print(len(items), "개")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
