"""전달 함수(감마) 민감도: 판정 규칙의 sRGB 가정(A1)을 BT.709/BT.1886 으로 바꾸면 결과가 얼마나 달라지나.

왜 재나
    measure.luminance_cd 는 화면을 sRGB 전달 함수로 가정한다(RULES_SPEC A1). 실제 방송·영화 영상(BT.709 부호화)은
    BT.1886 표시 장치(감마 2.4)에서 보는 것이 기준이다. 원색·휘도 계수(0.2126/0.7152/0.0722)는 sRGB 와 같고
    '코드값 → 선형 빛' 곡선만 다르다. 이 곡선이 정답 라벨을 얼마나 바꾸는지 숫자로 남긴다.

비교하는 전달 함수 (모두 채널별로 선형화 → 같은 휘도 계수 → × 200 cd/m²)
    srgb        : IEC 61966-2-1 (현재 판정기). 0.04045 이하 직선, 그 위 지수 2.4 + 오프셋 → 실효 감마 약 2.2
    bt1886      : ITU-R BT.1886, 검은색 0 cd/m² → L = V^2.4 (순수 지수 2.4)
    bt1886_lb02 : BT.1886, 검은색 0.2 cd/m² (명암비 1000:1 표시 장치)
    bt709_scene : BT.709 OETF 의 역함수 (카메라 쪽 장면 선형광. 표시 휘도가 아니라 참고용)

모드
    --curve   : 회색 코드값별 휘도 표 (데이터 불필요, 수 초)
    --oracle  : data/oracle_<tag> 의 샘플을 전달 함수마다 다시 판정해 라벨 변화를 센다 (수 분)
    --pse     : pse-test-media 외부 검증을 지정 전달 함수로 다시 돌린다 (validate_rules_pse.py 재사용, 약 1.5시간)
결과: checks/rules/gamma_sensitivity.md (curve·oracle), checks/rules/gamma_<이름>.md (pse)

주의: 합성 영상은 메모리에서만 다루고 화면에 띄우지 않는다(광과민성 위험 자극).
"""
from __future__ import annotations

import argparse
import json
import runpy
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from lumi_synth import measure  # noqa: E402

OUT = ROOT / "checks/rules/gamma_sensitivity.md"


# ── 1. 전달 함수: 코드값(0~1) → 상대 선형광(0~1, 흰색 = 1) ───────────────────────────
def _srgb(c):
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _bt1886(lb_cd: float):
    """BT.1886 부록 1 식. Lw = 200 (WHITE_CD), Lb = 검은색 휘도. 결과를 Lw 로 나눠 상대값으로 돌려준다."""
    lw, g = measure.WHITE_CD, 2.4
    a = (lw ** (1 / g) - lb_cd ** (1 / g)) ** g
    b = lb_cd ** (1 / g) / (lw ** (1 / g) - lb_cd ** (1 / g))
    return lambda c: a * np.maximum(c + b, 0.0) ** g / lw


def _bt709_scene(c):
    return np.where(c < 0.081, c / 4.5, ((c + 0.099) / 1.099) ** (1 / 0.45))


EOTF = {"srgb": _srgb, "bt1886": _bt1886(0.0), "bt1886_lb02": _bt1886(0.2), "bt709_scene": _bt709_scene}


def make_luminance(name: str):
    """measure.luminance_cd 와 같은 꼴(uint8 BGR → cd/m² float32)의 함수를 만든다."""
    f = EOTF[name]

    def luminance_cd(frames_bgr: np.ndarray) -> np.ndarray:
        c = frames_bgr.astype(np.float32) / 255.0
        lin = f(c)
        b, g, r = lin[..., 0], lin[..., 1], lin[..., 2]
        return (measure.WHITE_CD * (0.2126 * r + 0.7152 * g + 0.0722 * b)).astype(np.float32)

    return luminance_cd


def check_srgb_identical() -> None:
    """우리가 만든 'srgb' 가 현재 판정기 함수와 비트 단위로 같은지 확인 (비교의 기준점이 맞는지)."""
    x = np.arange(256, dtype=np.uint8).repeat(3).reshape(1, 1, 256, 3)
    assert np.allclose(make_luminance("srgb")(x), measure.luminance_cd(x), atol=1e-4), "srgb 기준점 불일치"


# ── 2. 곡선 표 ─────────────────────────────────────────────────────────────────
def curve_rows():
    codes = np.arange(256)
    lum = {n: measure.WHITE_CD * EOTF[n](codes / 255.0) for n in EOTF}
    rows = []
    for v in (0, 16, 32, 48, 64, 96, 128, 160, 192, 224, 255):
        rows.append([v] + [float(lum[n][v]) for n in EOTF])
    # 160 cd/m² (어두운 쪽 한계)에 해당하는 코드값, 20 cd/m² 를 넘기려면 필요한 회색 단계 수(몇몇 기준 밝기에서)
    limit_code = {n: int(np.argmax(lum[n] >= measure.DARK_LIMIT_CD)) for n in EOTF}
    steps = {}
    for base in (0, 32, 64, 128):
        steps[base] = {n: int(np.argmax(lum[n][base:] - lum[n][base] >= measure.DELTA_CD)) for n in EOTF}
    ratio = {n: float(np.max(np.abs(lum[n] - lum["srgb"]))) for n in EOTF}
    return rows, limit_code, steps, ratio


# ── 3. 오라클 데이터셋 다시 판정 ───────────────────────────────────────────────────
def rejudge_oracle(data_dir: Path, names: list[str]) -> dict:
    manifest = [json.loads(line) for line in (data_dir / "manifest.jsonl").read_text(encoding="utf-8").splitlines()]
    orig = measure.luminance_cd
    res = {n: [] for n in names}
    try:
        for m in manifest:
            z = np.load(data_dir / m["file"])
            for n in names:
                measure.luminance_cd = make_luminance(n)
                r = measure.analyze(z["flicker"], m["fps"])
                res[n].append({"id": m["id"], "split": m["split"], "intent": m["params"]["intent"],
                               "dangerous": bool(r["dangerous"]), "max_flashes": float(r["max_flashes"]),
                               "peak_area": float(r["peak_area"]), "hazard_mask": r["hazard_mask"].copy(),
                               "label_saved": bool(m["label_dangerous"])})
    finally:
        measure.luminance_cd = orig
    return {"manifest_n": len(manifest), "res": res}


def iou(a, b):
    u = np.logical_or(a, b).sum()
    return 1.0 if u == 0 else float(np.logical_and(a, b).sum() / u)


def summarize_oracle(o: dict) -> list[str]:
    base = o["res"]["srgb"]
    assert all(b["dangerous"] == b["label_saved"] for b in base), "srgb 재판정이 저장된 라벨과 다름 — 기준점 오류"
    lines = ["| 전달 함수 | 위험 라벨 수 | 라벨 바뀜 (안전→위험 / 위험→안전) | 의도-판정 일치 | 최대 깜빡임 변화 평균·최대 (회/초) | 최대 면적 변화 평균·최대 | 위험 마스크 IoU 평균(둘 다 위험) |",
             "|---|---|---|---|---|---|---|"]
    flips_detail = {}
    for n, rs in o["res"].items():
        up = [r for r, b in zip(rs, base) if r["dangerous"] and not b["dangerous"]]
        dn = [r for r, b in zip(rs, base) if b["dangerous"] and not r["dangerous"]]
        ok = sum((r["intent"] == "hazard") == r["dangerous"] for r in rs)
        df = np.array([abs(r["max_flashes"] - b["max_flashes"]) for r, b in zip(rs, base)])
        da = np.array([abs(r["peak_area"] - b["peak_area"]) for r, b in zip(rs, base)])
        ious = [iou(r["hazard_mask"], b["hazard_mask"]) for r, b in zip(rs, base) if r["dangerous"] and b["dangerous"]]
        lines.append(f"| `{n}` | {sum(r['dangerous'] for r in rs)}/{len(rs)} | {len(up)} / {len(dn)} | {ok}/{len(rs)} | "
                     f"{df.mean():.3f} · {df.max():.1f} | {da.mean():.4f} · {da.max():.3f} | "
                     f"{(np.mean(ious) if ious else float('nan')):.3f} |")
        flips_detail[n] = [(r["id"], r["intent"], b["max_flashes"], r["max_flashes"], b["peak_area"], r["peak_area"])
                           for r, b in zip(rs, base) if r["dangerous"] != b["dangerous"]]
    lines.append("")
    for n, fl in flips_detail.items():
        if n == "srgb" or not fl:
            continue
        lines.append(f"`{n}` 에서 라벨이 바뀐 샘플 (id, 의도, 깜빡임 sRGB→{n}, 면적 sRGB→{n}):")
        for f in fl:
            lines.append(f"- `{f[0]}` {f[1]}: {f[2]:.1f}→{f[3]:.1f} 회/초, 면적 {f[4]:.3f}→{f[5]:.3f}")
        lines.append("")
    by_intent = ["| 전달 함수 | " + " | ".join(f"{i} 위험 수" for i in ("hazard", "safe_area", "safe_count", "safe_lum")) + " |",
                 "|---|---|---|---|---|"]
    for n, rs in o["res"].items():
        cnt = [f"{sum(r['dangerous'] for r in rs if r['intent'] == i)}/{sum(r['intent'] == i for r in rs)}"
               for i in ("hazard", "safe_area", "safe_count", "safe_lum")]
        by_intent.append(f"| `{n}` | " + " | ".join(cnt) + " |")
    return lines + ["의도별 위험 판정 수:", ""] + by_intent


def write_report(curve, oracle_lines, data_dir, runtime):
    rows, limit_code, steps, maxdiff = curve
    names = list(EOTF)
    t = ["# 전달 함수(감마) 민감도 — sRGB(현재) vs BT.709/BT.1886", "",
         "> 스크립트: `scripts/gamma_sensitivity.py`. 판정 규칙(`measure.analyze`)의 다른 부분은 그대로 두고 "
         "'코드값 → 휘도' 함수만 바꿨다. 휘도 계수와 흰색 200 cd/m² 는 같다.", "",
         "## 1. 회색 코드값별 휘도 (cd/m²)", "",
         "| 코드값 | " + " | ".join(f"`{n}`" for n in names) + " |", "|---|" + "---|" * len(names)]
    for r in rows:
        t.append(f"| {r[0]} | " + " | ".join(f"{x:.2f}" for x in r[1:]) + " |")
    t += ["", "| 항목 | " + " | ".join(f"`{n}`" for n in names) + " |", "|---|" + "---|" * len(names),
          "| 160 cd/m²(어두운 쪽 한계)가 되는 회색 코드값 | " + " | ".join(str(limit_code[n]) for n in names) + " |"]
    for base, d in steps.items():
        t.append(f"| 회색 {base} 에서 ΔL ≥ 20 이 되려면 올려야 하는 코드값 | " + " | ".join(str(d[n]) for n in names) + " |")
    t.append("| sRGB 와의 최대 휘도 차이 (cd/m²) | " + " | ".join(f"{maxdiff[n]:.2f}" for n in names) + " |")
    if oracle_lines:
        t += ["", f"## 2. 오라클 데이터셋 다시 판정 (`{data_dir.relative_to(ROOT).as_posix()}`, 합성은 sRGB 로 한 그대로)", "",
              "합성(빛 더하기)은 sRGB 선형 공간에서 했으므로, 이것은 '같은 영상을 다른 표시 가정으로 판정하면 라벨이 바뀌는가'를 잰다.", ""]
        t += oracle_lines
    t += ["", f"실행 시간: {runtime:.0f}초 (CPU)"]
    OUT.write_text("\n".join(t) + "\n", encoding="utf-8")
    print(f"보고서: {OUT}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=ROOT / "data/oracle_128x72_24fps")
    ap.add_argument("--skip-oracle", action="store_true")
    ap.add_argument("--pse", choices=[n for n in EOTF if n != "srgb"], default=None,
                    help="pse-test-media 외부 검증을 이 전달 함수로 다시 돌린다 (나머지 인자는 -- 뒤에 validate_rules_pse.py 인자로)")
    args, rest = ap.parse_known_args()
    check_srgb_identical()

    if args.pse:
        # validate_rules_pse.py 는 import 시점의 measure.luminance_cd 를 '원래 함수'로 잡으므로 먼저 바꿔 둔다.
        measure.luminance_cd = make_luminance(args.pse)
        rest = [a for a in rest if a != "--"]
        if "--out" not in rest:
            rest += ["--out", str(ROOT / f"checks/rules/gamma_{args.pse}.md")]
        sys.argv = [str(ROOT / "scripts/validate_rules_pse.py")] + rest
        runpy.run_path(sys.argv[0], run_name="__main__")
        return

    t0 = time.time()
    curve = curve_rows()
    oracle_lines = None
    if not args.skip_oracle:
        o = rejudge_oracle(args.data, list(EOTF))
        oracle_lines = summarize_oracle(o)
    write_report(curve, oracle_lines, args.data, time.time() - t0)


if __name__ == "__main__":
    main()
