"""적색 판정 외부 검증: pse-test-media red_flash_patterns/30fps 의 CSV 208개를 화면 전체 색으로 재현해 판정한다.

설계상 기대 (pse-test-media README, 파일 이름 첫 글자):
  f  위험  — 매번 포화 빨강으로 돌아오고, Δu′v′ > 0.2, 1초에 3번 초과
  c  안전  — 횟수 미달
  s  안전  — 모든 깜빡임이 포화 빨강으로 돌아오지는 않음
  d  안전  — 색 차이 미달(적어도 한 번 Δu′v′ < 0.2)
  r 이 붙은 것(fr, cr, sr, dr)은 같은 설계의 순서 반전판.
CSV 한 줄 = 한 프레임(30 fps). 화면 전체를 그 색으로 칠한 32×18 영상으로 만든다(면적 조건은 항상 충족).
같은 영상을 휘도 판정에도 넣어, 적색 시험 패턴이 휘도 깜빡임으로 잡히지 않는지도 본다.

사용: python scripts\\validate_red_pse.py
"""
import csv
import time
from pathlib import Path

import numpy as np

from lumi_synth import measure

ROOT = Path(__file__).resolve().parents[1]
PSE = ROOT.parent / "tools/pse-test-media/red_flash_patterns/30fps"
OUT = ROOT / "checks/red/README.md"
FPS = 30.0
H, W = 18, 32


def load(path: Path) -> np.ndarray:
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    rgb = np.array([[int(r["r"]), int(r["g"]), int(r["b"])] for r in rows], np.uint8)
    frames = np.empty((len(rgb), H, W, 3), np.uint8)
    frames[:] = rgb[:, None, None, ::-1]                      # BGR
    return frames


def main() -> int:
    files = sorted(PSE.glob("*.csv"))
    rows = []
    t0 = time.time()
    for p in files:
        kind = p.name[0]
        expect_fail = kind == "f"
        v = load(p)
        red = measure.analyze_red(v, FPS)
        lum = measure.analyze(v, FPS)
        rows.append({
            "file": p.name, "kind": kind + ("r" if p.name[1] == "r" else ""),
            "expect": "위험" if expect_fail else "안전",
            "ours": "위험" if red["dangerous"] else "안전",
            "flashes": red["max_flashes"], "duv": red["max_delta_uv"],
            "lum": lum["dangerous"],
        })
    sec = time.time() - t0

    ok = [r for r in rows if r["expect"] == r["ours"]]
    miss = [r for r in rows if r["expect"] == "위험" and r["ours"] == "안전"]
    over = [r for r in rows if r["expect"] == "안전" and r["ours"] == "위험"]
    lum_hits = [r for r in rows if r["lum"]]
    kinds = sorted({r["kind"] for r in rows})

    def cell(k, want):
        sub = [r for r in rows if r["kind"] == k]
        return f"{sum(r['ours'] == want for r in sub)}/{len(sub)}"

    lines = [
        "# 적색 판정 외부 검증: pse-test-media 적색 패턴 208개",
        "",
        f"> {time.strftime('%Y-%m-%d')} · 코드: `src/lumi_synth/measure.py` `analyze_red` · 스크립트: `scripts/validate_red_pse.py`",
        "> 자료: pse-test-media `red_flash_patterns/30fps` (커밋 `ef78542`, BSD-3). CSV 한 줄 = 한 프레임, 화면 전체 색으로 재현(32×18, 30 fps)",
        "> 기준: 포화 빨강 선형 R/(R+G+B) ≥ 0.8, Δu′v′ > 0.2 (WCAG 2.2), 면적 > 25%, 1초 전이 ≥ 7 (ITU-R BT.1702-3)",
        "",
        f"**결과: {len(ok)}/{len(rows)} 일치 · 위험 놓침 {len(miss)} · 과잉 검출 {len(over)} · 휘도 깜빡임으로 잘못 잡힘 {len(lum_hits)}** (판정 {sec:.1f}초)",
        "",
        "| 설계 | 뜻 | 기대 | 개수 | 우리가 위험으로 판정 | 우리가 안전으로 판정 |",
        "|---|---|---|---|---|---|",
    ]
    meaning = {"f": "모두 넘음", "c": "횟수 미달", "s": "포화 빨강으로 안 돌아옴", "d": "색 차이 미달"}
    for k in kinds:
        n = sum(r["kind"] == k for r in rows)
        exp = "위험" if k[0] == "f" else "안전"
        lines.append(f"| {k} | {meaning[k[0]]}{' (순서 반전)' if len(k) > 1 else ''} | {exp} | {n} | {cell(k, '위험')} | {cell(k, '안전')} |")
    if miss or over:
        lines += ["", "## 불일치", "", "| 파일 | 기대 | 우리 | 1초 최대 깜빡임 | 최대 Δu′v′ |", "|---|---|---|---|---|"]
        for r in miss + over:
            lines.append(f"| `{r['file']}` | {r['expect']} | {r['ours']} | {r['flashes']:g} | {r['duv']:.4f} |")
    lines += [
        "",
        "## 주의 (과대평가 요인)",
        "",
        "- 화면 전체를 한 색으로 칠해서 **면적 조건은 항상 충족**됩니다. 이 검증은 '포화 빨강·색 차이·횟수' 판정만 시험합니다.",
        "  면적 조건은 `tests/test_red_flash.py`(20% 안전, 30% 위험)와 시연 영상(`checks/demo_hazards`)에서 따로 봅니다.",
        "- 실제 장면 위의 적색 깜빡임(배경 색이 섞인 경우)은 이 검증에 포함되지 않습니다.",
        "",
    ]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    return 0 if not miss and not over else 1


if __name__ == "__main__":
    raise SystemExit(main())
