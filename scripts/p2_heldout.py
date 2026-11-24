"""P2·흐름 면제 — 조정에 쓰지 않은 big1 val/test 합성 패턴 위험으로 확인 (2026-11-25, 실행 전 커밋).

배경: P2 규칙(점멸 2 회 이상 넘나듦·방향 전환 구간 2 프레임·대비 반전 2 프레임 이상)은 big1 **train** 진단 세트 결과를 보고 고쳤다.
같은 세트로 다시 재면 맞춘 것처럼 보일 수 있으므로, 손대지 않은 val/test 로 따로 본다.

정답(생성 레시피 기준, 실행 전에 고정)
  - pat_drift 없음(정지 줄무늬)                                       → '패턴 위험' (생성 의도 = 위험)
  - pat_drift 있음, 속도 상한 |drift|·cpd/ppd < 0.375 주기/프레임       → '면제(안전)' — 한 방향으로 매끄럽게 흐름(원문 면제)
  - pat_drift 있음, 속도 상한 ≥ 0.375                                  → 채점 제외(법선 방향 성분을 레시피로 알 수 없고,
                                                                          0.5 주기/프레임 이상은 앨리어싱으로 방향이 모호 — 해석 문제)
  0.375 = 반전 위상 문턱 ¾π 를 주기로 바꾼 값(¾π / 2π).
비교: G(P1 만, 흐름 면제 없음) vs G2(P1 흐름 면제 + P2).  출력: checks/judge/p2_heldout.md
"""
from __future__ import annotations

import glob
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from lumi_judge.cpu_limit import limit_cpu  # noqa: E402
limit_cpu()
import torch  # noqa: E402
from grid_pattern_eval import judge_clip  # noqa: E402
from lumi_judge.grid_motion import judge_pattern_motion  # noqa: E402
from lumi_judge.grid_pattern import GridPatternJudge, pixels_per_degree  # noqa: E402
from make_hazard7_dataset import FPS, decode  # noqa: E402

SPEED_EXEMPT = 0.375


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    rows = []
    Gs = {}
    for split in ("val", "test"):
        rec = {}
        for f in glob.glob(str(ROOT / f"data/hazard7/big1/{split}/manifest.s*.jsonl")):
            for l in open(f, encoding="utf-8"):
                r = json.loads(l); rc = r.get("recipe", r)
                rec[str(r.get("id") or r.get("name") or r.get("file", "")).replace(".mp4", "")] = rc
        for cid, rc in sorted(rec.items()):
            if rc.get("kind") != "pattern" or rc.get("intent") != "hazard":
                continue
            p = ROOT / f"data/hazard7/big1/{split}/{cid}.mp4"
            if not p.exists():
                continue
            x = decode(p)
            h, w = x.shape[1:3]
            ppd = pixels_per_degree(w, h)
            sp = math.hypot(*rc.get("pat_drift", [0.0, 0.0])) * rc["cpd"] / ppd
            truth = True if sp == 0 else (False if sp < SPEED_EXEMPT else None)
            if (h, w) not in Gs:
                Gs[(h, w)] = GridPatternJudge(h, w)
            g1 = judge_clip(x, FPS)["danger"]
            g2 = judge_pattern_motion(x, FPS, G=Gs[(h, w)])
            b = g2["best2"] or g2["best1"] or g2["best_any"]
            rows.append({"id": f"{split}/{cid}", "pattern": rc.get("pattern"), "mask": rc.get("mask_mode"), "speed_ppf": round(sp, 3), "truth": truth,
                         "G": g1, "G2": g2["p1"] or g2["p2"], "p1": g2["p1"], "p2": g2["p2"],
                         "why": None if b is None else {k: (round(v, 3) if isinstance(v, float) else v) for k, v in b.items()
                                                        if k in ("area_static", "area_change", "area_flow", "reason")}})
            print(rows[-1], flush=True)
            torch.cuda.empty_cache()
    S = [r for r in rows if r["truth"] is not None]
    acc = lambda k: sum(r[k] == r["truth"] for r in S)
    yn = lambda b: "-" if b is None else ("위험" if b else "안전")
    L = ["# P2·흐름 면제 — 조정에 쓰지 않은 big1 val/test 확인 (정답 규칙은 scripts/p2_heldout.py 머리말, 실행 전 커밋)", "",
         f"> 채점 {len(S)} (정지 {sum(r['truth'] for r in S)} · 느린 흐름 {sum(not r['truth'] for r in S)}) · 제외 {len(rows) - len(S)} · "
         f"정답 수 **G2 {acc('G2')}/{len(S)}** · G(P1 만) {acc('G')}/{len(S)}", "",
         "| 클립 | 무늬 | 마스크 | 속도 상한(주기/프레임) | 정답 | G | G2 | G2 근거 |", "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        mk = lambda k: yn(r[k]) + ("" if r["truth"] is None or r[k] == r["truth"] else " ✗")
        L.append(f"| {r['id']} | {r['pattern']} | {r['mask']} | {r['speed_ppf']} | {yn(r['truth'])} | {mk('G')} | {mk('G2')} | {r['why']} |")
    (ROOT / "checks" / "judge" / "p2_heldout.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L[:3]))


if __name__ == "__main__":
    main()
