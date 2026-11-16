"""탐지기의 '클립 판정 규칙'만 바꿔서 정밀도·재현율이 어떻게 변하는지 본다 (모델은 그대로, 다시 학습하지 않음).

    python scripts/analyze_clip_rule.py --tag v7_roi_c --data data/gdr/v7

입력: checks/gdr/<tag>_val_rows.jsonl, <tag>_val_rule_rows.jsonl (eval_roi.py 가 프레임별 예측 면적을 저장한 것)
규칙
    any12.5      지금 규칙: 어느 한 프레임이라도 예측 면적 ≥ 12.5% (종류 공통)
    rule(A)      판정기와 같은 꼴: 예측 면적 > A 인 프레임이 1초 창 안에 기준 시간 이상(줄무늬 0.5초, 휘도·적색 1프레임)
문턱 A 는 같은 G 검증(val)에서만 고르고(F1 최대), 규칙 생성기(val_rule)에는 그 값을 그대로 적용한다 — 시험 데이터로 고르지 않는다.
따로 세는 것
    경계 안전 오탐   판정기가 그 종류 '위험 없음'인데 생성 의도가 그 종류의 '근처 안전'인 표본(예: 면적 20~35% 줄무늬)
    다른 종류 오탐   그 종류를 넣지 않은 표본에서 그 종류로 잡은 것(예: 휘도 깜빡임 → 줄무늬)
    움직임별        줄무늬: 정지·반전·흐름 (규칙 생성기 레시피의 motion, G 의 gan_motion)
결과: checks/gdr/<tag>_clip_rule.md
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from gdr_common import ROOT, read_manifest
from lumi_synth import measure, roi

KO = {"lum": "휘도", "red": "적색", "stripe": "줄무늬"}
MOTION_KO = {"static": "정지", "reversing": "반전", "drifting": "흐름"}
MIN_SEC = {"lum": 0.0, "red": 0.0, "stripe": measure.PATTERN_MIN_SEC}
AREAS = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45)


def decide(rule: str, v: dict, kind: str, fps: float) -> bool:
    if rule == "any12.5":
        return v["pred"]
    area = float(rule.split("(")[1].rstrip(")"))
    return roi.clip_decision_rule(v["pred_series"], fps, area, MIN_SEC[kind])


def motion_of(meta: dict) -> str | None:
    r = meta.get("recipe") or {}
    return r.get("motion") if meta.get("generator") == "rule" else r.get("gan_motion")


def score(rows, metas, rule, kind, fps):
    c = {"tp": 0, "fp": 0, "fn": 0, "tn": 0, "near_fp": 0, "near_n": 0, "cross_fp": 0, "cross_n": 0}
    by_motion: dict = {}
    for r in rows:
        v, m = r[kind], metas[r["id"]]
        p = decide(rule, v, kind, fps)
        c["tp" if v["true"] and p else "fn" if v["true"] else "fp" if p else "tn"] += 1
        if not v["true"] and m["hazard"] == kind and m["intent"] == "near":
            c["near_n"] += 1
            c["near_fp"] += p
        if m["hazard"] not in (None, kind):
            c["cross_n"] += 1
            c["cross_fp"] += p and not v["true"]
        if kind == "stripe" and m["hazard"] == "stripe" and v["true"]:
            mo = motion_of(m) or "?"
            t = by_motion.setdefault(mo, [0, 0])
            t[0] += 1
            t[1] += p
    prec = c["tp"] / (c["tp"] + c["fp"]) if c["tp"] + c["fp"] else float("nan")
    rec = c["tp"] / (c["tp"] + c["fn"]) if c["tp"] + c["fn"] else float("nan")
    f1 = 2 * prec * rec / (prec + rec) if prec == prec and rec == rec and prec + rec else float("nan")
    return {**c, "precision": prec, "recall": rec, "f1": f1, "by_motion": by_motion}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--fps", type=float, default=24.0)
    args = ap.parse_args()
    out = ROOT / "checks" / "gdr"
    sets = {}
    for f in ("val", "val_rule"):
        rows = [json.loads(line) for line in (out / f"{args.tag}_{f}_rows.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
        metas = {m["id"]: m for m in read_manifest(args.data / f)}
        sets[f] = (rows, metas)
    rules = ["any12.5"] + [f"rule({a})" for a in AREAS]
    L = [f"# 클립 판정 규칙 분석 — `{args.tag}`", "",
         "> `scripts/analyze_clip_rule.py` · 모델은 그대로, 판정 규칙만 바꿈. 문턱은 같은 G 검증(val)에서 F1 최대로 고르고 규칙 생성기(val_rule)에 그대로 적용.", ""]
    chosen = {}
    for kind in roi.HAZARDS:
        res = {f: {r: score(*sets[f], r, kind, args.fps) for r in rules} for f in sets}
        best = max(rules[1:], key=lambda r: (res["val"][r]["f1"] if res["val"][r]["f1"] == res["val"][r]["f1"] else -1))
        chosen[kind] = best
        L += [f"## {KO[kind]}", "", f"- val 에서 고른 규칙: **{best}** (최소 지속 {MIN_SEC[kind]}초)", "",
              "| 규칙 | val 정밀도 / 재현율 | val_rule 정밀도 / 재현율 | val_rule 경계 안전 오탐 | val_rule 다른 종류 오탐 |",
              "|---|---|---|---|---|"]
        f = lambda x: "–" if x != x else f"{x:.2f}"     # noqa: E731
        for r in rules:
            a, b = res["val"][r], res["val_rule"][r]
            mark = " ←" if r == best else ""
            L.append(f"| {r}{mark} | {f(a['precision'])} / {f(a['recall'])} | {f(b['precision'])} / {f(b['recall'])} | "
                     f"{b['near_fp']} / {b['near_n']} | {b['cross_fp']} / {b['cross_n']} |")
        if kind == "stripe":
            L += ["", "움직임별 재현율 (val_rule, 찾음 / 위험)", "", "| 규칙 | " + " | ".join(MOTION_KO.values()) + " |", "|---|---|---|---|"]
            for r in ("any12.5", best):
                bm = res["val_rule"][r]["by_motion"]
                L.append(f"| {r} | " + " | ".join(f"{bm.get(m, [0, 0])[1]} / {bm.get(m, [0, 0])[0]}" for m in MOTION_KO) + " |")
            L += ["", "다른 종류 오탐 내역 (val_rule, 고른 규칙)", ""]
            rows, metas = sets["val_rule"]
            from collections import Counter
            cnt = Counter(f"{metas[r['id']]['hazard']}_{metas[r['id']]['intent']}" for r in rows
                          if metas[r["id"]]["hazard"] not in (None, "stripe") and not r["stripe"]["true"]
                          and decide(best, r["stripe"], "stripe", args.fps))
            L += [f"- {k}: {v}" for k, v in sorted(cnt.items())] or ["- 없음"]
        L.append("")
    L += ["- '경계 안전'은 그 종류의 '근처 안전' 의도이면서 판정기도 그 종류 위험 없음인 표본(예: 면적 20~35% 줄무늬, 초당 1~2.5회 깜빡임).",
          "- 판정기 자체의 줄무늬 면적 기준은 정지·반전·흐름 모두 40%(팀원 결정, 원문의 동적 25% 아님) — 이 분석은 탐지기 쪽 클립 판정만 바꾼다."]
    path = out / f"{args.tag}_clip_rule.md"
    path.write_text("\n".join(L) + "\n", encoding="utf-8")
    (out / f"{args.tag}_clip_rule.json").write_text(json.dumps(chosen), encoding="utf-8")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
