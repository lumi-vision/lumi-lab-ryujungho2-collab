"""data/gdr/<이름>/ 의 데이터 구분표를 만든다: 폴더마다 무엇으로·어느 바탕에서·무엇에 쓰려고 만들었고, 판정기가 뭐라 했는가.

    python scripts/gdr_dataset_card.py --name v1

쓰는 곳: data/gdr/<이름>/DATASET.md (데이터 옆, Git 제외) 와 checks/gdr/dataset_<이름>.md (같은 내용, Git 에 올림).

폴더 규칙 (make_gdr_dataset.py)
    train/       G 로 만든 것, train 바탕(영화 60클립)  → 탐지기·보정기 학습에만 쓴다
    val/         G 로 만든 것, val 바탕(다른 영화 12클립) → 평가에만 쓴다
    val_rule/    규칙 생성기로 만든 것, val 바탕        → 평가에만 쓴다(다른 생성기로 일반화 확인). 학습 금지
    test 바탕·South 2021 위험 163개는 이 폴더들에 없다(최종 시험용).
"""
from __future__ import annotations

import argparse
import collections
import time
from pathlib import Path

from gdr_common import ROOT, read_manifest

PURPOSE = {"train": "학습 전용", "val": "평가 전용", "val_rule": "평가 전용 (다른 생성기, 학습 금지)"}
KO = {"lum": "휘도", "red": "적색", "stripe": "줄무늬", None: "바탕 그대로"}


def source_of(m: dict) -> str:
    if m["hazard"] is None:
        return "없음 (바탕 그대로)"
    if m["generator"] == "gan":
        return f"G `{m['recipe'].get('checkpoint')}` (sha256 {m['recipe'].get('checkpoint_sha256', '')[:12]})"
    return f"규칙 생성기 `{m['recipe'].get('kind')}`"


def folder_section(folder: Path) -> tuple[list[str], list[str]]:
    metas = read_manifest(folder)
    problems = []
    splits = {m["split"] for m in metas}
    gens = {m["generator"] for m in metas if m["hazard"]}
    want_split = folder.name.split("_")[0]
    if splits != {want_split}:
        problems.append(f"{folder.name}: 바탕 분할이 섞임 {sorted(splits)}")
    if folder.name.endswith("_rule") and gens - {"rule"} or not folder.name.endswith("_rule") and gens - {"gan"}:
        problems.append(f"{folder.name}: 생성기가 폴더 규칙과 다름 {sorted(gens)}")
    outside = sum(m.get("outside_changed", 0) for m in metas)
    if outside:
        problems.append(f"{folder.name}: 마스크 밖 변화 {outside}")
    dirty = [m["id"] for m in metas if m["hazard"] is None and m["judge"]["dangerous"]]
    if dirty:
        problems.append(f"{folder.name}: 바탕 그대로인데 판정기가 위험 — {dirty} (바탕 자체가 위험)")

    L = [f"## `{folder.name}/` — {PURPOSE.get(folder.name, '?')}", "",
         f"- 바탕: {want_split} 분할 {len({m['bg'] for m in metas})}클립 · 표본 {len(metas)}개",
         "- 생성원: " + " · ".join(sorted({source_of(m) for m in metas})), "",
         "| 넣은 종류 | 생성 의도 | 표본 | 판정 휘도 위험 | 판정 적색 위험 | 판정 줄무늬 위험 | 판정 위험 없음 |",
         "|---|---|---|---|---|---|---|"]
    by = collections.defaultdict(list)
    for m in metas:
        by[(m["hazard"] or "", m["intent"] or "")].append(m)
    for (h, it), ms in sorted(by.items()):
        cnt = lambda k: sum(bool(x["judge"].get(k) and x["judge"][k]["dangerous"]) for x in ms)   # noqa: E731
        L.append(f"| {KO[h or None]} | {({'hazard': '위험', 'near': '근처 안전'}).get(it, '–')} | {len(ms)} | "
                 f"{cnt('lum')} | {cnt('red')} | {cnt('stripe')} | {sum(not x['judge']['dangerous'] for x in ms)} |")
    tot = {k: sum(bool(x["judge"].get(k) and x["judge"][k]["dangerous"]) for x in metas) for k in ("lum", "red", "stripe")}
    L += ["", f"- 판정기 기준 위험 표본(= 튜브가 있는 표본): 휘도 {tot['lum']} · 적색 {tot['red']} · 줄무늬 {tot['stripe']}", ""]
    return L, problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", default="v1")
    args = ap.parse_args()
    base = ROOT / "data" / "gdr" / args.name
    folders = [p for p in sorted(base.iterdir()) if (p / "manifest.jsonl").exists()]
    L = [f"# 탐지·보정 검증 데이터 `{args.name}` — 구분표", "",
         f"> {time.strftime('%Y-%m-%d %H:%M')} · `scripts/gdr_dataset_card.py` · 만든 스크립트 `scripts/make_gdr_dataset.py` · 설계 [docs/GDR_VERIFY.md](../../docs/GDR_VERIFY.md)", "",
         "- **정답(튜브)은 하드 판정기가 붙였다.** '생성 의도'는 어떤 시도였는지일 뿐 정답이 아니다.",
         "- 원본(clean)은 `data/oracle_768x432_24fps/backgrounds_<분할>.npz` 의 bg 번째 클립이다(표본 파일에 넣지 않음).",
         "- 탐지기·보정기 학습 스크립트는 train 분할·G 생성 표본만 받고, 평가 스크립트는 train 분할을 거부한다.", ""]
    problems = []
    for f in folders:
        sec, pr = folder_section(f)
        L += sec
        problems += pr
    L += ["## 점검", ""] + ([f"- ⚠ {p}" for p in problems] or ["- 분할·생성기 섞임 없음, 마스크 밖 변화 0, 바탕 그대로인 표본은 모두 위험 없음"])
    text = "\n".join(L) + "\n"
    (base / "DATASET.md").write_text(text, encoding="utf-8")
    out = ROOT / "checks" / "gdr" / f"dataset_{args.name}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(text)
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
