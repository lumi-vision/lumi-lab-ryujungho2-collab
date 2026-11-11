"""pse-test-media 휘도 세트(210개)에서 Jordan & Vanderheiden 2024 시간창 조항을 켠 판정과 기본 판정을 비교한다.

⚠️ 프레임은 메모리에서만 만들고 화면에 띄우거나 파일로 저장하지 않는다(validate_rules_pse.py 와 같음).

설정 6가지 (영상마다 휘도는 한 번만 계산)
  base          현재 판정기 (ITU-R BT.1702-3 구현, D1~D4)
  q_strict      + 단일 전이 허용 시간 66 ms, 30 fps 에서 1프레임
  q_lenient     + 단일 전이 허용 시간 66 ms, 30 fps 에서 2프레임(66.7 ms)
  sync20        + 영역 동기화 창 20 ms (30 fps 에서 같은 프레임만)
  jv24_strict   q_strict + sync20 + 빠른 점멸 병합 15 ms
  jv24_lenient  q_lenient + sync20 + 빠른 점멸 병합 15 ms
사용: python scripts/validate_jv24.py --pse-root <pse-test-media> [--workers 6]
결과: checks/rules/jv24.md (+ 영상별 원시 결과는 --save-results 로 저장소 밖에)
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from lumi_synth import measure  # noqa: E402

_spec = importlib.util.spec_from_file_location("vrp", ROOT / "scripts" / "validate_rules_pse.py")
vrp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(vrp)

VARIANTS = ["base", "q_strict", "q_lenient", "sync20", "jv24_strict", "jv24_lenient"]
LABEL = {"base": "현재 판정기", "q_strict": "+ 전이 시간 66 ms (1프레임)", "q_lenient": "+ 전이 시간 66 ms (2프레임)",
         "sync20": "+ 동기화 창 20 ms", "jv24_strict": "3조항 모두 (66 ms = 1프레임)",
         "jv24_lenient": "3조항 모두 (66 ms = 2프레임)"}


def one(job):
    json_path, padding, set_name = job
    spec = json.loads(Path(json_path).read_text())
    frames, fps = vrp.build_frames(Path(json_path), padding)
    lum = vrp._chunked_luminance(frames)
    del frames
    base_s = measure.pixel_transitions(lum)
    k1 = measure.pixel_transitions_windowed(lum, measure.qualifying_frames(fps, "strict"))
    k2 = measure.pixel_transitions_windowed(lum, measure.qualifying_frames(fps, "lenient"))
    c20 = measure.sync_frames(fps)
    res = {
        "base": measure.flash_from_transitions(base_s, fps),
        "q_strict": measure.flash_from_transitions(k1, fps),
        "q_lenient": measure.flash_from_transitions(k2, fps),
        "sync20": measure.flash_from_transitions(base_s, fps, c=c20),
        "jv24_strict": measure.flash_from_transitions(k1, fps, c=c20, fast_ms=measure.JV24_FAST_MS),
        "jv24_lenient": measure.flash_from_transitions(k2, fps, c=c20, fast_ms=measure.JV24_FAST_MS),
    }
    exp_raw = spec.get("expected_result", {}).get("itu_r1702_4")
    return {"set": set_name, "name": Path(json_path).stem, "fps": fps, "frames": int(len(lum)),
            "expected": vrp.norm_expected(exp_raw), "expected_raw": exp_raw,
            **{f"{v}": ("fail" if r["dangerous"] else "pass") for v, r in res.items()},
            **{f"{v}_flashes": float(r["max_flashes"]) for v, r in res.items()}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pse-root", required=True, type=Path)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--save-results", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=ROOT / "checks/rules/jv24.md")
    args = ap.parse_args()
    pse = args.pse_root.resolve(); vc = pse / "video_creation"
    padding = json.loads((pse / "video_config.json").read_text()).get("padding", 0)
    sets = [d.name for d in sorted(vc.iterdir()) if d.is_dir() and not vrp.set_has_color(d)]
    jobs = [(str(jp), padding, s) for s in sets for jp in sorted((vc / s).glob("*.json"))]
    print(f"세트 {len(sets)}개, 영상 {len(jobs)}개, 프로세스 {args.workers}", flush=True)
    t0 = time.time(); rows = []
    with ProcessPoolExecutor(args.workers) as ex:
        for i, r in enumerate(ex.map(one, jobs), 1):
            rows.append(r)
            if i % 10 == 0:
                print(f"  {i}/{len(jobs)} ({time.time() - t0:.0f}초)", flush=True)
    runtime = time.time() - t0
    if args.save_results:
        args.save_results.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")

    def cm(v):
        return {k: sum(1 for r in rows if (r["expected"], r[v]) == k)
                for k in [("fail", "fail"), ("fail", "pass"), ("pass", "fail"), ("pass", "pass")]}
    lines = ["# 시간창 3조항(Jordan & Vanderheiden 2024) 적용 비교 — pse-test-media 휘도 세트", "",
             f"> 실행 {time.strftime('%Y-%m-%d %H:%M')} · 영상 {len(rows)}개 · 세트 {len(sets)}개 · {runtime / 60:.0f}분 · "
             f"pse-test-media `{vrp.git_head(pse)}` · 스크립트 `scripts/validate_jv24.py`",
             "> 조항 원문: 단일 전이 허용 시간 ≤ 66 ms, 영역 동기화 창 ≤ 20 ms, 빠른 점멸 병합 ≤ 15 ms (pseGuidelines, 2024, 'working value').",
             "> 30 fps 영상에서 빠른 점멸 병합은 원리상 바뀌는 것이 없다(전이 간격 ≥ 33 ms). 기대값 = JSON `itu_r1702_4`.", "",
             "| 설정 | 일치 | 놓침 (기대 위험 → 우리 안전) | 과잉 (기대 안전 → 우리 위험) |", "|---|---|---|---|"]
    for v in VARIANTS:
        c = cm(v); agree = c[("fail", "fail")] + c[("pass", "pass")]
        lines.append(f"| {LABEL[v]} | **{agree}/{len(rows)}** | {c[('fail', 'pass')]} | {c[('pass', 'fail')]} |")
    lines += ["", "## 기본 판정과 결과가 달라진 영상", "",
              "| 세트 / 영상 | 기대 | " + " | ".join(LABEL[v] for v in VARIANTS) + " | 기존 원인 메모 |",
              "|---|---|" + "---|" * len(VARIANTS) + "---|"]
    for r in sorted(rows, key=lambda r: (r["set"], r["name"])):
        if len({r[v] for v in VARIANTS}) > 1 or r["base"] != r["expected"]:
            note = vrp.NOTES.get((r["set"], r["name"]), ("", ""))[0]
            cells = " | ".join(f"{r[v]} ({r[v + '_flashes']:g})" for v in VARIANTS)
            lines.append(f"| {r['set']} / {r['name']} | {r['expected']} | {cells} | {note} |")
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[8:8 + len(VARIANTS) + 2]))
    print("보고서:", args.out)


if __name__ == "__main__":
    main()
