"""오라클 데이터셋 만들기: 깨끗한 바탕 클립 × 설계 의도 4종 → (Y, X, M, 정답) 쌍.

실행 예 (판정·학습 해상도 768×432 통일, 2026-11-09 결정)
    python scripts/make_oracle_dataset.py --test <768×432 이상 시험 영상(data/ 기준 경로)>:<시작초>:<끝초>
    - train·val 기본 원본: data/sources/big_buck_bunny_1080p/big_buck_bunny_1080p_h264.mov (step0_check.py 가 받음)
    - test 기본 원본 Elephants Dream(data/raw)은 480×270 이라 768×432 로 쓸 수 없다(확대 금지 → 오류로 멈춤).
      다른 영화의 768×432 이상 영상을 --test 로 지정해야 한다(후보: Sintel 1080p, Elephants Dream HD — 미확보).
    128×72 예비 실험 재현: --width 128 --height 72 --train raw/BigBuckBunny_640x360.m4v:30:420
                          --val raw/BigBuckBunny_640x360.m4v:450:560 --test raw/elephantsdream-480-h264-st-aac.mov:20:620

출력
    data/oracle_<트랙>/{train,val,test}/<id>.npz   (트랙 = 예: 128x72_24fps)   clean(Y), flicker(X), mask_allow(M), mask_changed,
                                            w(시간 파형), hazard_frames, hazard_mask  (모두 무손실)
    data/oracle_<트랙>/manifest.jsonl             샘플마다 출처·분할·합성 조건·판정 결과 한 줄
    data/oracle_<트랙>/backgrounds_<split>.npz   바탕 클립 묶음 + fps·해상도 (GAN 학습 재료)
    checks/oracle/<트랙>.md                      요약표 (Git 에 올리는 결과 기록)

분할 (원본 영화 단위, 데이터 누수 방지)
    train : Big Buck Bunny 30~420초
    val   : Big Buck Bunny 450~560초   (같은 영화지만 시간 구간이 겹치지 않음 — 하이퍼파라미터 선택용)
    test  : Elephants Dream 전체 구간   (학습에 전혀 쓰지 않은 다른 영화 — 일반화 평가용)

주의: flicker 배열은 광과민성 위험 자극을 포함한다. 영상으로 재생하거나 미리보기하지 않는다.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

from lumi_synth import measure, oracle
from lumi_synth.data import ClipSpec, file_sha256, save_sample, scan_clean_clips

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {   # 경로는 --raw(기본 data/) 기준
    "train": ("sources/big_buck_bunny_1080p/big_buck_bunny_1080p_h264.mov", 30, 420),
    "val": ("sources/big_buck_bunny_1080p/big_buck_bunny_1080p_h264.mov", 450, 560),
    "test": ("raw/elephantsdream-480-h264-st-aac.mov", 20, 620),   # 480×270: 768×432 에는 못 씀 → --test 로 바꿀 것
}


def failed_condition(flicker: np.ndarray, fps: float) -> str:
    """판정이 의도와 다를 때, 네 줄 판정표에서 무너진 조건 이름을 모은다(원인 기록용)."""
    rows = measure.criteria_report(flicker, fps)
    bad = [r["조건"] for r in rows[:3] if not r["넘음"]]
    return ", ".join(bad) if bad else "세 조건 모두 넘음"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--raw", type=Path, default=ROOT / "data", help="원본 경로의 기준 폴더 (기본 data/)")
    ap.add_argument("--out", type=Path, default=None, help="생략하면 data/oracle_<트랙>")
    ap.add_argument("--width", type=int, default=768)
    ap.add_argument("--height", type=int, default=432)
    ap.add_argument("--frames", type=int, default=48)
    ap.add_argument("--fps", type=float, default=24.0)
    ap.add_argument("--train", default=None, help="파일:시작초:끝초 (생략하면 기본 SOURCES)")
    ap.add_argument("--val", default=None)
    ap.add_argument("--test", default=None)
    ap.add_argument("--train-clips", type=int, default=60)
    ap.add_argument("--val-clips", type=int, default=12)
    ap.add_argument("--test-clips", type=int, default=16)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    spec = ClipSpec(args.width, args.height, args.frames, args.fps)
    for split in ("train", "val", "test"):
        if getattr(args, split):
            f, s0, s1 = getattr(args, split).rsplit(":", 2)
            SOURCES[split] = (f, float(s0), float(s1))
    args.out = args.out or ROOT / "data" / f"oracle_{spec.tag}"
    rng = np.random.default_rng(args.seed)
    limits = {"train": args.train_clips, "val": args.val_clips, "test": args.test_clips}
    manifest, summary = [], []
    # 시작 전에 모든 원본을 검사: 없거나 목표 해상도보다 작으면(확대 금지) 아무것도 만들지 않고 멈춘다.
    import cv2
    for split, (fname, _, _) in SOURCES.items():
        src = args.raw / fname
        if not src.exists():
            print(f"[오류] {split} 원본이 없습니다: {src} (DATA_SOURCES.md 의 받는 법 참고)", file=sys.stderr)
            return 2
        cap = cv2.VideoCapture(str(src))
        sw, sh = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        if sw < spec.width or sh < spec.height:
            print(f"[오류] {split} 원본 {src.name} 이 {sw}×{sh} 로 목표 {spec.width}×{spec.height} 보다 작습니다(확대 금지). "
                  f"--{split} 로 더 큰 영상을 지정하세요.", file=sys.stderr)
            return 2
    t0 = time.time()
    for split, (fname, s0, s1) in SOURCES.items():
        src = args.raw / fname
        if not src.exists():
            print(f"[오류] 원본이 없습니다: {src} (DATA_SOURCES.md 의 받는 법 참고)", file=sys.stderr)
            return 2
        step = max(3.0, spec.frames / spec.fps + 1.0)   # 클립끼리 겹치지 않게 (클립 길이 + 1초 이상)
        clips = scan_clean_clips(src, spec, start_sec=s0, end_sec=s1, step_sec=step, limit=limits[split])
        np.savez_compressed(_mk(args.out) / f"backgrounds_{split}.npz",
                            clips=np.stack([c["clip"] for c in clips]) if clips else np.zeros((0,)),
                            fps=np.float32(spec.fps), width=np.int32(spec.width), height=np.int32(spec.height))
        ok_count, fail_reasons = Counter(), Counter()
        for i, c in enumerate(clips):
            for intent in oracle.INTENTS:            # 같은 바탕에 의도 4종 → 조건 하나만 다른 경계 쌍
                p = oracle.sample_params(intent, rng)
                r = oracle.synthesize(c["clip"], p, spec.fps)
                sid = f"{split}_{i:03d}_{intent}"
                reason = "" if r["intent_ok"] else failed_condition(r["flicker"], spec.fps)
                ok_count[(intent, r["intent_ok"])] += 1
                if reason:
                    fail_reasons[(intent, reason)] += 1
                meta = {"split": split, "source": c["source"], "start_frame": c["start_frame"],
                        "fps": spec.fps, "width": spec.width, "height": spec.height, "track": spec.tag, "background_check": c["check"], "params": r["params"],
                        "label_dangerous": r["judge"]["dangerous"],
                        "max_flashes": r["judge"]["max_flashes"], "peak_area": r["judge"]["peak_area"],
                        "mask_area": float(r["mask_allow"].mean()), "changed_area": float(r["mask_changed"].mean()),
                        "intent_ok": r["intent_ok"], "intent_fail_reason": reason,
                        "label_source": "lumi_synth.measure.analyze"}
                manifest.append(save_sample(args.out / split, sid, {
                    "clean": c["clip"], "flicker": r["flicker"], "mask_allow": r["mask_allow"],
                    "mask_changed": r["mask_changed"], "w": r["w"],
                    "hazard_frames": r["judge"]["hazard_frames"], "hazard_mask": r["judge"]["hazard_mask"],
                }, meta))
        summary.append((split, fname, len(clips), ok_count, fail_reasons))
        print(f"{split}: 깨끗한 바탕 {len(clips)}개 → 샘플 {len(clips) * len(oracle.INTENTS)}개 "
              f"({time.time() - t0:.0f}s)")

    with open(args.out / "manifest.jsonl", "w", encoding="utf-8") as f:
        for row in manifest:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    write_report(args, spec, summary, manifest, time.time() - t0)
    return 0


def _mk(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_report(args, spec, summary, manifest, secs) -> None:
    """checks/oracle/<트랙>.md — 사람이 읽는 결과 기록(실측값만)."""
    lines = [f"# 오라클 데이터셋 점검표 — 트랙 {spec.tag}", "",
             f"- 만든 날: {time.strftime('%Y-%m-%d %H:%M')}, 걸린 시간 {secs:.0f}초, seed {args.seed}",
             f"- 트랙 `{spec.tag}`: {spec.width}×{spec.height}, {spec.frames}프레임, {spec.fps:g} fps "
             f"({spec.frames / spec.fps:g}초), 데이터 폴더 `{args.out.relative_to(ROOT) if args.out.is_relative_to(ROOT) else args.out}`",
             f"- 정답 규칙: `lumi_synth.measure.analyze` (docs/RULES_SPEC.md)", "",
             "## 원본 (SHA256 앞 16자리)", "", "| 분할 | 파일 | 구간(초) | SHA256 |", "|---|---|---|---|"]
    for split, (fname, s0, s1) in SOURCES.items():
        lines.append(f"| {split} | {fname} | {s0}~{s1} | `{file_sha256(args.raw / fname)[:16]}…` |")
    lines += ["", "## 의도와 판정 일치율 (오라클 신뢰도)", "",
              "| 분할 | 깨끗한 바탕 | hazard | safe_area | safe_count | safe_lum |", "|---|---|---|---|---|---|"]
    for split, fname, n, okc, _ in summary:
        cells = [f"{okc[(i, True)]}/{okc[(i, True)] + okc[(i, False)]}" for i in oracle.INTENTS]
        lines.append(f"| {split} | {n} | " + " | ".join(cells) + " |")
    lines += ["", "## 의도와 판정이 다른 경우의 원인 (판정표에서 무너진 조건)", ""]
    any_fail = False
    for split, _, _, _, fr in summary:
        for (intent, reason), k in sorted(fr.items()):
            lines.append(f"- {split} / {intent}: {reason} — {k}개")
            any_fail = True
    if not any_fail:
        lines.append("- 없음")
    labels = Counter((r["split"], r["label_dangerous"]) for r in manifest)
    lines += ["", "## 최종 라벨 분포 (판정 규칙 기준)", "", "| 분할 | 위험 | 안전 |", "|---|---|---|"]
    for split in SOURCES:
        lines.append(f"| {split} | {labels[(split, True)]} | {labels[(split, False)]} |")
    lines += ["", "라벨은 설계 의도가 아니라 판정 규칙의 결과다. 의도와 다른 샘플도 버리지 않고 판정 결과로 라벨을 붙인다.",
              "데이터 파일(`data/`)은 Git 에 올리지 않는다."]
    out = ROOT / "checks" / "oracle"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{spec.tag}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
