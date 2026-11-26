"""회귀 벤치마크 만들기 — 위험 종류별 대표 영상을 한곳(data/benchmark)에 모으고 목록(manifest)을 만든다.

    python scripts/bench_build.py

구성(카테고리 · 정답 근거)
  실제 패턴   : 사용자 제공 패턴영상1·2 (정답 근거 = 사용자 지정 '패턴 영상')
  정지 패턴   : 적합성 시험 생성(ITU 원문으로 정답) — 직선 60 %, 면적 39/42 %, 동심원, 방사, 체커, 어두운 막대 165(SDR)
  변화 패턴   : 움직이는 패턴 시험(ITU 원문) — 흐름 60 %, 진동 30 %, 대비 반전 30 %, 점멸 30 %, 방향 바뀜 30 %, 동심원 줌 60 %
  휘도 깜빡임 : 시퀀스 시험(ITU 원문) — 사각 4 Hz, 맥동 22/60, 120 ms 오르내림, 3 Hz 경계, 면적 24 %
  적색·연장·장면 전환·교집합·정지 패턴(합성): big1 val 의 hazard/near (정답 근거 = 생성 레시피, 팀원 판 기준으로 만든 '문턱 근처')
  안전 일상   : BBB 바탕(나뭇잎) · South 줄무늬 셔츠(경계) · South DVD 진열장(경계) (정답 근거 = 사람 확인 전 — '미정')
  실제 위험   : Kaya 1 · 10 · 19 (정답 근거 = 외부 경고, 세부 라벨 미정)
합성 시험은 프레임을 그대로(uint8, 압축 없음) npz 로 저장 — 휘도가 정확해야 하므로. 실제 영상은 원본을 복사.
⚠️ 위험 자극 영상 — 재생 금지, 로컬 전용(깃·공개 금지). 목록(checks/bench/manifest.json)에는 영상이 없다.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
B = ROOT / "data" / "benchmark"
LV = Path(r"C:\dev\lumi-vision")


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def save_npz(name, frames, fps):
    p = B / "synth" / f"{name}.npz"
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(p, frames=frames, fps=np.float64(fps))
    return p


def copy(src: Path, sub: str, name: str | None = None):
    d = B / sub; d.mkdir(parents=True, exist_ok=True)
    dst = d / (name or src.name)
    if not dst.exists():
        shutil.copy2(src, dst)
    return dst


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    items = []
    add = lambda **k: items.append(k)
    # 실제 패턴 (사용자 제공)
    for i in (1, 2):
        p = copy(LV / f"패턴영상{i}.mp4", "real", f"pattern_video{i}.mp4")
        add(id=f"real_pattern{i}", cat="실제 패턴", path=p, expect="위험", expect_labels=["정적 공간 패턴|변화 공간 패턴"], truth="사용자 지정 패턴 영상")
    # 정지 패턴 (ITU 원문 생성)
    import pattern_conformance as PC
    for name, kw in [("straight60", dict(area=0.6)), ("area39", dict(area=0.39)), ("area42", dict(area=0.42)), ("concentric", dict(kind="concentric")),
                     ("radial", dict(kind="radial")), ("checker", dict(kind="checker")), ("dark165_sdr", dict(lo=165.0, hi=195.0))]:
        L, ar = PC.stripes(**kw); truth = PC.truth_of(kw, ar)
        x = np.repeat(PC.to_u8(L)[None], int(PC.FPS) * 2, 0)
        p = save_npz(f"static_{name}", x, PC.FPS)
        add(id=f"static_{name}", cat="정지 패턴", path=p, expect="위험" if truth else "안전", expect_labels=["정적 공간 패턴"] if truth else [],
            truth=f"ITU 원문 (실면적 {100 * ar:.0f}%)")
    # 변화 패턴
    import pattern_motion_conformance as PM
    for name, truth, kw in [("flow60", False, dict(motion="drift", area=0.6, speed=1.0)), ("osc30", True, dict(motion="osc", area=0.3, freq=2.0)),
                            ("reverse30", True, dict(motion="reverse", area=0.3, freq=2.0)), ("flash30", True, dict(motion="flash", area=0.3, freq=1.0)),
                            ("orient30", True, dict(motion="orient", area=0.3, freq=1.0)), ("zoom60", False, dict(motion="zoom", area=0.6, speed=1.0))]:
        x, ar = PM.clip(**kw)
        p = save_npz(f"motion_{name}", x, PM.FPS)
        add(id=f"motion_{name}", cat="변화 패턴", path=p, expect="위험" if truth else "안전", expect_labels=["변화 공간 패턴"] if truth else [],
            truth="ITU 원문 (흐름 면제 · 변화 25 %)")
    # 휘도 깜빡임
    import make_seq_figs as SQ
    tags = ["square4hz", "pulse22", "pulse60", "ramp120ms", "square3hz_boundary", "square4hz_area24"]
    for tag, (name, truth, build) in zip(tags, SQ.CASES):
        x, _ = build()
        p = save_npz(f"flash_{tag}", x, SQ.FPS)
        add(id=f"flash_{tag}", cat="휘도 깜빡임", path=p, expect="위험" if truth else "안전", expect_labels=["휘도 플래시"] if truth else [],
            truth="ITU 원문" + (" (JV24 66 ms 해석이면 안전)" if tag == "pulse22" else ""))
    # big1 합성 hazard/near
    kinds = {"lum_flash": "휘도 플래시", "red_flash": "적색 플래시", "lum_ext": "휘도 연장", "red_ext": "적색 연장",
             "scene": "빠른 장면 전환", "intersection": "교집합", "pattern": "정적 공간 패턴"}
    cats = {"lum_flash": "휘도 깜빡임", "red_flash": "적색 깜빡임", "lum_ext": "연장 깜빡임", "red_ext": "연장 깜빡임",
            "scene": "장면 전환", "intersection": "교집합", "pattern": "정지 패턴"}
    for k, lab in kinds.items():
        for intent in ("hazard", "near"):
            n = "003" if (k == "pattern" and intent == "hazard") else "000"          # 패턴 hazard_000 은 흐름(면제) → 정지인 003
            src = ROOT / f"data/hazard7/big1/val/val_{k}_{intent}_{n}.mp4"
            p = copy(src, "big1")
            add(id=f"big1_{k}_{intent}", cat=cats[k], path=p, expect="위험" if intent == "hazard" else "안전",
                expect_labels=[lab] if intent == "hazard" else [], truth="생성 레시피 (near = 팀원 판 기준 문턱 근처)")
    # 안전 일상(경계 포함)
    p = copy(ROOT / "data/hazard7/big1/train/train_red_flash_near_140.clean.mp4", "daily", "bbb_leaves.mp4")
    add(id="daily_bbb_leaves", cat="안전 일상", path=p, expect="안전", expect_labels=[], truth="합성 전 바탕(가정) — 팀원 판은 패턴이라 함")
    s = ROOT / "data/real/south2021/social-media-study-gifs"
    for key, note in (("giphy-253", "줄무늬 셔츠 흔들림 — 경계, 사람 확인 대기"), ("tenor-131", "DVD 진열장 세로선 — 경계, 사람 확인 대기"),
                      ("giphy-158", "PRIMO 로고 가는 가로줄 — G·팀원 판 모두 패턴")):
        src = next(s.glob(key + ".*"))
        p = copy(src, "daily", f"south_{key}{src.suffix}")
        add(id=f"south_{key}", cat="안전 일상" if key != "giphy-158" else "실제 패턴", path=p, expect="미정", expect_labels=[], truth=note)
    # 실제 위험 (Kaya)
    for k in ("1.mp4", "10.mp4", "19.avi"):
        p = copy(ROOT / "data/real/kaya2025/extracted" / k, "kaya", f"kaya_{k}")
        add(id=f"kaya_{Path(k).stem}", cat="실제 위험", path=p, expect="위험", expect_labels=[], truth="외부 경고(Kaya 2025) — 세부 라벨 미정")
    for it in items:
        it["sha256"] = sha(Path(it["path"])); it["path"] = str(Path(it["path"]).relative_to(B)).replace("\\", "/")
    man = {"version": 1, "built": __import__("datetime").datetime.now().isoformat(timespec="seconds"), "n": len(items), "items": items}
    (B / "manifest.json").write_text(json.dumps(man, ensure_ascii=False, indent=1), encoding="utf-8")
    out = ROOT / "checks" / "bench"; out.mkdir(parents=True, exist_ok=True)
    (out / "manifest.json").write_text(json.dumps(man, ensure_ascii=False, indent=1), encoding="utf-8")
    import collections
    print(len(items), dict(collections.Counter(i["cat"] for i in items)))


if __name__ == "__main__":
    main()
