"""깜빡임(F1 휘도 · F2 적색) 화질·해상도·주사율 강건성 — 사전 등록(이 머리말이 판단 규칙, 실행 전 커밋, 2026-11-26).

정지 패턴과 같은 틀: 규격 문턱(20 cd/m²·어두운 < 160·면적 25 %·초당 3 회)은 고정, '재는 방식'만 영상에 맞춤.
  - 근거 격자를 시각도(max(1°, 8 px))로, 판정 가능 범위(초당 fps/2 회)를 문장으로 — 판정을 바꾸지 않는 변경이라 실험 불필요(회귀 0/10 확인).
  - 이 실험이 정하는 것: **미세 평균**(WCAG 2.2 '0.1° 보다 작은 칸의 균형 잡힌 미세 무늬' 예외 — 측정 전 0.2° 상자 평균) 을 기본으로 켤지.
    ITU 에는 이 예외가 없으므로, 켜는 것은 '열화(디더링·압축 잡음)로 생기는 오탐을 줄이고 위험은 놓치지 않을 때'만.
열화: 원본 · 해상도 ½ · ¼ · H.264 CRF 23 · 35 · 45 · GIF(256 색 + 디더링) · 12 fps · 8 fps
세트: big1 val/test 의 휘도·적색 깜빡임·연장 hazard(위험)·near(문턱 근처 안전) + 진단 바탕 41(질감 많은 안전) + 시퀀스 비교 시험 6(정답 = ITU 원문)
변형: 기본 vs 미세 평균 0.2°
지표: 원본 판정과의 일치(세트·열화별) · 안전(바탕·near) 세트의 새 위험 수 · 시험 6 의 정답 수 ·
      놓친 위험 중 '판정 가능 범위 밖'(원본 초당 깜빡임 > 열화 후 fps/2)으로 설명되는 비율
판단 규칙(결과 보고 바꾸지 않음)
  1. 미세 평균을 기본으로 켬: 열화 합산에서 (a) 안전 세트 새 위험 수 < 기본 그리고 (b) 위험 세트 원본 일치 수 ≥ 기본 − 2 그리고 (c) 시험 6 정답 수 ≥ 기본.
  2. 아니면 끈 채로 두고(ITU 원문 그대로), WCAG 축 보고용 선택 사항으로만 남김.
  3. '범위 밖'으로 설명되지 않는 놓침은 결함 후보로 목록화.
출력: checks/judge/flash_robustness.md · data/judge/flash_robustness.jsonl
변경 기록
  2026-11-26 22:4x 범위 축소(결과를 보기 전): 클립당 약 1 분이라 287 개면 약 5 시간 → big1 위험·near 는 8 묶음(휘도/적색/휘도 연장/적색 연장
  × hazard/near)에서 묶음마다 10 개를 고정 난수(시드 0)로 뽑음(80) + 바탕 41 + 시험 6 = 127. 먼저 계산된 5 행은 보지 않고 지움. 판단 규칙은 그대로.
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from lumi_judge.cpu_limit import limit_cpu  # noqa: E402
limit_cpu()
from lumi_judge.grid_flash import judge_luminance, judge_red  # noqa: E402
from robustness_sweep import degrade  # noqa: E402

DEGS = ["orig", "half", "quarter", "crf23", "crf35", "crf45", "gif", "12fps", "8fps"]
FINE = 0.2


def judge(x, fps, fine):
    a = judge_luminance(x, fps, fine_avg_deg=fine); b = judge_red(x, fps, fine_avg_deg=fine)
    return bool(a["dangerous"] or b["dangerous"]), max(a["max_flashes"], b["max_flashes"])


def jobs():
    from make_hazard7_dataset import FPS, decode
    J = []
    groups: dict = {}
    for split in ("val", "test"):
        for p in sorted(glob.glob(str(ROOT / f"data/hazard7/big1/{split}/{split}_*.mp4"))):
            name = Path(p).name
            kind = next((k for k in ("lum_flash", "red_flash", "lum_ext", "red_ext") if f"_{k}_" in name), None)
            if ".clean." in name or kind is None:
                continue
            grp = "haz" if "_hazard_" in name else ("near" if "_near_" in name else None)
            if grp:
                groups.setdefault((kind, grp), []).append((split, name, p))
    rng = np.random.default_rng(0)                                     # 범위 축소: 묶음마다 10 개 (변경 기록 참고)
    for (kind, grp), items in sorted(groups.items()):
        for i in sorted(rng.choice(len(items), min(10, len(items)), replace=False)):
            split, name, p = items[i]
            J.append((grp, f"{split}/{name}", lambda p=p: (decode(Path(p)), FPS), None))
    dr = [json.loads(l) for l in (ROOT / "data/judge/pattern_diag_rows.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    for rid in sorted({r["id"] for r in dr if r["grp"] == "fp"}):
        J.append(("safe_bg", "bg/" + rid, lambda rid=rid: (decode(ROOT / "data/hazard7/big1/train" / rid), FPS), False))
    import make_seq_figs as SQ
    for name, truth, build in SQ.CASES:
        J.append(("test6", "seq/" + name, lambda build=build: (build()[0], SQ.FPS), truth))
    return J


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    cache = ROOT / "data/judge/flash_robustness.jsonl"
    done = {json.loads(l)["id"] for l in cache.read_text(encoding="utf-8").splitlines() if l.strip()} if cache.exists() else set()
    with cache.open("a", encoding="utf-8") as f:
        for i, (grp, cid, load, truth) in enumerate(jobs(), 1):
            if cid in done:
                continue
            try:
                x0, fps0 = load()
                row = {"id": cid, "grp": grp, "truth": truth, "res": {}}
                for d in DEGS:
                    x, fps = degrade(x0, fps0, d)
                    b, nb = judge(x, fps, None); a, na = judge(x, fps, FINE)
                    row["res"][d] = {"base": b, "fine": a, "flashes": nb, "fps": fps}
                    torch.cuda.empty_cache()
            except Exception as e:  # noqa: BLE001
                row = {"id": cid, "grp": grp, "error": f"{type(e).__name__}: {e}"}
            f.write(json.dumps(row, ensure_ascii=False) + "\n"); f.flush()
            if i % 10 == 0:
                print(f"  {i}", flush=True)
    report(cache)


def report(cache):
    rows = [json.loads(l) for l in cache.read_text(encoding="utf-8").splitlines() if l.strip()]
    err = [r for r in rows if "error" in r]; rows = [r for r in rows if "error" not in r]
    G = lambda g: [r for r in rows if r["grp"] == g]
    L = ["# 깜빡임 강건성 (사전 등록 = scripts/flash_robustness.py 머리말)", "",
         f"> 위험 {len(G('haz'))} · 문턱 근처 안전 {len(G('near'))} · 질감 바탕 {len(G('safe_bg'))} · 시험 {len(G('test6'))} (오류 {len(err)})", "",
         "| 열화 | 위험: 원본 일치 기본 | 미세 | 안전(near+바탕) 새 위험: 기본 | 미세 | 시험 6 정답: 기본 | 미세 |", "|---|---|---|---|---|---|---|"]
    tot = dict(hb=0, hf=0, sb=0, sf=0, tb=0, tf=0)
    for d in DEGS:
        H = G("haz"); S = G("near") + G("safe_bg"); T6 = G("test6")
        hb = sum(r["res"][d]["base"] == r["res"]["orig"]["base"] for r in H); hf = sum(r["res"][d]["fine"] == r["res"]["orig"]["base"] for r in H)
        sb = sum(r["res"][d]["base"] and not r["res"]["orig"]["base"] for r in S); sf = sum(r["res"][d]["fine"] and not r["res"]["orig"]["base"] for r in S)
        tb = sum(r["res"][d]["base"] == r["truth"] for r in T6); tf = sum(r["res"][d]["fine"] == r["truth"] for r in T6)
        if d != "orig":
            for k, v in zip(("hb", "hf", "sb", "sf", "tb", "tf"), (hb, hf, sb, sf, tb, tf)):
                tot[k] += v
        L.append(f"| {d} | {hb}/{len(H)} | {hf}/{len(H)} | {sb}/{len(S)} | {sf}/{len(S)} | {tb}/{len(T6)} | {tf}/{len(T6)} |")
    ok = tot["sf"] < tot["sb"] and tot["hf"] >= tot["hb"] - 2 and tot["tf"] >= tot["tb"]
    miss, expl, lst = 0, 0, []
    for r in G("haz"):
        if not r["res"]["orig"]["base"]:
            continue
        for d in DEGS[1:]:
            q = r["res"][d]
            if not q["base"]:
                miss += 1
                if r["res"]["orig"]["flashes"] > q["fps"] / 2:
                    expl += 1
                else:
                    lst.append(f"{r['id']} · {d} (원본 초당 {r['res']['orig']['flashes']}회, 열화 후 {q['flashes']}회, {q['fps']:.0f} fps)")
    L += ["", f"열화 합산 — 위험 원본 일치: 기본 {tot['hb']} · 미세 {tot['hf']} / 안전 새 위험: 기본 {tot['sb']} · 미세 {tot['sf']} / 시험 정답: 기본 {tot['tb']} · 미세 {tot['tf']}",
          f"원본에서 잡았는데 열화 후 놓친 경우 {miss} 중 '판정 가능 범위 밖'(초당 깜빡임 > fps/2)으로 설명 {expl}", "",
          "## 판정 (사전 규칙)", "", f"- **{'미세 평균을 기본으로 켬' if ok else '미세 평균은 끈 채 유지(ITU 원문) — WCAG 축 선택 사항'}**", "",
          "## 결함 후보 (범위 안인데 열화로 놓침)", ""] + [f"- {s}" for s in lst[:60]]
    (ROOT / "checks/judge/flash_robustness.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
