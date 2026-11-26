"""회귀 벤치마크 실행 — 같은 대표 영상 42 개를 판정기 버전마다 돌려 결과를 쌓는다.

    python scripts/bench_run.py --tag "G-P2+cpdA"                # 기본: G(기본) · G(적응) · 팀원 판 v1
    python scripts/bench_run.py --tag 실험명 --judges G --only real_pattern1,real_pattern2
    python scripts/bench_run.py --report                         # 표만 다시

기록: data/benchmark/history.jsonl (태그·git 커밋·변경 여부·시각·판정기·항목·라벨·시간) → checks/bench/history.md (항목 × 버전 표)
디코드: H.264 는 GPU(NVDEC h264_cuvid), 나머지는 CPU. 판정은 GPU.
긴 영상: 20 초 조각, 5 초 겹침(연장 깜빡임 5 초 창 유지)으로 나눠 판정하고 라벨을 합침 — 구현 선택(조각 경계의 첫 전이는 놓칠 수 있음).
⚠️ 재생 없음. 영상은 data/benchmark(로컬 전용).
"""
from __future__ import annotations

import argparse
import datetime
import json
import subprocess
import sys
import time
from pathlib import Path

import av
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from lumi_judge.cpu_limit import limit_cpu  # noqa: E402
limit_cpu()
B = ROOT / "data" / "benchmark"
HIST = B / "history.jsonl"
CHUNK_S, OVERLAP_S = 20.0, 5.0


def decode(p: Path):
    if p.suffix == ".npz":
        z = np.load(p)
        return z["frames"], float(z["fps"])
    frames = []
    with av.open(str(p)) as c:
        s = c.streams.video[0]
        fps = float(s.average_rate) if s.average_rate else 30.0
        if s.codec_context.name == "h264":
            try:
                cc = av.CodecContext.create("h264_cuvid", "r"); cc.extradata = s.codec_context.extradata
                for pkt in c.demux(s):
                    try:
                        for f in cc.decode(pkt):
                            frames.append(f.to_ndarray(format="bgr24"))
                    except av.error.EOFError:
                        pass
            except Exception:                                           # noqa: BLE001 — GPU 디코드 실패 시 CPU
                frames = []
        if not frames:
            c.seek(0)
            frames = [f.to_ndarray(format="bgr24") for f in c.decode(video=0)]
    if p.suffix.lower() == ".gif" or fps <= 0:
        from eval_real_hazards import decode as dec_any
        x, fps = dec_any(p); return x, fps
    return shrink(np.stack(frames)), fps


MAX_W = 640                                                            # 분석 폭 상한 — 팀원 판 analysis_max_width 와 같음(면적 % 는 그대로)


def shrink(x):
    """폭 > 640 이면 640 으로 줄임(GPU). 판정 가능 줄무늬 상한은 결과의 '판정 가능 범위'에 반영됨."""
    if x.shape[2] <= MAX_W:
        return x
    import torch.nn.functional as F
    h = int(round(x.shape[1] * MAX_W / x.shape[2]))
    out = np.empty((len(x), h, MAX_W, 3), np.uint8)
    for s in range(0, len(x), 64):
        t = torch.from_numpy(x[s:s + 64]).cuda().permute(0, 3, 1, 2).float()
        out[s:s + 64] = F.interpolate(t, size=(h, MAX_W), mode="area").round().clamp(0, 255).byte().permute(0, 2, 3, 1).cpu().numpy()
    return out


def chunks(T, fps):
    n, ov = int(round(CHUNK_S * fps)), int(round(OVERLAP_S * fps))
    if T <= n:
        return [(0, T)]
    out, s = [], 0
    while s < T:
        out.append((s, min(T, s + n)))
        if s + n >= T:
            break
        s += n - ov
    return out


def run_judge(name, x, fps):
    labels, sents, notes = set(), [], []
    for a, b in chunks(len(x), fps):
        seg = x[a:b]
        if name in ("G", "G_adapt"):
            from lumi_judge.grid_judge import judge_video
            r = judge_video(seg, fps, adapt=(name == "G_adapt"))
            labels |= set(r["labels"])
            sents += [f"[{a / fps:.0f}s~] " + f["sentence"] for f in r["findings"]][:2]
            notes += r.get("notes", [])[:1]
        elif name == "v1":
            import lumi_judge as J
            r = J.analyze_frames(list(seg), fps, device="cuda", mask_config=J.PartialMaskConfig())
            labels |= set(r.active_labels)
        torch.cuda.empty_cache()
    return sorted(labels), sents[:3], notes[:1]


def git_ver():
    g = lambda *a: subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    return g("rev-parse", "--short", "HEAD"), bool(g("status", "--porcelain", "--", "src"))


def report():
    man = json.loads((B / "manifest.json").read_text(encoding="utf-8"))
    rows = [json.loads(l) for l in HIST.read_text(encoding="utf-8").splitlines() if l.strip()] if HIST.exists() else []
    cols = []
    for r in rows:
        k = (r["tag"], r["judge"])
        if k not in cols:
            cols.append(k)
    last = {}
    for r in rows:
        last[(r["tag"], r["judge"], r["item"])] = r
    yn = lambda b: "위험" if b else "안전"
    L = ["# 회귀 벤치마크 — 항목 × 버전 (scripts/bench_run.py)", "",
         f"> 항목 {man['n']} · 정답 근거는 항목마다(ITU 원문 생성 / 생성 레시피 / 사용자 지정 / 외부 경고 / 미정). 셀 = 판정(✓ 정답 · ✗ 틀림 · · 정답 미정) + 라벨 약자", "",
         "| 버전 (태그 · 판정기 · 커밋) | " + " | ".join(f"{t} · {j}" for t, j in cols) + " |", "|---|" + "---|" * len(cols)]
    ab = {"휘도 플래시": "휘", "적색 플래시": "적", "휘도 연장": "휘연", "적색 연장": "적연", "정적 공간 패턴": "정패", "변화 공간 패턴": "변패",
          "빠른 장면 전환": "장면", "교집합": "교"}
    L.append("| 커밋 | " + " | ".join(next((r["git"] + ("*" if r["dirty"] else "") for r in rows if (r["tag"], r["judge"]) == c), "") for c in cols) + " |")
    score = {c: [0, 0] for c in cols}
    cur = None
    for it in man["items"]:
        if it["cat"] != cur:
            cur = it["cat"]; L.append(f"| **{cur}** |" + " |" * len(cols))
        cells = []
        for c in cols:
            r = last.get((c[0], c[1], it["id"]))
            if r is None:
                cells.append(""); continue
            d = bool(r["labels"])
            mark = "·" if it["expect"] == "미정" else ("✓" if (it["expect"] == "위험") == d else "✗")
            if it["expect"] != "미정":
                score[c][0] += mark == "✓"; score[c][1] += 1
            cells.append(f"{mark} {yn(d)} {'/'.join(ab.get(x, x) for x in r['labels'])}")
        L.append(f"| {it['id']} (정답 {it['expect']}) | " + " | ".join(cells) + " |")
    L.append("| **정답 수** | " + " | ".join(f"**{s[0]}/{s[1]}**" for s in score.values()) + " |")
    L += ["", "정답 근거·비고: `checks/bench/manifest.json`. 라벨 약자: 휘 휘도 플래시 · 적 적색 · 휘연/적연 연장 · 정패 정적 패턴 · 변패 변화 패턴 · 장면 빠른 장면 전환 · 교 교집합."]
    (ROOT / "checks" / "bench" / "history.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L[-8:]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default=None); ap.add_argument("--judges", default="G,G_adapt,v1"); ap.add_argument("--only", default="")
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    if a.report:
        report(); return
    man = json.loads((B / "manifest.json").read_text(encoding="utf-8"))
    ver, dirty = git_ver()
    only = set(filter(None, a.only.split(",")))
    with HIST.open("a", encoding="utf-8") as f:
        for it in man["items"]:
            if only and it["id"] not in only:
                continue
            x, fps = decode(B / it["path"])
            for j in a.judges.split(","):
                t0 = time.time()
                try:
                    labels, sents, notes = run_judge(j, x, fps)
                    row = {"tag": a.tag or ver, "judge": j, "git": ver, "dirty": dirty, "time": datetime.datetime.now().isoformat(timespec="seconds"),
                           "item": it["id"], "labels": labels, "sentences": sents, "notes": notes, "sec": round(time.time() - t0, 1)}
                except Exception as e:  # noqa: BLE001
                    row = {"tag": a.tag or ver, "judge": j, "git": ver, "dirty": dirty, "item": it["id"], "labels": [], "error": f"{type(e).__name__}: {e}"}
                f.write(json.dumps(row, ensure_ascii=False) + "\n"); f.flush()
                print(f"  {it['id']:28s} {j:8s} {row.get('labels')} {row.get('sec', row.get('error'))}", flush=True)
            del x; torch.cuda.empty_cache()
    report()


if __name__ == "__main__":
    main()
