"""위험 영상 분류 모음 — 실제 위험 영상을 종류별로 분류해 한곳(data/hazard_library)에 모은다.

    python scripts/library_build.py           # G 판정(없으면 GPU 로 실행) → 분류 → 폴더 복사 → 목록
    python scripts/library_build.py --index   # 판정은 건너뛰고 목록·폴더만 다시

대상(실제 영상만 — 합성은 생성 정답이 있으므로 벤치마크 쪽에서 다룸)
  Kaya 2025 22 개            출처 라벨 = Kaya 저자 보정 도구(out/*/correction_info.txt): general(휘도) · red(적색) · both(둘 다), 프레임 구간
  South 2021 '위험' 163 개   출처 라벨 = PhotosensitivityPal(WCAG 검출기): flash · red · pattern
  사용자 제공 패턴영상1       출처 라벨 = 사용자 지정 '패턴'(정지/변화 구분 없음)
  (패턴영상2 = Kaya 6 과 프레임 동일 → Kaya 6 으로 한 번만)
분류 규칙(구현 선택)
  종류: 휘도 깜빡임 · 적색 깜빡임 · 연장 깜빡임 · 빠른 장면 전환 · 정지 패턴 · 변화 패턴
  세 근거(출처 라벨 · G · 팀원 판 v1) 중 둘 이상이 같은 종류를 말하면 '확정'. 하나만 말하면 '보류'(사람 확인).
  사용자 '패턴'은 정지·변화 둘 다에 한 표. v1 '교집합'은 휘도 깜빡임으로, PSPal 'pattern' 은 정지 패턴으로 셈.
  폴더: 확정 종류가 하나면 그 종류, 둘 이상이면 '복합', 없으면 '분류 보류'.
출력: data/hazard_library/<폴더>/<id>.<ext> · data/hazard_library/index.json · checks/library/index.md(영상 없음)
⚠️ 위험 자극 영상 — 재생 금지, 로컬 전용(South·Kaya 재배포 금지).
"""
from __future__ import annotations

import csv
import json
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
LIB = ROOT / "data" / "hazard_library"
CATS = ["휘도 깜빡임", "적색 깜빡임", "연장 깜빡임", "빠른 장면 전환", "정지 패턴", "변화 패턴"]
FOLDER = {"휘도 깜빡임": "01_휘도깜빡임", "적색 깜빡임": "02_적색깜빡임", "연장 깜빡임": "03_연장깜빡임", "빠른 장면 전환": "04_장면전환",
          "정지 패턴": "05_정지패턴", "변화 패턴": "06_변화패턴", "복합": "07_복합", "보류": "99_분류보류"}
MAP_JUDGE = {"휘도 플래시": "휘도 깜빡임", "교집합": "휘도 깜빡임", "적색 플래시": "적색 깜빡임", "휘도 연장": "연장 깜빡임", "적색 연장": "연장 깜빡임",
             "빠른 장면 전환": "빠른 장면 전환", "정적 공간 패턴": "정지 패턴", "변화 공간 패턴": "변화 패턴"}


def sources():
    S = []
    kd = ROOT / "data/real/kaya2025/extracted"
    v1k = {}
    for line in (ROOT / "data/correction/kaya_review/index.md").read_text(encoding="utf-8").splitlines():
        m = re.match(r"\| (\S+\.(?:mp4|avi)) \| ([^|]*) \|", line)
        if m:
            v1k[m.group(1)] = [x.strip() for x in m.group(2).split(",") if x.strip() and x.strip() != "-"]
    for p in sorted(kd.glob("*.*"), key=lambda q: int(q.stem)):
        info = (kd / "out" / p.stem / "correction_info.txt").read_text(encoding="utf-8", errors="ignore")
        kinds = re.findall(r"\('(general|red|both)', (\d+), (\d+)\)", info.split("Corrected flashes")[0])
        src = set()
        for k, _, _ in kinds:
            src |= {"휘도 깜빡임"} if k == "general" else ({"적색 깜빡임"} if k == "red" else {"휘도 깜빡임", "적색 깜빡임"})
        S.append({"id": f"kaya_{p.stem}", "path": p, "origin": "Kaya 2025", "src_label": sorted(src),
                  "src_note": "Kaya 저자 보정 도구 " + ", ".join(sorted({k for k, _, _ in kinds})), "v1": v1k.get(p.name, [])})
    s = ROOT / "data/real/south2021"
    lab = {r["label"]: r for r in csv.DictReader(open(s / "social-media-study.csv", encoding="utf-8", errors="ignore"))}
    files = {p.stem: p for p in (s / "social-media-study-gifs").iterdir()}
    rows = {json.loads(l)["label"]: json.loads(l) for l in open(ROOT / "data/judge/south_partial/rows.jsonl", encoding="utf-8") if l.strip()}
    for k, r in sorted(lab.items()):
        if r["dangerous"] != "dangerous" or k not in files:
            continue
        src = set()
        if r["flash"] == "flash":
            src.add("휘도 깜빡임")
        if r["red"] == "red":
            src.add("적색 깜빡임")
        if r["pattern"] == "pattern":
            src.add("정지 패턴")
        rr = rows.get(k, {})
        S.append({"id": f"south_{k}", "path": files[k], "origin": "South 2021", "src_label": sorted(src), "src_note": "PhotosensitivityPal",
                  "v1": rr.get("labels", []) if "error" not in rr else []})
    S.append({"id": "user_pattern1", "path": Path(r"C:\dev\lumi-vision\패턴영상1.mp4"), "origin": "사용자 제공", "src_label": ["정지 패턴|변화 패턴"],
              "src_note": "사용자 지정 패턴 영상", "v1": None})
    return S


def run_G(S):
    from bench_run import decode, run_judge
    cache = LIB / "judge_cache.jsonl"
    LIB.mkdir(parents=True, exist_ok=True)
    done = {json.loads(l)["id"]: json.loads(l) for l in cache.read_text(encoding="utf-8").splitlines() if l.strip()} if cache.exists() else {}
    with cache.open("a", encoding="utf-8") as f:
        for i, s in enumerate(S, 1):
            need = ["G"] + (["v1"] if s["v1"] is None else [])
            if s["id"] in done and all(j in done[s["id"]] for j in need):
                continue
            x, fps = decode(Path(s["path"]))
            row = {"id": s["id"], "fps": fps, "frames": len(x), "w": int(x.shape[2]), "h": int(x.shape[1])}
            for j in need:
                labels, sents, notes = run_judge(j, x, fps)
                row[j] = {"labels": labels, "sentences": sents}
            f.write(json.dumps(row, ensure_ascii=False) + "\n"); f.flush()
            print(f"  {i}/{len(S)} {s['id']} G={row['G']['labels']}", flush=True)
    return {json.loads(l)["id"]: json.loads(l) for l in cache.read_text(encoding="utf-8").splitlines() if l.strip()}


def classify(s, g):
    votes = {c: [] for c in CATS}
    for c in s["src_label"]:
        for cc in c.split("|"):
            votes[cc].append("출처")
    gl = g.get("G", {}).get("labels", [])
    for x in {MAP_JUDGE[l] for l in gl if l in MAP_JUDGE}:
        votes[x].append("G")
    v1 = s["v1"] if s["v1"] is not None else g.get("v1", {}).get("labels", [])
    for x in {MAP_JUDGE[l] for l in v1 if l in MAP_JUDGE}:
        votes[x].append("v1")
    conf = [c for c in CATS if len(votes[c]) >= 2]
    pend = [c for c in CATS if len(votes[c]) == 1]
    folder = FOLDER[conf[0]] if len(conf) == 1 else (FOLDER["복합"] if conf else FOLDER["보류"])
    return conf, pend, votes, folder, gl, v1


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    S = sources()
    G = run_G(S) if "--index" not in sys.argv else {json.loads(l)["id"]: json.loads(l) for l in (LIB / "judge_cache.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()}
    idx = []
    for s in S:
        g = G.get(s["id"], {})
        conf, pend, votes, folder, gl, v1 = classify(s, g)
        dst = LIB / folder / f"{s['id']}{Path(s['path']).suffix}"
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            shutil.copy2(s["path"], dst)
        idx.append({"id": s["id"], "origin": s["origin"], "folder": folder, "confirmed": conf, "pending": pend,
                    "src_label": s["src_label"], "src_note": s["src_note"], "G": gl, "v1": v1,
                    "G_sentences": g.get("G", {}).get("sentences", []), "fps": g.get("fps"), "frames": g.get("frames"),
                    "size": f"{g.get('w')}x{g.get('h')}", "file": str(dst.relative_to(LIB)).replace("\\", "/")})
    (LIB / "index.json").write_text(json.dumps(idx, ensure_ascii=False, indent=1), encoding="utf-8")
    import collections
    byf = collections.Counter(r["folder"] for r in idx)
    L = ["# 위험 영상 분류 모음 (scripts/library_build.py)", "",
         f"> 실제 영상 {len(idx)} 개 (Kaya 22 · South 위험 163 · 사용자 1; 패턴영상2 = Kaya 6). 세 근거(출처 라벨 · G · 팀원 판 v1) 중 둘 이상 = 확정, 하나 = 보류.", "",
         "## 폴더별 개수", "", "| 폴더 | 개수 |", "|---|---|"] + [f"| {k} | {byf[k]} |" for k in sorted(byf)]
    cc = collections.Counter(c for r in idx for c in r["confirmed"]); pc = collections.Counter(c for r in idx for c in r["pending"])
    L += ["", "## 종류별 (확정 / 보류)", "", "| 종류 | 확정 | 보류 |", "|---|---|---|"] + [f"| {c} | {cc[c]} | {pc[c]} |" for c in CATS]
    L += ["", "## 항목", "", "| id | 폴더 | 확정 | 보류(근거 하나) | 출처 라벨 | G | 팀원 판 v1 |", "|---|---|---|---|---|---|---|"]
    for r in idx:
        L.append(f"| {r['id']} | {r['folder']} | {', '.join(r['confirmed']) or '-'} | {', '.join(r['pending']) or '-'} | "
                 f"{', '.join(r['src_label']) or '-'} ({r['src_note']}) | {', '.join(r['G']) or '-'} | {', '.join(r['v1'] or []) or '-'} |")
    out = ROOT / "checks" / "library"; out.mkdir(parents=True, exist_ok=True)
    (out / "index.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L[:30]))


if __name__ == "__main__":
    main()
