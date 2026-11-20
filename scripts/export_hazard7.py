"""hazard7 데이터를 공유용 폴더로 정리한다 (구글 드라이브 업로드용).

    python scripts/export_hazard7.py --name big1                    # 폴더 정리(하드링크 — 디스크 추가 사용 거의 없음)
    python scripts/export_hazard7.py --name big1 --pack 2           # + 분할별 zip 묶음(2 GB 단위, 압축 없이 저장)
    python scripts/export_hazard7.py --name big1 --extra big2       # 다른 묶음(추가 생성분)을 같은 데이터셋으로 합침

결과: data/export/LUMI_hazard7_<name>/
  README.md        만든 방법·라벨 체계·파일 형식·분할·개수·라이선스·주의 (자동 생성)
  index.csv        표본마다 한 줄(분할·종류·의도·라벨·바탕·레시피 요약·층 품질·판정 결과)
  south_bg_split.json  South 안전 영상을 바탕으로 쓸 때의 train/val 목록
  {train,val,test}/  <id>.mp4(합성) · <id>.clean.mp4(원본) · <id>.labels.npz · <id>.layers.npz · manifest.jsonl(조각 합침) · gen.jsonl
  zips/            --pack 일 때 <분할>_partNN.zip
판정이 끝나지 않은 표본은 labels.npz 가 없고 index.csv 의 judged = 0 이다.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import time
import zipfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ("train", "val", "test")


def read_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()] if p.exists() else []


def link(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    try:
        os.link(src, dst)                                    # 같은 디스크면 하드링크(복사 없음)
    except OSError:
        shutil.copy2(src, dst)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", default="big1")
    ap.add_argument("--extra", nargs="*", default=[], help="같은 데이터셋으로 합칠 추가 묶음 이름")
    ap.add_argument("--pack", type=float, default=0, help="분할별 zip 묶음 크기(GB). 0 이면 묶지 않음")
    a = ap.parse_args()
    out = ROOT / "data" / "export" / f"LUMI_hazard7_{a.name}"
    out.mkdir(parents=True, exist_ok=True)
    rows_all = []
    for split in SPLITS:
        (out / split).mkdir(exist_ok=True)
        gens, mans = [], []
        for nm in [a.name] + a.extra:
            d = ROOT / "data" / "hazard7" / nm / split
            if not d.exists():
                continue
            g = [r for r in read_jsonl(d / "gen.jsonl") if "error" not in r]
            m = {r["id"]: r for q in sorted(d.glob("manifest*.jsonl")) for r in read_jsonl(q) if "error" not in r}
            for r in g:
                for suf in (".mp4", ".clean.mp4", ".layers.npz", ".labels.npz"):
                    f = d / f"{r['id']}{suf}"
                    if f.exists():
                        link(f, out / split / f.name)
                gens.append(r)
                if r["id"] in m:
                    mans.append(m[r["id"]])
        with (out / split / "gen.jsonl").open("w", encoding="utf-8") as f:
            f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in gens)
        with (out / split / "manifest.jsonl").open("w", encoding="utf-8") as f:
            f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in mans)
        judged = {r["id"]: r for r in mans}
        for g in gens:
            rows_all.append(index_row(g, judged.get(g["id"])))
    sb = ROOT / "data" / "hazard7" / "south_bg_split.json"
    if sb.exists():
        shutil.copy2(sb, out / "south_bg_split.json")
    with (out / "index.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_all[0].keys()))
        w.writeheader(); w.writerows(rows_all)
    write_readme(out, rows_all, a)
    if a.pack:
        pack(out, a.pack)
    total = sum(p.stat().st_size for p in out.rglob("*") if p.is_file() and "zips" not in p.parts)
    print(f"정리: {out} · 표본 {len(rows_all)} · {total / 1e9:.1f} GB", flush=True)


def index_row(g: dict, m: dict | None) -> dict:
    rc, bg = g["recipe"], g.get("bg") or {}
    lay = g.get("layer") or {}
    return {"id": g["id"], "split": g["split"], "kind": g["kind"], "intent": g["intent"], "target_label": g["target_label"],
            "judged": int(m is not None), "hit_added": "" if m is None else int(bool(m.get("hit_added"))),
            "added_labels": "" if m is None else "|".join(m.get("added_labels") or []),
            "team_labels": "" if m is None else "|".join(m["team"]["labels"]),
            "clean_labels": "" if m is None else "|".join(m.get("clean_labels") or []),
            "ours_lum": "" if m is None else int(m["ours"]["lum"]), "ours_red": "" if m is None else int(m["ours"]["red"]),
            "ours_ext_lum": "" if m is None else int(m["ours"]["ext_lum"]), "ours_ext_red": "" if m is None else int(m["ours"]["ext_red"]),
            "bg_source": bg.get("source", ""), "bg_file": Path(str(bg.get("file", ""))).name, "bg_start": bg.get("start", ""),
            "mode": rc.get("mode", ""), "mask_mode": rc.get("mask_mode", ""), "rhythm": rc.get("rhythm", ""),
            "pattern": rc.get("pattern", ""), "near": rc.get("near") or "", "area": round(rc.get("area", 0), 4),
            "dur_s": round(rc.get("dur", 0), 3), "alpha": round(rc.get("alpha", rc.get("flat_alpha", 1.0)), 3),
            "bg_gain": round(rc.get("bg_gain", 1.0), 3), "seed": rc.get("seed", ""),
            "layer_fwd_p99": lay.get("fwd_p99", ""), "layer_lost_frac": lay.get("lost_frac", "")}


def write_readme(out: Path, rows: list[dict], a) -> None:
    c = Counter((r["split"]) for r in rows)
    cj = Counter(r["split"] for r in rows if r["judged"])
    kinds = Counter((r["split"], r["kind"], r["intent"]) for r in rows)
    hit = Counter((r["kind"]) for r in rows if r["judged"] and r["intent"] == "hazard" and r["hit_added"] == 1)
    nh = Counter((r["kind"]) for r in rows if r["judged"] and r["intent"] == "hazard")
    bgs = Counter((r["split"], r["bg_source"]) for r in rows)
    L = [f"# LUMI hazard7 데이터셋 `{a.name}`" + (f" (+ {', '.join(a.extra)})" if a.extra else ""), "",
         f"> 정리 {time.strftime('%Y-%m-%d %H:%M')} · 생성 코드 `lumi-lab-ryujungho2-collab` 의 `scripts/make_hazard7_dataset.py`, `src/lumi_synth/hazard7.py` · 실험 계획 `docs/CORRECTION_PROTOCOL.md`", "",
         "⚠️ **광과민성 위험 자극이 들어 있다.** 깜빡임·적색 점멸·줄무늬. 화면 밝기를 낮추고 작은 창으로 짧게 본다. 재배포·공개 금지(아래 라이선스).", "",
         "## 무엇인가", "",
         "광과민성 발작 위험 영상의 **탐지·보정 학습용 짝 데이터**. 표본마다 위험 영상(합성)과 원본(합성 전)이 짝이고, 위험 라벨과 합성 과정 자체(층)가 정답으로 붙어 있다.", "",
         "- 8 초 · 24 fps · 768×432 · H.264 crf 12 (yuv420p)",
         "- 라벨 체계 = 팀원 판별기 **Partial B2** (`lumi-lab-rockrovks/Partial/B2/vdcoreps`, 11-17 Update11)의 7종: 휘도 플래시 · 교집합 · 빠른 장면 전환 · 적색 플래시 · 휘도 연장 · 적색 연장 · 정적 공간 패턴",
         "- 라벨은 **저장한 mp4 를 다시 디코드해** 판별기로 붙였다(생성 의도가 아님). 원본만 판별한 라벨도 함께 있어, 합성으로 **새로 생긴 라벨**(`added`)을 쓸 수 있다 — 판별기가 깨끗한 바탕에서도 '정적 공간 패턴' 을 내는 경우가 있어서다.", "",
         "## 분할과 개수", "", "| 분할 | 표본 | 판정 완료 | 바탕 |", "|---|---|---|---|"]
    for s in SPLITS:
        L.append(f"| {s} | {c[s]} | {cj[s]} | " + ", ".join(f"{k[1]} {v}" for k, v in sorted(bgs.items()) if k[0] == s) + " |")
    L += ["", "바탕: movie = Big Buck Bunny(train 30–420 s · val 450–560 s) / Sintel(test, 학습에 없는 영화) · south = South 2021 안전 GIF·MP4(파일 해시 80:20 train/val, `south_bg_split.json`) · graphic = 합성 그래픽(그라데이션·도형·글자).", "",
          "종류 × 의도(분할별): 각 종류마다 hazard(위험을 노림) / near(기준 바로 아래를 노림) 같은 수.", "",
          "| 종류 | 목표 라벨 | hazard 적중(판정 끝난 것) |", "|---|---|---|"]
    lab = {"lum_flash": "휘도 플래시", "intersection": "교집합", "scene": "빠른 장면 전환", "red_flash": "적색 플래시",
           "lum_ext": "휘도 연장", "red_ext": "적색 연장", "pattern": "정적 공간 패턴"}
    for k, v in lab.items():
        L.append(f"| {k} | {v} | {hit[k]}/{nh[k]} |")
    L += ["", "## 파일", "",
          "| 파일 | 내용 |", "|---|---|",
          "| `<id>.mp4` | 합성한 위험 영상 (BGR 로 디코드) |",
          "| `<id>.clean.mp4` | 같은 인코딩의 원본(합성 전). 어두운 장면 레시피(`bg_gain` < 1)는 어둡게 한 것이 원본 |",
          "| `<id>.labels.npz` | `team` 7×T bool (합성본 프레임별 라벨) · `clean` 7×T (원본만 판별) · `added` = team ∧ ¬clean · `labels` 라벨 이름 · `weight` T×108×192 uint8 (Partial B2 픽셀 위험 마스크, 1/4) |",
          "| `<id>.layers.npz` | 합성의 아핀 층 X_lin = A·Y_lin + B (선형 빛, 1/4 해상도 T×108×192×3 float16). A ≈ 0 = 원본 정보 없음. 앞방향 오차는 index.csv `layer_fwd_p99` |",
          "| `*/manifest.jsonl` | 표본마다 레시피(재현 가능)·바탕·판정 요약·우리 판정기 결과 |",
          "| `*/gen.jsonl` | 생성 단계 기록(판정 전) |",
          "| `index.csv` | 위를 한 줄씩 요약 (엑셀로 열림) |", "",
          "## 읽는 법 (Python)", "", "```python",
          "import av, numpy as np",
          "def frames(p):",
          "    with av.open(p) as c: return np.stack([f.to_ndarray(format='bgr24') for f in c.decode(c.streams.video[0])])",
          "x, y = frames('train/<id>.mp4'), frames('train/<id>.clean.mp4')     # T×432×768×3",
          "lab = np.load('train/<id>.labels.npz'); added = lab['added']          # 7×T",
          "lay = np.load('train/<id>.layers.npz'); A, B = lay['A'], lay['B']   # T×108×192×3", "```", "",
          "## 다시 만들기·늘리기", "",
          "- 같은 표본: manifest 의 `recipe` + 바탕 정보로 `hazard7.render` 가 같은 영상을 만든다(seed 고정).",
          "- 더 만들기: `python scripts/make_hazard7_dataset.py --team-dir <rockrovks> --split train --per 200 --name big2 --seed 1` → id 가 `train_s1_...` 로 겹치지 않음 → `python scripts/export_hazard7.py --name big1 --extra big2`.", "",
          "## 라이선스·사용 제한", "",
          "- Big Buck Bunny, Sintel: © Blender Foundation, CC BY 3.0 (출처 표기).",
          "- **South 2021 (OSF 5a3dy) 바탕: 라이선스 표기 없음 → 내부 연구용, 재배포 금지.** 이 데이터셋을 팀 밖에 공유하거나 공개하지 않는다.",
          "- 합성 방식·라벨: LUMI 팀 내부.", "",
          "## 알려진 한계", "",
          "- 24 fps → 12 Hz 넘는 깜빡임 없음. 정답 판별기 하나(Partial B2)의 결과이며 임상 정답이 아니다.",
          "- 층 정답은 적색 물들이기(채널 섞임)·밝기 곱의 잘림에서 근사(`layer_fwd_p99` 가 큼)."]
    (out / "README.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def pack(out: Path, gb: float) -> None:
    zd = out / "zips"
    zd.mkdir(exist_ok=True)
    lim = gb * 1e9
    for split in SPLITS:
        files = sorted((out / split).iterdir())
        part, size, zf = 0, 0.0, None
        for f in files:
            if zf is None or size + f.stat().st_size > lim:
                if zf:
                    zf.close()
                part += 1; size = 0.0
                zf = zipfile.ZipFile(zd / f"{split}_part{part:02d}.zip", "w", zipfile.ZIP_STORED)
            zf.write(f, f"{split}/{f.name}"); size += f.stat().st_size
        if zf:
            zf.close()
    print("묶음:", zd, flush=True)


if __name__ == "__main__":
    main()
