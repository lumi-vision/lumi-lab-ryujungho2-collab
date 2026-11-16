"""학습한 매개변수(체크포인트) 목록: 무엇을, 어떤 데이터로, 어떻게 학습했고, 결과는 어디 있는가.

    python scripts/model_registry.py

읽는 곳: data/gan/*.pt (생성기 G), data/gdr/<버전>/*.pt (탐지기 roi_*, 손실 추정기 loss_*, 보정기 cor_*)
쓰는 곳: docs/MODELS.md (사람이 읽는 표), checks/models.json (기계용 — sha256 전체 포함)

체크포인트 파일은 data/ 안이라 Git 에 올라가지 않는다. 같은 이름으로 다시 학습하면 덮어쓰므로, 학습 스크립트는
이미 있는 이름이면 멈춘다(--overwrite 로만 덮어씀). 파일이 바뀌었는지는 이 목록의 sha256 으로 확인한다.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
HZ_KO = {"lum": "휘도", "red": "적색", "stripe": "줄무늬"}


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def gan_variant(c: dict) -> str:
    hz = c.get("hazard", "lum")
    bits = []
    if hz == "red":
        pal = c.get("red_palette", False)
        bits.append({False: "v1 장면↔같은 휘도 빨강", True: "v2 색 조건을 G 가 고름", "z": "v3 색 조건 무작위",
                     "z2": "v4 색 조건 무작위 + 등휘도"}.get(pal, str(pal)))
        if c.get("lambda_crisp"):
            bits.append(f"또렷함 λ {c['lambda_crisp']}")
    elif hz == "stripe":
        bits.append("v2 조건 무작위(정지·반전·흐름)" if c.get("stripe_cond") == "z" else "v1 조건을 G 가 고름(정지·흐름)")
    else:
        bits.append("v2 파형 흐림 무작위" if c.get("lum_smooth") else "v1")
    return ", ".join(bits)


def report_lines(path: Path, keys: tuple[str, ...]) -> dict:
    """평가 보고서 첫 표에서 keys 를 포함한 행의 '학습한 G' 칸을 뽑는다."""
    out = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        for k in keys:
            if line.startswith("|") and k in line and k not in out:
                cells = [c.strip().strip("*") for c in line.strip("|").split("|")]
                if len(cells) >= 2:
                    out[k] = cells[1]
    return out


def gan_rows() -> list[dict]:
    rows = []
    for p in sorted((ROOT / "data" / "gan").glob("*.pt"), key=lambda q: q.stat().st_mtime):
        ck = torch.load(p, map_location="cpu", weights_only=False)
        c = ck["config"]
        hz = c.get("hazard", "lum")
        rep = ROOT / "checks" / "gan" / f"{p.stem}.md"
        res = report_lines(rep, ("목표 '위험' →", "목표 '근처 안전' → 안전이면서", "목표 '근처 안전' → 안전 |"))
        rows.append({"kind": "G", "name": p.name, "path": str(p.relative_to(ROOT)), "sha256": sha(p),
                     "made": time.strftime("%Y-%m-%d %H:%M", time.localtime(p.stat().st_mtime)),
                     "hazard": hz, "variant": gan_variant(c),
                     "train": {"data": f"{c.get('data_track', '?')} backgrounds_train (영화 60클립)", "steps": c.get("steps"),
                               "batch": c.get("batch_size"), "width": c.get("width"), "zdim": c.get("zdim"),
                               "lambda_div": c.get("lambda_div"), "lambda_adv": c.get("lambda_adv"), "seed": c.get("seed"),
                               "train_hw": c.get("train_hw"), "fps": c.get("fps"), "judge_rule": c.get("judge_rule")},
                     "report": str(rep.relative_to(ROOT)) if rep.exists() else None, "result": res})
    return rows


def dataset_sources(ver_dir: Path) -> str:
    card = ROOT / "checks" / "gdr" / f"dataset_{ver_dir.name}.md"
    if not card.exists():
        return "?"
    txt = card.read_text(encoding="utf-8")
    m = re.search(r"## `train/`.*?- 생성원: ([^\n]+)", txt, re.S)
    return m.group(1) if m else "?"


def gdr_rows() -> list[dict]:
    rows = []
    for p in sorted((ROOT / "data" / "gdr").glob("v*/*.pt")):
        ck = torch.load(p, map_location="cpu", weights_only=False)
        c = ck["config"]
        ver = p.parent.name
        if p.stem.startswith("cor"):
            kind = "보정기"
            how = (f"목표 = 원본, 입력 = 영상 + {'화면 전체(E0, 마스크 없음)' if c.get('mask') == 'none' else '판정기 튜브(부드럽게)'}"
                   + (" + 정보 손실 지도(E3)" if c.get("loss_input") else "")
                   + f", 조각 {c.get('crop')}, 시간 손실 λ {c.get('lambda_t')}")
        elif c.get("target") == "loss":
            kind, how = "손실 추정기", f"목표 = α ≥ 0.95 (M_loss), 입력 특징 {c.get('features', 'rgb')}"
        else:
            kind, how = "탐지기", f"목표 = 판정기 튜브(휘도·적색·줄무늬), 입력 특징 {c.get('features', 'rgb')}"
        reps = sorted(str(q.relative_to(ROOT)) for q in (ROOT / "checks" / "gdr").glob(f"{ver}_{p.stem}_*.md"))
        reps += sorted(str(q.relative_to(ROOT)) for q in (ROOT / "checks" / "gdr").glob(f"{ver}_pipeline_*.md")) if p.stem.startswith("cor") else []
        rows.append({"kind": kind, "name": f"{ver}/{p.name}", "path": str(p.relative_to(ROOT)), "sha256": sha(p),
                     "made": time.strftime("%Y-%m-%d %H:%M", time.localtime(p.stat().st_mtime)),
                     "train": {"data": f"data/gdr/{ver}/train ({c.get('n_train')}개)", "sources": dataset_sources(p.parent),
                               "epochs": f"{c.get('epoch_done')}/{c.get('epochs')}", "width": c.get("width"), "lr": c.get("lr"),
                               "batch": c.get("batch_size"), "seed": c.get("seed")},
                     "how": how, "reports": reps})
    return rows


def main() -> int:
    g, d = gan_rows(), gdr_rows()
    (ROOT / "checks" / "models.json").write_text(json.dumps({"generators": g, "gdr": d}, ensure_ascii=False, indent=1), encoding="utf-8")
    L = ["# 학습한 모델 목록", "",
         f"> {time.strftime('%Y-%m-%d %H:%M')} · `python scripts/model_registry.py` 로 다시 만든다 · 기계용: `checks/models.json`", "",
         "- 매개변수는 체크포인트 파일(`.pt`)에 저장되어 있어 **다시 학습할 필요 없이** 불러 쓴다: 생성기는 `generate.gan_render` / 플레이어 '생성…', "
         "탐지기·보정기는 `scripts/eval_roi.py`·`eval_pipeline.py` 의 `--roi` `--cor` 등.",
         "- 파일은 `data/` 안이라 Git 에 없다. 이 PC 밖으로 옮길 때는 파일과 이 목록(sha256)을 함께 옮긴다.",
         "- 학습 스크립트는 같은 이름이 있으면 멈춘다(`--overwrite` 로만 덮어씀). 결과 보고서의 수치는 아래 sha256 의 파일로 낸 것이다.", "",
         "## 생성기 G (`data/gan/`)", "", "모두 768×432·24 fps, 학습 바탕 = `data/oracle_768x432_24fps/backgrounds_train.npz`(train 분할 60클립). "
         "평가는 val 분할 12클립 × 목표 2 × z 4 = 96개, 하드 판정.", "",
         "| 파일 | 종류 | 버전 | 스텝 | λ_div | z | 목표 위험 → 위험 | 근처 → 안전 | 만든 때 | sha256 |", "|---|---|---|---|---|---|---|---|---|---|"]
    for r in g:
        t, res = r["train"], r["result"]
        hz_res = next((v for k, v in res.items() if k.startswith("목표 '위험'")), "–")
        near = res.get("목표 '근처 안전' → 안전 |", "–")
        name = f"[`{r['name']}`]({'../' + r['report'].replace(chr(92), '/')})" if r["report"] else f"`{r['name']}`"
        L.append(f"| {name} | {HZ_KO.get(r['hazard'], r['hazard'])} | {r['variant']} | {t['steps']} | {t['lambda_div']} | {t['zdim']} | "
                 f"{hz_res} | {near} | {r['made']} | `{r['sha256'][:12]}` |")
    L += ["", "- `u_*`·`u2k_*`·`ufp32_*`: 2026-11-14 이전 판정 규칙(동시성 창 2프레임)으로 학습. 지금 규칙과 다르다.", "",
          "## 탐지·보정 (`data/gdr/<버전>/`)", "",
          "데이터 버전 구성은 [docs/GDR_VERIFY.md](GDR_VERIFY.md) '데이터 구분'과 `checks/gdr/dataset_<버전>.md`. 학습은 train 분할, 평가는 val·val_rule.", "",
          "| 파일 | 종류 | 학습 데이터 (생성원) | 학습 방법 | epoch | 결과 | 만든 때 | sha256 |", "|---|---|---|---|---|---|---|---|"]
    for r in d:
        t = r["train"]
        reps = " · ".join(f"[{Path(x).stem}]({'../' + x.replace(chr(92), '/')})" for x in r["reports"]) or "–"
        L.append(f"| `{r['name']}` | {r['kind']} | {t['data']}: {t['sources']} | {r['how']} | {t['epochs']} | {reps} | {r['made']} | `{r['sha256'][:12]}` |")
    (ROOT / "docs" / "MODELS.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"생성기 {len(g)}개 · 탐지·보정 {len(d)}개 → docs/MODELS.md, checks/models.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
