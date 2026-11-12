"""실제 위험 영상 데이터셋 받기 (재배포 금지, data/real/ 는 Git 제외).

  south : South, Saffo, Borkin, CHI 2021 — OSF 5a3dy (GIF 1,500 + MP4 500 + 라벨 CSV, 위험 163)
          라이선스 표기 없음 → 내부 연구용, 재배포 금지. 위험 163개는 시험 전용(학습 금지).
  kaya  : Kaya 외, SIViP 2025 — PSE_triggering_dataset.rar (약 2.43 GB), 데이터 라이선스 명시 없음 → 내부용.
⚠️ 받은 영상·GIF 는 광과민성 위험 자극을 포함한다. 열어 보거나 재생하지 않는다. 판정기에만 넣는다.
사용: python scripts/download_real_hazards.py south|kaya
"""
from __future__ import annotations

import json
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "real"
UA = {"User-Agent": "lumi-research-downloader/0.1"}
KAYA_URL = "http://web.cs.hacettepe.edu.tr/~aydinkaya/pse/dataset/PSE_triggering_dataset.rar"


def get_json(url: str) -> dict:
    for attempt in range(8):                         # OSF API 가 가끔 500 을 낸다
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
                return json.load(r)
        except Exception as e:  # noqa: BLE001
            print(f"  목록 재시도 {attempt + 1}: {e}", flush=True)
            time.sleep(15 * (attempt + 1))
    raise RuntimeError(f"목록 받기 실패: {url}")


def download(url: str, dest: Path, size: int | None = None) -> None:
    if dest.exists() and (size is None or dest.stat().st_size == size):
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(5):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=120) as r, open(tmp, "wb") as f:
                while chunk := r.read(1 << 20):
                    f.write(chunk)
            tmp.replace(dest)
            return
        except Exception as e:  # noqa: BLE001
            print(f"  재시도 {attempt + 1}: {dest.name} ({e})", flush=True)
            time.sleep(10 * (attempt + 1))
    raise RuntimeError(f"받기 실패: {url}")


def walk_osf(url: str, rel: Path, files: list) -> None:
    while url:
        page = get_json(url)
        for item in page["data"]:
            a = item["attributes"]
            if a["kind"] == "folder":
                walk_osf(item["relationships"]["files"]["links"]["related"]["href"], rel / a["name"], files)
            else:
                files.append((rel / a["name"], item["links"]["download"], a.get("size")))
        url = page["links"].get("next")


def south() -> None:
    dest = OUT / "south2021"
    dest.mkdir(parents=True, exist_ok=True)
    cache = dest / "file_list.json"
    if cache.exists():                               # 이미 받은 목록이 있으면 다시 묻지 않는다
        files = [(Path(p), u, s) for p, u, s in json.loads(cache.read_text())]
    else:
        files = []
        walk_osf("https://api.osf.io/v2/nodes/5a3dy/files/osfstorage/", Path(), files)
        cache.write_text(json.dumps([[str(p), u, s] for p, u, s in files], ensure_ascii=False, indent=0))
    total = sum(s or 0 for _, _, s in files)
    print(f"South 2021: 파일 {len(files)}개, {total / 1e9:.2f} GB", flush=True)
    # 라벨 CSV 를 먼저 받고, 위험 라벨(dangerous) 163개 → 나머지 순서로 받는다 (중간에 끊겨도 위험 영상부터 확보)
    import csv
    meta = [f for f in files if not str(f[0]).lower().endswith((".gif", ".mp4"))]
    for rel, url, size in meta:
        download(url, dest / rel, size)
    with open(dest / "social-media-study.csv", newline="", encoding="utf-8-sig") as f:
        danger = {r["label"] for r in csv.DictReader(f) if r["dangerous"] == "dangerous"}
    media = [f for f in files if f not in meta]
    files = sorted(media, key=lambda f: Path(f[0]).stem not in danger)       # 위험 먼저 (안정 정렬)
    print(f"위험 라벨 {sum(Path(f[0]).stem in danger for f in files)}개를 먼저 받음", flush=True)
    failed = []
    for i, (rel, url, size) in enumerate(files, 1):
        try:
            download(url, dest / rel, size)
        except RuntimeError:
            failed.append(str(rel))                  # 서버 오류(403/500)는 기록하고 넘어감 — 다시 실행하면 이어 받음
        time.sleep(0.3)                              # OSF 요청 간격
        if i % 100 == 0:
            print(f"  {i}/{len(files)} (실패 {len(failed)})", flush=True)
    (dest / "failed.json").write_text(json.dumps(failed, ensure_ascii=False, indent=0))
    print(f"끝: {dest} · 실패 {len(failed)}개", flush=True)


def kaya() -> None:
    dest = OUT / "kaya2025" / "PSE_triggering_dataset.rar"
    download(KAYA_URL, dest, 2429272890)
    print("끝:", dest, dest.stat().st_size, flush=True)


if __name__ == "__main__":
    {"south": south, "kaya": kaya}[sys.argv[1]]()
