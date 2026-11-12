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
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
        return json.load(r)


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
            time.sleep(3 * (attempt + 1))
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
    files: list = []
    walk_osf("https://api.osf.io/v2/nodes/5a3dy/files/osfstorage/", Path(), files)
    total = sum(s or 0 for _, _, s in files)
    print(f"South 2021: 파일 {len(files)}개, {total / 1e9:.2f} GB", flush=True)
    dest = OUT / "south2021"
    (dest / "file_list.json").parent.mkdir(parents=True, exist_ok=True)
    (dest / "file_list.json").write_text(json.dumps([[str(p), u, s] for p, u, s in files], ensure_ascii=False, indent=0))
    for i, (rel, url, size) in enumerate(files, 1):
        download(url, dest / rel, size)
        if i % 100 == 0:
            print(f"  {i}/{len(files)}", flush=True)
    print("끝:", dest, flush=True)


def kaya() -> None:
    dest = OUT / "kaya2025" / "PSE_triggering_dataset.rar"
    download(KAYA_URL, dest, 2429272890)
    print("끝:", dest, dest.stat().st_size, flush=True)


if __name__ == "__main__":
    {"south": south, "kaya": kaya}[sys.argv[1]]()
