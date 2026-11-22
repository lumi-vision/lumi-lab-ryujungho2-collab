"""Inventory South and reserve Kaya for external evaluation without moving media."""
from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REAL = ROOT / 'data' / 'real'
OUT = ROOT / 'data' / 'splits' / 'south_all_train_kaya_external_v2'


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    south = REAL / 'south2021'
    media = south / 'social-media-study-gifs'
    kaya = REAL / 'kaya2025' / 'extracted'
    kaya_files = sorted(p for p in kaya.iterdir() if p.suffix.lower() in ('.mp4', '.avi'))
    if not kaya_files:
        raise ValueError('Kaya original videos missing')
    external = [{'path': p.relative_to(ROOT).as_posix(), 'sha256': sha256(p),
                 'dataset': 'kaya2025', 'split': 'external_test', 'label': None} for p in kaya_files]
    external_hashes = {x['sha256'] for x in external}
    files = defaultdict(list)
    for p in media.iterdir():
        if p.suffix.lower() in ('.gif', '.mp4'):
            files[p.stem].append(p)
    groups = defaultdict(list)
    with (south / 'social-media-study.csv').open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    if len({r['label'] for r in rows}) != len(rows):
        raise ValueError('duplicate CSV labels')
    for row in rows:
        matches = files[row['label']]
        if len(matches) != 1:
            raise ValueError(f"expected one file for {row['label']}, found {len(matches)}")
        p = matches[0]
        digest = sha256(p)
        groups[digest].append({'path': p.relative_to(ROOT).as_posix(), 'dataset': 'south2021',
            'id': row['label'], 'source': row['source'], 'sha256': digest,
            'label': row['dangerous'], 'flash': row['flash'], 'red': row['red'], 'pattern': row['pattern']})
    buckets = defaultdict(list)
    excluded = []
    for digest, entries in sorted(groups.items()):
        labels = {e['label'] for e in entries}
        if digest in external_hashes or len(labels) != 1 or not labels <= {'safe', 'dangerous'}:
            for e in entries:
                e['split'] = 'excluded_kaya_duplicate' if digest in external_hashes else 'excluded_label_conflict'
                excluded.append(e)
        else:
            buckets[next(iter(labels))].append(entries)
    assigned = []
    for label, grouped in sorted(buckets.items()):
        for g in grouped:
            for e in g:
                e['split'] = 'train'
                assigned.append(e)
    OUT.mkdir(parents=True, exist_ok=True)
    for name, data in [('south.jsonl', assigned + excluded), ('kaya_external.jsonl', external)]:
        with (OUT / name).open('x', encoding='utf8', newline='\n') as f:
            for row in data:
                f.write(json.dumps(row, ensure_ascii=False) + '\n')
    counts = Counter((r['split'], r['label']) for r in assigned + excluded)
    print('South:', dict(counts), 'unique byte hashes:', len(groups))
    print('Kaya external:', len(external), 'path:', OUT)


if __name__ == '__main__':
    main()
