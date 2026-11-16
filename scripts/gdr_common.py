"""탐지·보정 검증 스크립트들이 함께 쓰는 데이터 읽기와 평가 (docs/GDR_VERIFY.md).

데이터: scripts/make_gdr_dataset.py 가 만든 data/gdr/<이름>/<폴더>/ (manifest.jsonl + <id>.npz).
원본(clean)은 data/oracle_768x432_24fps/backgrounds_<split>.npz 의 bg 번째 클립이다.
"""
from __future__ import annotations

import concurrent.futures as cf
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lumi_synth import generate, roi  # noqa: E402


def read_manifest(folder: Path) -> list[dict]:
    """manifest.jsonl → meta 목록(id 순). 각 meta 에 'folder' 를 붙인다."""
    rows = [json.loads(line) for line in (folder / "manifest.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    for r in rows:
        r["folder"] = str(folder)
    return sorted(rows, key=lambda r: r["id"])


def require_split(metas: list[dict], purpose: str) -> None:
    """데이터 구분을 강제한다. purpose='train': train 분할·G 생성 표본만. purpose='eval': train 분할 금지."""
    if not metas:
        raise SystemExit("표본이 없습니다")
    if purpose == "train":
        bad = [m["id"] for m in metas if m["split"] != "train" or (m["hazard"] and m["generator"] != "gan")]
        if bad:
            raise SystemExit(f"학습에는 train 분할의 G 표본만 씁니다. 섞인 표본 {len(bad)}개: {bad[:3]}")
    else:
        bad = [m["id"] for m in metas if m["split"] == "train"]
        if bad:
            raise SystemExit(f"평가에 train 분할 표본을 쓸 수 없습니다(학습 데이터): {len(bad)}개")


def group_of(meta: dict) -> str:
    """표본 묶음 이름: clean | <종류>_hazard | <종류>_near (생성 의도 기준 — 정답이 아니라 '어떤 시도였나')."""
    return "clean" if meta["hazard"] is None else f"{meta['hazard']}_{meta['intent']}"


def load_sample(meta: dict) -> dict:
    """meta → {'x': 입력 영상 uint8 T×H×W×3, 'clean': 원본, 'tubes': 3×T×H×W bool, 'mask_allow': H×W bool|None, 'meta'}."""
    clips, fps = generate.load_backgrounds(meta["split"])
    clean = clips[meta["bg"]]
    with np.load(Path(meta["folder"]) / f"{meta['id']}.npz") as z:
        tubes = z["tubes"]
        x = z["flicker"] if "flicker" in z else clean
        allow = z["mask_allow"] if "mask_allow" in z else None
    side = Path(meta["folder"]) / f"{meta['id']}.alpha.npz"            # 합성 비중 α (없으면 None — 바탕 그대로 등)
    alpha = None
    if side.exists():
        with np.load(side) as z:
            alpha = z["alpha"].astype(np.float32) / 255.0
    return {"x": x, "clean": clean, "tubes": tubes, "mask_allow": allow, "meta": meta, "fps": fps, "alpha": alpha}


def prefetch(metas: list[dict], threads: int = 4, ahead: int = 6):
    """표본을 순서대로 내놓되, 뒤의 것들을 스레드로 미리 읽어 둔다(npz 압축 풀기가 느림)."""
    for split in sorted({m["split"] for m in metas}):       # 바탕 파일은 스레드를 띄우기 전에 한 번만 읽는다
        generate.load_backgrounds(split)
    with cf.ThreadPoolExecutor(max_workers=threads) as pool:
        queue: list = []
        it = iter(metas)
        for m in it:
            queue.append(pool.submit(load_sample, m))
            if len(queue) > ahead:
                yield queue.pop(0).result()
        while queue:
            yield queue.pop(0).result()


def to_x(frames: list[np.ndarray], device) -> torch.Tensor:
    """[T×H×W×3 uint8] 목록 → [B,3,T,H,W] float 0~1 (BGR 그대로)."""
    return torch.from_numpy(np.stack(frames)).to(device).permute(0, 4, 1, 2, 3).float() / 255.0


@torch.no_grad()
def predict_tubes(net, x_u8: np.ndarray, device) -> np.ndarray:
    """영상 하나 → 튜브 확률 3×T×h×w float32 (1/4 해상도)."""
    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
        logits = net(to_x([x_u8], device))
    return torch.sigmoid(logits.float())[0].cpu().numpy()


def upsample_tubes(prob: np.ndarray, hw: tuple[int, int], thr: float = 0.5) -> np.ndarray:
    """튜브 확률 3×T×h×w → 전체 해상도 bool 3×T×H×W (확률을 쌍선형으로 키운 뒤 자름)."""
    t = torch.from_numpy(prob)
    up = torch.nn.functional.interpolate(t, size=hw, mode="bilinear", align_corners=False)
    return (up >= thr).numpy()


def eval_roi(net, metas: list[dict], device, thr: float = 0.5, log=None) -> list[dict]:
    """탐지기를 표본마다 재서 행 목록으로. IoU 는 1/4 해상도에서(정답은 면적 평균 ≥ 0.5)."""
    rows = []
    net.eval()
    for i, s in enumerate(prefetch(metas)):
        prob = predict_tubes(net, s["x"], device)
        true = roi.pool_tubes(torch.from_numpy(s["tubes"])[None])[0].numpy() >= 0.5
        row = {"id": s["meta"]["id"], "group": group_of(s["meta"])}
        for k, name in enumerate(roi.HAZARDS):
            pred = prob[k] >= thr
            row[name] = {"true": bool(true[k].any()), "pred": roi.clip_decision(prob[k], thr),
                         "iou": roi.tube_iou(pred, true[k]),
                         "pred_area": float(pred.mean()), "true_area": float(true[k].mean()),
                         # 프레임별 면적(문턱·지속 규칙을 나중에 바꿔 다시 판정하려고 저장 — analyze_clip_rule.py)
                         "pred_series": [round(float(v), 4) for v in pred.mean(axis=(1, 2))],
                         "true_series": [round(float(v), 4) for v in true[k].mean(axis=(1, 2))]}
        rows.append(row)
        if log and (i + 1) % 20 == 0:
            log(f"  평가 {i + 1}/{len(metas)}")
    return rows


def summarize_roi(rows: list[dict]) -> dict:
    """행 목록 → 종류별 {tp, fp, fn, tn, precision, recall, iou(정답이 있는 표본 평균), fp_area(정답이 빈 표본의 예측 면적 평균)}."""
    out = {}
    for name in roi.HAZARDS:
        c = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
        ious, fpa = [], []
        for r in rows:
            v = r[name]
            c["tp" if v["true"] and v["pred"] else "fn" if v["true"] else "fp" if v["pred"] else "tn"] += 1
            if v["true"]:
                ious.append(v["iou"])
            else:
                fpa.append(v["pred_area"])
        prec = c["tp"] / (c["tp"] + c["fp"]) if c["tp"] + c["fp"] else float("nan")
        rec = c["tp"] / (c["tp"] + c["fn"]) if c["tp"] + c["fn"] else float("nan")
        out[name] = {**c, "precision": prec, "recall": rec,
                     "iou": float(np.mean(ious)) if ious else float("nan"),
                     "fp_area": float(np.mean(fpa)) if fpa else float("nan")}
    return out
