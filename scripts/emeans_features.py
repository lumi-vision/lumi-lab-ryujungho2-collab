"""E-Means 검증 노트북용 특징 추출 (한 번만 실행, 결과는 data/emeans/ 에 캐시).

두 가지 크기의 특징을 만든다.
  1) 패치 특징 (수백만 개): 16×16 패치마다 휘도 시간 신호의 '깜빡임 지문'
       [평균 밝기, 밝기 범위, |FFT| 1~24 bin(24 fps·48프레임 → 0.5 Hz 간격, 0.5~12 Hz)]  (모두 cd/m² ÷ 200)
     → 군집화 알고리즘 비교(E-Means 의 속도 장점이 의미 있는 규모)와 ROI 후보 묶기에 쓴다.
  2) 클립 특징 (수천 개): 마스크 안 '넣은 빛'(flicker − clean)의 평균 시간 신호 지문 + 판정 결과
       [평균 추가 빛, 추가 빛 범위, 정규화 |FFT| 1~24, 마스크 면적, 최대 깜빡임 수/초, 최대 동시 면적]
     → 데이터 커버리지·불균형 분석(오라클 대 GAN, λ_div 별)에 쓴다.

입력 (재생하지 않음, 배열만 읽음)
  - 오라클: data/oracle_768x432_24fps/{train,val,test}/*.npz  (352개)
  - GAN 평가 샘플: data/gan/<실험>/val/*.npz                   (실험마다 96개)
출력
  - data/emeans/patch_feats.npy (float32 N×26), patch_meta.npz (클립 번호, 패치 위치, 위험 패치 여부)
  - data/emeans/clip_feats.npy (float32 M×29), clip_meta.csv (출처·조건·판정)
사용: python scripts/emeans_features.py  [--workers 12]
"""
from __future__ import annotations

import argparse
import csv
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from lumi_synth import measure

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "emeans"
P = 16                      # 패치 크기 (px)
NBIN = 24                   # FFT bin 1~24
W_CD = measure.WHITE_CD     # 200 cd/m² 로 나눠 0~1 근처로


def patch_signals(lum: np.ndarray) -> np.ndarray:
    """(T,H,W) cd/m² → (T, H/P, W/P) 패치 평균."""
    t, h, w = lum.shape
    return lum[:, : h // P * P, : w // P * P].reshape(t, h // P, P, w // P, P).mean(axis=(2, 4))


def fingerprint(sig: np.ndarray) -> np.ndarray:
    """(..., T) 시간 신호 → (..., 26) [평균, 범위, |FFT| 1~24] (÷200)."""
    mean = sig.mean(-1, keepdims=True)
    rng = sig.max(-1, keepdims=True) - sig.min(-1, keepdims=True)
    spec = np.abs(np.fft.rfft(sig - mean, axis=-1))[..., 1:NBIN + 1] / sig.shape[-1]
    return np.concatenate([mean, rng, spec], -1) / W_CD


def one(job):
    path, source, cond = job
    z = np.load(path)
    fl, cl = z["flicker"], z["clean"]
    mask = z["mask_allow"].astype(bool)
    hmask = z["hazard_mask"].astype(bool) if "hazard_mask" in z else np.zeros_like(mask)
    lum = measure.luminance_cd(fl)                              # (T,H,W)
    ps = patch_signals(lum)                                     # (T,h,w)
    pf = fingerprint(np.moveaxis(ps, 0, -1)).reshape(-1, 2 + NBIN).astype(np.float32)
    ph, pw = ps.shape[1:]
    hz = hmask[: ph * P, : pw * P].reshape(ph, P, pw, P).mean(axis=(1, 3)).reshape(-1) > 0.5
    yy, xx = np.meshgrid(np.arange(ph), np.arange(pw), indexing="ij")
    # 클립 특징: 마스크 안 넣은 빛의 평균 시간 신호
    add = (lum - measure.luminance_cd(cl))[:, mask].mean(axis=1) if mask.any() else np.zeros(len(lum))
    f = fingerprint(add)
    spec = f[2:] / (np.linalg.norm(f[2:]) + 1e-9)               # 리듬 모양(세기와 분리)
    r = measure.analyze(fl, 24.0)
    clip = np.concatenate([f[:2], spec, [mask.mean(), r["max_flashes"] / 12.0, r["peak_area"]]]).astype(np.float32)
    meta = {"source": source, "cond": cond, "file": str(path.relative_to(ROOT)), "dangerous": int(bool(r["dangerous"])),
            "max_flashes": float(r["max_flashes"]), "peak_area": float(r["peak_area"]), "mask_area": float(mask.mean()),
            "mean_add_cd": float(add.mean())}
    return pf, hz, yy.reshape(-1), xx.reshape(-1), clip, meta


def jobs():
    for split in ("train", "val", "test"):
        for p in sorted((ROOT / "data" / "oracle_768x432_24fps" / split).glob("*.npz")):
            yield p, "oracle", f"{split}:{p.stem.split('_', 2)[2]}"
    for d in sorted((ROOT / "data" / "gan").iterdir()):
        if (d / "val").is_dir():
            for p in sorted((d / "val").glob("*.npz")):
                yield p, "gan", f"{d.name}:{'hazard' if '_hazard_' in p.stem else 'near'}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    js = list(jobs())
    print(f"클립 {len(js)}개 처리", flush=True)
    pfs, hzs, ys, xs, cids, clips, metas = [], [], [], [], [], [], []
    with ProcessPoolExecutor(args.workers) as ex:
        for i, (pf, hz, yy, xx, clip, meta) in enumerate(ex.map(one, js, chunksize=4)):
            pfs.append(pf); hzs.append(hz); ys.append(yy); xs.append(xx)
            cids.append(np.full(len(pf), i, np.int32)); clips.append(clip); metas.append(meta)
            if (i + 1) % 100 == 0:
                print(f"  {i + 1}/{len(js)}", flush=True)
    np.save(OUT / "patch_feats.npy", np.concatenate(pfs))
    np.savez(OUT / "patch_meta.npz", clip=np.concatenate(cids), hazard=np.concatenate(hzs),
             py=np.concatenate(ys).astype(np.int16), px=np.concatenate(xs).astype(np.int16))
    np.save(OUT / "clip_feats.npy", np.stack(clips))
    with open(OUT / "clip_meta.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(metas[0]))
        w.writeheader(); w.writerows(metas)
    info = {"patch": P, "nbin": NBIN, "n_clips": len(js), "n_patches": int(sum(len(p) for p in pfs)),
            "patch_dims": ["mean", "range"] + [f"fft_{(i + 1) * 0.5:.1f}Hz" for i in range(NBIN)],
            "clip_dims": ["add_mean", "add_range"] + [f"rhythm_{(i + 1) * 0.5:.1f}Hz" for i in range(NBIN)]
                         + ["mask_area", "max_flashes/12", "peak_area"]}
    (OUT / "info.json").write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: info[k] for k in ("n_clips", "n_patches")}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
