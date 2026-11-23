"""CorrNet 앞부분을 고정한 채 탐지로 쓸 수 있는가 — 선형 탐침(linear probe) 실험 (2026-11-24, 사용자 요청).

질문: 합성 '층 벗기기'로 학습한 CorrNet(h_s0_e60)의 인코더·시공간 어텐션 특징이 실제 영상의 위험을 가르는가?
비교 (특징을 고정하고 로지스틱 회귀만 학습, South 2,000 영상 5 겹 층화 교차검증 AUC):
  A. CorrNet 시공간 특징(시공간 어텐션 출력의 공간·시간 평균/최대/시간 변화 에너지)
  B. CorrNet 위험 지도 head 의 zero-shot 점수(학습 없음: 시그모이드 최대·상위 1 % 평균)
  C. R(2+1)D-18 Kinetics-400 사전학습 특징(실제 영상으로 학습된 일반 특징, 기준선)
라벨: South 사람 라벨 dangerous / flash / pattern. Kaya 22(모두 위험)는 외부 확인: CV 로 정한 문턱에서 위험 판정 비율.
판단 규칙(결과 보기 전에 정함): A 의 AUC 가 C 보다 0.05 이상 높으면 'CorrNet 앞부분을 오탐 게이트로 쓴다',
  C 가 A 보다 0.05 이상 높으면 '합성 학습 특징은 실제로 일반화되지 않는다(이 방향 접음)', 그 사이면 '차이 없음'.
⚠️ 영상은 메모리에서만(재생·저장 없음). 특징은 data/probe/ 에 캐시.

    python scripts/probe_detection.py
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from lumi_judge.cpu_limit import limit_cpu  # noqa: E402
limit_cpu()
from eval_real_hazards import decode as dec_any  # noqa: E402
from lumi_synth.corrnet import CorrNet, srgb_to_lin  # noqa: E402

OUT = ROOT / "data" / "probe"
T_MAX = 192


def resample(x: np.ndarray, fps: float, fps_out: float = 24.0, t_max: int = T_MAX) -> np.ndarray:
    n = len(x)
    t = np.arange(min(t_max, max(1, int(n / fps * fps_out)))) / fps_out
    idx = np.minimum((t * fps).astype(int), n - 1)
    return x[idx]


def load_corrnet(name="h_s0_e60"):
    ck = torch.load(ROOT / "data" / "correction" / f"{name}.pt", map_location="cuda", weights_only=False)
    c = ck["config"]
    net = CorrNet(dim=c.get("dim", 96), depth=c.get("depth", 4), use_attn=not c.get("no_attn"), use_phys=not c.get("no_phys")).cuda().eval()
    net.load_state_dict(ck["model"])
    return net


@torch.no_grad()
def corrnet_feats(net, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    small = np.stack([cv2.resize(f, (384, 216), interpolation=cv2.INTER_AREA) for f in x])
    X = torch.from_numpy(small).cuda().permute(0, 3, 1, 2).float()[None] / 255.0
    b, t, _, H, W = X.shape
    with torch.autocast("cuda", dtype=torch.bfloat16):
        xl = srgb_to_lin(X)
        inp = torch.cat([X, xl], 2).flatten(0, 1)
        f1 = net.e1(inp); f2 = net.e2(f1); f3 = net.e3(f2)
        g = net.st(f3.view(b, t, *f3.shape[1:]))                     # [1,T,96,27,48] — 격자 특징
        out = net(X)
    g = g.float()[0]
    sp_mean = g.mean((2, 3))                                           # [T,96]
    feats = [sp_mean.mean(0), sp_mean.amax(0), g.amax((2, 3)).mean(0),
             (sp_mean[1:] - sp_mean[:-1]).abs().mean(0) if t > 1 else torch.zeros_like(sp_mean[0])]
    hz = torch.sigmoid(out["haz"].float()).flatten()
    k = max(1, int(0.01 * hz.numel()))
    zero = np.array([float(hz.max()), float(hz.topk(k).values.mean())])
    return torch.cat(feats).cpu().numpy(), zero


_R21 = None


@torch.no_grad()
def r21d_feats(x: np.ndarray) -> np.ndarray:
    global _R21
    import torchvision
    if _R21 is None:
        w = torchvision.models.video.R2Plus1D_18_Weights.KINETICS400_V1
        m = torchvision.models.video.r2plus1d_18(weights=w).cuda().eval()
        m.fc = torch.nn.Identity()
        _R21 = m
    fr = np.stack([cv2.resize(f, (171, 128), interpolation=cv2.INTER_AREA)[8:120, 30:142] for f in x])[..., ::-1].copy()
    v = torch.from_numpy(fr).cuda().float() / 255.0
    mean = torch.tensor([0.43216, 0.394666, 0.37645], device="cuda"); std = torch.tensor([0.22803, 0.22145, 0.216989], device="cuda")
    v = ((v - mean) / std).permute(3, 0, 1, 2)                           # [3,T,112,112]
    clips = [v[:, s:s + 16] for s in range(0, max(1, v.shape[1] - 15), 16)] or [v]
    clips = [c if c.shape[1] == 16 else F.pad(c, (0, 0, 0, 0, 0, 16 - c.shape[1]), mode="replicate") for c in clips]
    with torch.autocast("cuda", dtype=torch.bfloat16):
        f = _R21(torch.stack(clips)).float()
    return f.mean(0).cpu().numpy()


def extract():
    OUT.mkdir(parents=True, exist_ok=True)
    src = ROOT / "data" / "real" / "south2021"
    lab = {r["label"]: r for r in csv.DictReader(open(src / "social-media-study.csv", encoding="utf-8", errors="ignore"))}
    files = {p.stem: p for p in (src / "social-media-study-gifs").iterdir() if p.suffix.lower() in (".mp4", ".gif")}
    kaya = sorted(p for p in (ROOT / "data/real/kaya2025/extracted").glob("*.*") if p.suffix.lower() in (".mp4", ".avi"))
    items = [("south", k, files[k], lab[k]) for k in sorted(lab) if k in files] + [("kaya", p.stem, p, None) for p in kaya]
    cache = OUT / "feats.jsonl"
    done = {json.loads(l)["id"] for l in cache.read_text(encoding="utf-8").splitlines() if l.strip()} if cache.exists() else set()
    net = load_corrnet()
    with cache.open("a", encoding="utf-8") as f:
        for i, (srcn, k, p, L) in enumerate(items, 1):
            if f"{srcn}:{k}" in done:
                continue
            try:
                x, fps = dec_any(p)
                x = resample(x, fps)
                a, z = corrnet_feats(net, x)
                c = r21d_feats(x)
                row = {"id": f"{srcn}:{k}", "src": srcn, "A": a.round(5).tolist(), "B": z.round(5).tolist(), "C": c.round(5).tolist(),
                       "y": None if L is None else {q: L[q] for q in ("dangerous", "flash", "red", "pattern")}}
            except Exception as e:  # noqa: BLE001
                row = {"id": f"{srcn}:{k}", "src": srcn, "error": f"{type(e).__name__}: {e}"}
            f.write(json.dumps(row) + "\n"); f.flush()
            torch.cuda.empty_cache()
            if i % 100 == 0:
                print(f"  {i}/{len(items)}", flush=True)


def evaluate():
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    R = [json.loads(l) for l in (OUT / "feats.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    R = [r for r in R if "error" not in r]
    S = [r for r in R if r["src"] == "south"]; K = [r for r in R if r["src"] == "kaya"]
    L = ["# CorrNet 앞부분 고정 → 탐지 가능성 (선형 탐침)", "",
         f"> South {len(S)} 영상 5 겹 층화 CV, 로지스틱 회귀(표준화, C=1, class_weight=balanced) · Kaya {len(K)} 은 CV 문턱(위험 라벨, Youden)에서 위험 판정 비율 · 판단 규칙은 스크립트 머리말", "",
         "| 라벨 (양성 수) | A. CorrNet 특징 | B. CorrNet 위험 지도 zero-shot | C. R(2+1)D Kinetics | Kaya 위험 판정 A / C |", "|---|---|---|---|---|"]
    for lab, pos in (("dangerous", "dangerous"), ("flash", "flash"), ("pattern", "pattern")):
        y = np.array([r["y"][lab] == pos for r in S], int)
        row = [f"{lab} ({y.sum()})"]
        kay = {}
        for key in ("A", "C"):
            X = np.array([r[key] for r in S], np.float32)
            clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000))
            p = cross_val_predict(clf, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=0), method="predict_proba")[:, 1]
            auc = roc_auc_score(y, p)
            row.append(f"{auc:.3f}") if key == "A" else None
            if key == "A":
                zb = roc_auc_score(y, np.array([r["B"][1] for r in S]))
                row.append(f"{zb:.3f}")
            else:
                row.append(f"{auc:.3f}")
            if lab == "dangerous" and K:
                fpr_t = np.sort(np.unique(p))
                best = max(fpr_t, key=lambda th: ((p >= th) & (y == 1)).sum() / max(y.sum(), 1) - ((p >= th) & (y == 0)).sum() / max((y == 0).sum(), 1))
                clf.fit(X, y)
                kay[key] = int((clf.predict_proba(np.array([r[key] for r in K], np.float32))[:, 1] >= best).sum())
        row.append(f"{kay.get('A', '-')}/{len(K)} · {kay.get('C', '-')}/{len(K)}" if kay else "—")
        L.append("| " + " | ".join(row) + " |")
    p = ROOT / "checks" / "probe_detection.md"
    p.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    if "--eval-only" not in sys.argv:
        extract()
    evaluate()
