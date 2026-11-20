"""보정 방법 비교 평가 (docs/CORRECTION_PROTOCOL.md §3.3·§4·§5).

    python scripts/eval_correction.py --team-dir <rockrovks clone> --split val --limit 30 --models h_s0
    python scripts/eval_correction.py --team-dir ... --split test --models h_s0 h_s1 h_s2 abl_noattn ...

방법
  B0 보정 안 함 · B0v B0 를 mp4v 로 왕복(B2 압축 손실 대조) · B1 전체 감광(Partial B2 가 위험이라 한 프레임에서 화면 전체 ±0.5 s 시간 평균, 선형 빛)
  B2 팀원 Partial B2 감광(partial_coreops_1112.dim_video_partial_b2, 자막 끔, 코드 수정 없이 호출)
  B3 국소 규칙(Partial B2 픽셀 위험 마스크 안에서 ±0.5 s 시간 평균)
  B4 FlashGuard 식 재구현(arXiv 2507.19692 초록: CIELAB 프레임 간 변화로 깜빡임 검출 → 휘도 낮추기 + 색 전환 시간 평균)
  모델: data/correction/<이름>.pt (384×216 에서 추론 → 보정량(잔차)을 원래 해상도로 키워 더함)
지표 (원래 해상도 768×432)
  안전: 보정 뒤 Partial B2 로 다시 판별해 '바탕에 없던 라벨'이 남았는가(무늬는 0.5 s 이상) · 우리 판정기(휘도·적색·IRIS 연장)
  충실도: PSNR·SSIM·LPIPS(원본 대비) — 영역: 바뀐 곳(층 정답 A<0.98 또는 B>0.005) / 복원 가능(바뀐 곳 ∩ A ≥ 0.05) / 정보 없음 / 바뀌지 않은 곳
  시간: 원본 흐름 워핑 오차 비율(출력/원본, 바뀐 곳) · 처리 시간
결과: checks/correction/eval_<split>_<tag>.md · _rows.jsonl (이어 하기)
⚠️ 영상은 광과민성 위험 자극이다. 재생·미리보기하지 않는다.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from lumi_synth import hazard7, quality  # noqa: E402
from lumi_synth.corrnet import CorrNet, lin_to_srgb, srgb_to_lin  # noqa: E402
from make_hazard7_dataset import FPS, H, W, decode, encode, ours_judge  # noqa: E402

LUT = hazard7._LUT
PAT = hazard7.LABELS.index("정적 공간 패턴")


def u8_from_lin(y: np.ndarray) -> np.ndarray:
    return hazard7.u8(y.astype(np.float32))


# ---------------- 규칙 기준선 ----------------

def time_mean(x_lin: np.ndarray, frames: np.ndarray, half: int = 12) -> np.ndarray:
    """프레임마다 ±half 프레임 이동 평균(선형) — frames 가 True 인 프레임만 바꿈."""
    cs = np.cumsum(np.concatenate([np.zeros_like(x_lin[:1]), x_lin], 0), 0)
    n = len(x_lin)
    out = x_lin.copy()
    for t in np.nonzero(frames)[0]:
        a, b = max(0, t - half), min(n, t + half + 1)
        out[t] = (cs[b] - cs[a]) / (b - a)
    return out


def b1_global(x: np.ndarray, team: dict) -> np.ndarray:
    fr = team["masks"].any(0)
    return u8_from_lin(time_mean(LUT[x], fr))


def b3_local(x: np.ndarray, team: dict) -> np.ndarray:
    xl = LUT[x]
    wm = team["weight_full"]                                     # T×H×W 0~1
    fr = wm.reshape(len(wm), -1).max(1) > 0.05
    avg = time_mean(xl, fr)
    return u8_from_lin(xl * (1 - wm[..., None]) + avg * wm[..., None])


def b4_flashguard(x: np.ndarray, thr: float = 20.0, dim: float = 0.6, ema: float = 0.25) -> np.ndarray:
    """FlashGuard 초록의 두 단계를 재구현: CIELAB 프레임 간 ΔE 가 thr 넘는 픽셀 = 깜빡임 →
    그 픽셀의 휘도를 dim 배(선형)로 낮추고, 색은 지수 이동 평균(ema)으로 전환을 부드럽게. 원 논문 코드가 아님(재구현)."""
    lab = np.stack([cv2.cvtColor(f, cv2.COLOR_BGR2LAB).astype(np.float32) for f in x])
    xl = LUT[x].copy()
    out = xl.copy()
    run = xl[0].copy()
    for t in range(1, len(x)):
        de = np.linalg.norm(lab[t] - lab[t - 1], axis=-1)
        m = (de > thr)[..., None].astype(np.float32)
        run = (1 - ema) * run + ema * xl[t]
        out[t] = xl[t] * (1 - m) + (run * dim) * m
    return u8_from_lin(out)


def mp4v_roundtrip(x: np.ndarray, tmp: Path) -> np.ndarray:
    """B0v: 입력을 팀원 B2 와 같은 OpenCV mp4v 로 저장했다가 다시 읽음 — B2 충실도에 섞인 압축 손실의 크기를 재는 대조군."""
    p = tmp.with_name(tmp.name + "_b0v.mp4")
    vw = cv2.VideoWriter(str(p), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (x.shape[2], x.shape[1]))
    for fr in x:
        vw.write(fr)
    vw.release()
    y = decode(p)
    p.unlink(missing_ok=True)
    return y


def b2_partial(x_path: Path, tmp: Path, team_dir: str) -> np.ndarray:
    b2 = Path(team_dir) / "Partial" / "B2"
    if str(b2) not in sys.path:
        sys.path.insert(0, str(b2))
    import partial_coreops_1112 as pc
    cfg = pc.B2PartialConfig(show_hazard_caption=False)
    out = tmp.with_suffix(".mp4")
    pc.dim_video_partial_b2(str(x_path), str(out), b2_config=cfg, device="cuda")
    y = decode(out)
    out.unlink(missing_ok=True)
    return y


# ---------------- 모델 ----------------
_NETS: dict = {}


def model_correct(name: str, x: np.ndarray) -> np.ndarray:
    if name not in _NETS:
        ck = torch.load(ROOT / "data" / "correction" / f"{name}.pt", map_location="cuda", weights_only=False)
        c = ck["config"]
        net = CorrNet(dim=c.get("dim", 96), depth=c.get("depth", 4), use_attn=not c.get("no_attn"), use_phys=not c.get("no_phys")).cuda().eval()
        net.load_state_dict(ck["model"])
        _NETS[name] = net
    net = _NETS[name]
    small = np.stack([cv2.resize(f, (384, 216), interpolation=cv2.INTER_AREA) for f in x])
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        X = torch.from_numpy(small).cuda().permute(0, 3, 1, 2).float()[None] / 255.0
        Yl = net(X)["Y"].float()[0]                                          # [T,3,h,w] 선형
        delta = Yl - srgb_to_lin(X[0])                                      # 보정량(잔차)
        up = torch.nn.functional.interpolate(delta, size=(H, W), mode="bilinear", align_corners=False)
        xl = torch.as_tensor(LUT, device="cuda")[torch.from_numpy(x).cuda().long()].permute(0, 3, 1, 2)
        out = (lin_to_srgb((xl + up).clamp(0, 1)) * 255).round().byte().permute(0, 2, 3, 1).cpu().numpy()
    return out


# ---------------- 판정·지표 ----------------

def team_judge(frames: np.ndarray, team_dir: str) -> dict:
    import make_hazard7_dataset as m
    m.VARIANT = "partial"
    j = m.team_judge(frames, team_dir)
    wm = j["weight"]
    j["weight_full"] = None if wm is None else np.stack([cv2.resize(f, (W, H), interpolation=cv2.INTER_LINEAR) for f in wm]).astype(np.float32) / 255.0
    return j


def safe_after(masks_after: np.ndarray, masks_clean: np.ndarray) -> tuple[bool, list]:
    added = masks_after & ~masks_clean
    labs = [hazard7.LABELS[i] for i in range(len(hazard7.LABELS)) if added[i].sum() >= (12 if i == PAT else 1)]
    return not labs, labs


def metrics(y: np.ndarray, clean: np.ndarray, regs: dict, flow, base_warp: float, lp_every: int = 4) -> dict:
    out = {}
    lp = quality.lpips_map(y[::lp_every], clean[::lp_every])
    ss = quality.ssim_map(y, clean)
    for k, m in regs.items():
        out[f"psnr_{k}"] = round(quality.psnr(y, clean, m), 3)
        out[f"ssim_{k}"] = round(quality.masked_mean(ss, m), 4)
        mm = m[::lp_every]
        out[f"lpips_{k}"] = round(quality.masked_mean(lp, mm), 4)
    we = quality.warp_error(y, flow, regs["changed"])
    out["warp_ratio"] = round(we / base_warp, 4) if base_warp and base_warp > 0 else float("nan")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--team-dir", required=True)
    ap.add_argument("--data", default="big1")
    ap.add_argument("--split", default="val")
    ap.add_argument("--models", nargs="*", default=[])
    ap.add_argument("--baselines", nargs="*", default=["B0", "B0v", "B1", "B2", "B3", "B4"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    d = ROOT / "data" / "hazard7" / a.data / a.split
    rows_g = {json.loads(l)["id"]: json.loads(l) for q in sorted(d.glob("manifest*.jsonl")) for l in q.read_text(encoding="utf-8").splitlines() if l.strip()}
    ids = sorted(k for k, r in rows_g.items() if "error" not in r)
    if a.limit:
        ids = ids[::max(1, len(ids) // a.limit)][:a.limit]
    tag = a.tag or "_".join(a.models) or "baselines"
    out_rows = ROOT / "checks" / "correction" / f"eval_{a.split}_{tag}_rows.jsonl"
    out_rows.parent.mkdir(parents=True, exist_ok=True)
    done = {(json.loads(l)["id"], json.loads(l)["method"]) for l in out_rows.read_text(encoding="utf-8").splitlines() if l.strip()} if out_rows.exists() else set()
    methods = a.baselines + a.models
    tmp = ROOT / "data" / "correction" / "tmp_eval"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    print(f"표본 {len(ids)} · 방법 {methods}", flush=True)
    with out_rows.open("a", encoding="utf-8") as f:
        for i, sid in enumerate(ids, 1):
            g = rows_g[sid]
            todo = [m for m in methods if (sid, m) not in done]
            if not todo:
                continue
            x, clean = decode(d / f"{sid}.mp4"), decode(d / f"{sid}.clean.mp4")
            with np.load(d / f"{sid}.labels.npz") as z:
                clean_masks = z["clean"]
            with np.load(d / f"{sid}.layers.npz") as z:
                A, B = z["A"].astype(np.float32), z["B"].astype(np.float32)
            up = lambda v: np.stack([cv2.resize(fr, (W, H), interpolation=cv2.INTER_NEAREST) for fr in v])
            Am, Bm = up(A.mean(-1)), up(B.max(-1))
            changed = (Am < 0.98) | (Bm > 0.005)
            regs = {"changed": changed, "recoverable": changed & (Am >= 0.05), "lost": changed & (Am < 0.05), "unchanged": ~changed}
            flow = quality.clean_flow(clean)
            base_warp = quality.warp_error(clean, flow, changed)
            team_x = team_judge(x, a.team_dir)
            for m in todo:
                t0 = time.time()
                try:
                    if m == "B0":
                        y = x
                    elif m == "B0v":
                        y = mp4v_roundtrip(x, tmp)
                    elif m == "B1":
                        y = b1_global(x, team_x)
                    elif m == "B2":
                        y = b2_partial(d / f"{sid}.mp4", tmp, a.team_dir)
                    elif m == "B3":
                        y = b3_local(x, team_x)
                    elif m == "B4":
                        y = b4_flashguard(x)
                    else:
                        y = model_correct(m, x)
                    sec = time.time() - t0
                    n = min(len(y), len(clean))
                    y = y[:n]
                    tj = team_judge(y, a.team_dir)
                    safe, left = safe_after(tj["masks"], clean_masks[:, :n])
                    oj = ours_judge(y)
                    row = {"id": sid, "method": m, "kind": g["kind"], "intent": g["intent"], "hazard_before": bool(g.get("added_labels")),
                           "labels_before": g.get("added_labels"), "safe_team": safe, "left_labels": left,
                           "safe_ours": not (oj["lum"] or oj["red"] or oj["ext_lum"] or oj["ext_red"]), "sec": round(sec, 2),
                           **metrics(y, clean[:n], {k: v[:n] for k, v in regs.items()}, flow[:n - 1], base_warp)}
                except Exception as e:  # noqa: BLE001
                    import traceback
                    row = {"id": sid, "method": m, "error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()[-800:]}
                f.write(json.dumps(row, ensure_ascii=False) + "\n"); f.flush()
            print(f"  {i}/{len(ids)} {sid}", flush=True)
    report(out_rows, ROOT / "checks" / "correction" / f"eval_{a.split}_{tag}.md", methods)


def report(rows_path: Path, out: Path, methods: list) -> None:
    rows = [json.loads(l) for l in rows_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    rows = [r for r in rows if "error" not in r]
    L = [f"# 보정 방법 비교 — {rows_path.stem.replace('_rows', '')}", "",
         f"> {time.strftime('%Y-%m-%d %H:%M')} · `scripts/eval_correction.py` · 값은 중앙값 · 안전 = 보정 뒤 Partial B2 에서 바탕에 없던 라벨이 없음", "",
         "## 위험 표본 (보정 전 Partial B2 가 새 라벨을 붙인 표본)", "",
         "| 방법 | n | 안전(팀원) | 안전(우리) | PSNR 바뀐 곳 | PSNR 복원 가능 | LPIPS 바뀐 곳 | SSIM 바뀐 곳 | PSNR 안 바뀐 곳 | 워핑 비율 | 초 |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    med = lambda v: float(np.nanmedian([x for x in v if x is not None and np.isfinite(x)])) if any(x is not None and np.isfinite(x) for x in v) else float("nan")
    for m in methods:
        rs = [r for r in rows if r["method"] == m and r["hazard_before"]]
        if not rs:
            continue
        L.append(f"| {m} | {len(rs)} | {sum(r['safe_team'] for r in rs)}/{len(rs)} | {sum(r['safe_ours'] for r in rs)}/{len(rs)} | "
                 f"{med([r['psnr_changed'] for r in rs]):.2f} | {med([r['psnr_recoverable'] for r in rs]):.2f} | {med([r['lpips_changed'] for r in rs]):.3f} | "
                 f"{med([r['ssim_changed'] for r in rs]):.3f} | {med([r['psnr_unchanged'] for r in rs]):.2f} | {med([r['warp_ratio'] for r in rs]):.2f} | {med([r['sec'] for r in rs]):.1f} |")
    L += ["", "## 경계 표본 (보정 전 안전) — 덜 건드릴수록 좋음", "", "| 방법 | n | 그대로 안전 | PSNR 바뀐 곳 | PSNR 안 바뀐 곳 |", "|---|---|---|---|---|"]
    for m in methods:
        rs = [r for r in rows if r["method"] == m and not r["hazard_before"]]
        if rs:
            L.append(f"| {m} | {len(rs)} | {sum(r['safe_team'] for r in rs)}/{len(rs)} | {med([r['psnr_changed'] for r in rs]):.2f} | {med([r['psnr_unchanged'] for r in rs]):.2f} |")
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("보고서:", out, flush=True)


if __name__ == "__main__":
    main()
