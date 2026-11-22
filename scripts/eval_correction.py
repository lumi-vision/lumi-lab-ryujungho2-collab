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


def limit_cpu(mask: int = 0x00FF) -> None:
    """2026-11-23 사용자 요청: CPU 사용률·온도를 낮춘다 — 논리 코어 8 개(0xFF)·우선순위 '보통 이하'·스레드 수 제한.
    결과에는 영향 없음(속도만). 환경변수 LUMI_CPU_MASK 로 바꿀 수 있다."""
    import os
    mask = int(os.environ.get("LUMI_CPU_MASK", str(mask)), 0)
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.GetCurrentProcess.restype = wintypes.HANDLE
        k32.SetProcessAffinityMask.argtypes = [wintypes.HANDLE, ctypes.c_size_t]
        k32.SetPriorityClass.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        h = k32.GetCurrentProcess()
        k32.SetProcessAffinityMask(h, mask)
        k32.SetPriorityClass(h, 0x00004000)                  # BELOW_NORMAL_PRIORITY_CLASS
    n = max(1, bin(mask).count("1") // 2)
    torch.set_num_threads(n)
    cv2.setNumThreads(n)


limit_cpu()
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


# 우리 판정기(numpy, 방법당 ~11 s CPU)는 스레드에서 GPU 작업과 겹쳐 돌린다(결과 동일, 실행 순서만 다름)
import concurrent.futures as _cf  # noqa: E402
POOL = _cf.ThreadPoolExecutor(3)
POOL_DEPTH = 4                                             # 끝나지 않은 판정이 이보다 많으면 기다림(메모리 상한)

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
    torch.cuda.empty_cache()
    return out


# ---------------- 판정·지표 ----------------

def team_judge(frames: np.ndarray, team_dir: str) -> dict:
    import make_hazard7_dataset as m
    m.VARIANT = "partial"
    j = m.team_judge(frames, team_dir)
    wm = j["weight"]
    j["weight_full"] = None if wm is None else np.stack([cv2.resize(f, (frames.shape[2], frames.shape[1]), interpolation=cv2.INTER_LINEAR) for f in wm]).astype(np.float32) / 255.0
    return j


def safe_after(masks_after: np.ndarray, masks_clean: np.ndarray) -> tuple[bool, list]:
    added = masks_after & ~masks_clean
    labs = [hazard7.LABELS[i] for i in range(len(hazard7.LABELS)) if added[i].sum() >= (12 if i == PAT else 1)]
    return not labs, labs


def metrics(y: np.ndarray, clean: np.ndarray, regs: dict, flow, base_warp: float, lp_every: int = 4) -> dict:
    out = {}
    lp = quality.lpips_map(y[::lp_every], clean[::lp_every])
    ss = quality.ssim_map(y, clean)
    se = ((y.astype(np.float32) - clean.astype(np.float32)) ** 2).sum(-1)   # 픽셀별 채널 합 — quality.psnr 과 같은 값, 한 번만 계산
    for k, m in regs.items():
        if not m.any():
            out[f"psnr_{k}"] = float("nan")
        else:
            mse = float(se[m].sum() / (m.sum() * 3))
            out[f"psnr_{k}"] = round(float("inf") if mse == 0 else float(10 * np.log10(255.0 ** 2 / mse)), 3)
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
    ap.add_argument("--real", default="", choices=("", "kaya"), help="kaya: 실제 위험 영상 22 개(원본 없음)")
    ap.add_argument("--shard", default="0/1", help="i/k: 표본을 k 조각으로 나눠 i 번째만(여러 프로세스가 GPU 를 함께 씀). 행 파일은 조각별")
    ap.add_argument("--report-only", action="store_true", help="조각별 행 파일을 모아 보고서만 다시 씀")
    ap.add_argument("--save-models", nargs="*", default=[], help="(Kaya) 이 방법들의 보정 결과 영상을 data/correction/kaya_out/<방법>/ 에 저장(육안 확인용)")
    a = ap.parse_args()
    a.si, a.sk = (int(v) for v in a.shard.split("/"))
    if a.real:
        return main_real(a)
    d = ROOT / "data" / "hazard7" / a.data / a.split
    rows_g = {json.loads(l)["id"]: json.loads(l) for q in sorted(d.glob("manifest*.jsonl")) for l in q.read_text(encoding="utf-8").splitlines() if l.strip()}
    ids = sorted(k for k, r in rows_g.items() if "error" not in r)
    if a.limit:
        ids = ids[::max(1, len(ids) // a.limit)][:a.limit]
    tag = a.tag or "_".join(a.models) or "baselines"
    methods = a.baselines + a.models
    base_rows = ROOT / "checks" / "correction" / f"eval_{a.split}_{tag}_rows.jsonl"
    if a.report_only:
        return report(merge_rows(base_rows), ROOT / "checks" / "correction" / f"eval_{a.split}_{tag}.md", methods)
    ids = ids[a.si::a.sk]
    out_rows = base_rows if a.sk == 1 else base_rows.with_name(base_rows.stem + f".s{a.si}.jsonl")
    out_rows.parent.mkdir(parents=True, exist_ok=True)
    done = done_set(base_rows, "id")                       # 조각 수를 바꿔 다시 띄워도 끝난 (표본, 방법)은 건너뜀
    tmp = ROOT / "data" / "correction" / f"tmp_eval_s{a.si}"
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
            pend = []                                              # (행, 우리 판정 future) — CPU 판정을 GPU 작업과 겹쳐 돌림
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
                    fut = POOL.submit(ours_judge, y)
                    tj = team_judge(y, a.team_dir)
                    safe, left = safe_after(tj["masks"], clean_masks[:, :n])
                    row = {"id": sid, "method": m, "kind": g["kind"], "intent": g["intent"], "hazard_before": bool(g.get("added_labels")),
                           "labels_before": g.get("added_labels"), "safe_team": safe, "left_labels": left, "sec": round(sec, 2),
                           **metrics(y, clean[:n], {k: v[:n] for k, v in regs.items()}, flow[:n - 1], base_warp)}
                    pend.append((row, fut))
                except Exception as e:  # noqa: BLE001
                    import traceback
                    row = {"id": sid, "method": m, "error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()[-800:]}
                    pend.append((row, None))
                y = None
                while sum(1 for _, q in pend if q is not None and not q.done()) > POOL_DEPTH:
                    time.sleep(0.2)
            for row, fut in pend:
                if fut is not None:
                    try:
                        oj = fut.result()
                        row["safe_ours"] = not (oj["lum"] or oj["red"] or oj["ext_lum"] or oj["ext_red"])
                    except Exception as e:  # noqa: BLE001
                        row = {"id": row["id"], "method": row["method"], "error": f"ours_judge {type(e).__name__}: {e}"}
                f.write(json.dumps(row, ensure_ascii=False) + "\n"); f.flush()
            print(f"  {i}/{len(ids)} {sid}", flush=True)
    if a.sk == 1:
        report(out_rows, ROOT / "checks" / "correction" / f"eval_{a.split}_{tag}.md", methods)


def done_set(base: Path, key: str) -> set:
    """<base> 와 모든 조각 파일에서 오류 없이 끝난 (key, method)."""
    out = set()
    for q in sorted(base.parent.glob(base.stem + ".s*.jsonl")) + ([base] if base.exists() else []):
        for l in q.read_text(encoding="utf-8").splitlines():
            if l.strip():
                r = json.loads(l)
                if "error" not in r:
                    out.add((r[key], r["method"]))
    return out


def merge_rows(base: Path) -> Path:
    """조각별 행 파일(<base>.s<i>.jsonl)과 <base> 를 하나로 모은다."""
    parts = sorted(base.parent.glob(base.stem + ".s*.jsonl")) + ([base] if base.exists() else [])
    merged = base.with_name(base.stem + ".merged.jsonl")
    merged.write_text("".join(q.read_text(encoding="utf-8") for q in parts), encoding="utf-8")
    return merged


# ---------------- 실제 영상 (Kaya 2025 원본 22 개, 원본 없음) ----------------

def model_correct_long(name: str, x: np.ndarray, chunk: int = 192, over: int = 48) -> np.ndarray:
    """긴 영상: 192 프레임 조각(48 겹침)으로 나눠 보정하고 겹친 곳은 선형으로 섞는다. 임의 해상도 → 384×216 에서 추론."""
    n, h, w = x.shape[:3]
    if n <= chunk:
        return model_correct_res(name, x)
    acc = np.zeros((n, h, w, 3), np.float32)
    wsum = np.zeros((n, 1, 1, 1), np.float32)
    s = 0
    while s < n:
        e = min(n, s + chunk)
        s0 = max(0, e - chunk)
        y = model_correct_res(name, x[s0:e]).astype(np.float32)
        wt = np.ones(e - s0, np.float32)
        r = min(over, e - s0)
        if s0 > 0:
            wt[:r] = np.linspace(0, 1, r)
        acc[s0:e] += y * wt[:, None, None, None]
        wsum[s0:e, 0, 0, 0] += wt
        if e == n:
            break
        s = e - over
    return (acc / np.maximum(wsum, 1e-6)).round().clip(0, 255).astype(np.uint8)


def model_correct_res(name: str, x: np.ndarray) -> np.ndarray:
    """model_correct 의 임의 해상도판(보정량을 입력 해상도로 키움)."""
    global H, W
    h0, w0 = H, W
    H, W = x.shape[1], x.shape[2]
    try:
        return model_correct(name, x)
    finally:
        H, W = h0, w0


def main_real(a) -> None:
    sys.path.insert(0, str(ROOT / "scripts"))
    from eval_real_hazards import decode as dec_any
    files = sorted((ROOT / "data" / "real" / "kaya2025" / "extracted").glob("*.*"), key=lambda p: int(p.stem) if p.stem.isdigit() else 999)
    files = [p for p in files if p.suffix.lower() in (".mp4", ".avi")]
    if a.limit:
        files = files[:a.limit]
    tag = a.tag or "_".join(a.models) or "baselines"
    base_rows = ROOT / "checks" / "correction" / f"eval_kaya_{tag}_rows.jsonl"
    methods = [m for m in a.baselines if m not in ("B0v",)] + a.models
    if a.report_only:
        return report_real(merge_rows(base_rows), ROOT / "checks" / "correction" / f"eval_kaya_{tag}.md", methods)
    files = files[a.si::a.sk]
    out_rows = base_rows if a.sk == 1 else base_rows.with_name(base_rows.stem + f".s{a.si}.jsonl")
    done = done_set(base_rows, "file")
    tmp = ROOT / "data" / "correction" / f"tmp_eval_s{a.si}"
    import make_hazard7_dataset as mk
    print(f"Kaya {len(files)} · 방법 {methods}", flush=True)
    with out_rows.open("a", encoding="utf-8") as f:
        for i, p in enumerate(files, 1):
            todo = [m for m in methods if (p.name, m) not in done]
            if not todo:
                continue
            x, fps = dec_any(p)                                        # 가로 640 이하
            mk.FPS = fps                                               # 판별기에 원래 fps
            tj0 = team_judge(x, a.team_dir)
            lab0 = sorted(tj0["labels"])
            xl = LUT[x]
            dl0 = float(np.abs(np.diff(xl.mean(-1), axis=0)).mean())
            for m in todo:
                t0 = time.time()
                try:
                    if m == "B0":
                        y = x
                    elif m == "B1":
                        y = b1_global(x, tj0)
                    elif m == "B2":
                        src = tmp.with_name(f"kaya_{p.stem}_in.mp4")
                        encode_any(x, src, fps)
                        y = b2_partial(src, tmp, a.team_dir)
                        src.unlink(missing_ok=True)
                    elif m == "B3":
                        y = b3_local(x, tj0)
                    elif m == "B4":
                        y = b4_flashguard(x)
                    else:
                        y = model_correct_long(m, x)
                    sec = time.time() - t0
                    n = min(len(x), len(y)); y = y[:n]
                    if m in a.save_models:                         # 육안 확인용(사용자가 본다). 원본 해상도·fps
                        od = ROOT / "data" / "correction" / "kaya_out" / m
                        od.mkdir(parents=True, exist_ok=True)
                        encode_any(y, od / f"{p.stem}.mp4", fps)
                    tj = team_judge(y, a.team_dir)
                    oj = ours_judge_fps(y, fps)
                    lp = quality.lpips_map(y[:n:8], x[:n:8])
                    ev = np.ones(y.shape[:3], bool)
                    row = {"file": p.name, "method": m, "fps": round(fps, 2), "frames": n, "labels_before": lab0,
                           "labels_after": sorted(tj["labels"]),
                           "flash_labels_after": sorted(l for l in tj["labels"] if l != "정적 공간 패턴"),
                           "safe_team": not tj["labels"], "safe_team_flash": not [l for l in tj["labels"] if l != "정적 공간 패턴"],
                           "safe_ours": not (oj["lum"] or oj["red"] or oj["ext_lum"] or oj["ext_red"]),
                           "lpips_vs_input": round(float(lp.mean()), 4), "psnr_vs_input": round(quality.psnr(y, x[:n], ev), 3),
                           "flicker_ratio": round(float(np.abs(np.diff(LUT[y].mean(-1), axis=0)).mean()) / max(dl0, 1e-6), 4),
                           "sec": round(sec, 2)}
                except Exception as e:  # noqa: BLE001
                    import traceback
                    row = {"file": p.name, "method": m, "error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()[-800:]}
                f.write(json.dumps(row, ensure_ascii=False) + "\n"); f.flush()
            print(f"  {i}/{len(files)} {p.name}", flush=True)
    if a.sk == 1:
        report_real(out_rows, ROOT / "checks" / "correction" / f"eval_kaya_{tag}.md", methods)


def encode_any(x: np.ndarray, path: Path, fps: float) -> None:
    import av
    with av.open(str(path), "w") as c:
        st = c.add_stream("libx264", rate=max(1, int(round(fps))))
        st.width, st.height, st.pix_fmt = x.shape[2] // 2 * 2, x.shape[1] // 2 * 2, "yuv420p"
        st.options = {"crf": "12"}
        for fr in x:
            for pk in st.encode(av.VideoFrame.from_ndarray(np.ascontiguousarray(fr[:st.height, :st.width]), format="bgr24")):
                c.mux(pk)
        for pk in st.encode():
            c.mux(pk)


def ours_judge_fps(y: np.ndarray, fps: float) -> dict:
    from lumi_synth import measure
    r = measure.analyze(y, fps); rr = measure.analyze_red(y, fps)
    return {"lum": bool(r["dangerous"]), "red": bool(rr["dangerous"]),
            "ext_lum": bool(measure.extended_flash(r["transitions"], fps)["extended_dangerous"]),
            "ext_red": bool(measure.extended_flash(rr["transitions"], fps)["extended_dangerous"])}


def report_real(rows_path: Path, out: Path, methods: list) -> None:
    rows = [json.loads(l) for l in rows_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    rows = [r for r in rows if "error" not in r]
    med = lambda v: float(np.nanmedian(v)) if v else float("nan")
    L = [f"# 실제 위험 영상(Kaya 2025 원본) 보정 — {rows_path.stem.replace('_rows', '')}", "",
         f"> {time.strftime('%Y-%m-%d %H:%M')} · 원본(정답) 없음 → 보정 전후 재판정과 '입력을 얼마나 바꿨나'만 · 깜빡임 라벨 = 정적 공간 패턴을 뺀 6종(패턴은 판별기 오탐이 많음)", "",
         "| 방법 | n | 안전(팀원, 7라벨) | 안전(팀원, 깜빡임 6라벨) | 안전(우리) | LPIPS 입력 대비 | PSNR 입력 대비 | 프레임 간 변화 비율 | 초 |",
         "|---|---|---|---|---|---|---|---|---|"]
    for m in methods:
        rs = [r for r in rows if r["method"] == m]
        if rs:
            L.append(f"| {m} | {len(rs)} | {sum(r['safe_team'] for r in rs)}/{len(rs)} | {sum(r['safe_team_flash'] for r in rs)}/{len(rs)} | "
                     f"{sum(r['safe_ours'] for r in rs)}/{len(rs)} | {med([r['lpips_vs_input'] for r in rs]):.3f} | "
                     f"{med([r['psnr_vs_input'] for r in rs if np.isfinite(r['psnr_vs_input'])]):.2f} | {med([r['flicker_ratio'] for r in rs]):.3f} | {med([r['sec'] for r in rs]):.1f} |")
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("보고서:", out, flush=True)


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
