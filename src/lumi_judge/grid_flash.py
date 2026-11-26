"""격자 판정기 G — 휘도 깜빡임(F1): 픽셀에서 규격대로 재고, 격자로 판단·설명한다 (docs/GRID_JUDGE_DESIGN.md §2–§4).

측정은 `lumi_synth.measure` 의 휘도 깜빡임 판정(ITU-R BT.1702-3, 설계 결정 D1–D4, pse-test-media 검증)을
**같은 정의로 GPU(PyTorch)로 옮긴 것**이다. 동일성: `scripts/grid_judge_parity.py`.
그 위에 격자 요약과 근거 출력을 더한다:
  - 격자 32×18(화면 비율 기준, docs §1). 칸 값 = 그 칸에서 위험 창의 화면 전이에 참여한 픽셀 비율.
  - 근거: 위험 구간(초), 그 구간의 1 초 최대 깜빡임 수, 화면 전이 순간의 방향별 면적(%), 전이 픽셀의 밝기 변화(cd/m²) 중앙값,
    조항 이름, 칸 목록.
"""
from __future__ import annotations

import math

import numpy as np
import torch

WHITE_CD, DELTA_CD, DARK_LIMIT_CD, AREA_LIMIT, MAX_FLASHES = 200.0, 20.0, 160.0, 0.25, 3
D3_SYNC_MS = 20.0
CLAUSE_F1 = "ITU-R BT.1702-3 Annex 1 Guideline 1 (일반 깜빡임: 20 cd/m², 어두운 쪽 < 160 cd/m², 면적 > 25 %, 1 초 > 3 회)"


def concurrency_frames(fps: float) -> int:
    return max(1, int(math.floor(D3_SYNC_MS / 1000.0 * fps + 1e-9)) + 1)


def window_frames(fps: float) -> int:
    return max(1, int(math.ceil(fps - 1e-9)))


@torch.no_grad()
def luminance_cd(frames_bgr: torch.Tensor) -> torch.Tensor:
    """uint8 BGR [T,H,W,3] (GPU) → cd/m² float32 [T,H,W] (measure.luminance_cd 와 같은 식).
    32 프레임씩 나눠 계산 — 결과는 같고, 긴·고해상도 영상에서 GPU 메모리 최대 사용량만 줄인다(2026-11-27)."""
    out = torch.empty(frames_bgr.shape[:-1], dtype=torch.float32, device=frames_bgr.device)
    for s in range(0, frames_bgr.shape[0], 32):
        c = frames_bgr[s:s + 32].to(torch.float32) / 255.0
        lin = torch.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
        out[s:s + 32] = WHITE_CD * (0.2126 * lin[..., 2] + 0.7152 * lin[..., 1] + 0.0722 * lin[..., 0])
    return out


@torch.no_grad()
def pixel_transitions(lum: torch.Tensor) -> torch.Tensor:
    """measure.pixel_transitions 의 GPU 판 — 픽셀은 병렬, 시간만 순차. int8 [T,H,W]."""
    ref = lum[0].clone(); lo = lum[0].clone(); hi = lum[0].clone()
    direction = torch.zeros(lum.shape[1:], dtype=torch.int8, device=lum.device)
    out = torch.zeros(lum.shape, dtype=torch.int8, device=lum.device)
    one = torch.ones_like(direction); m1 = -one; zero = torch.zeros_like(direction)
    for t in range(1, lum.shape[0]):
        v = lum[t]
        unknown = direction == 0
        extend = ((direction == 1) & (v > ref)) | ((direction == -1) & (v < ref))
        ref = torch.where(extend, v, ref)
        diff = v - ref
        big = (diff.abs() >= DELTA_CD) & (torch.minimum(v, ref) < DARK_LIMIT_CD) & ~extend & ~unknown
        new_dir = torch.sign(diff).to(torch.int8)
        up_mag, dn_mag = v - lo, hi - v
        up = unknown & (up_mag >= DELTA_CD) & (lo < DARK_LIMIT_CD)
        dn = unknown & (dn_mag >= DELTA_CD) & (v < DARK_LIMIT_CD)
        first = torch.where(up & dn, torch.where(up_mag >= dn_mag, one, m1), torch.where(up, one, torch.where(dn, m1, zero)))
        new_dir = torch.where(unknown, first, new_dir)
        big = big | (first != 0)
        trans = big & (new_dir != direction)
        out[t] = torch.where(trans, new_dir, zero)
        direction = torch.where(trans, new_dir, direction)
        ref = torch.where(big, v, ref)
        lo = torch.minimum(lo, v); hi = torch.maximum(hi, v)
    return out


@torch.no_grad()
def screen_transitions(signed: torch.Tensor, c: int):
    """measure.directional_areas + screen_transitions(c 소비 규칙 포함)의 GPU 판 → (e int8[T], a_up, a_dn).

    순차 상태 기계는 CPU 에서 숫자로만 돌리고, 필요한 면적은 GPU 에서 한 번에 미리 센다:
    '마지막 전이 프레임 > t − j' ⇔ '(t−j, t] 안에 그 방향 전이가 있음' 이므로, j = 1..c 마다 이동 창 합집합 면적을 구해 두면
    소비 규칙의 문턱 max(t − c, used) (used 는 t − c 보다 크면 t−c+1..t−1 중 하나)에 해당하는 면적을 바로 꺼낼 수 있다.
    """
    T = signed.shape[0]
    n_px = float(signed[0].numel())
    up_f = (signed == 1); dn_f = (signed == -1)
    cu = torch.cumsum(up_f.to(torch.int16), 0); cd = torch.cumsum(dn_f.to(torch.int16), 0)

    def win_area(cs, j):                       # (t−j, t] 안에 전이가 있는 픽셀 비율, t = 0..T−1
        prev = torch.cat([torch.zeros_like(cs[:j]), cs[:-j]], 0) if j < T else torch.zeros_like(cs)
        return ((cs - prev) > 0).flatten(1).sum(1).double().div(n_px).cpu().numpy()
    A_up = {j: win_area(cu, j) for j in range(1, c + 1)}
    A_dn = {j: win_area(cd, j) for j in range(1, c + 1)}
    a_up, a_dn = A_up[c], A_dn[c]
    up_any = up_f.flatten(1).any(1).cpu().numpy(); dn_any = dn_f.flatten(1).any(1).cpu().numpy()
    never = -(2 ** 30)
    used_up = used_dn = never
    e = np.zeros(T, dtype=np.int8)
    state = 0
    for t in range(T):
        th_u = max(t - c, used_up); th_d = max(t - c, used_dn)
        area_up = A_up[t - th_u][t] if t - th_u >= 1 else 0.0
        area_dn = A_dn[t - th_d][t] if t - th_d >= 1 else 0.0
        up = area_up > AREA_LIMIT and bool(up_any[t])
        dn = area_dn > AREA_LIMIT and bool(dn_any[t])
        if up and dn:
            d = -state if state != 0 else (1 if a_up[t] >= a_dn[t] else -1)
        elif up:
            d = 1
        elif dn:
            d = -1
        else:
            continue
        if d != state:
            e[t] = d; state = d
            if d == 1 or (up and dn):
                used_up = t
            if d == -1 or (up and dn):
                used_dn = t
    return e, a_up, a_dn


def window_counts(e: np.ndarray, fps: float) -> np.ndarray:
    w = window_frames(fps)
    csum = np.concatenate([[0], np.cumsum(e != 0)])
    n = len(e)
    return csum[np.minimum(np.arange(n) + w, n)] - csum[:n]


def grid_shape(h: int, w: int, cols: int = 32, grid_deg: float | None = None, view=None) -> tuple[int, int]:
    """근거 격자 (행, 열). grid_deg 를 주면 칸 = 시각도 grid_deg (해상도·화면·거리에 따라 칸 수가 정해짐 — 정지 패턴과 같은 방식)."""
    if grid_deg:
        from .grid_pattern import VIEW_DEFAULT, field_of_view
        fw, fh = field_of_view(w, h, view or VIEW_DEFAULT)
        return max(2, int(round(fh / grid_deg))), max(4, int(round(fw / grid_deg)))
    rows = max(1, int(round(cols * h / w)))
    return rows, cols


def fine_kernel_px(h: int, w: int, fine_avg_deg: float | None, view=None) -> int:
    """미세 평균 창(px, 홀수). WCAG 2.2 의 '0.1° 보다 작은 칸의 균형 잡힌 미세 무늬(백색 잡음·체커)' 예외를 구현할 때
    한 주기(0.2°)를 평균하면 그런 무늬의 밝기 변화가 상쇄된다. ITU 에는 이 예외가 없다 → 선택 사항(기본 끔, 실험으로 결정)."""
    if not fine_avg_deg:
        return 1
    from .grid_pattern import VIEW_DEFAULT, pixels_per_degree
    k = int(round(fine_avg_deg * pixels_per_degree(w, h, view=view or VIEW_DEFAULT)))
    return max(1, k + (1 - k % 2))


def fine_average(img: torch.Tensor, k: int) -> torch.Tensor:
    """[T,H,W] 또는 [T,H,W,C] 를 k×k 상자 평균(가장자리 복제)."""
    if k <= 1:
        return img
    ch_last = img.dim() == 4
    z = img.permute(0, 3, 1, 2) if ch_last else img[:, None]
    z = torch.nn.functional.avg_pool2d(torch.nn.functional.pad(z.float(), (k // 2,) * 4, mode="replicate"), k, stride=1)
    return z.permute(0, 2, 3, 1) if ch_last else z[:, 0]


@torch.no_grad()
def judge_luminance(frames_bgr: np.ndarray, fps: float, device: str = "cuda", grid_cols: int = 32,
                    grid_deg: float | None = None, view=None, fine_avg_deg: float | None = None) -> dict:
    """F1 판정 + 근거. frames_bgr uint8 [T,H,W,3] numpy.
    grid_deg: 근거 격자를 시각도로(정지 패턴과 같은 방식) · fine_avg_deg: WCAG 미세 무늬 예외(측정 전 미세 평균) — 둘 다 판정 문턱은 바꾸지 않음."""
    x = torch.from_numpy(np.ascontiguousarray(frames_bgr)).to(device)
    lum = fine_average(luminance_cd(x), fine_kernel_px(x.shape[1], x.shape[2], fine_avg_deg, view))
    signed = pixel_transitions(lum)
    c = concurrency_frames(fps)
    e, a_up, a_dn = screen_transitions(signed, c)
    counts = window_counts(e, fps)
    w = window_frames(fps)
    starts = np.flatnonzero(counts >= 2 * MAX_FLASHES + 1)
    hazard_frames = np.zeros(len(e), bool)
    for s in starts:
        hazard_frames[s:s + w] = True
    hz_t = np.flatnonzero((e != 0) & hazard_frames)
    hazard_mask = torch.zeros(signed.shape[1:], dtype=torch.bool, device=device)
    dl_vals = []
    for t in hz_t:
        lo_i = max(0, t - c + 1)
        part = (signed[lo_i:t + 1] == int(e[t])).any(0)
        hazard_mask |= part
        if t >= 1:
            d = (lum[t] - lum[max(0, t - 1)]).abs()[part]
            if d.numel():
                dl_vals.append(float(d.median()))
    flashes = float(counts.max()) / 2.0 if len(counts) else 0.0
    H, W = signed.shape[1:]
    gr, gc = grid_shape(H, W, grid_cols, grid_deg, view)
    cell = torch.nn.functional.adaptive_avg_pool2d(hazard_mask.float()[None, None], (gr, gc))[0, 0].cpu().numpy()
    segs = []
    if starts.size:
        fr = np.flatnonzero(hazard_frames)
        br = np.flatnonzero(np.diff(fr) > 1)
        for a_, b_ in zip(np.r_[fr[0], fr[br + 1]], np.r_[fr[br], fr[-1]]):
            segs.append({"start_s": round(a_ / fps, 2), "end_s": round((b_ + 1) / fps, 2),
                         "max_flashes_per_s": float(counts[a_:b_ + 1].max()) / 2.0})
    return {
        "dangerous": flashes > MAX_FLASHES,
        "max_flashes": flashes,
        "transitions": e, "area_up": a_up, "area_down": a_dn,
        "hazard_frames": hazard_frames,
        "hazard_mask": hazard_mask.cpu().numpy(),
        "grid": cell,
        "evidence": {
            "clause": CLAUSE_F1,
            "segments": segs,
            "peak_area_pct": round(100 * float(np.maximum(a_up, a_dn).max()), 1) if len(a_up) else 0.0,
            "hazard_area_pct": round(100 * float(hazard_mask.float().mean()), 1),
            "delta_cd_m2_median": round(float(np.median(dl_vals)), 1) if dl_vals else None,
            "cells": [[int(r), int(q)] for r, q in zip(*np.nonzero(cell > 0.1))],
            "grid_shape": [gr, gc],
        },
    }


# ============================== F2 적색 깜빡임 ==============================
# measure.red_state / red_transitions / analyze_red 와 같은 정의(WCAG 2.2 적색 정의 + BT.1702-3 면적·횟수):
#   포화 빨강: 선형 R/(R+G+B) ≥ 0.8 이고 R − G − B ≥ 0.0625 · 색도 u′v′(검정·거의 검정은 D65 고정)
#   전이: 바로 앞 프레임 대비 Δu′v′ > 0.2 이고 포화 빨강으로/에서 바뀜 · 반대 방향 변화끼리 짝지은 것만 셈
CLAUSE_F2 = "ITU-R BT.1702-3 Annex 1 Guideline 1 (포화 적색으로의/으로부터의 전이) · 적색 정의 WCAG 2.2 (R/(R+G+B) ≥ 0.8, Δu′v′ > 0.2)"
RED_SAT_RATIO, RED_DELTA_UV, RED_MIN_EXCESS, RED_CHANGE_EPS, RED_BLACK_Y = 0.8, 0.2, 0.0625, 0.02, 0.005
D65_UV = (0.1978, 0.4683)
_XYZ = torch.tensor([[0.4124564, 0.3575761, 0.1804375],
                     [0.2126729, 0.7151522, 0.0721750],
                     [0.0193339, 0.1191920, 0.9503041]], dtype=torch.float64)


@torch.no_grad()
def red_state(fr: torch.Tensor):
    c = fr.to(torch.float32) / 255.0
    lin = torch.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4).to(torch.float64)
    b, g, r = lin[..., 0], lin[..., 1], lin[..., 2]
    total = r + g + b
    sat = (total > 0) & (r >= RED_SAT_RATIO * total) & (r - g - b >= RED_MIN_EXCESS)
    M = _XYZ.to(fr.device)
    x = M[0, 0] * r + M[0, 1] * g + M[0, 2] * b
    y = M[1, 0] * r + M[1, 1] * g + M[1, 2] * b
    z = M[2, 0] * r + M[2, 1] * g + M[2, 2] * b
    d = x + 15.0 * y + 3.0 * z
    ok = (d > 0) & (y >= RED_BLACK_Y)
    dd = torch.where(ok, d, torch.ones_like(d))
    u = torch.where(ok, 4.0 * x / dd, torch.full_like(d, D65_UV[0])).to(torch.float32)
    v = torch.where(ok, 9.0 * y / dd, torch.full_like(d, D65_UV[1])).to(torch.float32)
    return sat, u, v


@torch.no_grad()
def red_transitions(frames: torch.Tensor):
    T = frames.shape[0]
    ps, pu, pv = red_state(frames[0])
    shape = ps.shape
    event = torch.zeros((T,) + shape, dtype=torch.bool, device=frames.device)
    qual = torch.zeros((T,) + shape, dtype=torch.int8, device=frames.device)
    duv_at = torch.zeros((T,) + shape, dtype=torch.float32, device=frames.device)
    one = torch.ones(shape, dtype=torch.int8, device=frames.device); zero = torch.zeros_like(one)
    for t in range(1, T):
        s, u, v = red_state(frames[t])
        duv = torch.hypot(u - pu, v - pv)
        nd = torch.where(s & ~ps, one, torch.where(~s & ps, -one, zero))
        event[t] = (duv >= RED_CHANGE_EPS) | (nd != 0)
        qual[t] = torch.where((duv > RED_DELTA_UV) & (nd != 0), nd, zero)
        duv_at[t] = duv
        ps, pu, pv = s, u, v
    k = torch.cumsum(event.to(torch.int32), 0)
    tix = torch.arange(T, device=frames.device, dtype=torch.int32).view((T,) + (1,) * len(shape)).expand((T,) + shape)
    ev_t = torch.where(event, tix, torch.full_like(tix, -1))
    prev_ev = torch.cummax(ev_t, 0).values
    prev_ev = torch.cat([torch.full((1,) + shape, -1, dtype=torch.int32, device=frames.device), prev_ev[:-1]])
    ev_rev = torch.where(event, tix, torch.full_like(tix, T)).flip(0)
    next_ev = torch.cummin(ev_rev, 0).values.flip(0)
    next_ev = torch.cat([next_ev[1:], torch.full((1,) + shape, T, dtype=torch.int32, device=frames.device)])
    q_pad = torch.cat([qual, torch.zeros((1,) + shape, dtype=torch.int8, device=frames.device)])
    first = (k % 2 == 1)
    partner = torch.where(first, next_ev, prev_ev)
    idx = torch.where((partner >= 0) & (partner <= T), partner, torch.full_like(partner, T)).long()
    partner_q = torch.gather(q_pad, 0, idx)
    unpaired_last = first & (next_ev >= T)
    valid = event & (qual != 0) & (((partner_q == -qual) & (partner >= 0)) | unpaired_last)
    out = torch.where(valid, qual, torch.zeros_like(qual))
    max_duv = float(duv_at[valid].max()) if bool(valid.any()) else 0.0
    return out, max_duv


@torch.no_grad()
def judge_red(frames_bgr: np.ndarray, fps: float, device: str = "cuda", grid_cols: int = 32,
              grid_deg: float | None = None, view=None, fine_avg_deg: float | None = None) -> dict:
    x = torch.from_numpy(np.ascontiguousarray(frames_bgr)).to(device)
    k = fine_kernel_px(x.shape[1], x.shape[2], fine_avg_deg, view)
    if k > 1:                                                          # 색은 sRGB 값 평균(근사) 후 다시 8 비트로
        x = fine_average(x, k).round().clamp(0, 255).to(torch.uint8)
    signed, max_duv = red_transitions(x)
    return _judge_from_signed(signed, fps, grid_cols, CLAUSE_F2, extra={"max_delta_uv": round(max_duv, 3)}, grid_deg=grid_deg, view=view)


@torch.no_grad()
def _judge_from_signed(signed: torch.Tensor, fps: float, grid_cols: int, clause: str, extra: dict | None = None,
                       grid_deg: float | None = None, view=None) -> dict:
    c = concurrency_frames(fps)
    e, a_up, a_dn = screen_transitions(signed, c)
    counts = window_counts(e, fps)
    w = window_frames(fps)
    starts = np.flatnonzero(counts >= 2 * MAX_FLASHES + 1)
    hazard_frames = np.zeros(len(e), bool)
    for s in starts:
        hazard_frames[s:s + w] = True
    hazard_mask = torch.zeros(signed.shape[1:], dtype=torch.bool, device=signed.device)
    for t in np.flatnonzero((e != 0) & hazard_frames):
        hazard_mask |= (signed[max(0, t - c + 1):t + 1] == int(e[t])).any(0)
    flashes = float(counts.max()) / 2.0 if len(counts) else 0.0
    H, W = signed.shape[1:]
    gr, gc = grid_shape(H, W, grid_cols, grid_deg, view)
    cell = torch.nn.functional.adaptive_avg_pool2d(hazard_mask.float()[None, None], (gr, gc))[0, 0].cpu().numpy()
    segs = []
    if starts.size:
        fr = np.flatnonzero(hazard_frames)
        br = np.flatnonzero(np.diff(fr) > 1)
        for a_, b_ in zip(np.r_[fr[0], fr[br + 1]], np.r_[fr[br], fr[-1]]):
            segs.append({"start_s": round(float(a_) / fps, 2), "end_s": round(float(b_ + 1) / fps, 2),
                         "max_flashes_per_s": float(counts[a_:b_ + 1].max()) / 2.0})
    ev = {"clause": clause, "segments": segs,
          "peak_area_pct": round(100 * float(np.maximum(a_up, a_dn).max()), 1) if len(a_up) else 0.0,
          "hazard_area_pct": round(100 * float(hazard_mask.float().mean()), 1),
          "cells": [[int(r), int(q)] for r, q in zip(*np.nonzero(cell > 0.1))], "grid_shape": [gr, gc]}
    ev.update(extra or {})
    return {"dangerous": flashes > MAX_FLASHES, "max_flashes": flashes, "transitions": e, "area_up": a_up, "area_down": a_dn,
            "hazard_frames": hazard_frames, "hazard_mask": hazard_mask.cpu().numpy(), "grid": cell, "evidence": ev}
