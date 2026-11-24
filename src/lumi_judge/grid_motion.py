"""격자 판정기 G — 패턴의 움직임: 정지 / 흐름(면제) / 변화(P2) 를 칸마다 가른다 (docs/GRID_JUDGE_DESIGN.md §3 P2).

규격 원문(ITU-R BT.1702-3 Att.1 = Ofcom Annex 1 = ITC 2001):
  - "the stripes are stationary and the pattern occupies more than 40%"                         → P1
  - "the stripes change direction, oscillate, flash, or reverse in contrast ... more than 25%"   → P2
  - "If the patterns obviously flow smoothly across, into, or out of the screen in one direction they are [exempt]"
근거(의학): Binnie, Findlay & Wilkins 1985 — 진동 격자 PPR 0.97, 정지 0.38, 한 방향 흐름 0–0.04 ("drifting least epileptogenic").

방법(구현 선택 — 민감도 분석 대상)
  1. 표본 프레임(초당 8)에서 P1 과 같은 절차로 줄무늬 영역을 찾되 면적 문턱은 25 %(변화 패턴 후보).
  2. 그 프레임을 가운데 둔 1 초 창의 **모든 프레임**에서, 픽셀마다 표본 프레임의 필터(방향·주파수)로 복소 응답 z_t 를 구한다.
     줄무늬가 법선 방향으로 움직이면 z 의 위상이 돈다 — 한 프레임 위상 변화 v_t = arg Σ_칸 z_t·conj(z_{t−1}) (칸마다).
     줄무늬의 세기 a_t = 칸 평균 |z_t|.
  3. 칸 판정
     - 흐름  : 세기가 남아 있는 프레임(a_t ≥ ½ a_0)의 절반 이상에서 |v_t| ≥ ε 이고, 그 방향(부호)이 모두 같고, 반전(|v_t| > ¾π)이 없음
     - 변화  : (흐름이 아니고) 대비 반전(|v_t| > ¾π) 또는 진동·방향 전환(움직이는 프레임끼리 부호가 바뀜)
               또는 점멸·방향 바뀜(세기가 ½ a_0 문턱을 2 번 이상 넘나듦 — 다른 방향 줄무늬로 바뀌어도 원래 필터 세기가 떨어진다;
               한 번 나타나거나 사라지는 것은 점멸이 아님: 2026-11-25 합성 진단에서 정지 무늬의 등장을 점멸로 오분류한 것을 고침)
     - 정지  : 그 밖(움직임 중앙값 < ε)
     ε = 0.5 주기/초 를 프레임당 위상으로 (그보다 느린 움직임은 정지로 봄).
  4. 면적(확인 비율을 곱한 줄무늬 픽셀) — 변화 > 25 % → P2 · 정지 + 변화 > 40 % → P1(변화는 정지보다 덜 위험하지 않으므로 합산) · 흐름 칸은 둘 다에서 뺌.
  왜 칸마다인가: 동심원이 커지는 줌·방사 무늬는 화면 전체로 평균하면 반대쪽끼리 위상이 상쇄된다. 칸 안에서는 방향이 하나.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from .grid_flash import luminance_cd
from .grid_pattern import GridPatternJudge

CLAUSE_P2 = ("ITU-R BT.1702-3 Att.1 · Ofcom Annex 1 (줄무늬가 방향 전환·진동·점멸·대비 반전, > 5 쌍, ΔL ≥ 20 cd/m², 면적 > 25 %; "
             "한 방향으로 매끄럽게 흐르면 면제)")
FLOW_MIN_PPS = 0.5          # 주기/초 — 이보다 느리면 정지 (구현 선택)
REV_PHASE = 0.75 * math.pi  # 한 프레임 위상 변화가 이보다 크면 대비 반전 (반전 = π)
AMP_DROP = 0.5              # 세기가 표본 프레임의 절반 아래 → 점멸·방향 바뀜 (문턱을 2 번 이상 넘나들 때만)
REV_MIN = 2                 # 대비 반전으로 볼 반전 프레임 수 하한(1 초 창) — 점멸과 같은 '반복' 기준
MIN_RUN = 2                 # 방향 전환으로 셀 움직임 구간의 최소 길이(프레임)
FLOW_FRAC = 0.5             # 흐름: 세기가 남은 프레임 중 움직이는 비율 하한
CELL_MIN_PX = 0.10          # 칸 픽셀 중 줄무늬 픽셀이 이 비율 이상인 칸만 판정
AREA_P2, AREA_P1 = 0.25, 0.40


@torch.no_grad()
def _responses(bank, lum: torch.Tensor, ks: torch.Tensor) -> torch.Tensor:
    X = torch.fft.fft2(lum - lum.mean())
    return torch.fft.ifft2(X[None] * bank.F[ks])                     # [len(ks), h, w] complex


@torch.no_grad()
def classify_motion(G: GridPatternJudge, lums: list[torch.Tensor], a_idx: int, best: dict, fps: float) -> dict:
    """lums: 1 초 창 프레임 밝기(cd/m²), a_idx: 그중 표본 프레임 위치. → 칸별 class 와 면적."""
    h, w = G.h, G.w
    mask = torch.from_numpy(best["mask"]).to(lums[0].device)
    kmap = best["k_map"]
    ks, inv = torch.unique(kmap[mask], return_inverse=True)
    ys, xs = torch.nonzero(mask, as_tuple=True)
    cell_h, cell_w = h / G.rows, w / G.cols
    cid = (torch.div(ys, cell_h, rounding_mode="floor").long().clamp_max(G.rows - 1) * G.cols
           + torch.div(xs, cell_w, rounding_mode="floor").long().clamp_max(G.cols - 1))
    n_cell = G.rows * G.cols
    cnt = torch.zeros(n_cell, device=mask.device).index_add_(0, cid, torch.ones_like(cid, dtype=torch.float32))
    z = []
    for L in lums:
        Z = _responses(G.bank, L, ks)                                  # [K', h, w]
        z.append(Z[inv, ys, xs])                                       # 픽셀별 자기 필터 응답
    z = torch.stack(z)                                                 # [T, P]
    amp = torch.zeros(len(lums), n_cell, device=mask.device)
    for t in range(len(lums)):
        amp[t].index_add_(0, cid, z[t].abs().float())
    amp = amp / cnt.clamp_min(1)
    prod = z[1:] * z[:-1].conj()
    re = torch.zeros(len(lums) - 1, n_cell, device=mask.device); im = torch.zeros_like(re)
    for t in range(len(lums) - 1):
        re[t].index_add_(0, cid, prod[t].real.float()); im[t].index_add_(0, cid, prod[t].imag.float())
    v = torch.atan2(im, re)                                            # [T-1, cells] 프레임당 위상 변화
    eps = 2 * math.pi * FLOW_MIN_PPS / fps
    a0 = amp[a_idx]
    cell_px = (h * w) / n_cell
    valid = cnt >= CELL_MIN_PX * cell_px
    klass = np.full(n_cell, "", dtype=object); reason = np.full(n_cell, "", dtype=object)
    A = amp.cpu().numpy(); V = v.cpu().numpy(); A0 = a0.cpu().numpy()
    for c in np.flatnonzero(valid.cpu().numpy()):
        strong = A[1:, c] >= AMP_DROP * A0[c]                          # v_t 는 (t−1,t) 사이 — 뒤 프레임 기준
        vv = V[:, c]
        moving = np.abs(vv) >= eps
        rev = int((np.abs(vv[strong]) > REV_PHASE).sum()) >= REV_MIN    # 반복된 반전만(한 프레임 위상 튐 제외) — 2026-11-25 변경
        mv = vv[strong & moving]
        # 방향 전환: 같은 부호로 2 프레임 이상 이어진 움직임 구간끼리 부호가 바뀜(한 프레임 잡음 부호는 무시) — 2026-11-25 변경
        runs = []
        for sg in np.sign(mv):
            if runs and runs[-1][0] == sg:
                runs[-1][1] += 1
            else:
                runs.append([sg, 1])
        kept = [r_[0] for r_ in runs if r_[1] >= MIN_RUN]
        sgn_change = any(a_ != b_ for a_, b_ in zip(kept[:-1], kept[1:]))
        # 점멸: 세기가 문턱(½ a_0)을 두 번 이상 넘나듦(켜짐→꺼짐→켜짐 또는 그 반대). 한 번 나타나거나 사라지는 것은 점멸이 아님 — 2026-11-25 변경
        on = A[:, c] >= AMP_DROP * A0[c]
        drop = int((on[1:] != on[:-1]).sum()) >= 2
        if strong.sum() and (strong & moving).sum() >= FLOW_FRAC * strong.sum() and not sgn_change and not rev:
            klass[c] = "flow"
        elif rev:
            klass[c], reason[c] = "change", "대비 반전"
        elif sgn_change:
            klass[c], reason[c] = "change", "진동·방향 전환"
        elif drop:
            klass[c], reason[c] = "change", "점멸·방향 바뀜"
        else:
            klass[c] = "static"
    frac_px = (cnt / (h * w)).cpu().numpy() * float(best["verified"])
    area = {k: float(frac_px[klass == k].sum()) for k in ("static", "change", "flow")}
    rs = [r for r in reason[klass == "change"] if r]
    top = max(set(rs), key=rs.count) if rs else ""
    cells = lambda k: [[int(c // G.cols), int(c % G.cols)] for c in np.flatnonzero(klass == k)]
    return {"area": area, "reason": top, "cells": {k: cells(k) for k in ("static", "change", "flow")}}


@torch.no_grad()
def judge_pattern_motion(frames_bgr: np.ndarray, fps: float, sample_fps: float = 8.0, G: GridPatternJudge | None = None) -> dict:
    """P1(정지, 흐름 면제) + P2(변화) 클립 판정. 반환: p1/p2 위험 여부·최고 근거·프레임 표시."""
    T, h, w = frames_bgr.shape[:3]
    G = G or GridPatternJudge(h, w)
    step = max(1, int(round(fps / sample_fps)))
    half = max(1, int(round(0.5 * fps)))
    p1 = np.zeros(T, bool); p2 = np.zeros(T, bool)
    best1 = best2 = best_any = None
    lum_cache: dict[int, torch.Tensor] = {}

    def lum(i):
        if i not in lum_cache:
            lum_cache[i] = luminance_cd(torch.from_numpy(np.ascontiguousarray(frames_bgr[i:i + 1])).cuda())[0]
            if len(lum_cache) > 4 * half + 8:
                lum_cache.pop(min(lum_cache))
        return lum_cache[i]

    for i in range(0, T, step):
        b = G.frame(lum(i), area_min=AREA_P2, maps=True)
        if not b["is_pat"]:
            continue
        lo, hi = max(0, i - half), min(T, i + half + 1)
        m = classify_motion(G, [lum(j) for j in range(lo, hi)], i - lo, b, fps)
        a = m["area"]
        info = {"t": round(i / fps, 2), "pairs": b["pairs"], "cpd": b["cpd"], "kind": b["kind"], "verified_seeds": b["verified_seeds"],
                "area_static": a["static"], "area_change": a["change"], "area_flow": a["flow"], "reason": m["reason"], "cells": m["cells"]}
        if best_any is None or sum(a.values()) > sum(best_any[k] for k in ("area_static", "area_change", "area_flow")):
            best_any = info
        if a["static"] + a["change"] > AREA_P1:
            p1[i:i + step] = True
            if best1 is None or a["static"] + a["change"] > best1["area_static"] + best1["area_change"]:
                best1 = info
        if a["change"] > AREA_P2:
            p2[lo:hi] = True
            if best2 is None or a["change"] > best2["area_change"]:
                best2 = info
    win = max(1, int(math.ceil(fps - 1e-9))); need = int(math.ceil(0.5 * fps - 1e-9))
    cs = np.concatenate([[0], np.cumsum(p1)])
    counts = cs[np.minimum(np.arange(T) + win, T)] - cs[:T]
    return {"p1": bool((counts >= need).any()) if T else False, "p2": bool(p2.any()), "best1": best1, "best2": best2, "best_any": best_any,
            "p1_frames": p1, "p2_frames": p2}
