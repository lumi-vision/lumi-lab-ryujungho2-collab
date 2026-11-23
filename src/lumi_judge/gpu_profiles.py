"""GPU 일괄 처리판 — 1D 밝기 단면의 히스테리시스 극값과 '줄무늬 쌍' 최장 구간 (판정기 v2, 2026-11-23).

원본(`spatial.extract_hysteresis_extrema_1d` + `_best_open_run_from_extrema`)은 단면 하나씩 파이썬 반복문으로 돈다
(192 프레임 클립 하나에 약 6 만 번 호출, CPU 시간의 대부분). 여기서는 N 개 단면을 [N, L] 텐서로 묶어
길이 L 방향으로만 순차 진행하고 N 방향은 GPU 에서 동시에 처리한다. 결과는 원본과 같아야 한다
(`scripts/judge_v2_parity.py` 로 무작위·실제 단면에서 대조).

동등성 근거: 원본은 (1) 같은 값이 이어지는 구간의 첫 표본만 남기고 (2) 엄격한 국소 극값만 남긴 뒤 상태 기계를 돈다.
엄격 비교(>, <)로 갱신하면 평탄 구간에서 첫 위치가 유지되어 (1) 과 같고, 국소 극값이 아닌 표본은 단조 구간 안에 있어
극값 갱신이나 확정 위치를 바꾸지 않으므로 (2) 를 생략해도 같은 극값 열이 나온다. 단, 원본은 (1) 뒤 표본이 3 개 미만이면
빈 결과를 내므로 그 조건은 따로 맞춘다.
"""
from __future__ import annotations

import torch


@torch.no_grad()
def hysteresis_extrema_batched(values: torch.Tensor, theta: torch.Tensor, lengths: torch.Tensor, h: float):
    """values, theta: [N, L] (왼쪽 정렬, lengths[n] 이후는 무시) → (ext_t [N, K], ext_v [N, K], n_ext [N]).

    K = 최대 극값 수(패딩은 NaN). 원본 extract_hysteresis_extrema_1d 와 같은 극값 열.
    """
    dev = values.device
    N, L = values.shape
    v = values.to(torch.float64)
    t = theta.to(torch.float64)
    idx = torch.arange(L, device=dev)
    inb = idx[None, :] < lengths[:, None]
    # 원본의 조기 종료: 길이 < 3, 범위 < h, 평탄 제거 뒤 표본 < 3
    vmax = torch.where(inb, v, torch.full_like(v, -torch.inf)).amax(1)
    vmin = torch.where(inb, v, torch.full_like(v, torch.inf)).amin(1)
    change = torch.zeros_like(inb)
    change[:, 1:] = (v[:, 1:] != v[:, :-1]) & inb[:, 1:]
    m = 1 + change.sum(1)
    ok = (lengths >= 3) & ((vmax - vmin) >= h) & (m >= 3)

    cur_max = v[:, 0].clone(); cur_min = v[:, 0].clone()
    max_t = t[:, 0].clone(); min_t = t[:, 0].clone()
    state = torch.zeros(N, dtype=torch.int8, device=dev)
    # 확정 극값을 위치별로 기록: rec_v[n, k] = 그 단계에서 확정된 극값(없으면 NaN)
    rec_v = torch.full((N, L), torch.nan, dtype=torch.float64, device=dev)
    rec_t = torch.full((N, L), torch.nan, dtype=torch.float64, device=dev)
    for k in range(1, L):
        act = ok & (k < lengths)
        val = v[:, k]; pos = t[:, k]
        s0, s1, sm = act & (state == 0), act & (state == 1), act & (state == -1)
        # --- state 0 ---
        up0 = s0 & (val > cur_max); dn0 = s0 & (val < cur_min)
        cur_max = torch.where(up0, val, cur_max); max_t = torch.where(up0, pos, max_t)
        cur_min = torch.where(dn0, val, cur_min); min_t = torch.where(dn0, pos, min_t)
        a = s0 & ((cur_max - val) >= h) & (max_t < pos)
        b = s0 & ~a & ((val - cur_min) >= h) & (min_t < pos)
        # --- state 1 (봉우리 찾는 중) ---
        up1 = s1 & (val > cur_max)
        c = s1 & ~up1 & ((cur_max - val) >= h)
        cur_max = torch.where(up1, val, cur_max); max_t = torch.where(up1, pos, max_t)
        # --- state -1 (골 찾는 중) ---
        dn1 = sm & (val < cur_min)
        d = sm & ~dn1 & ((val - cur_min) >= h)
        cur_min = torch.where(dn1, val, cur_min); min_t = torch.where(dn1, pos, min_t)
        # 봉우리 확정(a, c) → 골 찾기 / 골 확정(b, d) → 봉우리 찾기
        peak = a | c; trough = b | d
        rec_v[:, k] = torch.where(peak, cur_max, torch.where(trough, cur_min, rec_v[:, k]))
        rec_t[:, k] = torch.where(peak, max_t, torch.where(trough, min_t, rec_t[:, k]))
        cur_min = torch.where(peak, val, cur_min); min_t = torch.where(peak, pos, min_t)
        cur_max = torch.where(trough, val, cur_max); max_t = torch.where(trough, pos, max_t)
        state = torch.where(peak, torch.full_like(state, -1), torch.where(trough, torch.full_like(state, 1), state))
    # 확정 극값을 앞으로 모음
    has = ~torch.isnan(rec_v)
    n_ext = has.sum(1)
    K = int(n_ext.max().item()) + 1 if N else 1
    order = torch.argsort((~has).to(torch.int8), dim=1, stable=True)[:, :K]
    ext_v = torch.gather(rec_v, 1, order); ext_t = torch.gather(rec_t, 1, order)
    kk = torch.arange(K, device=dev)[None, :]
    ext_v = torch.where(kk < n_ext[:, None], ext_v, torch.nan); ext_t = torch.where(kk < n_ext[:, None], ext_t, torch.nan)
    # 끝 극값: 마지막 확정과 h 이상 차이면 추가
    last_v = torch.gather(ext_v, 1, (n_ext - 1).clamp_min(0)[:, None])[:, 0]
    tail_peak = ok & (n_ext > 0) & (state == 1) & ((cur_max - last_v) >= h)
    tail_tr = ok & (n_ext > 0) & (state == -1) & ((last_v - cur_min) >= h)
    tail = tail_peak | tail_tr
    rows = torch.nonzero(tail, as_tuple=True)[0]
    ext_v[rows, n_ext[rows]] = torch.where(tail_peak, cur_max, cur_min)[rows]
    ext_t[rows, n_ext[rows]] = torch.where(tail_peak, max_t, min_t)[rows]
    n_ext = n_ext + tail.to(n_ext.dtype)
    n_ext = torch.where(ok, n_ext, torch.zeros_like(n_ext))
    return ext_t, ext_v, n_ext


@torch.no_grad()
def best_open_run_batched(ext_t, ext_v, n_ext, delta_min, dark_limit, michelson_min, cpd_lo, cpd_hi):
    """원본 _best_open_run_from_extrema 의 일괄판 → (best_pairs [N] int, median_cpd [N] float)."""
    N, K = ext_v.shape
    dev = ext_v.device
    if K < 2:
        return torch.zeros(N, dtype=torch.long, device=dev), torch.zeros(N, dtype=torch.float64, device=dev)
    v0, v1 = ext_v[:, :-1], ext_v[:, 1:]
    lo, hi = torch.minimum(v0, v1), torch.maximum(v0, v1)
    dl = hi - lo
    contrast = torch.where(lo < dark_limit, dl >= delta_min, (dl / torch.clamp(hi + lo, min=1e-6)) > michelson_min)
    dth = ext_t[:, 1:] - ext_t[:, :-1]
    cpds = torch.where(dth > 1e-6, 0.5 / torch.clamp(dth, min=1e-6), torch.zeros_like(dth))
    seg_ok = torch.arange(K - 1, device=dev)[None, :] < (n_ext - 1).clamp_min(0)[:, None]
    valid = contrast & (cpds >= cpd_lo) & (cpds <= cpd_hi) & seg_ok
    # 연속 True 구간 길이: 각 위치에서 끝나는 run 길이
    run = torch.zeros_like(valid, dtype=torch.long)
    acc = torch.zeros(N, dtype=torch.long, device=dev)
    for j in range(K - 1):
        acc = torch.where(valid[:, j], acc + 1, torch.zeros_like(acc))
        run[:, j] = acc
    best_len, best_end = run.max(1)                       # 첫 최댓값(원본 argmax 와 같은 첫 구간)
    # 원본 argmax 는 '시작 순서상 첫 최장 구간' — 끝 위치 기준 첫 최댓값과 같음(구간은 겹치지 않음)
    pairs = (best_len + 1) // 2
    # 최장 구간의 cpd 중앙값 (numpy median: 짝수 개면 가운데 둘 평균)
    jj = torch.arange(K - 1, device=dev)[None, :]
    start = best_end - best_len + 1
    inrun = (jj >= start[:, None]) & (jj <= best_end[:, None]) & (best_len[:, None] > 0)
    srt = torch.sort(torch.where(inrun, cpds, torch.full_like(cpds, torch.inf)), dim=1).values
    cnt = best_len
    lo_i = ((cnt - 1) // 2).clamp_min(0); hi_i = (cnt // 2).clamp_min(0)
    med = 0.5 * (torch.gather(srt, 1, lo_i[:, None])[:, 0] + torch.gather(srt, 1, hi_i[:, None])[:, 0])
    med = torch.where(pairs > 0, med, torch.zeros_like(med))
    pairs = torch.where(best_len > 0, pairs, torch.zeros_like(pairs))
    return pairs, med
