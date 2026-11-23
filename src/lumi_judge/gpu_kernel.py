"""단면당 GPU 스레드 1 개 CUDA 커널(CuPy RawKernel) — 히스테리시스 극값 + 최장 줄무늬 구간 (판정기 v2).

원본 `spatial.extract_hysteresis_extrema_1d` → `_best_open_run_from_extrema` 를 C 로 그대로 옮겼다(순서·비교·경계 조건 동일).
한 번 호출로 단면 N 개를 동시에 처리하고, 두 cpd 하한(줄무늬 band, wedge)의 결과를 함께 낸다.
동일성은 `scripts/judge_v2_parity.py --kernel` 로 확인한다.
"""
from __future__ import annotations

import numpy as np

_SRC = r"""
extern "C" __global__
void hyst_runs(const double* __restrict__ V, const double* __restrict__ T, const int* __restrict__ LEN,
               int N, int L, double h, double dmin, double dark, double mich, double cpd_lo, double cpd_lo2, double cpd_hi,
               double* ET, double* EV, int* NEXT, int* PAIRS, double* CPD, int* PAIRS2, double* SCR)
{
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;
    const double* v = V + (size_t)n * L;
    const double* t = T + (size_t)n * L;
    double* et = ET + (size_t)n * L;
    double* ev = EV + (size_t)n * L;
    double* scr = SCR + (size_t)n * L;
    int len = LEN[n];
    NEXT[n] = 0; PAIRS[n] = 0; CPD[n] = 0.0; PAIRS2[n] = 0;
    if (len < 3) return;
    double mx = v[0], mn = v[0]; int m = 1;
    for (int i = 1; i < len; ++i) { if (v[i] > mx) mx = v[i]; if (v[i] < mn) mn = v[i]; if (v[i] != v[i-1]) ++m; }
    if ((mx - mn) < h || m < 3) return;

    // --- 원본 상태 기계 (평탄 첫 표본·국소 극값 압축과 동등: 엄격 비교로 처리) ---
    double cur_max = v[0], cur_min = v[0], max_t = t[0], min_t = t[0];
    int state = 0, k = 0;
    for (int i = 1; i < len; ++i) {
        double val = v[i], pos = t[i];
        if (state == 0) {
            if (val > cur_max) { cur_max = val; max_t = pos; }
            if (val < cur_min) { cur_min = val; min_t = pos; }
            if ((cur_max - val) >= h && max_t < pos) { et[k] = max_t; ev[k] = cur_max; ++k; cur_min = val; min_t = pos; state = -1; }
            else if ((val - cur_min) >= h && min_t < pos) { et[k] = min_t; ev[k] = cur_min; ++k; cur_max = val; max_t = pos; state = 1; }
        } else if (state == 1) {
            if (val > cur_max) { cur_max = val; max_t = pos; }
            else if ((cur_max - val) >= h) { et[k] = max_t; ev[k] = cur_max; ++k; cur_min = val; min_t = pos; state = -1; }
        } else {
            if (val < cur_min) { cur_min = val; min_t = pos; }
            else if ((val - cur_min) >= h) { et[k] = min_t; ev[k] = cur_min; ++k; cur_max = val; max_t = pos; state = 1; }
        }
    }
    if (k > 0) {
        if (state == 1 && (cur_max - ev[k-1]) >= h) { et[k] = max_t; ev[k] = cur_max; ++k; }
        else if (state == -1 && (ev[k-1] - cur_min) >= h) { et[k] = min_t; ev[k] = cur_min; ++k; }
    }
    NEXT[n] = k;
    if (k < 2) return;

    // --- 원본 _best_open_run_from_extrema (두 cpd 하한) ---
    for (int pass = 0; pass < 2; ++pass) {
        double lo_c = pass == 0 ? cpd_lo : cpd_lo2;
        int best_len = 0, best_end = -1, run = 0;
        for (int j = 0; j < k - 1; ++j) {
            double a = ev[j], b = ev[j+1];
            double l_low = a < b ? a : b, l_high = a < b ? b : a, dl = l_high - l_low;
            bool contrast = (l_low < dark) ? (dl >= dmin) : ((dl / fmax(l_high + l_low, 1e-6)) > mich);
            double dth = et[j+1] - et[j];
            double cpd = dth > 1e-6 ? 0.5 / fmax(dth, 1e-6) : 0.0;
            bool ok = contrast && cpd >= lo_c && cpd <= cpd_hi;
            run = ok ? run + 1 : 0;
            if (run > best_len) { best_len = run; best_end = j; }
        }
        int pairs = (best_len + 1) / 2;
        if (best_len <= 0 || pairs <= 0) continue;
        if (pass == 1) { PAIRS2[n] = pairs; continue; }
        PAIRS[n] = pairs;
        int s = best_end - best_len + 1;
        for (int j = 0; j < best_len; ++j) {          // 구간 cpd 를 모아 삽입 정렬 → 중앙값(numpy median 과 같은 정의)
            double dth = et[s+j+1] - et[s+j];
            double c = dth > 1e-6 ? 0.5 / fmax(dth, 1e-6) : 0.0;
            int q = j; while (q > 0 && scr[q-1] > c) { scr[q] = scr[q-1]; --q; } scr[q] = c;
        }
        CPD[n] = (best_len % 2) ? scr[best_len/2] : 0.5 * (scr[best_len/2 - 1] + scr[best_len/2]);
    }
}
"""
_K = None


def _kernel():
    global _K
    if _K is None:
        import cupy as cp
        _K = cp.RawKernel(_SRC, "hyst_runs")
    return _K


def hyst_runs(values, theta, lengths, pse, wedge_cpd_min: float = 0.05):
    """values/theta: [N, L] (cupy 또는 numpy, 왼쪽 정렬) → (band_pairs, band_cpd, wedge_pairs, n_ext, ext_t, ext_v) cupy."""
    import cupy as cp
    V = cp.ascontiguousarray(cp.asarray(values, dtype=cp.float64))
    T = cp.ascontiguousarray(cp.asarray(theta, dtype=cp.float64))
    LEN = cp.ascontiguousarray(cp.asarray(lengths, dtype=cp.int32))
    N, L = V.shape
    ET = cp.zeros((N, L), cp.float64); EV = cp.zeros((N, L), cp.float64); SCR = cp.zeros((N, L), cp.float64)
    NEXT = cp.zeros(N, cp.int32); P = cp.zeros(N, cp.int32); C = cp.zeros(N, cp.float64); P2 = cp.zeros(N, cp.int32)
    if N:
        bs = 128
        _kernel()(((N + bs - 1) // bs,), (bs,), (V, T, LEN, np.int32(N), np.int32(L), np.float64(pse.pattern_hysteresis_cd_m2),
                                                 np.float64(pse.delta_lum_min_cd_m2), np.float64(pse.dark_lum_limit_cd_m2),
                                                 np.float64(pse.bright_michelson_min), np.float64(pse.pattern_cpd_min),
                                                 np.float64(wedge_cpd_min), np.float64(pse.pattern_cpd_max),
                                                 ET, EV, NEXT, P, C, P2, SCR))
    return P, C, P2, NEXT, ET, EV
