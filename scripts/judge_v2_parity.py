"""판정기 v2 동일성 검사 — GPU 일괄판이 원본(팀원 Partial B2) 함수와 같은 답을 내는가.

1) 무작위 단면(평탄 구간·잡음·줄무늬 섞음) 수천 개에서 extract_hysteresis_extrema_1d / _best_open_run_from_extrema 대조
2) 실제 판정 중 원본 함수에 들어간 단면을 가로채 같은 대조 (--clip 지정 시)

    python scripts/judge_v2_parity.py [--n 5000] [--clip data/hazard7/big1/val/val_pattern_hazard_000.mp4]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from lumi_judge import spatial as S  # noqa: E402
from lumi_judge.config import PSEConfig  # noqa: E402
from lumi_judge.gpu_profiles import best_open_run_batched, hysteresis_extrema_batched  # noqa: E402


def random_profiles(n, rng):
    out = []
    for _ in range(n):
        L = int(rng.integers(2, 400))
        kind = rng.integers(0, 4)
        x = np.linspace(0, 1, L)
        if kind == 0:
            v = 100 + 80 * np.sign(np.sin(2 * np.pi * rng.uniform(2, 40) * x))
        elif kind == 1:
            v = 100 + rng.uniform(5, 120) * np.sin(2 * np.pi * rng.uniform(1, 60) * x) + rng.normal(0, rng.uniform(0, 15), L)
        elif kind == 2:
            r = int(rng.integers(1, 8)); v = np.repeat(rng.uniform(0, 200, L // r + 1), r)[:L]
        else:
            v = rng.uniform(0, 200, L)
        v = np.round(v.astype(np.float64), int(rng.integers(0, 3)))
        th = np.cumsum(rng.uniform(0.01, 0.2, L))
        out.append((v, th))
    return out


def compare(profiles, pse, dev="cuda"):
    h = pse.pattern_hysteresis_cd_m2
    N = len(profiles); L = max(len(v) for v, _ in profiles)
    V = np.zeros((N, L)); T = np.zeros((N, L)); lens = np.array([len(v) for v, _ in profiles])
    for i, (v, t) in enumerate(profiles):
        V[i, :len(v)] = v; T[i, :len(t)] = t
    t0 = time.time()
    et, ev, ne = hysteresis_extrema_batched(torch.tensor(V, device=dev), torch.tensor(T, device=dev), torch.tensor(lens, device=dev), h)
    pairs, cpd = best_open_run_batched(et, ev, ne, pse.delta_lum_min_cd_m2, pse.dark_lum_limit_cd_m2, pse.bright_michelson_min,
                                       pse.pattern_cpd_min, pse.pattern_cpd_max)
    torch.cuda.synchronize(); t_gpu = time.time() - t0
    et, ev, ne, pairs, cpd = (z.cpu().numpy() for z in (et, ev, ne, pairs, cpd))
    t0 = time.time()
    bad_ext = bad_run = 0
    for i, (v, t) in enumerate(profiles):
        rt, rv = S.extract_hysteresis_extrema_1d(v, t, h)
        n = int(ne[i])
        if n != len(rv) or not (np.allclose(et[i, :n], rt) and np.allclose(ev[i, :n], rv)):
            bad_ext += 1
            continue
        rp, rc = S._best_open_run_from_extrema(rt, rv, pse, pse.pattern_cpd_min)
        if int(pairs[i]) != rp or not np.isclose(cpd[i], rc):
            bad_run += 1
    t_cpu = time.time() - t0
    return bad_ext, bad_run, t_gpu, t_cpu


def compare_kernel(profiles, pse):
    """CUDA 커널판(gpu_kernel.hyst_runs) 대조 — band·wedge 쌍 수와 band cpd."""
    import cupy as cp
    from lumi_judge.gpu_kernel import hyst_runs
    h = pse.pattern_hysteresis_cd_m2
    N = len(profiles); L = max(len(v) for v, _ in profiles)
    V = np.zeros((N, L)); T = np.zeros((N, L)); lens = np.array([len(v) for v, _ in profiles])
    for i, (v, t) in enumerate(profiles):
        V[i, :len(v)] = v; T[i, :len(t)] = t
    hyst_runs(V[:2], T[:2], lens[:2], pse)            # 컴파일 예열
    cp.cuda.Device().synchronize(); t0 = time.time()
    P, C, P2, NE, ET, EV = hyst_runs(V, T, lens, pse)
    cp.cuda.Device().synchronize(); tg = time.time() - t0
    P, C, P2, NE, ET, EV = (cp.asnumpy(z) for z in (P, C, P2, NE, ET, EV))
    t0 = time.time(); bad = 0
    for i, (v, t) in enumerate(profiles):
        bp, bc, wp = S.evaluate_1d_open_dual(v, t, pse)
        rt, rv = S.extract_hysteresis_extrema_1d(v, t, h)
        n = int(NE[i])
        if n != len(rv) or not (np.allclose(ET[i, :n], rt) and np.allclose(EV[i, :n], rv)) or int(P[i]) != bp or int(P2[i]) != wp or not np.isclose(C[i], bc):
            bad += 1
    return bad, tg, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--clip", default="")
    ap.add_argument("--kernel", action="store_true", help="CUDA 커널판(CuPy)도 대조")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    pse = PSEConfig()
    prof = random_profiles(a.n, np.random.default_rng(0))
    be, br, tg, tc = compare(prof, pse)
    print(f"무작위 {len(prof)}: 극값 불일치 {be} · 최장 구간 불일치 {br} · GPU {tg:.2f}s vs 원본 {tc:.2f}s")
    if a.kernel:
        bk, tg, tc = compare_kernel(prof, pse)
        print(f"  커널: 불일치 {bk} · 커널 {tg*1000:.1f}ms vs 원본 {tc:.2f}s")
    if a.clip:
        import lumi_judge as J
        from make_hazard7_dataset import FPS, decode
        caught = []
        orig = S.extract_hysteresis_extrema_1d

        def spy(values, theta_deg, hysteresis):
            if len(caught) < 20000:
                caught.append((np.array(values, np.float64), np.array(theta_deg, np.float64)))
            return orig(values, theta_deg, hysteresis)
        S.extract_hysteresis_extrema_1d = spy
        x = decode(Path(a.clip))
        J.analyze_frames(list(x[:48]), FPS, device="cuda")
        S.extract_hysteresis_extrema_1d = orig
        be, br, tg, tc = compare(caught, pse)
        print(f"실제 단면 {len(caught)} ({a.clip}): 극값 불일치 {be} · 최장 구간 불일치 {br} · GPU {tg:.2f}s vs 원본 {tc:.2f}s")
        if a.kernel:
            bk, tg, tc = compare_kernel(caught, pse)
            print(f"  커널: 불일치 {bk} · 커널 {tg*1000:.1f}ms vs 원본 {tc:.2f}s")


if __name__ == "__main__":
    main()
