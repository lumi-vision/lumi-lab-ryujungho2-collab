"""줄무늬(규칙 패턴) 주파수 탐지기 — FFT·DCT 패치 특징 + E-Means 군집 (2026-11-13, 실험용 두 번째 탐지기).

measure.analyze_pattern(스캔선 위에서 밝은·어두운 줄을 하나씩 세는 방식)과 **같은 규칙 상수**(PATTERN_CPD, PATTERN_MIN_PAIRS,
면적·지속, 명암 ΔL ≥ 20 cd/m² · 어두운 쪽 < 160 cd/m², 시청 조건)를 쓰고, '줄을 찾는 방법'만 주파수 영역으로 바꾼다.
정답 규칙은 여전히 measure.py 이고, 이 모듈은 그것과 비교하려는 두 번째 탐지기다(scripts/eval_stripe_freq.py).

왜 주파수인가: 스캔선 방식은 '직선 위에서 폭이 일정한 줄이 11개 이어질 것'을 요구해 곡선·방사형·원근 줄무늬에서 끊긴다.
패치(시야각 약 2도) 하나 안에서는 그런 줄무늬도 거의 직선·등간격이라, 패치 스펙트럼에 날카로운 봉우리가 생긴다.

절차
  ① 패치    시야각 PATCH_DEG(2도 = 1 cpd 줄무늬 2주기) 크기 P(짝수), 간격 P/2 로 휘도(cd/m²) 패치를 자른다.
  ② FFT     평균을 빼고 2D Hann 창 → 파워 스펙트럼 |F|². 가장 센 봉우리 = (공간 주파수 → cpd, 방향 θ).
            ρ1 = 봉우리 3×3 과 그 대칭점 3×3 의 에너지 / 전체 교류 에너지   (규칙 줄무늬면 크다: 사각파 기본파 ≈ 81%)
            ρ2 = 첫 봉우리와 45° 넘게 다른 방향에서 같은 방식으로 잰 둘째 봉우리 집중도   (체커보드·격자면 크다)
            대역 에너지 6칸: cpd < 1, 1–2, 2–4, 4–8, 8–10, > 10 의 교류 에너지 비율
  ③ DCT     2D DCT-II(cv2.dct). 첫 행·첫 열(직류 제외) 에너지 비율 = 세로·가로 줄 방향성,
            가장 큰 계수 1개·4개의 에너지 비율 = 에너지 집중도(코사인 몇 개로 설명되는 반복 무늬).
  ④ 명암    패치 휘도 5·95 백분위의 차 ΔL ≥ 20 cd/m², 5 백분위 < 160 cd/m² (깜빡임·패턴 공통 명암 기준).
  ⑤ 규칙    ρ1 ≥ RHO_MIN 이고 ρ2 ≤ RHO2_MAX 이고 cpd ∈ PATTERN_CPD 이고 명암 조건 → '줄무늬 패치'.
  ⑥ E-Means ②·③·④ 특징 16차원을 군집(학습: fit_model). 합성 줄무늬 비율이 CLUSTER_STRIPE_MIN 이상인 군집의 패치는
            ρ1 문턱을 RHO_LOOSE 로 낮춰 인정 — 패치 안에서 주파수가 조금씩 변해 봉우리가 퍼지는 곡선·방사형·원근 줄무늬용.
            모델 파일이 없으면 ⑥ 없이 규칙만 쓴다(결과의 'emeans' = False).
  ⑦ 쌍 수   줄무늬 픽셀의 연결 성분마다, 법선 방향(성분의 θ 중앙값) 길이 / 한 주기 ≥ PATTERN_MIN_PAIRS 여야 인정.
  ⑧ 면적·지속 measure.analyze_pattern 과 같은 규칙: 면적 > PATTERN_AREA_STATIC(움직임 분류는 하지 않음 — 현재 상수에서는
            정지·반전·흐름 모두 40% 이고 흐름 제외가 꺼져 있어 판정에 영향이 없다), 1초 창 안 ≥ 0.5초.

E-Means: Bessa 외 2026, "e-means: outperforming k-means (in one-tenth the epochs)", arXiv:2611.09501 의 식 2(질량·합 EMA),
식 5(α 코사인 감쇠), 식 6·7(배치 B·α₀ 법칙), 부록 H.1(죽은 군집 복구), 마지막 epoch 평균 마무리를 numpy 로 직접 구현했다
(수식 출처: scripts/emeans_formula_figure.py, 원문 확인 2026-11-11). 공식 faiss 구현(NumpyEmeans)과 세부(초기화, 배치 법칙의
d > 16 분기 등)가 달라 결과가 같지 않을 수 있다.

구현 선택(근거 없음, 민감도 분석 대상): PATCH_DEG, RHO_MIN, RHO_LOOSE, CLUSTER_STRIPE_MIN, EMEANS_K, 특징 구성, 합성 학습 분포.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from lumi_synth import measure

PATCH_DEG = 2.0                   # 패치 한 변의 시야각(도) — 1 cpd 줄무늬가 2주기 들어감
PATCH_MIN, PATCH_MAX = 8, 64      # 패치 크기(px) 범위
# 문턱 근거(2026-11-13 측정, 384×216·패치 24 px, scripts/eval_stripe_freq.py 와 같은 생성기):
#   합성 줄무늬(직선·동심원·방사·나선·원근) ρ1 10 백분위 0.63~0.78, 중앙값 0.77~0.80 /
#   자연 영상(BBB·ED) 패치에 규칙 전체를 적용했을 때 오탐 패치 비율 ρ1 ≥ 0.35: 4.6~7.1%, ≥ 0.5: 2.1~3.9%.
#   둘째 봉우리 중앙값: 줄무늬 0.02~0.03, 체커보드 0.31, 밝기 경계 0.35.
RHO_MIN = 0.5                     # 줄무늬 패치: 첫 봉우리 집중도 하한 (규칙만 쓸 때)
RHO_LOOSE = 0.3                   # 줄무늬 군집에 속한 패치의 집중도 하한 (E-Means 사용 시)
RHO2_MAX = 0.2                    # 둘째(다른 방향) 봉우리 집중도 상한 — 체커보드·밝기 경계 제외.
                                  # 주의: 체커보드를 줄무늬로 보지 않는 것은 Harding FPA·Wilkins 1979(직교 성분이 위험을 낮춤)
                                  # 쪽 해석이고, ITU 의 'rows of repetitive elements' 와는 다르다. 스캔선 방식은 체커보드도 셈
CLUSTER_STRIPE_MIN = 0.5          # 군집 안 합성 줄무늬 비율이 이 이상이면 '줄무늬 군집'
EMEANS_K = 32
BANDS = (0.0, 1.0, 2.0, 4.0, 8.0, 10.0, np.inf)   # 대역 에너지 칸 (cpd)
N_FEAT = 10 + len(BANDS) - 1      # 특징 차원 = 16
MODEL_PATH = Path(__file__).resolve().parents[2] / "data" / "stripe_freq" / "emeans_model.npz"


# ---- ① 패치 ------------------------------------------------------------------------------------
def patch_size(width: int, height: int) -> int:
    """시청 조건(measure.pixels_per_degree)에서 PATCH_DEG 도에 해당하는 짝수 패치 크기."""
    p = int(round(PATCH_DEG * measure.pixels_per_degree(width, height) / 2.0)) * 2
    return int(np.clip(p, PATCH_MIN, PATCH_MAX))


def _grid(n: int, p: int, s: int) -> np.ndarray:
    """길이 n 을 크기 p·간격 s 패치로 덮는 시작 위치 (마지막 패치는 끝에 맞춤)."""
    if n <= p:
        return np.array([0])
    starts = list(range(0, n - p + 1, s))
    if starts[-1] != n - p:
        starts.append(n - p)
    return np.array(starts)


def extract_patches(lum2d: np.ndarray, p: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """휘도 한 장(H×W) → (패치 N×p×p, 행 시작, 열 시작). 화면이 p 보다 작으면 가장자리를 복제해 채운다."""
    h, w = lum2d.shape
    if h < p or w < p:
        lum2d = np.pad(lum2d, ((0, max(0, p - h)), (0, max(0, p - w))), mode="edge")
        h, w = lum2d.shape
    ys, xs = _grid(h, p, p // 2), _grid(w, p, p // 2)
    win = np.lib.stride_tricks.sliding_window_view(lum2d, (p, p))
    patches = win[ys[:, None], xs[None, :]].reshape(-1, p, p)
    return patches.astype(np.float32), ys, xs


# ---- ②·③·④ 특징 -------------------------------------------------------------------------------
_HANN_CACHE: dict[int, np.ndarray] = {}


def _hann2(p: int) -> np.ndarray:
    if p not in _HANN_CACHE:
        h = np.hanning(p).astype(np.float32)
        _HANN_CACHE[p] = np.outer(h, h)
    return _HANN_CACHE[p]


def _peak_energy(power: np.ndarray, iy: np.ndarray, ix: np.ndarray) -> np.ndarray:
    """(N×p×p, fftshift 된) 파워에서 (iy, ix) 3×3 과 대칭점 3×3 의 에너지 합 (원형으로 감쌈)."""
    n, p, _ = power.shape
    c = p // 2
    tot = np.zeros(n, np.float64)
    rows = np.arange(n)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            y1, x1 = (iy + dy) % p, (ix + dx) % p
            y2, x2 = (2 * c - iy + dy) % p, (2 * c - ix + dx) % p
            tot += power[rows, y1, x1]
            same = (y1 == y2) & (x1 == x2)                      # 봉우리가 대칭점과 겹치면(나이퀴스트) 두 번 세지 않음
            tot += np.where(same, 0.0, power[rows, y2, x2])
    return tot


def patch_features(patches: np.ndarray, ppd: float) -> dict:
    """패치(N×p×p, cd/m²) → 특징 사전.

    cpd, theta(도, 0 = 밝기가 x 방향으로 바뀜 = 세로 줄), rho1, rho2, dct_row, dct_col, dct_top1, dct_top4,
    contrast(ΔL, cd/m²), dark(5 백분위, cd/m²), bands(N×6), feat(N×16, 군집용).
    """
    n, p, _ = patches.shape
    lo = np.percentile(patches, 5, axis=(1, 2))
    hi = np.percentile(patches, 95, axis=(1, 2))
    x = patches - patches.mean(axis=(1, 2), keepdims=True)

    # ② FFT
    spec = np.fft.fftshift(np.fft.fft2(x * _hann2(p)), axes=(1, 2))
    power = (spec.real ** 2 + spec.imag ** 2).astype(np.float64)
    c = p // 2
    ky, kx = np.meshgrid(np.arange(p) - c, np.arange(p) - c, indexing="ij")
    rad = np.hypot(ky, kx)
    power[:, rad <= 1.0] = 0.0                                   # 직류와 Hann 창이 번진 바로 옆 칸 제외
    total = power.reshape(n, -1).sum(axis=1) + 1e-12
    flat = power.reshape(n, -1).argmax(axis=1)
    iy, ix = np.unravel_index(flat, (p, p))
    e1 = _peak_energy(power, iy, ix)
    rho1 = e1 / total
    # 봉우리 위치를 3×3 무게중심으로 다듬어 주파수·방향 추정
    rows = np.arange(n)
    wsum = np.zeros(n); fy = np.zeros(n); fx = np.zeros(n)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            wgt = power[rows, (iy + dy) % p, (ix + dx) % p]
            wsum += wgt; fy += wgt * (iy - c + dy); fx += wgt * (ix - c + dx)
    fy /= wsum + 1e-12; fx /= wsum + 1e-12
    cyc_px = np.hypot(fy, fx) / p                                # cycles / pixel
    cpd = cyc_px * ppd
    theta = (np.degrees(np.arctan2(fy, fx)) % 180.0)
    # 둘째 봉우리: 첫 봉우리 방향과 45° 넘게 다른 칸만
    ang = np.degrees(np.arctan2(ky, kx))[None] % 180.0
    dang = np.abs(ang - theta[:, None, None]) % 180.0
    dang = np.minimum(dang, 180.0 - dang)
    p2 = np.where(dang > 45.0, power, 0.0)
    flat2 = p2.reshape(n, -1).argmax(axis=1)
    iy2, ix2 = np.unravel_index(flat2, (p, p))
    rho2 = _peak_energy(power, iy2, ix2) / total
    # 대역 에너지
    cpd_grid = rad / p * ppd
    bands = np.stack([power[:, (cpd_grid >= a) & (cpd_grid < b)].sum(axis=1)
                      for a, b in zip(BANDS[:-1], BANDS[1:])], axis=1) / total[:, None]

    # ③ DCT
    d = np.stack([cv2.dct(np.ascontiguousarray(xi)) for xi in x]) ** 2
    d[:, 0, 0] = 0.0
    dtot = d.reshape(n, -1).sum(axis=1) + 1e-12
    dct_row = d[:, 0, 1:].sum(axis=1) / dtot                     # 첫 행 = 가로로만 변함 = 세로 줄
    dct_col = d[:, 1:, 0].sum(axis=1) / dtot
    srt = -np.sort(-d.reshape(n, -1), axis=1)
    dct_top1 = srt[:, 0] / dtot
    dct_top4 = srt[:, :4].sum(axis=1) / dtot

    contrast = hi - lo
    feat = np.column_stack([
        np.clip(cpd / 10.0, 0, 2), np.cos(np.radians(2 * theta)), np.sin(np.radians(2 * theta)),
        rho1, rho2, dct_row, dct_col, dct_top1, dct_top4, np.log10(1.0 + contrast) / 2.5, bands,
    ]).astype(np.float32)
    return {"cpd": cpd, "theta": theta, "rho1": rho1, "rho2": rho2, "dct_row": dct_row, "dct_col": dct_col,
            "dct_top1": dct_top1, "dct_top4": dct_top4, "contrast": contrast, "dark": lo, "bands": bands,
            "feat": feat}


def rule_ok(f: dict, rho_min: float | np.ndarray = RHO_MIN) -> np.ndarray:
    """⑤ 패치 규칙: 집중도(한 방향 봉우리)·공간 주파수·명암."""
    lo, hi = measure.PATTERN_CPD
    return ((f["rho1"] >= rho_min) & (f["rho2"] <= RHO2_MAX) & (f["cpd"] >= lo) & (f["cpd"] <= hi)
            & (f["contrast"] >= measure.DELTA_CD) & (f["dark"] < measure.DARK_LIMIT_CD))


# ---- ⑥ E-Means ----------------------------------------------------------------------------------
def emeans_recipe(n: int, k: int, epochs: int) -> tuple[int, float]:
    """식 6·7: 배치 크기 B 와 초기 갱신 비율 α₀ (B 는 [k, n], α₀ 는 1 이하로 자름)."""
    b = 0.0351 * k ** 0.135 * n ** 0.876 * epochs ** 0.813
    b = int(np.clip(round(b), min(k, n), n))
    a0 = 0.824 * k ** -0.462 * b ** 0.900 * n ** -0.450 * epochs ** -0.106
    return b, float(min(1.0, a0))


def _assign(x: np.ndarray, c: np.ndarray) -> np.ndarray:
    d = (x * x).sum(1)[:, None] - 2.0 * x @ c.T + (c * c).sum(1)[None, :]
    return d.argmin(axis=1)


def emeans(x: np.ndarray, k: int, epochs: int = 5, seed: int = 0, batch: int | None = None,
           alpha0: float | None = None, dead_frac: float = 0.1, eps: float = 0.1, split_eps: float = 1e-3
           ) -> tuple[np.ndarray, dict]:
    """E-Means (numpy 직접 구현) → (중심 k×d, 설정).

    질량 N_j·합 S_j 를 γ = 1 − α(t) 로 지수이동평균하고 c_j = S_j / N_j (식 2).
    α(t) = α₀ [ε + (1 − ε) · ½ (1 + cos(π t / M))], ε = 0.1, M = 전체 갱신 횟수 (식 5).
    5배치마다 N_j < 0.1 · 평균 질량인 군집을 가장 무거운 군집을 둘로 나눠 되살림: c ← c_donor ⊙ (1 ± ε'), N = N_donor / 2.
    마지막 epoch 에 배정된 점들의 평균으로 마무리.
    """
    x = np.asarray(x, np.float64)
    n, d = x.shape
    k = min(k, n)
    rng = np.random.default_rng(seed)
    b_law, a_law = emeans_recipe(n, k, epochs)
    b = batch or b_law
    a0 = alpha0 if alpha0 is not None else a_law
    c = x[rng.choice(n, k, replace=False)].copy()
    mass = np.full(k, b / k)
    sums = c * mass[:, None]
    steps = int(np.ceil(n / b)) * epochs
    t = 0
    last_sum = np.zeros_like(c); last_cnt = np.zeros(k)
    for ep in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, b):
            xb = x[order[s:s + b]]
            a = _assign(xb, c)
            cnt = np.bincount(a, minlength=k).astype(np.float64)
            sb = np.zeros_like(c); np.add.at(sb, a, xb)
            if ep == epochs - 1:
                last_sum += sb; last_cnt += cnt
            alpha = a0 * (eps + (1 - eps) * 0.5 * (1 + np.cos(np.pi * t / max(steps, 1))))
            mass = (1 - alpha) * mass + alpha * cnt
            sums = (1 - alpha) * sums + alpha * sb
            ok = mass > 1e-12
            c[ok] = sums[ok] / mass[ok, None]
            t += 1
            if t % 5 == 0:                                         # 죽은 군집 복구
                for j in np.flatnonzero(mass < dead_frac * mass.mean()):
                    donor = int(mass.argmax())
                    if donor == j:
                        continue
                    sign = rng.choice([-1.0, 1.0], size=d)
                    c[j] = c[donor] * (1 + split_eps * sign)
                    c[donor] = c[donor] * (1 - split_eps * sign)
                    mass[donor] /= 2.0; mass[j] = mass[donor]
                    sums[j] = c[j] * mass[j]; sums[donor] = c[donor] * mass[donor]
    ok = last_cnt > 0
    c[ok] = last_sum[ok] / last_cnt[ok, None]
    return c.astype(np.float32), {"batch": b, "alpha0": a0, "steps": steps, "batch_law": b_law, "alpha0_law": a_law}


def fit_model(feat: np.ndarray, is_stripe: np.ndarray, k: int = EMEANS_K, epochs: int = 5, seed: int = 0,
              path: Path | None = MODEL_PATH) -> dict:
    """패치 특징(N×16)과 합성 정답(줄무늬 여부)으로 E-Means 군집과 군집별 줄무늬 비율을 만든다(저장은 path)."""
    mu = feat.mean(axis=0); sd = feat.std(axis=0) + 1e-6
    z = (feat - mu) / sd
    cent, info = emeans(z, k, epochs=epochs, seed=seed)
    lab = _assign(z.astype(np.float64), cent.astype(np.float64))
    cnt = np.bincount(lab, minlength=len(cent))
    ratio = np.bincount(lab, weights=is_stripe.astype(np.float64), minlength=len(cent)) / np.maximum(cnt, 1)
    model = {"centroids": cent, "mu": mu.astype(np.float32), "sd": sd.astype(np.float32),
             "stripe_ratio": ratio.astype(np.float32), "count": cnt, **{f"info_{a}": v for a, v in info.items()}}
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, **model)
    return model


def load_model(path: Path = MODEL_PATH) -> dict | None:
    if not path.exists():
        return None
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def stripe_cluster(feat: np.ndarray, model: dict) -> tuple[np.ndarray, np.ndarray]:
    """패치 특징 → (군집 번호, 그 군집의 줄무늬 비율)."""
    z = (feat - model["mu"]) / model["sd"]
    lab = _assign(z.astype(np.float64), model["centroids"].astype(np.float64))
    return lab, model["stripe_ratio"][lab]


# ---- ⑤~⑦ 한 프레임 ------------------------------------------------------------------------------
def frame_stripes(lum2d: np.ndarray, model: dict | None = None) -> dict:
    """휘도 한 장 → 줄무늬 픽셀 마스크와 패치별 결과."""
    h, w = lum2d.shape
    p = patch_size(w, h)
    ppd = measure.pixels_per_degree(w, h)
    patches, ys, xs = extract_patches(lum2d, p)
    f = patch_features(patches, ppd)
    rho_min = np.full(len(patches), RHO_MIN)
    cluster = ratio = None
    if model is not None:
        cluster, ratio = stripe_cluster(f["feat"], model)
        rho_min = np.where(ratio >= CLUSTER_STRIPE_MIN, RHO_LOOSE, RHO_MIN)
    ok = rule_ok(f, rho_min).reshape(len(ys), len(xs))
    # 픽셀 ← 가장 가까운 패치 중심
    s = p // 2
    iy = np.clip(np.round((np.arange(h) - p / 2 + 0.5) / s).astype(int), 0, len(ys) - 1)
    ix = np.clip(np.round((np.arange(w) - p / 2 + 0.5) / s).astype(int), 0, len(xs) - 1)
    pix = ok[iy[:, None], ix[None, :]]
    cpd_map = f["cpd"].reshape(len(ys), len(xs))[iy[:, None], ix[None, :]]
    th_map = f["theta"].reshape(len(ys), len(xs))[iy[:, None], ix[None, :]]
    keep = _keep_by_pairs(pix, cpd_map, th_map)
    return {"mask": keep, "patch_ok": ok, "patch": p, "features": f, "cluster": cluster, "cluster_ratio": ratio,
            "rows": ys, "cols": xs}


def _keep_by_pairs(pix: np.ndarray, cpd_map: np.ndarray, th_map: np.ndarray) -> np.ndarray:
    """⑦ 줄무늬 픽셀의 연결 성분마다 법선 방향 길이 / 한 주기 ≥ PATTERN_MIN_PAIRS 인 성분만 남긴다."""
    h, w = pix.shape
    keep = np.zeros_like(pix)
    n_lab, lab = cv2.connectedComponents(pix.astype(np.uint8), connectivity=8)
    yy, xx = np.mgrid[:h, :w]
    for j in range(1, n_lab):
        m = lab == j
        th = np.radians(np.median(th_map[m]))
        cpd = float(np.median(cpd_map[m]))
        period = measure.pixels_per_degree(w, h, np.degrees(th)) / max(cpd, 1e-6)
        proj = xx[m] * np.cos(th) + yy[m] * np.sin(th)
        extent = np.percentile(proj, 99) - np.percentile(proj, 1)
        if extent / period >= measure.PATTERN_MIN_PAIRS:
            keep |= m
    return keep


# ---- 쿼드트리 (2026-11-13) ----------------------------------------------------------------------
QT_TOP_DEG = 8.0                  # 처음 블록 한 변의 시야각(도)
QT_MIN_PX = 8                     # 블록 최소 크기(px)
QT_MIN_PERIODS = 2.0              # 쪼갠 블록에 측정 주파수의 주기가 이 수 이상 들어가야 쪼갬(봉우리를 잴 수 있는 최소)
QT_UNIFORM = 0.7                  # 블록 전체 인정: 네 사분면 표준편차의 최소/최대 비 하한(명암이 고르게 퍼짐) — 구현 선택.
                                  # 0.3 이면 사분면에 줄무늬가 4/23 만 걸쳐도(비 0.42) 통과해 경계를 넘쳐 칠했다(2026-11-13 확인)
QT_SPLIT_RHO = 0.0                # 사분면이 고르고 규칙도 못 넘은 블록은 첫 봉우리 집중도가 이 이상일 때만 더 쪼갬 — 구현 선택.
                                  # 0.2 로 두었더니 동심원 중심(고리가 사방으로 휘어 스펙트럼이 원으로 퍼짐, ρ1 < 0.2) 블록을
                                  # 쪼개지 않고 버려 중심 띠가 통째로 빠졌다(2026-11-13) → 명암이 있으면 항상 쪼갬


QT_REFINE_REL = 0.5               # 가버 다듬기: 블록 안 응답 90 백분위의 이 비율 이상인 픽셀만 남김 — 구현 선택
QT_REFINE_ABS = measure.DELTA_CD / (2 * np.pi)   # 응답 하한(cd/m²): 명암 ΔL 사각파의 기본파 진폭은 ΔL/π, 그 절반 = ΔL 20 기준


REFINE_ANGLES = tuple(range(0, 180, 15))   # 가버 다듬기 방향 묶음 (도)
REFINE_SCALES = (0.8, 1.0, 1.25)            # 블록 cpd 에 곱하는 주파수 묶음


def gabor_refine(lum2d: np.ndarray, blocks: list[tuple[int, int, int, float, float]]) -> np.ndarray:
    """쿼드트리가 인정한 블록 (y, x, 크기, cpd, θ) 안에서, 그 블록이 잰 주파수 근처의 복소 가버 필터 응답 세기로
    줄무늬가 실제로 있는 픽셀만 남긴다 → H×W bool.

    가버 핵 g = G_σ(x, y) · exp(i 2π (x cosφ + y sinφ) / P), Σ G = 1, P = 주기(px) = 1도당 픽셀 / (cpd · 배율), σ = P.
    진폭 A 인 같은 주파수 사인 무늬에 대한 응답 세기는 A/2 (σ = P 면 직류가 새는 양 exp(−2π²) 은 무시할 만함).
    방향 φ ∈ REFINE_ANGLES(15° 간격), 배율 ∈ REFINE_SCALES 묶음의 최대 응답을 쓴다 — 블록 하나 안에서도 방향·간격이
    위치마다 바뀌는 동심원·방사형 줄무늬 때문(2026-11-13: 블록 방향 하나로만 거르면 동심원 IoU 0.96 → 0.80,
    방사형 0.72 → 0.25 로 지워졌다). 줄무늬 여부는 쿼드트리가 이미 정했고, 여기서는 '어디까지인가'만 정한다.
    블록마다 |응답| ≥ max(QT_REFINE_REL · 90 백분위, QT_REFINE_ABS) 인 픽셀만 남긴다. 경계 오차는 대략 ±σ/2.
    블록 밖은 건드리지 않으므로 넘친 곳을 깎기만 하고 빠진 곳을 채우지는 않는다.
    """
    from collections import defaultdict
    h, w = lum2d.shape
    keep = np.zeros((h, w), bool)
    ppd = measure.pixels_per_degree(w, h)
    groups: dict[float, list] = defaultdict(list)
    for y, x, s, cpd, th in blocks:                         # 비슷한 주파수 블록은 한 번에 거름
        groups[round(cpd * 4) / 4].append((y, x, s))
    for cpd, bl in groups.items():
        periods = [ppd / max(cpd * k, 1e-6) for k in REFINE_SCALES]
        r = int(np.ceil(3 * max(periods)))
        y0 = max(0, min(b[0] for b in bl) - r); y1 = min(h, max(b[0] + b[2] for b in bl) + r)
        x0 = max(0, min(b[1] for b in bl) - r); x1 = min(w, max(b[1] + b[2] for b in bl) + r)
        sub = np.ascontiguousarray(lum2d[y0:y1, x0:x1], dtype=np.float32)
        amp = np.zeros_like(sub)
        for period in periods:
            rr = int(np.ceil(3 * period))
            yy, xx = np.mgrid[-rr:rr + 1, -rr:rr + 1].astype(np.float32)
            g = np.exp(-(xx ** 2 + yy ** 2) / (2 * period ** 2)); g /= g.sum()
            for ang in REFINE_ANGLES:
                ph = 2 * np.pi / period * (xx * np.cos(np.radians(ang)) + yy * np.sin(np.radians(ang)))
                re = cv2.filter2D(sub, -1, (g * np.cos(ph)).astype(np.float32), borderType=cv2.BORDER_REFLECT)
                im = cv2.filter2D(sub, -1, (g * np.sin(ph)).astype(np.float32), borderType=cv2.BORDER_REFLECT)
                np.maximum(amp, np.hypot(re, im), out=amp)
        for y, x, s in bl:
            a = amp[y - y0:y - y0 + s, x - x0:x - x0 + s]
            thr = max(QT_REFINE_REL * float(np.percentile(a, 90)), QT_REFINE_ABS)
            keep[y:y + s, x:x + s] |= a >= thr
    return keep


def frame_stripes_quadtree(lum2d: np.ndarray, model: dict | None = None, refine: bool = False) -> dict:
    """쿼드트리 방식: 큰 블록에서 시작해 애매한 블록만 4개로 쪼갠다 → 줄무늬 픽셀 마스크와 블록 기록.

    블록마다 (같은 크기 블록끼리 한 번에 FFT·DCT)
      1. 명암 문 1~99 백분위 차 < 20 cd/m²                          → 버림 (FFT 하지 않음)
      2. 줄무늬 규칙(⑤, E-Means 사용 시 군집별 문턱) 통과 이고
         네 사분면 모두 명암 5~95 백분위 차 ≥ 20, 표준편차 최소/최대 ≥ QT_UNIFORM → 블록 전체 인정
      3. 그 밖에, 쪼갠 블록 한 변 ≥ max(QT_MIN_PX, QT_MIN_PERIODS · 주기) 이고
         (사분면이 고르지 않거나 ρ1 ≥ QT_SPLIT_RHO)                      → 4개로 쪼갬
      4. 더 못 쪼개면 규칙 통과일 때, 또는 조상 블록이 규칙을 통과했고 이 블록에 명암(ΔL ≥ 20·어두운 쪽 < 160)과
         ρ1 ≥ RHO_LOOSE 가 있을 때 인정한다(줄무늬 여부는 큰 척도에서, 위치는 작은 블록에서 정함 — 작은 블록은 주기가
         적어 혼자서는 규칙을 못 넘기 쉽다). 이때 cpd·방향은 조상 블록에서 잰 값을 쓴다.
    주기는 그 블록에서 잰 cpd 로 계산하고, cpd 가 하한(1) 미만이면 하한으로 본다. 그래서 1 cpd 무늬는 2도보다 작게
    쪼개지 않는다(블록 안에 두 주기가 있어야 봉우리를 잼).
    refine=True 이면 인정된 블록을 통째로 칠하지 않고 gabor_refine 으로 줄무늬가 실제로 있는 픽셀만 남긴다
    (블록 크기 하한 때문에 경계에 걸친 블록이 영역 밖까지 칠하던 문제용, 2026-11-13).
    """
    from collections import defaultdict
    h, w = lum2d.shape
    ppd = measure.pixels_per_degree(w, h)
    lo, _ = measure.PATTERN_CPD
    s0 = int(min(max(QT_MIN_PX, round(QT_TOP_DEG * ppd)), h, w))
    s0 -= s0 % 2
    # (y, x, 크기, 부모가 확인한 (cpd, θ) 또는 None)
    pending = [(int(y), int(x), s0, None) for y in _grid(h, s0, s0) for x in _grid(w, s0, s0)]
    pix = np.zeros((h, w), bool)
    cpd_map = np.zeros((h, w), np.float32)
    th_map = np.zeros((h, w), np.float32)
    leaves: list[tuple[int, int, int, str]] = []         # (y, x, 크기, 'stripe' | 'reject' | 'flat')
    accepted: list[tuple[int, int, int, float, float]] = []   # 인정된 블록 (y, x, 크기, cpd, θ)
    n_fft = 0
    while pending:
        groups: dict[int, list] = defaultdict(list)
        for b in pending:
            groups[b[2]].append(b)
        pending = []
        for s, bl in groups.items():
            blocks = np.stack([lum2d[y:y + s, x:x + s] for y, x, _, _ in bl]).astype(np.float32)
            gate = (np.percentile(blocks, 99, axis=(1, 2)) - np.percentile(blocks, 1, axis=(1, 2))) >= measure.DELTA_CD
            for i in np.flatnonzero(~gate):
                leaves.append((*bl[i][:2], s, "flat"))
            idx = np.flatnonzero(gate)
            if not idx.size:
                continue
            sub = blocks[idx]
            f = patch_features(sub, ppd)
            n_fft += len(idx)
            rho_min = np.full(len(idx), RHO_MIN)
            if model is not None:
                _, ratio = stripe_cluster(f["feat"], model)
                rho_min = np.where(ratio >= CLUSTER_STRIPE_MIN, RHO_LOOSE, RHO_MIN)
            ok = rule_ok(f, rho_min)
            q2 = s // 2
            quads = [sub[:, :q2, :q2], sub[:, :q2, q2:2 * q2], sub[:, q2:2 * q2, :q2], sub[:, q2:2 * q2, q2:2 * q2]]
            h2 = q2 - (q2 % 2)                        # 자식 크기는 짝수로(cv2.dct 는 홀수 크기를 못 함): h2 와 s − h2
            qc = np.stack([np.percentile(a, 95, axis=(1, 2)) - np.percentile(a, 5, axis=(1, 2)) for a in quads], 1)
            qs = np.stack([a.std(axis=(1, 2)) for a in quads], 1)
            uniform = (qc.min(1) >= measure.DELTA_CD) & (qs.min(1) >= QT_UNIFORM * np.maximum(qs.max(1), 1e-6))
            period = ppd / np.maximum(f["cpd"], lo)
            can_split = (s - h2) >= np.maximum(QT_MIN_PX, QT_MIN_PERIODS * period)
            want_split = ~uniform | (f["rho1"] >= QT_SPLIT_RHO)
            # 부모가 줄무늬로 확인된 블록(inherit)은 더 못 쪼갤 때 명암·완화 문턱만 보면 인정(줄무늬 여부는 큰 척도에서 판단)
            inherit_ok = (f["contrast"] >= measure.DELTA_CD) & (f["dark"] < measure.DARK_LIMIT_CD) & (f["rho1"] >= RHO_LOOSE)
            for j, i in enumerate(idx):
                y, x, _, parent = bl[i]
                mine = (float(f["cpd"][j]), float(f["theta"][j])) if ok[j] else parent
                leaf_ok = ok[j] or (parent is not None and inherit_ok[j])
                if ok[j] and uniform[j] or (leaf_ok and not can_split[j]):
                    pix[y:y + s, x:x + s] = True
                    cpd_map[y:y + s, x:x + s] = mine[0]
                    th_map[y:y + s, x:x + s] = mine[1]
                    leaves.append((y, x, s, "stripe"))
                    accepted.append((y, x, s, mine[0], mine[1]))
                elif can_split[j] and want_split[j]:
                    pending += [(y, x, h2, mine), (y, x + h2, s - h2, mine), (y + h2, x, s - h2, mine),
                                (y + h2, x + h2, s - h2, mine)]
                else:
                    leaves.append((y, x, s, "reject"))
    if refine and accepted:
        pix = pix & gabor_refine(lum2d, accepted)
    keep = _keep_by_pairs(pix, cpd_map, th_map)
    return {"mask": keep, "leaves": leaves, "accepted": accepted, "n_fft": n_fft, "top": s0}


# ---- ⑧ 영상 ----------------------------------------------------------------------------------------
def analyze_pattern_freq(frames_bgr: np.ndarray, fps: float, model: dict | None | str = "auto",
                         lum: np.ndarray | None = None, method: str = "grid", refine: bool = False,
                         keep_blocks: bool = False) -> dict:
    """주파수 방식 줄무늬 판정. 돌려주는 값은 measure.analyze_pattern 과 같은 꼴(움직임 관련 값 제외).

    model='auto' 이면 MODEL_PATH 의 E-Means 모델을 쓰고, 없으면 규칙만 쓴다. None 이면 규칙만.
    method: 'grid'(고정 2도 패치, frame_stripes) | 'quadtree'(frame_stripes_quadtree)
    refine: 쿼드트리에서 가버 다듬기(gabor_refine) 사용. keep_blocks: 쿼드트리 블록 기록(프레임별 leaves)을 함께 돌려줌.
    """
    if method not in ("grid", "quadtree"):
        raise ValueError(f"method 는 'grid' 또는 'quadtree': {method}")
    if method == "grid":
        one = frame_stripes
    else:
        def one(l2, mdl):
            return frame_stripes_quadtree(l2, mdl, refine=refine)
    blocks = []
    measure._check_input(frames_bgr, fps)
    if lum is None:
        lum = measure.luminance_cd(frames_bgr)
    if isinstance(model, str):
        model = load_model()
    t_len = len(lum)
    stripes = np.zeros(lum.shape, dtype=bool)
    for t in range(t_len):
        r1 = one(lum[t], model)
        stripes[t] = r1["mask"]
        if keep_blocks:
            blocks.append(r1.get("leaves", []))
    area = stripes.mean(axis=(1, 2))
    cand = area > measure.PATTERN_AREA_STATIC
    w = measure.window_frames(fps)
    need = max(1, int(np.ceil(measure.PATTERN_MIN_SEC * fps - 1e-9)))
    cs = np.concatenate([[0], np.cumsum(cand)])
    counts = cs[np.minimum(np.arange(t_len) + w, t_len)] - cs[:t_len]
    hazard_frames = np.zeros(t_len, dtype=bool)
    for s in np.flatnonzero(counts >= need):
        hazard_frames[s:s + w] = True
    hazard_frames &= cand
    hazard_mask = stripes[hazard_frames].any(axis=0) if hazard_frames.any() else np.zeros(lum.shape[1:], bool)
    return {"stripe_area": area, "stripes": stripes, "candidate": cand,
            "max_stripe_sec": float(counts.max()) / fps if t_len else 0.0,
            "peak_stripe_area": float(area.max()) if t_len else 0.0,
            "dangerous": bool((counts >= need).any()), "hazard_frames": hazard_frames, "hazard_mask": hazard_mask,
            "emeans": model is not None, "method": method, "refine": refine,
            **({"blocks": blocks} if keep_blocks else {})}


# ---- 합성 패턴 (학습·평가용) -----------------------------------------------------------------------
PATTERN_KINDS = ("straight", "concentric", "radial", "spiral", "chirp", "checker")


def pattern_phase(kind: str, h: int, w: int, ppd: float, cpd: float, angle: float = 0.0,
                  center: tuple[float, float] = (0.5, 0.5), spokes: int = 24, chirp: float = 1.0
                  ) -> tuple[np.ndarray, np.ndarray]:
    """줄무늬 위상 φ(y, x)(라디안)와 국소 공간 주파수(cpd) 지도. 줄 = 사각파(φ).

    straight   φ = 2π f (x cosθ + y sinθ)                       국소 f 일정
    concentric φ = 2π f r                                        국소 f 일정
    radial     φ = spokes · atan2                                국소 f = spokes / (2π r)   (중심 가까울수록 높음)
    spiral     φ = 2π f r + spokes · atan2                       국소 f = |∇φ| / 2π
    chirp      φ = 2π f (u + chirp · u² / (2 L)), u = 회전 좌표   원근처럼 한쪽으로 갈수록 주파수가 1 + chirp 배까지 커짐
    checker    두 방향 사각파의 곱 (국소 f 는 한 방향 값)
    (f = cpd / ppd cycles/px)
    """
    f = cpd / ppd
    yy, xx = np.mgrid[:h, :w].astype(np.float64)
    cy, cx = center[0] * h, center[1] * w
    dy, dx = yy - cy, xx - cx
    r = np.hypot(dy, dx) + 1e-6
    a = np.radians(angle)
    u = dx * np.cos(a) + dy * np.sin(a)
    if kind in ("straight", "checker"):
        phi = 2 * np.pi * f * u
        loc = np.full((h, w), cpd)
    elif kind == "concentric":
        phi = 2 * np.pi * f * r
        loc = np.full((h, w), cpd)
    elif kind == "radial":
        phi = spokes * np.arctan2(dy, dx)
        loc = spokes / (2 * np.pi * r) * ppd
    elif kind == "spiral":
        phi = 2 * np.pi * f * r + spokes * np.arctan2(dy, dx)
        loc = np.hypot(f, spokes / (2 * np.pi * r)) * ppd
    elif kind == "chirp":
        span = float(max(h, w))
        u0 = u - u.min()
        phi = 2 * np.pi * f * (u0 + chirp * u0 ** 2 / (2 * span))
        loc = f * (1 + chirp * u0 / span) * ppd
    else:
        raise ValueError(kind)
    return phi, loc


def render_pattern(kind: str, h: int, w: int, ppd: float, cpd: float, lo_cd: float, hi_cd: float,
                   **kw) -> tuple[np.ndarray, np.ndarray]:
    """합성 줄무늬 휘도(cd/m²)와 국소 cpd 지도."""
    phi, loc = pattern_phase(kind, h, w, ppd, cpd, **kw)
    s = np.sin(phi) >= 0
    if kind == "checker":
        phi2, _ = pattern_phase("straight", h, w, ppd, cpd, angle=kw.get("angle", 0.0) + 90.0,
                                center=kw.get("center", (0.5, 0.5)))
        s = s ^ (np.sin(phi2) >= 0)
    return np.where(s, hi_cd, lo_cd).astype(np.float32), loc


def cd_to_bgr(lum: np.ndarray) -> np.ndarray:
    """회색 휘도(cd/m²) → sRGB 8비트 BGR."""
    lin = np.clip(lum / measure.WHITE_CD, 0, 1)
    c = np.where(lin <= 0.0031308, lin * 12.92, 1.055 * np.power(lin, 1 / 2.4) - 0.055)
    g = np.clip(np.rint(c * 255.0), 0, 255).astype(np.uint8)
    return np.repeat(g[..., None], 3, axis=-1)


def training_patches(n: int, width: int, height: int, backgrounds: list[np.ndarray] | None = None,
                     seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """E-Means 학습용 패치 (특징 N×16, 정답 줄무늬 여부 N, 종류 이름 N).

    정답 = 줄무늬 종류(체커보드 제외)이고 패치 중심의 국소 cpd ∈ PATTERN_CPD 이고 명암 조건을 만족.
    배경(backgrounds: 휘도 H×W 목록)이 있으면 절반은 자연 영상 패치(정답 0), 없으면 평탄·잡음·경계 패치를 쓴다.
    """
    rng = np.random.default_rng(seed)
    ppd = measure.pixels_per_degree(width, height)
    p = patch_size(width, height)
    lo_cpd, hi_cpd = measure.PATTERN_CPD
    pats, labels, kinds = [], [], []
    n_neg = n // 2
    for _ in range(n - n_neg):
        kind = rng.choice(PATTERN_KINDS, p=[0.35, 0.15, 0.15, 0.1, 0.15, 0.1])
        cpd = float(np.exp(rng.uniform(np.log(0.4), np.log(16.0))))
        lo = float(rng.uniform(1, 120)); hi = lo + float(rng.uniform(5, 160))
        size = 4 * p
        img, loc = render_pattern(kind, size, size, ppd, cpd, lo, hi, angle=float(rng.uniform(0, 180)),
                                  center=(float(rng.uniform(-0.5, 1.5)), float(rng.uniform(-0.5, 1.5))),
                                  spokes=int(rng.integers(8, 60)), chirp=float(rng.uniform(0.2, 2.0)))
        img = img + rng.normal(0, rng.uniform(0, 4), img.shape).astype(np.float32)
        y0, x0 = rng.integers(0, size - p, 2)
        patch = img[y0:y0 + p, x0:x0 + p]
        c_loc = float(loc[y0 + p // 2, x0 + p // 2])
        ok = (kind != "checker") and lo_cpd <= c_loc <= hi_cpd and (hi - lo) >= measure.DELTA_CD \
            and lo < measure.DARK_LIMIT_CD
        pats.append(patch); labels.append(ok); kinds.append(str(kind))
    for _ in range(n_neg):
        if backgrounds:
            bg = backgrounds[int(rng.integers(len(backgrounds)))]
            y0 = int(rng.integers(0, bg.shape[0] - p + 1)); x0 = int(rng.integers(0, bg.shape[1] - p + 1))
            patch = bg[y0:y0 + p, x0:x0 + p].astype(np.float32); kind = "natural"
        else:
            choice = rng.integers(3)
            base = float(rng.uniform(5, 180))
            if choice == 0:
                patch = np.full((p, p), base, np.float32); kind = "flat"
            elif choice == 1:
                patch = (base + rng.normal(0, rng.uniform(2, 40), (p, p))).astype(np.float32); kind = "noise"
            else:
                yy, xx = np.mgrid[:p, :p]
                a = rng.uniform(0, np.pi)
                patch = np.where(np.cos(a) * (xx - p / 2) + np.sin(a) * (yy - p / 2) > 0, base, base + 60).astype(np.float32)
                kind = "edge"
        pats.append(patch); labels.append(False); kinds.append(kind)
    f = patch_features(np.stack(pats), ppd)
    return f["feat"], np.array(labels), np.array(kinds)
