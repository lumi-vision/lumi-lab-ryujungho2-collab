"""격자 판정기 G — 정지 패턴(P1): 후보는 필터로, 판정은 규격대로, 면적은 실제 줄무늬 픽셀로 (docs/GRID_JUDGE_DESIGN.md §3 P1).

절차(프레임마다)
  1. 밝기 L (cd/m², 흰색 200 · sRGB→선형, 깜빡임과 같은 정의)
  2. 후보 — 방향 12(15° 간격) × 주파수 8(1–10 cpd, 로그 간격) 복소 필터(주파수 영역, GPU)로 픽셀마다 가장 강한
     줄무늬 성분의 방향 θ·주파수 f·진폭 a(cd/m²) 를 구한다. 진폭으로 본 밝기 폭(2a)이 문턱의 일부(기본 0.6×20 cd/m²)
     이상이면 '줄무늬 후보 픽셀'. (필터는 후보만 낸다 — 판정은 4 단계의 정확한 확인으로)
  3. 격자 32×18 — 칸마다 후보 픽셀 비율·주된 θ·f. 이웃 칸끼리 f 비 ≤ 1.5, 방향 차 ≤ 20° 이면 같은 영역(곡선·방사는
     방향이 칸마다 조금씩 변하므로 '이웃끼리만' 비교).
  4. 확인 — 영역에서 진폭이 큰 점들을 씨앗으로, **줄무늬에 수직인 방향(국소 법선)을 따라 경로를 그리며** 밝기 단면을 읽고
     원본 판정기와 같은 히스테리시스 쌍 세기(밝기 차 ≥ 20 cd/m², 어두운 막대 < 160 cd/m², 1–10 cpd, `gpu_kernel.hyst_runs`)
     로 쌍 수를 센다. 직선·동심·방사·곡선 줄무늬 모두 경로가 법선을 따라 휘므로 같은 방법으로 센다.
  5. 판정 — 쌍 > 5 로 확인된 영역의 **후보 픽셀 면적이 화면 > 40 %** 이면 그 프레임은 정지 패턴. 1 초 창 안에 0.5 초 이상이면 위험.
근거: ①④⑤ 규격(ITC/Ofcom Annex 1 §5·5.1, BT.1702-3 Att.1: > 5 쌍, ΔL ≥ 20, 어두운 막대 < 160, 면적 > 40 %) ·
      0.5 초 지속(Harding FPA 해설) · 1–10 cpd·시청 조건(팀원 판·EF 2026 합의 1–8 cpd 에 여유) · 나머지는 구현 선택(민감도 분석 대상).
"""
from __future__ import annotations

import math

import numpy as np
import torch

from .gpu_kernel import hyst_runs

CLAUSE_P1 = "ITC/Ofcom Annex 1 §5·5.1(i) · ITU-R BT.1702-3 Att.1 (정지 줄무늬 > 5 쌍, ΔL ≥ 20 cd/m², 어두운 막대 < 160, 면적 > 40 %)"
N_THETA, N_FREQ = 12, 8
CPD_LO, CPD_HI = 1.0, 10.0
CAND_FRAC = 0.6               # 후보 문턱 = 0.6 × 20 cd/m² (구현 선택: 필터 진폭은 파형에 따라 과소추정 → 확인은 4 단계에서 정확히)
CELL_MIN_FRAC = 0.30          # 칸이 '줄무늬 칸'이 되는 후보 픽셀 비율 (구현 선택)
LINK_FREQ_RATIO, LINK_DTHETA = 1.5, 30.0   # 이웃 칸 연결 조건 (구현 선택; 30° = 큰 굴곡의 물결무늬도 한 영역으로 — 2026-11-24 진단)
LOCAL_REG_MAX = 0.25         # 규칙성: 이웃한 한 주기 폭끼리의 상대 변화 |Δp|/p 의 중앙값 상한 — '명확히 식별되는 (규칙) 줄무늬'(원문)의 구현.
                             # 서서히 변하는 처프·원근 줄무늬는 통과, 간격이 제각각인 자연 질감은 탈락 (구현 선택; 팀원 판 전체 CV ≤ 0.35 와 다른 '국소' 정의)
TRACE_STEP = 0.25            # 경로 표본 간격(px): 1 px 이면 7 cpd 줄의 반 주기(≈1.6 px)가 1·2 px 로 잘려 cpd > 10 으로 오판 → 쌍이 끊김
COHERENCE_2D = 0.5            # 칸 안 방향 일관성(|평균 e^{i2θ}|)이 이보다 낮으면 2 차원 반복 무늬(체커·점 열) → 방향 대신 주파수로만 연결 (구현 선택)
HALF_AMP = 0.5                # 패턴 경계 = 영역 핵심 진폭의 절반(대칭 필터의 계단 응답이 실제 경계에서 50 %) — 면적은 이 경계 안의 픽셀
AREA_MIN, MIN_PAIRS = 0.40, 6


def pixels_per_degree(w: int, h: int, angle_deg: float = 0.0) -> float:
    """measure.pixels_per_degree 와 같은 시청 조건(56 cm, 15인치 16:9)."""
    diag, k, dist = 15.0 * 2.54, math.hypot(16.0, 9.0), 56.0
    fov_w = 2.0 * math.degrees(math.atan(diag * 16.0 / k / (2.0 * dist)))
    fov_h = 2.0 * math.degrees(math.atan(diag * 9.0 / k / (2.0 * dist)))
    a = math.radians(angle_deg)
    return 1.0 / math.hypot(math.cos(a) / (w / fov_w), math.sin(a) / (h / fov_h))


class PatternBank:
    """주파수 영역 복소 필터 묶음 (방향 θ = 파수 벡터 방향 = 줄무늬의 법선 방향)."""

    def __init__(self, h: int, w: int, device="cuda", cpd_lo: float = CPD_LO, cpd_hi: float = CPD_HI):
        self.h, self.w, self.device = h, w, device
        ppd = pixels_per_degree(w, h)
        nyq_cpd = 0.5 * ppd
        n_f = N_FREQ if cpd_lo >= CPD_LO else N_FREQ + int(math.ceil(math.log(CPD_LO / cpd_lo) / math.log(10.0 ** (1 / 7))))   # 아래로 넓히면 같은 간격으로 필터 추가
        self.cpds = np.exp(np.linspace(math.log(cpd_lo), math.log(min(cpd_hi, 0.9 * nyq_cpd)), n_f))
        self.n_freq = n_f
        self.thetas = np.arange(N_THETA) * (180.0 / N_THETA)
        fy = torch.fft.fftfreq(h, device=device)[:, None].expand(h, w)
        fx = torch.fft.fftfreq(w, device=device)[None, :].expand(h, w)
        rad = torch.hypot(fx, fy).clamp_min(1e-9)
        ang = torch.atan2(fy, fx)
        filt = []
        s_log = math.log(self.cpds[1] / self.cpds[0]) if n_f > 1 else 0.5   # 이웃 주파수 간격 = 1 σ
        s_th = math.radians(180.0 / N_THETA) * 0.8
        for th in self.thetas:
            t = math.radians(th)
            d = torch.atan2(torch.sin(ang - t), torch.cos(ang - t))           # 한쪽(해석적) 필터 → 복소 응답
            ang_w = torch.exp(-0.5 * (d / s_th) ** 2)
            for c in self.cpds:
                f0 = c / ppd                                                   # cycles / pixel
                rad_w = torch.exp(-0.5 * (torch.log(rad / f0) / s_log) ** 2)
                filt.append((ang_w * rad_w).to(torch.complex64))
        self.F = torch.stack(filt)                                             # [K,h,w]
        # 진폭 보정: 사인파 진폭 A 에 대한 응답 크기가 A 가 되도록 필터 중심 이득으로 나눔(해석적 필터는 A/2 → ×2)
        self.gain = 2.0

    @torch.no_grad()
    def respond(self, lum: torch.Tensor):
        """lum [h,w] → (best_k [h,w], amp [h,w] cd/m², theta_deg [h,w], cpd [h,w])."""
        X = torch.fft.fft2(lum - lum.mean())
        R = torch.fft.ifft2(X[None] * self.F).abs() * self.gain                # [K,h,w]
        amp, k = R.max(0)
        th = torch.as_tensor(self.thetas, device=lum.device, dtype=torch.float32)[k // self.n_freq]
        cp = torch.as_tensor(self.cpds, device=lum.device, dtype=torch.float32)[k % self.n_freq]
        return k, amp, th, cp


def _best_run_regularity(et, ev, n, pse):
    """한 단면의 최장 유효 구간(원본 _best_open_run 과 같은 규칙)에서 이웃 한 주기 폭의 상대 변화 중앙값."""
    if n < 4:
        return 1.0
    et, ev = et[:n], ev[:n]
    lo, hi = np.minimum(ev[:-1], ev[1:]), np.maximum(ev[:-1], ev[1:])
    dl = hi - lo
    ok = np.where(lo < pse.dark_lum_limit_cd_m2, dl >= pse.delta_lum_min_cd_m2, (dl / np.maximum(hi + lo, 1e-6)) > pse.bright_michelson_min)
    dth = np.diff(et)
    cpd = np.where(dth > 1e-6, 0.5 / np.maximum(dth, 1e-6), 0.0)
    valid = ok & (cpd >= pse.pattern_cpd_min) & (cpd <= pse.pattern_cpd_max)
    best, cur, end = 0, 0, -1
    for j, v in enumerate(valid):
        cur = cur + 1 if v else 0
        if cur > best:
            best, end = cur, j
    if best < 4:
        return 1.0
    s0 = end - best + 1
    t_run = et[s0:end + 2]
    per = t_run[2:] - t_run[:-2]                                               # 한 주기 = 극값 두 칸
    if len(per) < 2:
        return 1.0
    return float(np.median(np.abs(np.diff(per)) / np.maximum(per[:-1], 1e-9)))


def _cells(mask, th, cp, rows, cols):
    """칸별 후보 비율·주된 방향(2θ 원평균)·주파수(로그 평균)."""
    m = mask.float()
    pool = lambda z: torch.nn.functional.adaptive_avg_pool2d(z[None, None], (rows, cols))[0, 0]
    frac = pool(m)
    a2 = torch.deg2rad(th) * 2
    c2, s2 = pool(m * torch.cos(a2)), pool(m * torch.sin(a2))
    th_c = (torch.rad2deg(torch.atan2(s2, c2)) / 2) % 180
    coh = torch.hypot(c2, s2) / frac.clamp_min(1e-6)
    lf = pool(m * torch.log(cp)) / frac.clamp_min(1e-6)
    return frac.cpu().numpy(), th_c.cpu().numpy(), torch.exp(lf).cpu().numpy(), coh.cpu().numpy()


def _regions(on, th_c, f_c, coh=None):
    """이웃 칸 연결(방향 차 ≤ 20°, 주파수 비 ≤ 1.5) → 영역 번호 [rows, cols] (-1 = 없음)."""
    R, C = on.shape
    lab = -np.ones((R, C), int)
    nid = 0
    for r0 in range(R):
        for c0 in range(C):
            if not on[r0, c0] or lab[r0, c0] >= 0:
                continue
            stack = [(r0, c0)]; lab[r0, c0] = nid
            while stack:
                r, c = stack.pop()
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    q, p = r + dr, c + dc
                    if 0 <= q < R and 0 <= p < C and on[q, p] and lab[q, p] < 0:
                        dth = abs((th_c[q, p] - th_c[r, c] + 90) % 180 - 90)
                        if coh is not None and min(coh[q, p], coh[r, c]) < COHERENCE_2D:
                            dth = 0.0                                          # 2 차원 반복 무늬: 방향 대신 주파수로만
                        ratio = max(f_c[q, p], f_c[r, c]) / max(min(f_c[q, p], f_c[r, c]), 1e-6)
                        if dth <= LINK_DTHETA and ratio <= LINK_FREQ_RATIO:
                            lab[q, p] = nid; stack.append((q, p))
            nid += 1
    return lab, nid


@torch.no_grad()
def _trace_profiles(lum, th_px, inreg, seeds, ppd, step=TRACE_STEP, max_len=16000):
    """씨앗마다 국소 법선(θ)을 따라 양쪽으로 경로를 그리며 밝기 단면(cd/m²)과 누적 거리(도)를 얻는다."""
    h, w = lum.shape
    L = lum.cpu().numpy(); TH = th_px.cpu().numpy(); IN = inreg
    out = []
    for (y0, x0) in seeds:
        prof = []
        for sgn in (-1.0, 1.0):
            y, x = float(y0), float(x0)
            pv = None
            pts = []
            for it in range(max_len):
                yi, xi = int(round(y)), int(round(x))
                if not (0 <= yi < h and 0 <= xi < w) or not IN[yi, xi]:
                    break
                if it * step > 30 and math.hypot(y - y0, x - x0) < 2.0:          # 닫힌 경로(동심원을 도는 경우): 한 바퀴에서 멈춤
                    break
                a = math.radians(float(TH[yi, xi]))
                d = np.array([math.sin(a), math.cos(a)])                     # (dy, dx): 파수 벡터 방향
                if pv is not None and float(d @ pv) < 0:
                    d = -d
                if pv is None:
                    d = d * sgn
                pv = d
                pts.append((y, x))
                y += step * d[0]; x += step * d[1]
            prof.append(pts[::-1] if sgn < 0 else pts[1:])
        pts = prof[0] + prof[1]
        if len(pts) < 12:
            continue
        ys = np.array([p[0] for p in pts]); xs = np.array([p[1] for p in pts])
        y0i, x0i = np.clip(np.floor(ys).astype(int), 0, h - 2), np.clip(np.floor(xs).astype(int), 0, w - 2)
        fy, fx = ys - y0i, xs - x0i
        v = (L[y0i, x0i] * (1 - fy) * (1 - fx) + L[y0i + 1, x0i] * fy * (1 - fx) + L[y0i, x0i + 1] * (1 - fy) * fx + L[y0i + 1, x0i + 1] * fy * fx)
        seg = np.hypot(np.diff(ys), np.diff(xs))
        ang = np.degrees(np.arctan2(np.diff(ys), np.diff(xs)))
        ppd_s = np.array([pixels_per_degree(w, h, float(a_)) for a_ in ang]) if len(ang) else np.array([ppd])
        theta = np.concatenate([[0.0], np.cumsum(seg / ppd_s)])
        out.append((v.astype(np.float64), theta))
    return out


def _sdr(pse):
    """SDR: 어두운 막대 ≥ 160 cd/m² 의 1/17 Michelson 규칙은 HDR 에만 적용(BT.1702-3 Guideline 1 "applicable for HDR programmes only").
    팀원 판은 SDR 에도 적용 → 적합성 시험에서 어두운 막대 165 cd/m² 를 위험으로 냄. G 는 SDR 기본에서 이 규칙을 끔."""
    import dataclasses
    return dataclasses.replace(pse, bright_michelson_min=1e9)


class GridPatternJudge:
    def __init__(self, h: int, w: int, device="cuda", grid_cols: int = 32, hdr: bool = False,
                 cpd_lo: float = CPD_LO, cpd_hi: float = CPD_HI):
        """cpd_lo/hi: 줄무늬 공간 주파수 대역 — 기본 1–10(팀원 값, 출처 없음) · 1–8(Fisher 2026 EF 합의) · 하한 없음(BT.1702-3 원문, 실제로는 0.15)."""
        self.hdr = hdr
        self.h, self.w, self.device = h, w, device
        self.cpd_lo, self.cpd_hi = cpd_lo, cpd_hi
        self.bank = PatternBank(h, w, device, cpd_lo, cpd_hi)
        self.rows = max(1, int(round(grid_cols * h / w))); self.cols = grid_cols
        self.ppd = pixels_per_degree(w, h)

    @torch.no_grad()
    def frame(self, lum: torch.Tensor, pse=None, n_seeds: int = 12, area_min: float = AREA_MIN, maps: bool = False) -> dict:
        """area_min: 정지 0.40(기본) · 변화 패턴 후보(P2) 0.25. maps=True 면 best 에 'mask'(줄무늬 픽셀, np bool)·'k_map'(픽셀별 필터 번호)·'verified' 를 넣는다."""
        from .config import PSEConfig
        pse = pse or PSEConfig()
        k, amp, th, cp = self.bank.respond(lum)
        cand = (2.0 * amp >= CAND_FRAC * pse.delta_lum_min_cd_m2)
        frac, th_c, f_c, coh = _cells(cand, th, cp, self.rows, self.cols)
        on = frac >= CELL_MIN_FRAC
        lab, n = _regions(on, th_c, f_c, coh)
        best = {"is_pat": False, "area": 0.0, "pairs": 0, "cpd": 0.0, "cells": [], "kind": None}
        if n == 0:
            return best
        cell_h, cell_w = self.h / self.rows, self.w / self.cols
        cand_np = cand.cpu().numpy(); amp_np = amp.cpu().numpy()
        for rid in range(n):
            cells = np.argwhere(lab == rid)
            if len(cells) == 0:
                continue
            reg = np.zeros((self.h, self.w), bool)
            for r, c in cells:
                reg[int(r * cell_h):int((r + 1) * cell_h), int(c * cell_w):int((c + 1) * cell_w)] = True
            # 반값 경계: 칸 경계에 묶이지 않도록 영역을 한 칸 넓힌 뒤, 핵심 진폭(상위 10 %)의 절반 이상인 후보 픽셀만 면적으로
            core = amp_np[reg & cand_np]
            if core.size == 0:
                continue
            a_core = float(np.percentile(core, 90))
            reg_d = reg.copy()
            for r, c in cells:
                reg_d[max(0, int((r - 1) * cell_h)):int((r + 2) * cell_h), max(0, int((c - 1) * cell_w)):int((c + 2) * cell_w)] = True
            pat = reg_d & cand_np & (amp_np >= HALF_AMP * a_core)
            area = float(pat.mean())
            if area <= area_min:                                               # 면적 미달 영역은 확인 생략(판정 불변)
                continue
            # 씨앗 = 영역 칸 전체에서 고르게 뽑은 칸마다 그 칸의 줄무늬 픽셀 중 진폭 최대 점 (확인 비율이 면적의 치우치지 않은 추정이 되도록)
            seeds = []
            order = np.linspace(0, len(cells) - 1, min(n_seeds, len(cells))).round().astype(int)
            for ci in order:
                r, c = cells[ci]
                y0_, y1_ = int(r * cell_h), int((r + 1) * cell_h); x0_, x1_ = int(c * cell_w), int((c + 1) * cell_w)
                blk = np.where(pat[y0_:y1_, x0_:x1_], amp_np[y0_:y1_, x0_:x1_], -1)
                if blk.max() <= 0:
                    continue
                yy_, xx_ = np.unravel_index(int(np.argmax(blk)), blk.shape)
                seeds.append((y0_ + yy_, x0_ + xx_))
            profs = _trace_profiles(lum, th, pat, seeds, self.ppd)
            if not profs:
                continue
            Lmax = max(len(p[0]) for p in profs)
            V = np.zeros((len(profs), Lmax)); T = np.zeros((len(profs), Lmax)); lens = np.array([len(p[0]) for p in profs])
            for i, (v, t) in enumerate(profs):
                V[i, :len(v)] = v; T[i, :len(t)] = t
            pse_u = pse if self.hdr else _sdr(pse)
            if (self.cpd_lo, self.cpd_hi) != (CPD_LO, CPD_HI):
                import dataclasses
                pse_u = dataclasses.replace(pse_u, pattern_cpd_min=self.cpd_lo, pattern_cpd_max=self.cpd_hi)
            P, C, _, NE, ET, EV = hyst_runs(V, T, lens, pse_u)
            P = P.get(); C = C.get(); NE = NE.get(); ET = ET.get(); EV = EV.get()
            reg_ok = np.array([_best_run_regularity(ET[i], EV[i], int(NE[i]), pse_u) <= LOCAL_REG_MAX for i in range(len(P))])
            Pq = np.where(reg_ok, P, 0)
            j = int(np.argmax(Pq))
            pairs, cpd = int(Pq[j]), float(C[j])
            # 표본 확인 면적: 영역 곳곳의 씨앗 중 규격대로(쌍 > 5·규칙적) 확인된 비율만큼만 면적으로 인정 — 질감이 한두 곳에서 우연히 맞는 경우 차단
            verified = float((Pq >= MIN_PAIRS).mean())
            area = area * verified
            dth = np.std([th_c[r, c] for r, c in cells])
            kind = "직선" if dth < 10 else "곡선·방사"
            if pairs >= MIN_PAIRS and area > area_min and area > best["area"]:
                best = {"is_pat": True, "area": area, "pairs": pairs, "cpd": cpd, "kind": kind, "verified_seeds": f"{int((Pq >= MIN_PAIRS).sum())}/{len(Pq)}",
                        "cells": [[int(r), int(c)] for r, c in cells]}
                if maps:
                    best.update(mask=pat, k_map=k, verified=verified)
        return best
