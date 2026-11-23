"""Pixel-level static spatial pattern localization & optional enhanced detector (`spatial_mask.py`).

Supports two modes via `PartialMaskConfig.enhanced_spatial_pattern`:
  1) Default legacy-compatible mode (`enhanced_spatial_pattern=False`):
     - Preserves 100% identical scalar verdicts `(is_pat, effective_area, best_pairs, best_cpd)`
       with `vdcorefs.spatial.StaticPatternDetector`.
     - Back-projects exact 1D periodic segments intersected with local contrast support.
  2) Opt-in enhanced mode (`enhanced_spatial_pattern=True`, used by `B2/partial` per `4-a`):
     - `(3-1)` Expands linear offset coverage (`+-0.75 -> +-0.95`, `9 -> 15` lines) + `3x3`
       overlapping ROI Sobel dominant normal orientation targeted parallel bundles (keeping
       the international full-screen `> 40%` contrast area gate `1-a`).
     - `(3-2)` Pre-scan mild Gaussian anti-aliasing (`sigma = 0.60 px`), full-period (`peak-to-peak`
       / `trough-to-trough`) regularity check (`CV <= 0.35`, `2-a`), and `>= 2` parallel scanline
       spatial/frequency consistency (eliminating single-line cartoon outline & staircase false positives).
     - `(3-3a)` Decouples Sobel Hough center voting into Top-2 Tangent centers (`Path 1` concentric
       rings only) and Top-2 Normal centers (`Path 2` radial spokes only), retaining `(cx0, cy0)`
       safeguard.
     - `(3-4a)` Retains Global `2D FFT` fallback and supplements it with a 24-channel frequency-domain
       Analytic Quadrature `2D Log-Gabor Wavelet Bank` (`4 scales x 6 orientations`) to catch curved
       gratings whose global spectral ratio is diluted (`< 35%`) by strong background steps.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F

from .config import PSEConfig, PartialMaskConfig, ViewingGeometry
from .spatial import (
    StaticPatternDetector,
    evaluate_1d_closed_dual,
    evaluate_1d_open_dual,
    extract_hysteresis_extrema_1d,
    filter_static_pattern_runs,
    separable_max_min_2d,
)


def morphological_close_2d(mask_2d: torch.Tensor, radius: int) -> torch.Tensor:
    """Exact separable 2D binary morphological closing on `2*radius` replicate-padded canvas."""
    if radius <= 0 or not bool(mask_2d.any().item()):
        return mask_2d

    r = int(radius)
    k = 2 * r + 1
    pad2 = 2 * r
    x = mask_2d.to(torch.float32).unsqueeze(0).unsqueeze(0)
    x_pad = F.pad(x, (pad2, pad2, pad2, pad2), mode="replicate")

    d_h = F.max_pool2d(x_pad, kernel_size=(1, k), stride=1)
    dilated = F.max_pool2d(d_h, kernel_size=(k, 1), stride=1)

    e_h = -F.max_pool2d(-dilated, kernel_size=(1, k), stride=1)
    eroded = -F.max_pool2d(-e_h, kernel_size=(k, 1), stride=1)
    return mask_2d | (eroded[0, 0] > 0.5)


def _find_valid_open_runs_1d(
    values: np.ndarray,
    theta_deg: np.ndarray,
    pse: PSEConfig,
    cpd_lo: float | None = None,
) -> list[tuple[float, float, int, float]]:
    """Return `[(t_start_deg, t_end_deg, pairs, median_cpd), ...]` for all runs `>= min_pairs`."""
    ext_t, ext_v = extract_hysteresis_extrema_1d(
        values, theta_deg, pse.pattern_hysteresis_cd_m2
    )
    if len(ext_v) < 2:
        return []

    lo_f = pse.pattern_cpd_min if cpd_lo is None else cpd_lo
    v0 = ext_v[:-1]
    v1 = ext_v[1:]
    l_low = np.minimum(v0, v1)
    l_high = np.maximum(v0, v1)
    dl = l_high - l_low
    contrast_ok = np.where(
        l_low < pse.dark_lum_limit_cd_m2,
        dl >= pse.delta_lum_min_cd_m2,
        (dl / np.maximum(l_high + l_low, 1e-6)) > pse.bright_michelson_min,
    )
    d_theta = ext_t[1:] - ext_t[:-1]
    cpds = np.where(d_theta > 1e-6, 0.5 / np.maximum(d_theta, 1e-6), 0.0)
    valid = contrast_ok & (cpds >= lo_f) & (cpds <= pse.pattern_cpd_max)
    if not np.any(valid):
        return []

    padded = np.concatenate(([False], valid, [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    out: list[tuple[float, float, int, float]] = []
    for k in range(0, len(edges), 2):
        rs = int(edges[k])
        re = int(edges[k + 1])
        pairs = (re - rs + 1) // 2
        if pairs >= pse.pattern_min_pairs:
            med_cpd = float(np.median(cpds[rs:re]))
            out.append((float(ext_t[rs]), float(ext_t[re]), pairs, med_cpd))
    return out


def find_regular_open_runs_1d(
    values: np.ndarray,
    theta_deg: np.ndarray,
    pse: PSEConfig,
    *,
    cpd_lo: float | None = None,
    full_period_cv_max: float = 0.35,
) -> list[tuple[float, float, int, float]]:
    """Return regular `[1..10] cpd` runs `>= min_pairs` whose full periods satisfy `CV <= full_period_cv_max`.

    Uses full-period intervals `T_j = ext_t[j+2] - ext_t[j]` (peak-to-peak and trough-to-trough)
    so asymmetric duty-cycle gratings (`narrow dark / wide bright`) remain `CV ~ 0`, and extracts
    the longest regular sub-run via prefix sums so perspective-foreshortened gratings remain intact.
    """
    ext_t, ext_v = extract_hysteresis_extrema_1d(
        values, theta_deg, pse.pattern_hysteresis_cd_m2
    )
    if len(ext_v) < 2:
        return []

    lo_f = pse.pattern_cpd_min if cpd_lo is None else cpd_lo
    v0 = ext_v[:-1]
    v1 = ext_v[1:]
    l_low = np.minimum(v0, v1)
    l_high = np.maximum(v0, v1)
    dl = l_high - l_low
    contrast_ok = np.where(
        l_low < pse.dark_lum_limit_cd_m2,
        dl >= pse.delta_lum_min_cd_m2,
        (dl / np.maximum(l_high + l_low, 1e-6)) > pse.bright_michelson_min,
    )
    d_theta = ext_t[1:] - ext_t[:-1]
    cpds = np.where(d_theta > 1e-6, 0.5 / np.maximum(d_theta, 1e-6), 0.0)
    valid = contrast_ok & (cpds >= lo_f) & (cpds <= pse.pattern_cpd_max)
    if not np.any(valid):
        return []

    min_half_steps = 2 * pse.pattern_min_pairs - 1
    padded = np.concatenate(([False], valid, [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    out: list[tuple[float, float, int, float]] = []

    for k in range(0, len(edges), 2):
        rs = int(edges[k])
        re = int(edges[k + 1])
        run_len = re - rs
        if run_len < min_half_steps:
            continue

        full_t = ext_t[rs + 2 : re + 1] - ext_t[rs : re - 1]
        if len(full_t) < 2:
            continue

        mean_t = float(np.mean(full_t))
        cv_whole = float(np.std(full_t) / max(mean_t, 1e-6))
        if cv_whole <= full_period_cv_max:
            pairs = (run_len + 1) // 2
            med_cpd = float(np.median(cpds[rs:re]))
            out.append((float(ext_t[rs]), float(ext_t[re]), pairs, med_cpd))
            continue

        best_sub: tuple[int, int] | None = None
        best_sub_len = 0
        n_t = len(full_t)
        min_t_len = max(2, min_half_steps - 1)
        cs1 = np.concatenate(([0.0], np.cumsum(full_t)))
        cs2 = np.concatenate(([0.0], np.cumsum(full_t * full_t)))
        for a in range(0, n_t - min_t_len + 1):
            if (n_t - a) + 1 <= best_sub_len:
                break
            for b in range(n_t, a + min_t_len - 1, -1):
                cnt = b - a
                if cnt + 1 <= best_sub_len:
                    break
                mu = (cs1[b] - cs1[a]) / cnt
                var = max(0.0, ((cs2[b] - cs2[a]) / cnt) - mu * mu)
                if math.sqrt(var) / max(mu, 1e-6) <= full_period_cv_max:
                    best_sub_len = cnt + 1
                    best_sub = (rs + a, rs + b + 1)
                    break

        if best_sub is not None:
            ss, se = best_sub
            pairs = (se - ss + 1) // 2
            if pairs >= pse.pattern_min_pairs:
                med_cpd = float(np.median(cpds[ss:se]))
                out.append((float(ext_t[ss]), float(ext_t[se]), pairs, med_cpd))

    return out


def _find_valid_closed_runs_1d(
    values: np.ndarray,
    cum_theta: np.ndarray,
    ring_perimeter_deg: float,
    pse: PSEConfig,
    cpd_lo: float | None = None,
) -> list[tuple[float, float, int, float]]:
    """Return `[(t_start_deg, t_end_deg, pairs, median_cpd), ...]` on a closed `360 deg` ring."""
    n = len(values)
    if n < 12:
        return []
    half_n = max(4, n // 2)
    wrapped_v = np.concatenate([values, values[:half_n]])
    wrapped_t = np.concatenate([cum_theta, cum_theta[:half_n] + ring_perimeter_deg])

    ext_t, ext_v = extract_hysteresis_extrema_1d(
        wrapped_v, wrapped_t, pse.pattern_hysteresis_cd_m2
    )
    if len(ext_v) < 4:
        return []

    lo_f = pse.pattern_cpd_min if cpd_lo is None else cpd_lo
    v0 = ext_v[:-1]
    v1 = ext_v[1:]
    l_low = np.minimum(v0, v1)
    l_high = np.maximum(v0, v1)
    dl = l_high - l_low
    contrast_ok = np.where(
        l_low < pse.dark_lum_limit_cd_m2,
        dl >= pse.delta_lum_min_cd_m2,
        (dl / np.maximum(l_high + l_low, 1e-6)) > pse.bright_michelson_min,
    )
    d_theta = ext_t[1:] - ext_t[:-1]
    cpds = np.where(d_theta > 1e-6, 0.5 / np.maximum(d_theta, 1e-6), 0.0)
    valid = contrast_ok & (cpds >= lo_f) & (cpds <= pse.pattern_cpd_max)
    if not np.any(valid):
        return []

    padded = np.concatenate(([False], valid, [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    limit_deg = ring_perimeter_deg + 1e-4
    out: list[tuple[float, float, int, float]] = []

    for k in range(0, len(edges), 2):
        rs0 = int(edges[k])
        re0 = int(edges[k + 1])
        best_p = 0
        best_sl = (rs0, re0)
        run_s = rs0
        for j in range(rs0, re0):
            while run_s <= j and (ext_t[j + 1] - ext_t[run_s]) > limit_deg:
                run_s += 1
            run_len = (j + 1) - run_s
            span_deg = ext_t[j + 1] - ext_t[run_s]
            if span_deg >= 0.92 * ring_perimeter_deg:
                in_one_turn = int(
                    np.searchsorted(ext_t, ext_t[run_s] + ring_perimeter_deg - 1e-4, side="left")
                    - np.searchsorted(ext_t, ext_t[run_s] - 1e-5, side="left")
                )
                pairs = min((run_len + 1) // 2, in_one_turn // 2)
            else:
                pairs = (run_len + 1) // 2
            if pairs > best_p and run_len > 0:
                best_p = pairs
                best_sl = (run_s, j + 1)

        if best_p >= pse.pattern_min_pairs and best_sl[1] > best_sl[0]:
            med_cpd = float(np.median(cpds[best_sl[0] : best_sl[1]]))
            out.append((float(ext_t[best_sl[0]]), float(ext_t[best_sl[1]]), best_p, med_cpd))

    return out


def _in_azimuth_arc(ang_2d: torch.Tensor, a0: float, a1: float) -> torch.Tensor:
    """Boolean mask on `ang_2d in [0, 2*pi)` for counter-clockwise arc `a0 -> a1`."""
    two_pi = 2.0 * math.pi
    span = a1 - a0
    if span >= two_pi - 1e-4:
        return torch.ones_like(ang_2d, dtype=torch.bool)
    lo = a0 % two_pi
    hi = a1 % two_pi
    if lo <= hi:
        return (ang_2d >= lo) & (ang_2d <= hi)
    return (ang_2d >= lo) | (ang_2d <= hi)


class PartialStaticPatternDetector(StaticPatternDetector):
    """Extends `StaticPatternDetector` with exact 2D segment localization and opt-in `(3-1..3-4a)` upgrades."""

    def __init__(
        self,
        geometry: ViewingGeometry,
        pse: PSEConfig = PSEConfig(),
        device: torch.device | None = None,
        mask_config: PartialMaskConfig | None = None,
    ) -> None:
        super().__init__(geometry=geometry, pse=pse, device=device)
        m_cfg = mask_config or PartialMaskConfig()
        self.mask_config = m_cfg
        self.enhanced_spatial_pattern = bool(m_cfg.enhanced_spatial_pattern)
        self._prev_mask: torch.Tensor | None = None
        self._prev_close_scale: float | None = None

        w, h = self.geom.width, self.geom.height
        self._lin_cx = 0.5 * (w - 1)
        self._lin_cy = 0.5 * (h - 1)
        self._lin_s0 = -float(math.ceil(0.5 * math.hypot(w, h)))
        self._lin_angle_meta: list[tuple[float, float, float, np.ndarray, float]] = []
        for m in range(12):
            phi = math.radians(15.0 * m)
            c_p, s_p = math.cos(phi), math.sin(phi)
            p_phi = float(self.geom.pixels_per_degree(phi))
            o_max = 0.5 * (w - 1) * abs(s_p) + 0.5 * (h - 1) * abs(c_p)
            offsets = np.linspace(-0.75 * o_max, 0.75 * o_max, 9)
            self._lin_angle_meta.append((c_p, s_p, p_phi, offsets, o_max))

        self._ring_alpha_luts: list[tuple[np.ndarray, np.ndarray]] = []
        for (s_idx, e_idx), (cum_theta, perim_deg) in zip(self._ring_slices, self._ring_theta):
            n_alpha = e_idx - s_idx
            alphas = np.linspace(0.0, 2.0 * math.pi, n_alpha, endpoint=False, dtype=np.float64)
            self._ring_alpha_luts.append(
                (np.append(cum_theta, perim_deg), np.append(alphas, 2.0 * math.pi))
            )

        self._spoke_alphas = np.linspace(0.0, 2.0 * math.pi, 24, endpoint=False, dtype=np.float64)

        # Precompute enhanced grids & Analytic Quadrature 2D Log-Gabor bank only when enabled
        if self.enhanced_spatial_pattern:
            self._enh_n_offsets = max(9, int(m_cfg.linear_num_offsets))
            self._enh_offset_limit = float(np.clip(m_cfg.linear_offset_limit, 0.75, 0.99))
            self._enh_cv_max = float(m_cfg.pattern_full_period_cv_max)
            self._enh_min_parallel = max(1, int(m_cfg.pattern_min_parallel_lines))
            self._enh_sobel_top_k = max(1, int(m_cfg.sobel_top_k_per_dir))
            self._enh_enable_gabor = bool(m_cfg.enable_gabor_fallback)

            sigma = float(m_cfg.pattern_antialias_sigma_px)
            if sigma > 1e-3:
                rad_aa = max(1, int(math.ceil(3.0 * sigma)))
                xs_aa = torch.arange(-rad_aa, rad_aa + 1, dtype=torch.float32, device=self.device)
                k1 = torch.exp(-(xs_aa * xs_aa) / (2.0 * sigma * sigma))
                self._aa_kernel_1d: torch.Tensor | None = k1 / k1.sum()
            else:
                self._aa_kernel_1d = None

            (
                self._lin_grid_enh,
                self._lin_valid_enh,
                self._lin_theta_enh,
                self._lin_meta_enh,
                self._s_vals_np,
            ) = self._build_enhanced_linear_cache()

            if self._enh_enable_gabor:
                self._gabor_bank_fft, self._gabor_freqs, self._gabor_thetas = (
                    self._build_gabor_bank(w, h)
                )
                self._gabor_var_rad = max(
                    6, int(round(1.0 * max(geometry.px_per_deg_x, geometry.px_per_deg_y)))
                )

    def reset(self) -> None:
        super().reset()
        self._prev_mask = None
        self._prev_close_scale = None

    def _antialias_4d(self, lum_4d: torch.Tensor) -> torch.Tensor:
        if not self.enhanced_spatial_pattern or self._aa_kernel_1d is None:
            return lum_4d
        pad = self._aa_kernel_1d.numel() // 2
        x = F.pad(lum_4d, (pad, pad, pad, pad), mode="replicate")
        kh = self._aa_kernel_1d.view(1, 1, 1, -1)
        kv = self._aa_kernel_1d.view(1, 1, -1, 1)
        return F.conv2d(F.conv2d(x, kh), kv)

    def _build_enhanced_linear_cache(
        self,
    ) -> tuple[
        torch.Tensor,
        np.ndarray,
        np.ndarray,
        list[tuple[float, float, float, np.ndarray, float]],
        np.ndarray,
    ]:
        w, h = self.geom.width, self.geom.height
        cx = 0.5 * (w - 1)
        cy = 0.5 * (h - 1)
        s_max = math.ceil(0.5 * math.hypot(w, h))
        s_vals = np.arange(-s_max, s_max + 1e-6, self._linear_step_px, dtype=np.float64)
        l_len = len(s_vals)

        n_angles = 12
        n_offsets = self._enh_n_offsets
        total_lines = n_angles * n_offsets
        x_all = np.zeros((total_lines, l_len), dtype=np.float64)
        y_all = np.zeros((total_lines, l_len), dtype=np.float64)
        theta_all = np.zeros((total_lines, l_len), dtype=np.float64)
        meta: list[tuple[float, float, float, np.ndarray, float]] = []

        line_idx = 0
        for m in range(n_angles):
            phi = math.radians(15.0 * m)
            c_p, s_p = math.cos(phi), math.sin(phi)
            nx, ny = -s_p, c_p
            o_max = 0.5 * (w - 1) * abs(s_p) + 0.5 * (h - 1) * abs(c_p)
            offsets = np.linspace(
                -self._enh_offset_limit * o_max, self._enh_offset_limit * o_max, n_offsets
            )
            p_phi = float(self.geom.pixels_per_degree(phi))
            theta_line = (s_vals - s_vals[0]) / p_phi
            meta.append((c_p, s_p, p_phi, offsets, o_max))

            for off in offsets:
                x_all[line_idx] = cx + off * nx + s_vals * c_p
                y_all[line_idx] = cy + off * ny + s_vals * s_p
                theta_all[line_idx] = theta_line
                line_idx += 1

        valid = (x_all >= 0.0) & (x_all <= (w - 1)) & (y_all >= 0.0) & (y_all <= (h - 1))
        return self._to_norm_grid(x_all, y_all), valid, theta_all, meta, s_vals

    def _build_gabor_bank(
        self, w: int, h: int
    ) -> tuple[torch.Tensor, list[float], list[float]]:
        """Precompute 24-channel (`4 scales x 6 orientations`) one-sided Analytic Log-Gabor filters."""
        fy = torch.fft.fftfreq(h, d=1.0, device=self.device) * self.geom.px_per_deg_y
        fx = torch.fft.fftfreq(w, d=1.0, device=self.device) * self.geom.px_per_deg_x
        f_rad = torch.hypot(fy[:, None], fx[None, :])
        ang = torch.atan2(fy[:, None], fx[None, :])

        f_centers = [1.4, 2.6, 4.8, 8.2]
        n_orient = 6
        sigma_ln_f = 0.42
        sigma_ang = math.radians(18.0)

        filters: list[torch.Tensor] = []
        freqs_meta: list[float] = []
        thetas_meta: list[float] = []
        in_band = (f_rad >= 0.85 * self.pse.pattern_cpd_min) & (
            f_rad <= 1.10 * self.pse.pattern_cpd_max
        )
        log_f = torch.log(torch.clamp(f_rad, min=1e-6))

        for fk in f_centers:
            radial_w = (
                torch.exp(-0.5 * ((log_f - math.log(fk)) / sigma_ln_f) ** 2) * in_band.float()
            )
            radial_w[0, 0] = 0.0
            for m in range(n_orient):
                th = math.radians(30.0 * m)
                d_ang = torch.remainder(ang - th + math.pi, 2.0 * math.pi) - math.pi
                ang_w = (
                    torch.exp(-0.5 * (d_ang / sigma_ang) ** 2)
                    * (d_ang.abs() < (0.5 * math.pi)).float()
                )
                filters.append(2.0 * radial_w * ang_w)
                freqs_meta.append(fk)
                thetas_meta.append(th)

        return torch.stack(filters, dim=0), freqs_meta, thetas_meta

    def _compute_contrast_mask(self, lum_4d: torch.Tensor) -> torch.Tensor:
        l_max_loc, l_min_loc = separable_max_min_2d(lum_4d, self.pool_radius)
        dl_loc = l_max_loc - l_min_loc
        return torch.where(
            l_min_loc < self.pse.dark_lum_limit_cd_m2,
            dl_loc >= self.pse.delta_lum_min_cd_m2,
            (dl_loc / torch.clamp(l_max_loc + l_min_loc, min=1e-6)) > self.pse.bright_michelson_min,
        )

    def _evaluate_2d_fft_wave_mask(self, lum: torch.Tensor) -> tuple[int, float, torch.Tensor]:
        """Step 4 2D FFT fallback returning `(est_pairs, f_dom_cpd, active_wave_mask)`."""
        empty = torch.zeros_like(lum, dtype=torch.bool)
        centered = (lum - lum.mean()) * self._hann_2d
        spec = torch.fft.rfft2(centered)
        power = spec.real * spec.real + spec.imag * spec.imag
        power[0, 0] = 0.0
        total_energy = float(power.sum().item())
        if total_energy <= 1e-6:
            return 0, 0.0, empty

        bp_power = power * self._fft_bp_mask
        bp_ratio = float(bp_power.sum().item()) / total_energy
        if bp_ratio < self.pse.pattern_fft_energy_ratio_min:
            return 0, 0.0, empty

        wave = torch.fft.irfft2(spec * self._fft_bp_mask, s=lum.shape)
        w4 = wave.unsqueeze(0).unsqueeze(0)
        w_max, w_min = separable_max_min_2d(w4, self.pool_radius)
        active_wave = (w_max - w_min) >= self.pse.delta_lum_min_cd_m2
        wave_area = float(active_wave.float().mean().item())
        if wave_area <= self.pse.pattern_area_min:
            return 0, 0.0, empty

        flat_idx = int(torch.argmax(bp_power.reshape(-1)).item())
        rw = bp_power.shape[1]
        iy, ix = flat_idx // rw, flat_idx % rw
        f_dom_cpd = float(self._fft_f_cpd[iy, ix].item())
        phi_dom = float(self._fft_phi[iy, ix].item())

        ys, xs = torch.where(active_wave)
        if xs.numel() < 32:
            return 0, 0.0, empty
        proj_deg = (
            (xs.float() * math.cos(phi_dom)) / self.geom.px_per_deg_x
            + (ys.float() * math.sin(phi_dom)) / self.geom.px_per_deg_y
        )
        q_lo, q_hi = torch.quantile(
            proj_deg, torch.tensor([0.02, 0.98], device=self.device)
        ).tolist()
        span_deg = max(0.0, float(q_hi - q_lo))
        est_pairs = int(math.floor(f_dom_cpd * span_deg))
        return est_pairs, f_dom_cpd, active_wave

    # -----------------------------------------------------------------------
    # Enhanced `(3-1 .. 3-4a)` Helpers
    # -----------------------------------------------------------------------

    def _verify_fine_parallel_neighbors(
        self,
        lum_aa_4d: torch.Tensor,
        singleton_candidates: list[tuple[int, int, float, float, float, float, int, float]],
    ) -> list[tuple[int, int, float, float, int, float]]:
        """Verify isolated coarse-grid hits via fine lateral parallel lines (`+-0.25, +-0.50, +-0.75 deg`)."""
        if not singleton_candidates:
            return []
        w, h = self.geom.width, self.geom.height
        shifts_deg = (-0.75, -0.50, -0.25, 0.25, 0.50, 0.75)
        n_sh = len(shifts_deg)
        s_vals = self._s_vals_np
        l_len = len(s_vals)
        s0 = float(s_vals[0])

        total_rows = len(singleton_candidates) * n_sh
        x_all = np.zeros((total_rows, l_len), dtype=np.float64)
        y_all = np.zeros((total_rows, l_len), dtype=np.float64)
        t_all = np.zeros((total_rows, l_len), dtype=np.float64)

        cx, cy = self._lin_cx, self._lin_cy
        row = 0
        for _, _, phi, base_off_px, _, _, _, _ in singleton_candidates:
            c_p, s_p = math.cos(phi), math.sin(phi)
            nx, ny = -s_p, c_p
            p_phi = float(self.geom.pixels_per_degree(phi))
            p_perp = float(self.geom.pixels_per_degree((phi + 0.5 * math.pi) % math.pi))
            t_line = (s_vals - s0) / p_phi
            for sd in shifts_deg:
                off_px = base_off_px + sd * p_perp
                x_all[row] = cx + off_px * nx + s_vals * c_p
                y_all[row] = cy + off_px * ny + s_vals * s_p
                t_all[row] = t_line
                row += 1

        valid = (x_all >= 0.0) & (x_all <= (w - 1)) & (y_all >= 0.0) & (y_all <= (h - 1))
        profs = (
            F.grid_sample(
                lum_aa_4d,
                self._to_norm_grid(x_all, y_all),
                mode="bilinear",
                padding_mode="border",
                align_corners=True,
            )
            .squeeze(0)
            .squeeze(0)
            .cpu()
            .numpy()
        )
        confirmed: list[tuple[int, int, float, float, int, float]] = []
        need_nbrs = max(0, self._enh_min_parallel - 1)
        for c_i, (m_ang, k_off, _, _, t0, t1, prs, med_cpd) in enumerate(singleton_candidates):
            if need_nbrs == 0:
                confirmed.append((m_ang, k_off, t0, t1, prs, med_cpd))
                continue
            nbr_hits = 0
            for k in range(n_sh):
                r_i = c_i * n_sh + k
                vm = valid[r_i]
                if int(np.sum(vm)) < 12:
                    continue
                vals = profs[r_i, vm]
                if float(np.ptp(vals)) < self.pse.pattern_hysteresis_cd_m2:
                    continue
                runs = find_regular_open_runs_1d(
                    vals, t_all[r_i, vm], self.pse, full_period_cv_max=self._enh_cv_max
                )
                matched = False
                for nt0, nt1, _, ncpd in runs:
                    ov = min(t1, nt1) - max(t0, nt0)
                    span_min = max(0.5, min(t1 - t0, nt1 - nt0))
                    if ov >= 0.40 * span_min and (0.55 * med_cpd <= ncpd <= 1.80 * med_cpd):
                        matched = True
                        break
                if matched:
                    nbr_hits += 1
                    if nbr_hits >= need_nbrs:
                        break
            if nbr_hits >= need_nbrs:
                confirmed.append((m_ang, k_off, t0, t1, prs, med_cpd))
        return confirmed

    def _scan_sobel_targeted_bundles(
        self, lum_aa_4d: torch.Tensor, contrast_mask: torch.Tensor
    ) -> tuple[int, float, list[tuple[float, float, float, float, float, float, float, float]]]:
        """`(3-1)` `3x3` overlapping ROI + global Sobel dominant normal angle targeted scan."""
        w, h = self.geom.width, self.geom.height
        grads = F.conv2d(lum_aa_4d, self._sobel, padding=1)[0]
        gx, gy = grads[0], grads[1]
        mag = torch.hypot(gx, gy) * contrast_mask.float()

        xs_t = [(0, w // 2), (w // 4, (3 * w) // 4), (w // 2, w)]
        ys_t = [(0, h // 2), (h // 4, (3 * h) // 4), (h // 2, h)]
        rois = [(0, w, 0, h)] + [(x0, x1, y0, y1) for y0, y1 in ys_t for x0, x1 in xs_t]

        bundles: list[tuple[float, float, float]] = []
        min_roi_px = max(48, int(0.03 * w * h))

        for x0, x1, y0, y1 in rois:
            cm_sub = contrast_mask[y0:y1, x0:x1]
            if int(cm_sub.sum().item()) < min_roi_px:
                continue
            mag_sub = mag[y0:y1, x0:x1]
            strong = mag_sub > 20.0
            if int(strong.sum().item()) < 28:
                continue
            ys_s, xs_s = torch.where(strong)
            wts = mag_sub[strong]
            w_sum = float(wts.sum().item())
            if w_sum <= 1e-6:
                continue
            cx_r = float(x0) + float((xs_s.float() * wts).sum().item() / w_sum)
            cy_r = float(y0) + float((ys_s.float() * wts).sum().item() / w_sum)

            gx_s = gx[y0:y1, x0:x1][strong]
            gy_s = gy[y0:y1, x0:x1][strong]
            ang_deg = torch.remainder(torch.atan2(gy_s, gx_s) * (180.0 / math.pi), 180.0)
            bins = torch.clamp((ang_deg / 3.0).long(), 0, 59)
            hist = torch.zeros(60, dtype=torch.float32, device=self.device)
            hist.scatter_add_(0, bins, wts)
            hist_pad = torch.cat([hist[-1:], hist, hist[:1]])
            hist_sm = 0.25 * hist_pad[:-2] + 0.50 * hist_pad[1:-1] + 0.25 * hist_pad[2:]
            top_b = int(torch.argmax(hist_sm).item())
            if float(hist_sm[top_b].item()) < 0.12 * w_sum:
                continue
            phi_perp = math.radians((top_b + 0.5) * 3.0)

            dup = False
            for bx, by, bp in bundles:
                d_ang = abs((phi_perp - bp + 0.5 * math.pi) % math.pi - 0.5 * math.pi)
                if math.hypot(cx_r - bx, cy_r - by) < 0.10 * min(w, h) and d_ang < math.radians(8.0):
                    dup = True
                    break
            if not dup:
                bundles.append((cx_r, cy_r, phi_perp))

        if not bundles:
            return 0, 0.0, []

        s_vals = self._s_vals_np
        s0 = float(s_vals[0])
        l_len = len(s_vals)
        off_degs = np.linspace(-1.35, 1.35, 7)
        n_par = len(off_degs)
        total_lines = len(bundles) * n_par
        x_all = np.zeros((total_lines, l_len), dtype=np.float64)
        y_all = np.zeros((total_lines, l_len), dtype=np.float64)
        theta_all = np.zeros((total_lines, l_len), dtype=np.float64)

        row = 0
        for cx_r, cy_r, phi in bundles:
            c_p, s_p = math.cos(phi), math.sin(phi)
            nx, ny = -s_p, c_p
            p_phi = float(self.geom.pixels_per_degree(phi))
            p_perp = float(self.geom.pixels_per_degree((phi + 0.5 * math.pi) % math.pi))
            t_line = (s_vals - s0) / p_phi
            for od in off_degs:
                off_px = od * p_perp
                x_all[row] = cx_r + off_px * nx + s_vals * c_p
                y_all[row] = cy_r + off_px * ny + s_vals * s_p
                theta_all[row] = t_line
                row += 1

        valid = (x_all >= 0.0) & (x_all <= (w - 1)) & (y_all >= 0.0) & (y_all <= (h - 1))
        profs = (
            F.grid_sample(
                lum_aa_4d,
                self._to_norm_grid(x_all, y_all),
                mode="bilinear",
                padding_mode="border",
                align_corners=True,
            )
            .squeeze(0)
            .squeeze(0)
            .cpu()
            .numpy()
        )

        best_p, best_c = 0, 0.0
        # `(cx_r, cy_r, phi, s_lo_px, s_hi_px, n_lo_px, n_hi_px, med_cpd)`
        corridors: list[tuple[float, float, float, float, float, float, float, float]] = []

        for b_idx, (cx_r, cy_r, phi) in enumerate(bundles):
            p_phi = float(self.geom.pixels_per_degree(phi))
            p_perp = float(self.geom.pixels_per_degree((phi + 0.5 * math.pi) % math.pi))
            d_perp_px = 0.45 * p_perp
            seg_hits: list[tuple[int, float, float, int, float]] = []
            for k in range(n_par):
                r_idx = b_idx * n_par + k
                vm = valid[r_idx]
                if int(np.sum(vm)) < 12:
                    continue
                vals = profs[r_idx, vm]
                if float(np.ptp(vals)) < self.pse.pattern_hysteresis_cd_m2:
                    continue
                runs = find_regular_open_runs_1d(
                    vals, theta_all[r_idx, vm], self.pse, full_period_cv_max=self._enh_cv_max
                )
                for rt0, rt1, rprs, rcpd in runs:
                    seg_hits.append((k, rt0, rt1, rprs, rcpd))

            for k1, t0_1, t1_1, p1, c1 in seg_hits:
                partners = 1
                for k2, t0_2, t1_2, _, c2 in seg_hits:
                    if k1 == k2 or abs(k1 - k2) > 3:
                        continue
                    ov = min(t1_1, t1_2) - max(t0_1, t0_2)
                    span_m = max(0.5, min(t1_1 - t0_1, t1_2 - t0_2))
                    if ov >= 0.45 * span_m and (0.55 * c1 <= c2 <= 1.80 * c1):
                        partners += 1
                if partners >= self._enh_min_parallel:
                    if p1 > best_p:
                        best_p, best_c = p1, c1
                    pad_s = 0.80 * (p_phi / max(c1, 1.0))
                    s_lo = s0 + t0_1 * p_phi - pad_s
                    s_hi = s0 + t1_1 * p_phi + pad_s
                    off_px = float(off_degs[k1]) * p_perp
                    corridors.append(
                        (
                            cx_r,
                            cy_r,
                            phi,
                            s_lo,
                            s_hi,
                            off_px - 0.85 * d_perp_px,
                            off_px + 0.85 * d_perp_px,
                            c1,
                        )
                    )

        return best_p, best_c, corridors

    def _estimate_separated_centers(
        self, lum_aa_4d: torch.Tensor
    ) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
        """`(3-3a)` Separate Sobel Top-2 Tangent -> Rings, Top-2 Normal -> Spokes, + `(cx0, cy0)`."""
        w, h = self.geom.width, self.geom.height
        cx0, cy0 = 0.5 * (w - 1), 0.5 * (h - 1)
        min_sep = 0.08 * min(w, h)

        grads = F.conv2d(lum_aa_4d, self._sobel, padding=1)[0]
        gx_sub = grads[0, 2:-2:4, 2:-2:4]
        gy_sub = grads[1, 2:-2:4, 2:-2:4]
        mag = torch.hypot(gx_sub, gy_sub)
        flat_mag = mag.reshape(-1)
        if flat_mag.numel() == 0:
            return [(cx0, cy0)], [(cx0, cy0)]

        k_top = min(512, flat_mag.numel())
        top_vals, top_idx = torch.topk(flat_mag, k=k_top)
        valid_g = top_vals > 15.0
        if int(valid_g.sum().item()) < 24:
            return [(cx0, cy0)], [(cx0, cy0)]

        top_idx = top_idx[valid_g]
        sub_w = gx_sub.shape[1]
        py = ((top_idx // sub_w).float() * 4.0) + 2.0
        px = ((top_idx % sub_w).float() * 4.0) + 2.0
        gxv = gx_sub.reshape(-1)[top_idx]
        gyv = gy_sub.reshape(-1)[top_idx]
        gnorm = torch.clamp(torch.hypot(gxv, gyv), min=1e-6)
        nx, ny = gxv / gnorm, gyv / gnorm
        tx, ty = -ny, nx

        bins_y, bins_x = 24, 32
        x_min, x_max = -0.15 * w, 1.15 * w
        y_min, y_max = -0.15 * h, 1.15 * h
        diag = math.hypot(w, h)
        t_steps = torch.linspace(-diag, diag, 41, device=self.device)[None, :]

        def _top_k(vx: torch.Tensor, vy: torch.Tensor) -> list[tuple[float, float]]:
            out_c: list[tuple[float, float]] = []
            ux, uy = -vy, vx
            rhs_proj = ux * px + uy * py
            bin_tol = 1.15 * max((x_max - x_min) / bins_x, (y_max - y_min) / bins_y)
            cand_x = px[:, None] + vx[:, None] * t_steps
            cand_y = py[:, None] + vy[:, None] * t_steps
            bx = ((cand_x - x_min) / max(x_max - x_min, 1.0) * bins_x).long()
            by = ((cand_y - y_min) / max(y_max - y_min, 1.0) * bins_y).long()
            in_b = (bx >= 0) & (bx < bins_x) & (by >= 0) & (by < bins_y)
            flat_bins = (by * bins_x + bx)[in_b]
            if flat_bins.numel() > 0:
                hist = torch.bincount(flat_bins, minlength=bins_y * bins_x)
                for b_i in torch.argsort(hist, descending=True).tolist():
                    if int(hist[b_i].item()) < 12 or len(out_c) >= self._enh_sobel_top_k:
                        break
                    vy_i, vx_i = b_i // bins_x, b_i % bins_x
                    ex = float(
                        np.clip(x_min + (vx_i + 0.5) * (x_max - x_min) / bins_x, -0.15 * w, 1.15 * w)
                    )
                    ey = float(
                        np.clip(y_min + (vy_i + 0.5) * (y_max - y_min) / bins_y, -0.15 * h, 1.15 * h)
                    )
                    # Sub-pixel 2x2 least-squares intersection of rays passing near `(ex, ey)`
                    perp_d = (ux * (ex - px) + uy * (ey - py)).abs()
                    inliers = perp_d <= bin_tol
                    if int(inliers.sum().item()) >= 8:
                        u_x = ux[inliers]
                        u_y = uy[inliers]
                        r_p = rhs_proj[inliers]
                        a00 = float((u_x * u_x).sum().item())
                        a01 = float((u_x * u_y).sum().item())
                        a11 = float((u_y * u_y).sum().item())
                        det_a = a00 * a11 - a01 * a01
                        if det_a > 1e-3 * (a00 + a11) ** 2:
                            b0 = float((u_x * r_p).sum().item())
                            b1 = float((u_y * r_p).sum().item())
                            rx = (a11 * b0 - a01 * b1) / det_a
                            ry = (a00 * b1 - a01 * b0) / det_a
                            if math.hypot(rx - ex, ry - ey) <= 1.5 * bin_tol:
                                ex = float(np.clip(rx, -0.15 * w, 1.15 * w))
                                ey = float(np.clip(ry, -0.15 * h, 1.15 * h))
                    if all(math.hypot(ex - x0, ey - y0) > min_sep for x0, y0 in out_c):
                        out_c.append((ex, ey))
            if all(math.hypot(cx0 - x0, cy0 - y0) > min_sep for x0, y0 in out_c):
                out_c.append((cx0, cy0))
            return out_c

        return _top_k(tx, ty), _top_k(nx, ny)

    def _scan_rings_enhanced(
        self,
        lum_aa_4d: torch.Tensor,
        ring_centers: list[tuple[float, float]],
        closed_contrast: torch.Tensor | None = None,
        *,
        min_shell_area: float | None = None,
    ) -> tuple[int, float, float, torch.Tensor | None]:
        w, h = self.geom.width, self.geom.height
        area_gate = self.pse.pattern_area_min if min_shell_area is None else float(min_shell_area)
        mask_acc = (
            torch.zeros((h, w), dtype=torch.bool, device=self.device)
            if closed_contrast is not None
            else None
        )
        if not ring_centers:
            return 0, 0.0, 0.0, mask_acc

        c_np = np.asarray(ring_centers, dtype=np.float32)
        c_gpu = torch.from_numpy(c_np).to(self.device)
        ring_xy = self._ring_unit_xy[None, :, :] + c_gpu[:, None, :]
        gx = (2.0 * ring_xy[..., 0] / max(w - 1, 1)) - 1.0
        gy = (2.0 * ring_xy[..., 1] / max(h - 1, 1)) - 1.0
        ring_vals_batch = (
            F.grid_sample(
                lum_aa_4d,
                torch.stack([gx, gy], dim=-1).unsqueeze(0),
                mode="bilinear",
                padding_mode="border",
                align_corners=True,
            )
            .squeeze(0)
            .squeeze(0)
            .cpu()
            .numpy()
        )
        ring_xy_np = self._ring_unit_xy_np[None, :, :] + c_np[:, None, :]
        in_bounds_batch = (
            (ring_xy_np[..., 0] >= 0.0)
            & (ring_xy_np[..., 0] <= (w - 1))
            & (ring_xy_np[..., 1] >= 0.0)
            & (ring_xy_np[..., 1] <= (h - 1))
        )

        best_pairs, best_cpd, best_radial_area = 0, 0.0, 0.0
        radii = self._ring_radii
        n_r = len(radii)

        for c_idx, (cx, cy) in enumerate(ring_centers):
            ring_vals_all = ring_vals_batch[c_idx]
            in_bounds_all = in_bounds_batch[c_idx]
            c_best_p, c_best_c, band_hits = 0, 0.0, 0
            active_shells: list[int] = []
            ring_arc_specs: list[tuple[int, float, float, bool]] = []

            for k_idx, ((s_idx, e_idx), (cum_theta, perim_deg)) in enumerate(
                zip(self._ring_slices, self._ring_theta)
            ):
                vals = ring_vals_all[s_idx:e_idx]
                valid = in_bounds_all[s_idx:e_idx]
                if not np.any(valid):
                    continue
                theta_lut, alpha_lut = self._ring_alpha_luts[k_idx]

                if mask_acc is None:
                    rbp, rbc, rwp = 0, 0.0, 0
                    if np.all(valid):
                        if float(np.ptp(vals)) < self.pse.pattern_hysteresis_cd_m2:
                            continue
                        rbp, rbc, rwp = evaluate_1d_closed_dual(vals, cum_theta, perim_deg, self.pse)
                    else:
                        padded = np.concatenate(([False], valid, [False]))
                        edges = np.flatnonzero(padded[1:] != padded[:-1])
                        runs = [(int(edges[k]), int(edges[k + 1])) for k in range(0, len(edges), 2)]
                        arc_list: list[tuple[np.ndarray, np.ndarray]] = []
                        if len(runs) >= 2 and runs[0][0] == 0 and runs[-1][1] == len(valid):
                            fs, fe = runs[0]
                            ls, le = runs[-1]
                            mv = np.concatenate([vals[ls:le], vals[fs:fe]])
                            mt = np.concatenate([cum_theta[ls:le], cum_theta[fs:fe] + perim_deg])
                            runs = runs[1:-1]
                            if len(mv) >= 12:
                                arc_list.append((mv, mt))
                        for rs, re in runs:
                            if (re - rs) >= 12:
                                arc_list.append((vals[rs:re], cum_theta[rs:re]))
                        for av, at in arc_list:
                            if float(np.ptp(av)) < self.pse.pattern_hysteresis_cd_m2:
                                continue
                            pb, cb, pw = evaluate_1d_open_dual(av, at, self.pse)
                            if pb > rbp:
                                rbp, rbc = pb, cb
                            if pw > rwp:
                                rwp = pw

                    if rbp >= self.pse.pattern_min_pairs:
                        band_hits += 1
                        if rbp > c_best_p:
                            c_best_p, c_best_c = rbp, rbc
                    if max(rbp, rwp) >= self.pse.pattern_min_pairs:
                        active_shells.append(k_idx)
                else:
                    if np.all(valid):
                        if float(np.ptp(vals)) < self.pse.pattern_hysteresis_cd_m2:
                            continue
                        band_runs = _find_valid_closed_runs_1d(vals, cum_theta, perim_deg, self.pse)
                        wedge_runs = _find_valid_closed_runs_1d(
                            vals, cum_theta, perim_deg, self.pse, cpd_lo=0.05
                        )
                    else:
                        padded = np.concatenate(([False], valid, [False]))
                        edges = np.flatnonzero(padded[1:] != padded[:-1])
                        runs_ib = [
                            (int(edges[k]), int(edges[k + 1])) for k in range(0, len(edges), 2)
                        ]
                        arc_list = []
                        if len(runs_ib) >= 2 and runs_ib[0][0] == 0 and runs_ib[-1][1] == len(valid):
                            fs, fe = runs_ib[0]
                            ls, le = runs_ib[-1]
                            mv = np.concatenate([vals[ls:le], vals[fs:fe]])
                            mt = np.concatenate([cum_theta[ls:le], cum_theta[fs:fe] + perim_deg])
                            runs_ib = runs_ib[1:-1]
                            if len(mv) >= 12:
                                arc_list.append((mv, mt))
                        for rs, re in runs_ib:
                            if (re - rs) >= 12:
                                arc_list.append((vals[rs:re], cum_theta[rs:re]))
                        band_runs, wedge_runs = [], []
                        for av, at in arc_list:
                            if float(np.ptp(av)) < self.pse.pattern_hysteresis_cd_m2:
                                continue
                            band_runs.extend(_find_valid_open_runs_1d(av, at, self.pse))
                            wedge_runs.extend(_find_valid_open_runs_1d(av, at, self.pse, cpd_lo=0.05))

                    if band_runs:
                        band_hits += 1
                    if band_runs or wedge_runs:
                        active_shells.append(k_idx)
                    for t0, t1, pairs, med_cpd in band_runs:
                        if pairs > c_best_p:
                            c_best_p, c_best_c = pairs, med_cpd
                        pad_deg = 0.80 / max(med_cpd, 1.0)
                        t0_p = t0 - pad_deg
                        t1_p = t1 + pad_deg
                        if (t1_p - t0_p) >= 0.88 * perim_deg:
                            ring_arc_specs.append((k_idx, 0.0, 2.0 * math.pi, True))
                        else:
                            a0 = float(np.interp(t0_p % perim_deg, theta_lut, alpha_lut))
                            a1 = float(np.interp(t1_p % perim_deg, theta_lut, alpha_lut))
                            if a1 < a0:
                                a1 += 2.0 * math.pi
                            ring_arc_specs.append((k_idx, a0, a1, False))

                    # Contiguous outer radial-sunburst wedge continuation ONLY when inner full-360 band exists
                    if (
                        not band_runs
                        and wedge_runs
                        and any(is_f and ki == (k_idx - 1) for ki, _, _, is_f in ring_arc_specs)
                    ):
                        for t0, t1, wpairs, med_cpd in wedge_runs:
                            if wpairs >= max(self.pse.pattern_min_pairs, c_best_p - 2):
                                pad_deg = 0.80 / max(med_cpd, 0.4)
                                if ((t1 + pad_deg) - (t0 - pad_deg)) >= 0.88 * perim_deg:
                                    ring_arc_specs.append((k_idx, 0.0, 2.0 * math.pi, True))
                                    break

            if band_hits >= 2 and c_best_p >= self.pse.pattern_min_pairs and active_shells:
                dist_2d = torch.hypot(self._xx_2d - cx, self._yy_2d - cy)
                shell_mask = torch.zeros_like(dist_2d, dtype=torch.bool)
                for k_idx in active_shells:
                    r_lo = 0.0 if k_idx == 0 else 0.5 * (radii[k_idx - 1] + radii[k_idx])
                    r_hi = (
                        (1.25 * radii[-1])
                        if (k_idx == n_r - 1)
                        else 0.5 * (radii[k_idx] + radii[k_idx + 1])
                    )
                    shell_mask |= (dist_2d >= r_lo) & (dist_2d < r_hi)
                rad_area = float(shell_mask.float().mean().item())
                if rad_area > area_gate:
                    if c_best_p > best_pairs:
                        best_pairs, best_cpd = c_best_p, c_best_c
                    best_radial_area = max(best_radial_area, rad_area)
                    if mask_acc is not None and closed_contrast is not None:
                        ang_2d = torch.remainder(
                            torch.atan2(self._yy_2d - cy, self._xx_2d - cx), 2.0 * math.pi
                        )
                        center_ring_mask = torch.zeros_like(mask_acc)
                        has_inner_full = any(is_f and k_i <= 2 for k_i, _, _, is_f in ring_arc_specs)
                        for k_idx, a0, a1, is_full in ring_arc_specs:
                            r_lo = 0.0 if k_idx == 0 else 0.5 * (radii[k_idx - 1] + radii[k_idx])
                            r_hi = (
                                (1.25 * radii[-1])
                                if (k_idx == n_r - 1)
                                else 0.5 * (radii[k_idx] + radii[k_idx + 1])
                            )
                            annulus = (dist_2d >= r_lo) & (dist_2d <= r_hi)
                            if is_full:
                                if k_idx <= 2:
                                    center_ring_mask |= annulus
                                else:
                                    center_ring_mask |= annulus & closed_contrast
                            else:
                                center_ring_mask |= (
                                    annulus & _in_azimuth_arc(ang_2d, a0, a1) & closed_contrast
                                )
                        if has_inner_full:
                            center_ring_mask |= dist_2d <= (1.2 * radii[0])
                        mask_acc |= center_ring_mask

        return best_pairs, best_cpd, best_radial_area, mask_acc

    def _scan_spokes_enhanced(
        self,
        lum_aa_4d: torch.Tensor,
        spoke_centers: list[tuple[float, float]],
        closed_contrast: torch.Tensor | None = None,
    ) -> tuple[int, float, torch.Tensor | None]:
        w, h = self.geom.width, self.geom.height
        mask_acc = (
            torch.zeros((h, w), dtype=torch.bool, device=self.device)
            if closed_contrast is not None
            else None
        )
        if not spoke_centers:
            return 0, 0.0, mask_acc

        c_np = np.asarray(spoke_centers, dtype=np.float32)
        c_gpu = torch.from_numpy(c_np).to(self.device)
        c_count = len(spoke_centers)
        spoke_xy = (self._spoke_unit_xy[None, :, :, :] + c_gpu[:, None, None, :]).view(
            c_count * 24, -1, 2
        )
        sgx = (2.0 * spoke_xy[..., 0] / max(w - 1, 1)) - 1.0
        sgy = (2.0 * spoke_xy[..., 1] / max(h - 1, 1)) - 1.0
        spoke_vals_all = (
            F.grid_sample(
                lum_aa_4d,
                torch.stack([sgx, sgy], dim=-1).unsqueeze(0),
                mode="bilinear",
                padding_mode="border",
                align_corners=True,
            )
            .squeeze(0)
            .squeeze(0)
            .cpu()
            .numpy()
        )
        spoke_xy_np = (
            self._spoke_unit_xy_np[None, :, :, :] + c_np[:, None, None, :]
        ).reshape(c_count * 24, -1, 2)
        spoke_valid_all = (
            (spoke_xy_np[..., 0] >= 0.0)
            & (spoke_xy_np[..., 0] <= (w - 1))
            & (spoke_xy_np[..., 1] >= 0.0)
            & (spoke_xy_np[..., 1] <= (h - 1))
        )

        best_pairs, best_cpd = 0, 0.0
        sp_ptp = np.ptp(spoke_vals_all, axis=1)
        d_sp = 0.68 * (2.0 * math.pi / 24.0)
        spokes_by_center: dict[int, list[tuple[int, float, float, int, float]]] = {}

        for row_idx in np.flatnonzero(sp_ptp >= self.pse.pattern_hysteresis_cd_m2):
            vm = spoke_valid_all[row_idx]
            if int(np.sum(vm)) < 12:
                continue
            c_idx = int(row_idx // 24)
            sp_idx = int(row_idx % 24)
            p_phi_sp = float(self._spoke_p_phi[sp_idx])
            runs = find_regular_open_runs_1d(
                spoke_vals_all[row_idx, vm],
                self._spoke_s_px[vm] / p_phi_sp,
                self.pse,
                full_period_cv_max=self._enh_cv_max,
            )
            if not runs:
                continue
            lst = spokes_by_center.setdefault(c_idx, [])
            for t0, t1, prs, med_cpd in runs:
                pad_r = 0.80 * (p_phi_sp / max(med_cpd, 1.0))
                r_lo = max(0.0, t0 * p_phi_sp - pad_r)
                r_hi = t1 * p_phi_sp + pad_r
                lst.append((sp_idx, r_lo, r_hi, prs, med_cpd))

        for c_idx, sp_list in spokes_by_center.items():
            distinct_orientations = {x[0] % 12 for x in sp_list}
            if len(distinct_orientations) < 2:
                continue
            top_sp = max(sp_list, key=lambda x: x[3])
            if top_sp[3] > best_pairs:
                best_pairs, best_cpd = top_sp[3], top_sp[4]
            if mask_acc is not None and closed_contrast is not None:
                cx, cy = spoke_centers[c_idx]
                dist_2d = torch.hypot(self._xx_2d - cx, self._yy_2d - cy)
                ang_2d = torch.remainder(
                    torch.atan2(self._yy_2d - cy, self._xx_2d - cx), 2.0 * math.pi
                )
                sp_corridor = torch.zeros_like(mask_acc)
                for sp_idx, r_lo, r_hi, _, _ in sp_list:
                    a_mid = float(self._spoke_alphas[sp_idx])
                    sp_corridor |= (
                        (dist_2d >= r_lo)
                        & (dist_2d <= r_hi)
                        & _in_azimuth_arc(ang_2d, a_mid - d_sp, a_mid + d_sp)
                    )
                mask_acc |= sp_corridor & closed_contrast

        return best_pairs, best_cpd, mask_acc

    def _evaluate_gabor_fallback(
        self, lum: torch.Tensor, contrast_mask: torch.Tensor
    ) -> tuple[int, float, torch.Tensor]:
        """`(3-4a)` 24-channel Frequency-Domain Analytic Quadrature 2D Log-Gabor Wavelet Bank fallback."""
        empty = torch.zeros_like(lum, dtype=torch.bool)
        if not self._enh_enable_gabor:
            return 0, 0.0, empty

        spec_full = torch.fft.fft2(lum - lum.mean())
        z_bank = torch.fft.ifft2(spec_full.unsqueeze(0) * self._gabor_bank_fft)
        swing_bank = 2.0 * torch.abs(z_bank)
        max_swing, best_ch = torch.max(swing_bank, dim=0)

        r = self._gabor_var_rad
        k = 2 * r + 1
        l4 = lum.unsqueeze(0).unsqueeze(0)
        l_pad = F.pad(l4, (r, r, r, r), mode="replicate")
        loc_mean = F.avg_pool2d(F.avg_pool2d(l_pad, (1, k), stride=1), (k, 1), stride=1)[0, 0]
        l2_pad = F.pad(l4 * l4, (r, r, r, r), mode="replicate")
        loc_m2 = F.avg_pool2d(F.avg_pool2d(l2_pad, (1, k), stride=1), (k, 1), stride=1)[0, 0]
        loc_var = torch.clamp(loc_m2 - loc_mean * loc_mean, min=1e-4)

        coherence = (0.125 * max_swing * max_swing) / loc_var
        coherent_wave = (
            (max_swing >= self.pse.delta_lum_min_cd_m2) & (coherence >= 0.35) & contrast_mask
        )
        if float(coherent_wave.float().mean().item()) <= self.pse.pattern_area_min:
            return 0, 0.0, empty

        ys, xs = torch.where(coherent_wave)
        if xs.numel() < 64:
            return 0, 0.0, empty

        ch_counts = torch.bincount(best_ch[coherent_wave], minlength=len(self._gabor_freqs))
        dom_ch = int(torch.argmax(ch_counts).item())
        f_dom = float(self._gabor_freqs[dom_ch])
        phi_dom = float(self._gabor_thetas[dom_ch])

        dom_m = dom_ch % 6
        adj_ms = {(dom_m - 1) % 6, dom_m, (dom_m + 1) % 6}
        keep_chs = [i for i in range(len(self._gabor_freqs)) if (i % 6) in adj_ms]
        bw_4d = (loc_mean + z_bank[keep_chs].real.sum(dim=0)).unsqueeze(0).unsqueeze(0)

        cx_w, cy_w = float(xs.float().mean().item()), float(ys.float().mean().item())
        s_vals = self._s_vals_np
        c_p, s_p = math.cos(phi_dom), math.sin(phi_dom)
        nx, ny = -s_p, c_p
        p_phi = float(self.geom.pixels_per_degree(phi_dom))
        p_perp = float(self.geom.pixels_per_degree((phi_dom + 0.5 * math.pi) % math.pi))
        t_line = (s_vals - s_vals[0]) / p_phi
        off_degs = np.linspace(-2.4, 2.4, 9)
        x_b = np.zeros((9, len(s_vals)), dtype=np.float64)
        y_b = np.zeros((9, len(s_vals)), dtype=np.float64)
        for i_o, od in enumerate(off_degs):
            off_px = od * p_perp
            x_b[i_o] = cx_w + off_px * nx + s_vals * c_p
            y_b[i_o] = cy_w + off_px * ny + s_vals * s_p
        v_b = (
            (x_b >= 0.0)
            & (x_b <= (self.geom.width - 1))
            & (y_b >= 0.0)
            & (y_b <= (self.geom.height - 1))
        )
        profs_b = (
            F.grid_sample(
                bw_4d,
                self._to_norm_grid(x_b, y_b),
                mode="bilinear",
                padding_mode="border",
                align_corners=True,
            )
            .squeeze(0)
            .squeeze(0)
            .cpu()
            .numpy()
        )
        par_hits: list[tuple[int, float]] = []
        for i_o in range(9):
            vm = v_b[i_o]
            if int(np.sum(vm)) < 12:
                continue
            runs = find_regular_open_runs_1d(
                profs_b[i_o, vm], t_line[vm], self.pse, full_period_cv_max=0.42
            )
            if runs:
                top_r = max(runs, key=lambda r_item: r_item[2])
                par_hits.append((top_r[2], top_r[3]))

        if len(par_hits) >= 2:
            best_g = max(par_hits, key=lambda x: x[0])
            return best_g[0], best_g[1], coherent_wave

        # Highly curved horseshoe/vortex grating safeguard when narrowband coherence is very high (>= 0.55)
        strong_coh = (
            (max_swing >= self.pse.delta_lum_min_cd_m2) & (coherence >= 0.55) & contrast_mask
        )
        if float(strong_coh.float().mean().item()) > self.pse.pattern_area_min:
            proj_deg = (
                (xs.float() * math.cos(phi_dom)) / self.geom.px_per_deg_x
                + (ys.float() * math.sin(phi_dom)) / self.geom.px_per_deg_y
            )
            q_lo, q_hi = torch.quantile(
                proj_deg, torch.tensor([0.02, 0.98], device=self.device)
            ).tolist()
            est_pairs = int(math.floor(f_dom * max(0.0, float(q_hi - q_lo))))
            if est_pairs >= self.pse.pattern_min_pairs:
                return est_pairs, f_dom, coherent_wave

        return 0, 0.0, empty

    def _inspect_frame_enhanced(self, lum: torch.Tensor) -> tuple[bool, float, int, float]:
        lum_4d = lum.unsqueeze(0).unsqueeze(0)
        lum_aa_4d = self._antialias_4d(lum_4d)
        contrast_mask = self._compute_contrast_mask(lum_4d)
        fine_area = float(contrast_mask.float().mean().item())

        if fine_area <= self.pse.pattern_area_min:
            if fine_area < 0.12:
                return False, fine_area, 0, 0.0
            sub = lum_4d[..., ::4, ::4]
            c_max, c_min = separable_max_min_2d(sub, self._coarse_radius)
            c_dl = c_max - c_min
            coarse_ok = torch.where(
                c_min < self.pse.dark_lum_limit_cd_m2,
                c_dl >= self.pse.delta_lum_min_cd_m2,
                (c_dl / torch.clamp(c_max + c_min, min=1e-6)) > self.pse.bright_michelson_min,
            )
            if float(coarse_ok.float().mean().item()) <= self.pse.pattern_area_min:
                return False, fine_area, 0, 0.0
            ring_centers, _ = self._estimate_separated_centers(lum_aa_4d)
            rad_pairs, rad_cpd, rad_area, _ = self._scan_rings_enhanced(lum_aa_4d, ring_centers)
            if rad_pairs >= self.pse.pattern_min_pairs and rad_area > self.pse.pattern_area_min:
                return True, rad_area, rad_pairs, rad_cpd
            return False, fine_area, 0, 0.0

        best_pairs, best_cpd = 0, 0.0
        effective_area = fine_area

        # Step A-1: Fixed 12 x 15 (+-0.95) linear scan with >= 2 parallel lines & full-period CV <= 0.35
        lin_profs = (
            F.grid_sample(
                lum_aa_4d,
                self._lin_grid_enh,
                mode="bilinear",
                padding_mode="border",
                align_corners=True,
            )
            .squeeze(0)
            .squeeze(0)
            .cpu()
            .numpy()
        )
        ptp = np.ptp(lin_profs, axis=1)
        hits_by_ang: dict[int, list[tuple[int, float, float, int, float]]] = {}
        for idx in np.flatnonzero(ptp >= self.pse.pattern_hysteresis_cd_m2):
            vm = self._lin_valid_enh[idx]
            if int(np.sum(vm)) < 12:
                continue
            runs = find_regular_open_runs_1d(
                lin_profs[idx, vm],
                self._lin_theta_enh[idx, vm],
                self.pse,
                full_period_cv_max=self._enh_cv_max,
            )
            if runs:
                m_ang = int(idx // self._enh_n_offsets)
                k_off = int(idx % self._enh_n_offsets)
                lst = hits_by_ang.setdefault(m_ang, [])
                for rt0, rt1, rprs, rcpd in runs:
                    lst.append((k_off, rt0, rt1, rprs, rcpd))

        singletons: list[tuple[int, int, float, float, float, float, int, float]] = []
        for m_ang, h_list in hits_by_ang.items():
            phi = math.radians(15.0 * m_ang)
            _, _, _, offsets, _ = self._lin_meta_enh[m_ang]
            for k1, t0_1, t1_1, p1, c1 in h_list:
                partners = 1
                for k2, t0_2, t1_2, _, c2 in h_list:
                    if k1 == k2 or abs(k1 - k2) > 3:
                        continue
                    ov = min(t1_1, t1_2) - max(t0_1, t0_2)
                    span_m = max(0.5, min(t1_1 - t0_1, t1_2 - t0_2))
                    if ov >= 0.45 * span_m and (0.55 * c1 <= c2 <= 1.80 * c1):
                        partners += 1
                if partners >= self._enh_min_parallel:
                    if p1 > best_pairs:
                        best_pairs, best_cpd = p1, c1
                else:
                    singletons.append((m_ang, k1, phi, float(offsets[k1]), t0_1, t1_1, p1, c1))

        if singletons:
            for _, _, _, _, prs, cpd_v in self._verify_fine_parallel_neighbors(lum_aa_4d, singletons):
                if prs > best_pairs:
                    best_pairs, best_cpd = prs, cpd_v

        # Step A-2: 3x3 overlapping Sobel dominant normal targeted bundles
        sob_p, sob_c, _ = self._scan_sobel_targeted_bundles(lum_aa_4d, contrast_mask)
        if sob_p > best_pairs:
            best_pairs, best_cpd = sob_p, sob_c

        # Step B: Separated Sobel Top-2 Tangent -> Rings, Top-2 Normal -> Spokes
        ring_centers, spoke_centers = self._estimate_separated_centers(lum_aa_4d)
        r_pairs, r_cpd, r_area, _ = self._scan_rings_enhanced(
            lum_aa_4d, ring_centers, min_shell_area=0.10
        )
        if r_pairs > best_pairs:
            best_pairs, best_cpd = r_pairs, r_cpd
            effective_area = max(effective_area, r_area)

        sp_pairs, sp_cpd, _ = self._scan_spokes_enhanced(lum_aa_4d, spoke_centers)
        if sp_pairs > best_pairs:
            best_pairs, best_cpd = sp_pairs, sp_cpd

        # Step C: `(3-4a)` Retain Global 2D FFT fallback + Quadrature 2D Gabor Wavelet Bank fallback
        if best_pairs < self.pse.pattern_min_pairs:
            fft_pairs, fft_cpd = self._evaluate_2d_fft_fallback(lum)
            if fft_pairs >= self.pse.pattern_min_pairs:
                best_pairs, best_cpd = fft_pairs, fft_cpd
            else:
                gab_pairs, gab_cpd, _ = self._evaluate_gabor_fallback(lum, contrast_mask)
                if gab_pairs > best_pairs:
                    best_pairs, best_cpd = gab_pairs, gab_cpd

        return best_pairs >= self.pse.pattern_min_pairs, effective_area, best_pairs, best_cpd

    def _extract_enhanced_segments_mask(
        self, lum: torch.Tensor, close_scale: float
    ) -> torch.Tensor:
        w, h = self.geom.width, self.geom.height
        seg_union = torch.zeros((h, w), dtype=torch.bool, device=self.device)
        lum_4d = lum.unsqueeze(0).unsqueeze(0)
        lum_aa_4d = self._antialias_4d(lum_4d)
        contrast_mask = self._compute_contrast_mask(lum_4d)
        if not bool(contrast_mask.any().item()):
            return seg_union

        r_close = max(3, int(round(max(0.55, close_scale) * self.pool_radius)))
        closed_contrast = morphological_close_2d(contrast_mask, r_close)
        fine_area = float(contrast_mask.float().mean().item())
        found_1d = False

        dx = self._xx_2d - self._lin_cx
        dy = self._yy_2d - self._lin_cy
        s0 = float(self._s_vals_np[0])

        # 1) Linear fixed 12 x 15 + Sobel targeted bundles (when `fine_area > pattern_area_min`)
        if fine_area > self.pse.pattern_area_min:
            lin_profs = (
                F.grid_sample(
                    lum_aa_4d,
                    self._lin_grid_enh,
                    mode="bilinear",
                    padding_mode="border",
                    align_corners=True,
                )
                .squeeze(0)
                .squeeze(0)
                .cpu()
                .numpy()
            )
            ptp = np.ptp(lin_profs, axis=1)
            hits_by_ang: dict[int, list[tuple[int, float, float, int, float]]] = {}
            for idx in np.flatnonzero(ptp >= self.pse.pattern_hysteresis_cd_m2):
                vm = self._lin_valid_enh[idx]
                if int(np.sum(vm)) < 12:
                    continue
                runs = find_regular_open_runs_1d(
                    lin_profs[idx, vm],
                    self._lin_theta_enh[idx, vm],
                    self.pse,
                    full_period_cv_max=self._enh_cv_max,
                )
                if runs:
                    m_ang = int(idx // self._enh_n_offsets)
                    k_off = int(idx % self._enh_n_offsets)
                    lst = hits_by_ang.setdefault(m_ang, [])
                    for rt0, rt1, rprs, rcpd in runs:
                        lst.append((k_off, rt0, rt1, rprs, rcpd))

            confirmed_by_ang: dict[int, list[tuple[int, float, float, float]]] = {}
            singletons: list[tuple[int, int, float, float, float, float, int, float]] = []
            for m_ang, h_list in hits_by_ang.items():
                phi = math.radians(15.0 * m_ang)
                _, _, _, offsets, _ = self._lin_meta_enh[m_ang]
                for k1, t0_1, t1_1, p1, c1 in h_list:
                    partners = 1
                    for k2, t0_2, t1_2, _, c2 in h_list:
                        if k1 == k2 or abs(k1 - k2) > 3:
                            continue
                        ov = min(t1_1, t1_2) - max(t0_1, t0_2)
                        span_m = max(0.5, min(t1_1 - t0_1, t1_2 - t0_2))
                        if ov >= 0.45 * span_m and (0.55 * c1 <= c2 <= 1.80 * c1):
                            partners += 1
                    if partners >= self._enh_min_parallel:
                        confirmed_by_ang.setdefault(m_ang, []).append((k1, t0_1, t1_1, c1))
                    else:
                        singletons.append((m_ang, k1, phi, float(offsets[k1]), t0_1, t1_1, p1, c1))

            if singletons:
                for m_ang, k_off, t0_s, t1_s, _, med_cpd in self._verify_fine_parallel_neighbors(
                    lum_aa_4d, singletons
                ):
                    confirmed_by_ang.setdefault(m_ang, []).append((k_off, t0_s, t1_s, med_cpd))

            for m_ang, seg_list in confirmed_by_ang.items():
                c_p, s_p, p_phi, offsets, o_max = self._lin_meta_enh[m_ang]
                u_2d = dx * c_p + dy * s_p
                n_2d = -dx * s_p + dy * c_p
                d_off = (
                    (2.0 * self._enh_offset_limit * o_max) / max(self._enh_n_offsets - 1, 1)
                    if o_max > 0.0
                    else 1.0
                )
                ang_corridor = torch.zeros_like(seg_union)
                for k_off, t0, t1, med_cpd in seg_list:
                    pad_s = 0.80 * (p_phi / max(med_cpd, 1.0))
                    s_lo = s0 + t0 * p_phi - pad_s
                    s_hi = s0 + t1 * p_phi + pad_s
                    n_lo = (
                        (-1.05 * o_max)
                        if k_off == 0
                        else (float(offsets[k_off]) - 0.75 * d_off)
                    )
                    n_hi = (
                        (1.05 * o_max)
                        if k_off == (self._enh_n_offsets - 1)
                        else (float(offsets[k_off]) + 0.75 * d_off)
                    )
                    ang_corridor |= (u_2d >= s_lo) & (u_2d <= s_hi) & (n_2d >= n_lo) & (n_2d <= n_hi)
                seg_union |= ang_corridor & closed_contrast
                found_1d = True

            # Sobel targeted parallel bundle corridors
            _, _, sob_corridors = self._scan_sobel_targeted_bundles(lum_aa_4d, contrast_mask)
            if sob_corridors:
                sob_mask = torch.zeros_like(seg_union)
                for cx_r, cy_r, phi, s_lo, s_hi, n_lo, n_hi, _ in sob_corridors:
                    c_p, s_p = math.cos(phi), math.sin(phi)
                    dx_r = self._xx_2d - cx_r
                    dy_r = self._yy_2d - cy_r
                    u_r = dx_r * c_p + dy_r * s_p
                    n_r = -dx_r * s_p + dy_r * c_p
                    sob_mask |= (u_r >= s_lo) & (u_r <= s_hi) & (n_r >= n_lo) & (n_r <= n_hi)
                seg_union |= sob_mask & closed_contrast
                found_1d = True

        # 2) Separated Sobel Top-2 Tangent -> Rings & Top-2 Normal -> Spokes
        ring_centers, spoke_centers = self._estimate_separated_centers(lum_aa_4d)
        min_ring_area = 0.10 if (fine_area > self.pse.pattern_area_min) else self.pse.pattern_area_min
        r_pairs, _, _, ring_mask = self._scan_rings_enhanced(
            lum_aa_4d,
            ring_centers,
            closed_contrast=closed_contrast,
            min_shell_area=min_ring_area,
        )
        if r_pairs >= self.pse.pattern_min_pairs and ring_mask is not None and bool(ring_mask.any().item()):
            seg_union |= ring_mask
            found_1d = True

        if fine_area > self.pse.pattern_area_min:
            sp_pairs, _, spoke_mask = self._scan_spokes_enhanced(
                lum_aa_4d, spoke_centers, closed_contrast=closed_contrast
            )
            if (
                sp_pairs >= self.pse.pattern_min_pairs
                and spoke_mask is not None
                and bool(spoke_mask.any().item())
            ):
                seg_union |= spoke_mask
                found_1d = True

        # 3) `(3-4a)` Global 2D FFT fallback + Quadrature 2D Log-Gabor Wavelet Bank fallback
        if not found_1d and fine_area > self.pse.pattern_area_min:
            fft_pairs, _, active_wave = self._evaluate_2d_fft_wave_mask(lum)
            if fft_pairs >= self.pse.pattern_min_pairs:
                seg_union |= morphological_close_2d(active_wave & contrast_mask, r_close)
            else:
                gab_pairs, _, gab_wave = self._evaluate_gabor_fallback(lum, contrast_mask)
                if gab_pairs >= self.pse.pattern_min_pairs:
                    seg_union |= morphological_close_2d(gab_wave & contrast_mask, r_close)

        return seg_union

    # -----------------------------------------------------------------------
    # Legacy `(enhanced_spatial_pattern=False)` Segment Mask Back-Projection
    # -----------------------------------------------------------------------

    def _extract_legacy_segments_mask(
        self,
        lum: torch.Tensor,
        close_scale: float,
    ) -> torch.Tensor:
        """Extract 2D periodic pattern mask matching upstream `vdcorefs` (`enhanced_spatial_pattern=False`)."""
        w, h = self.geom.width, self.geom.height
        seg_union = torch.zeros((h, w), dtype=torch.bool, device=self.device)
        lum_4d = lum.unsqueeze(0).unsqueeze(0)
        contrast_mask = self._compute_contrast_mask(lum_4d)
        if not bool(contrast_mask.any().item()):
            return seg_union

        r_close = max(3, int(round(max(0.55, close_scale) * self.pool_radius)))
        closed_contrast = morphological_close_2d(contrast_mask, r_close)
        fine_area = float(contrast_mask.float().mean().item())
        found_1d = False

        if fine_area > self.pse.pattern_area_min:
            lin_profiles = (
                F.grid_sample(
                    lum_4d,
                    self._linear_grid,
                    mode="bilinear",
                    padding_mode="border",
                    align_corners=True,
                )
                .squeeze(0)
                .squeeze(0)
                .cpu()
                .numpy()
            )
            ptp = np.ptp(lin_profiles, axis=1)
            active_by_angle: dict[int, list[tuple[int, float, float, float]]] = {}
            for idx in np.flatnonzero(ptp >= self.pse.pattern_hysteresis_cd_m2):
                v_mask = self._linear_valid[idx]
                if int(np.sum(v_mask)) < 12:
                    continue
                runs = _find_valid_open_runs_1d(
                    lin_profiles[idx, v_mask], self._linear_theta[idx, v_mask], self.pse
                )
                if not runs:
                    continue
                m_ang = int(idx // 9)
                k_off = int(idx % 9)
                lst = active_by_angle.setdefault(m_ang, [])
                for t0, t1, _, med_cpd in runs:
                    lst.append((k_off, t0, t1, med_cpd))

            if active_by_angle:
                dx = self._xx_2d - self._lin_cx
                dy = self._yy_2d - self._lin_cy
                for m_ang, seg_list in active_by_angle.items():
                    c_p, s_p, p_phi, offsets, o_max = self._lin_angle_meta[m_ang]
                    u_2d = dx * c_p + dy * s_p
                    n_2d = -dx * s_p + dy * c_p
                    d_off = (1.5 * o_max) / 8.0 if o_max > 0.0 else 1.0
                    ang_corridor = torch.zeros_like(seg_union)

                    for k_off, t0, t1, med_cpd in seg_list:
                        pad_s = 0.80 * (p_phi / max(med_cpd, 1.0))
                        s_lo = self._lin_s0 + t0 * p_phi - pad_s
                        s_hi = self._lin_s0 + t1 * p_phi + pad_s
                        n_lo = (-1.05 * o_max) if k_off == 0 else (float(offsets[k_off]) - 0.65 * d_off)
                        n_hi = (1.05 * o_max) if k_off == 8 else (float(offsets[k_off]) + 0.65 * d_off)
                        ang_corridor |= (u_2d >= s_lo) & (u_2d <= s_hi) & (n_2d >= n_lo) & (n_2d <= n_hi)

                    seg_union |= ang_corridor & closed_contrast
                    found_1d = True

        centers = self._estimate_candidate_centers(lum_4d, contrast_mask)
        if centers:
            c_np = np.asarray(centers, dtype=np.float32)
            c_gpu = torch.from_numpy(c_np).to(self.device)

            ring_xy = self._ring_unit_xy[None, :, :] + c_gpu[:, None, :]
            gx = (2.0 * ring_xy[..., 0] / max(w - 1, 1)) - 1.0
            gy = (2.0 * ring_xy[..., 1] / max(h - 1, 1)) - 1.0
            ring_grid = torch.stack([gx, gy], dim=-1).unsqueeze(0)
            ring_vals_batch = (
                F.grid_sample(lum_4d, ring_grid, mode="bilinear", padding_mode="border", align_corners=True)
                .squeeze(0)
                .squeeze(0)
                .cpu()
                .numpy()
            )
            ring_xy_np = self._ring_unit_xy_np[None, :, :] + c_np[:, None, :]
            in_bounds_batch = (
                (ring_xy_np[..., 0] >= 0.0)
                & (ring_xy_np[..., 0] <= (w - 1))
                & (ring_xy_np[..., 1] >= 0.0)
                & (ring_xy_np[..., 1] <= (h - 1))
            )

            radii = self._ring_radii
            n_r = len(radii)

            for c_idx, (cx, cy) in enumerate(centers):
                ring_vals_all = ring_vals_batch[c_idx]
                in_bounds_all = in_bounds_batch[c_idx]
                ring_arc_specs: list[tuple[int, float, float, bool]] = []
                wedge_shell_indices: list[int] = []
                c_best_ring_pairs = 0

                for k_idx, ((s_idx, e_idx), (cum_theta, perim_deg)) in enumerate(
                    zip(self._ring_slices, self._ring_theta)
                ):
                    vals = ring_vals_all[s_idx:e_idx]
                    valid = in_bounds_all[s_idx:e_idx]
                    if not np.any(valid):
                        continue
                    theta_lut, alpha_lut = self._ring_alpha_luts[k_idx]

                    if np.all(valid):
                        if float(np.ptp(vals)) < self.pse.pattern_hysteresis_cd_m2:
                            continue
                        band_runs = _find_valid_closed_runs_1d(vals, cum_theta, perim_deg, self.pse)
                        wedge_runs = _find_valid_closed_runs_1d(
                            vals, cum_theta, perim_deg, self.pse, cpd_lo=0.05
                        )
                    else:
                        padded = np.concatenate(([False], valid, [False]))
                        edges = np.flatnonzero(padded[1:] != padded[:-1])
                        runs_ib = [(int(edges[k]), int(edges[k + 1])) for k in range(0, len(edges), 2)]
                        arc_list: list[tuple[np.ndarray, np.ndarray]] = []
                        if len(runs_ib) >= 2 and runs_ib[0][0] == 0 and runs_ib[-1][1] == len(valid):
                            fs, fe = runs_ib[0]
                            ls, le = runs_ib[-1]
                            mv = np.concatenate([vals[ls:le], vals[fs:fe]])
                            mt = np.concatenate([cum_theta[ls:le], cum_theta[fs:fe] + perim_deg])
                            runs_ib = runs_ib[1:-1]
                            if len(mv) >= 12:
                                arc_list.append((mv, mt))
                        for rs, re in runs_ib:
                            if (re - rs) >= 12:
                                arc_list.append((vals[rs:re], cum_theta[rs:re]))

                        band_runs = []
                        wedge_runs = []
                        for av, at in arc_list:
                            if float(np.ptp(av)) < self.pse.pattern_hysteresis_cd_m2:
                                continue
                            band_runs.extend(_find_valid_open_runs_1d(av, at, self.pse))
                            wedge_runs.extend(
                                _find_valid_open_runs_1d(av, at, self.pse, cpd_lo=0.05)
                            )

                    if band_runs or wedge_runs:
                        wedge_shell_indices.append(k_idx)

                    for t0, t1, pairs, med_cpd in band_runs:
                        if pairs > c_best_ring_pairs:
                            c_best_ring_pairs = pairs
                        pad_deg = 0.80 / max(med_cpd, 1.0)
                        t0_p = t0 - pad_deg
                        t1_p = t1 + pad_deg
                        if (t1_p - t0_p) >= 0.88 * perim_deg:
                            ring_arc_specs.append((k_idx, 0.0, 2.0 * math.pi, True))
                        else:
                            a0 = float(np.interp(t0_p % perim_deg, theta_lut, alpha_lut))
                            a1 = float(np.interp(t1_p % perim_deg, theta_lut, alpha_lut))
                            if a1 < a0:
                                a1 += 2.0 * math.pi
                            ring_arc_specs.append((k_idx, a0, a1, False))

                if c_best_ring_pairs >= self.pse.pattern_min_pairs and wedge_shell_indices:
                    dist_2d = torch.hypot(self._xx_2d - cx, self._yy_2d - cy)
                    full_shell_check = torch.zeros_like(dist_2d, dtype=torch.bool)
                    for k_idx in wedge_shell_indices:
                        r_lo = 0.0 if k_idx == 0 else 0.5 * (radii[k_idx - 1] + radii[k_idx])
                        r_hi = (
                            (1.25 * radii[-1])
                            if (k_idx == n_r - 1)
                            else 0.5 * (radii[k_idx] + radii[k_idx + 1])
                        )
                        full_shell_check |= (dist_2d >= r_lo) & (dist_2d < r_hi)

                    if float(full_shell_check.float().mean().item()) > self.pse.pattern_area_min:
                        ang_2d = torch.remainder(
                            torch.atan2(self._yy_2d - cy, self._xx_2d - cx), 2.0 * math.pi
                        )
                        center_ring_mask = torch.zeros_like(seg_union)
                        has_inner_full = any(is_f and k_i <= 2 for k_i, _, _, is_f in ring_arc_specs)

                        for k_idx, a0, a1, is_full in ring_arc_specs:
                            r_lo = 0.0 if k_idx == 0 else 0.5 * (radii[k_idx - 1] + radii[k_idx])
                            r_hi = (
                                (1.25 * radii[-1])
                                if (k_idx == n_r - 1)
                                else 0.5 * (radii[k_idx] + radii[k_idx + 1])
                            )
                            annulus = (dist_2d >= r_lo) & (dist_2d <= r_hi)
                            if is_full:
                                if k_idx <= 2:
                                    center_ring_mask |= annulus
                                else:
                                    center_ring_mask |= annulus & closed_contrast
                            else:
                                center_ring_mask |= (
                                    annulus & _in_azimuth_arc(ang_2d, a0, a1) & closed_contrast
                                )

                        if has_inner_full:
                            center_ring_mask |= dist_2d <= (1.2 * radii[0])

                        seg_union |= center_ring_mask
                        found_1d = True

            if fine_area > self.pse.pattern_area_min:
                c_count = len(centers)
                spoke_xy = self._spoke_unit_xy[None, :, :, :] + c_gpu[:, None, None, :]
                spoke_xy_flat = spoke_xy.view(c_count * 24, -1, 2)
                sgx = (2.0 * spoke_xy_flat[..., 0] / max(w - 1, 1)) - 1.0
                sgy = (2.0 * spoke_xy_flat[..., 1] / max(h - 1, 1)) - 1.0
                spoke_grid = torch.stack([sgx, sgy], dim=-1).unsqueeze(0)
                spoke_vals_all = (
                    F.grid_sample(
                        lum_4d, spoke_grid, mode="bilinear", padding_mode="border", align_corners=True
                    )
                    .squeeze(0)
                    .squeeze(0)
                    .cpu()
                    .numpy()
                )
                spoke_xy_np = (self._spoke_unit_xy_np[None, :, :, :] + c_np[:, None, None, :]).reshape(
                    c_count * 24, -1, 2
                )
                spoke_valid_all = (
                    (spoke_xy_np[..., 0] >= 0.0)
                    & (spoke_xy_np[..., 0] <= (w - 1))
                    & (spoke_xy_np[..., 1] >= 0.0)
                    & (spoke_xy_np[..., 1] <= (h - 1))
                )
                sp_ptp = np.ptp(spoke_vals_all, axis=1)
                d_sp = 0.68 * (2.0 * math.pi / 24.0)

                spokes_by_center: dict[int, list[tuple[int, float, float]]] = {}
                for row_idx in np.flatnonzero(sp_ptp >= self.pse.pattern_hysteresis_cd_m2):
                    v_mask = spoke_valid_all[row_idx]
                    if int(np.sum(v_mask)) < 12:
                        continue
                    c_idx = int(row_idx // 24)
                    sp_idx = int(row_idx % 24)
                    vals = spoke_vals_all[row_idx, v_mask]
                    p_phi_sp = float(self._spoke_p_phi[sp_idx])
                    theta_sp = self._spoke_s_px[v_mask] / p_phi_sp
                    runs_sp = _find_valid_open_runs_1d(vals, theta_sp, self.pse)
                    if not runs_sp:
                        continue
                    lst_sp = spokes_by_center.setdefault(c_idx, [])
                    for t0, t1, _, med_cpd in runs_sp:
                        pad_r = 0.80 * (p_phi_sp / max(med_cpd, 1.0))
                        r_lo = max(0.0, t0 * p_phi_sp - pad_r)
                        r_hi = t1 * p_phi_sp + pad_r
                        lst_sp.append((sp_idx, r_lo, r_hi))

                for c_idx, sp_list in spokes_by_center.items():
                    cx, cy = centers[c_idx]
                    dist_2d = torch.hypot(self._xx_2d - cx, self._yy_2d - cy)
                    ang_2d = torch.remainder(
                        torch.atan2(self._yy_2d - cy, self._xx_2d - cx), 2.0 * math.pi
                    )
                    sp_corridor = torch.zeros_like(seg_union)
                    for sp_idx, r_lo, r_hi in sp_list:
                        a_mid = float(self._spoke_alphas[sp_idx])
                        sp_corridor |= (
                            (dist_2d >= r_lo)
                            & (dist_2d <= r_hi)
                            & _in_azimuth_arc(ang_2d, a_mid - d_sp, a_mid + d_sp)
                        )
                    seg_union |= sp_corridor & closed_contrast
                    found_1d = True

        if not found_1d and fine_area > self.pse.pattern_area_min:
            fft_pairs, _, active_wave = self._evaluate_2d_fft_wave_mask(lum)
            if fft_pairs >= self.pse.pattern_min_pairs:
                seg_union |= morphological_close_2d(active_wave & contrast_mask, r_close)

        return seg_union

    # -----------------------------------------------------------------------
    # Public Inspection Entry Points
    # -----------------------------------------------------------------------

    def _extract_active_segments_mask(
        self,
        lum: torch.Tensor,
        close_scale: float,
    ) -> torch.Tensor:
        if self.enhanced_spatial_pattern:
            return self._extract_enhanced_segments_mask(lum, close_scale)
        return self._extract_legacy_segments_mask(lum, close_scale)

    def inspect_frame(self, lum: torch.Tensor) -> tuple[bool, float, int, float]:
        """Inspect `lum` `(H, W)` -> `(is_pattern, area, pairs, cpd)`."""
        if not self.enhanced_spatial_pattern:
            return super().inspect_frame(lum)

        if (
            self._prev_lum is not None
            and self._prev_result is not None
            and float((lum - self._prev_lum).abs().max().item()) < 0.5
        ):
            return self._prev_result

        res = self._inspect_frame_enhanced(lum)
        self._prev_lum = lum.clone()
        self._prev_result = res
        return res

    def _probe_is_pattern_stateless(self, lum: torch.Tensor) -> bool:
        """Evaluate `inspect_frame` without mutating `(self._prev_lum, self._prev_result)`."""
        saved_lum = self._prev_lum
        saved_res = self._prev_result
        try:
            self._prev_lum = None
            self._prev_result = None
            is_p, _, _, _ = self.inspect_frame(lum)
            return is_p
        finally:
            self._prev_lum = saved_lum
            self._prev_result = saved_res

    def inspect_frame_with_mask(
        self,
        lum: torch.Tensor,
        close_scale: float = 0.50,
    ) -> tuple[bool, float, int, float, torch.Tensor]:
        """Inspect `lum` `(H, W)` -> `(is_pattern, area, pairs, cpd, pattern_core_mask)`."""
        if (
            self._prev_lum is not None
            and self._prev_result is not None
            and self._prev_mask is not None
            and self._prev_close_scale == close_scale
            and float((lum - self._prev_lum).abs().max().item()) < 0.5
        ):
            is_p, ar, pr, cp = self._prev_result
            return is_p, ar, pr, cp, self._prev_mask

        self._prev_lum = None
        self._prev_result = None
        is_pat, effective_area, best_pairs, best_cpd = self.inspect_frame(lum)
        res = (is_pat, effective_area, best_pairs, best_cpd)

        if not is_pat:
            empty_mask = torch.zeros_like(lum, dtype=torch.bool)
            self._prev_lum = lum.clone()
            self._prev_result = res
            self._prev_mask = empty_mask
            self._prev_close_scale = close_scale
            return (*res, empty_mask)

        pat_mask = self._extract_active_segments_mask(lum, close_scale=close_scale)

        for _ in range(2):
            if not bool(pat_mask.any().item()):
                break
            max_masked = float(lum[pat_mask].max().item())
            alpha_sim = min(1.0, 18.0 / max(max_masked, 1e-6))
            w_sim = F.avg_pool2d(
                F.pad(
                    pat_mask.to(torch.float32).unsqueeze(0).unsqueeze(0),
                    (2, 2, 2, 2),
                    mode="replicate",
                ),
                kernel_size=5,
                stride=1,
            )[0, 0]
            w_sim = torch.maximum(pat_mask.to(torch.float32), w_sim)
            lum_sim = lum * (1.0 - w_sim * (1.0 - alpha_sim))
            if not self._probe_is_pattern_stateless(lum_sim):
                break
            extra_mask = self._extract_active_segments_mask(lum_sim, close_scale=close_scale)
            if not bool((extra_mask & (~pat_mask)).any().item()):
                lum_4d = lum.unsqueeze(0).unsqueeze(0)
                c_mask = self._compute_contrast_mask(lum_4d)
                pat_mask = morphological_close_2d(pat_mask, max(4, self.pool_radius)) & (
                    morphological_close_2d(c_mask, max(4, self.pool_radius)) | pat_mask
                )
                if self._probe_is_pattern_stateless(
                    lum * (1.0 - pat_mask.to(torch.float32) * (1.0 - alpha_sim))
                ):
                    _, _, active_w = self._evaluate_2d_fft_wave_mask(lum_sim)
                    if bool(active_w.any().item()):
                        pat_mask |= active_w
                    elif self.enhanced_spatial_pattern:
                        _, _, gab_w = self._evaluate_gabor_fallback(lum_sim, c_mask)
                        pat_mask |= gab_w
                break
            pat_mask |= extra_mask

        if getattr(self.mask_config, "pattern_area_from_mask", False):
            # 판정기 v2: 규격대로 '패턴이 차지한 면적' 으로 판정 (원본은 화면 전체 국소 대비 면적 fine_area 를 씀)
            mask_area = float(pat_mask.float().mean().item())
            if mask_area <= self.pse.pattern_area_min:
                res = (False, mask_area, best_pairs, best_cpd)
                pat_mask = torch.zeros_like(lum, dtype=torch.bool)
            else:
                res = (True, mask_area, best_pairs, best_cpd)

        self._prev_lum = lum.clone()
        self._prev_result = res
        self._prev_mask = pat_mask
        self._prev_close_scale = close_scale
        return (*res, pat_mask)


__all__ = [
    "PartialStaticPatternDetector",
    "find_regular_open_runs_1d",
    "morphological_close_2d",
    "extract_hysteresis_extrema_1d",
    "separable_max_min_2d",
    "filter_static_pattern_runs",
]
