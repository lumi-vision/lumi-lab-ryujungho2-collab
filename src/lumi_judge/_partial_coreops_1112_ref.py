"""LUMI B2 Partial-Screen Risk-Proportional + Separated Red-Channel Dimming Core Engine (`partial_coreops_1112.py`).

Extends `B2/fullscreen/fullscreen_coreops_1112.py` to partial-screen spatiotemporal mitigation on top of
standalone `vdcoreps` (Structure B):
  - 2-Mask channel-separated linear-light transformation (`1-b`):
      `M_GB(t, y, x) = 1.0 - W_t(y, x) * (1.0 - alpha_t)`
      `M_R(t, y, x)  = M_GB(t, y, x) * (1.0 - W_{R,t}(y, x) * (1.0 - beta_{R,t}))`
      `(G'_lin, B'_lin) = M_GB * (G_lin, B_lin)`,  `R'_lin = M_R * R_lin`
  - Brightness attenuation strength (`Method B`, `risk_calibrated` proportional):
      `s_t = clip((R_t - 50.0) / 50.0, 0.0, 1.0)`
      `alpha_risk[t] = alpha_at_50 - s_t * (alpha_at_50 - alpha_at_100)` (`0.50 -> 0.09`)
  - Masked red-channel attenuation (`Choice 2` + `1-ii` red-mask interior statistics):
      Computes `gamma_abs` and `gamma_ratio` strictly over saturated-red pixels inside the red
      hazard core mask (`red_mask_sat_red_excess`, `red_mask_red_escape_beta`) so background
      non-red pixels never dilute localized red-flash suppression, applying `beta_{R,t}` solely
      within the dedicated red soft mask `W_{R,t}` to avoid cyan color shift on non-red hazards.
  - Pure B2 proportional default (`2-a`, `enforce_zero_residual=False` by default, with optional
    closed-loop reinforcement ready when `enforce_zero_residual=True`).
  - Top-left Korean outline-stroke hazard caption (`[위험 감지] ...`) rendered strictly onto the
    saved `.mp4` stream while verifying pure dimmed frames in `vdcoreps` (`4-a`).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from pathlib import Path
import sys
from typing import Iterable, Literal

import cv2
import numpy as np
import torch

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from vdcoreps import (
    CachedCaptionSprite,
    DEFAULT_CAPTION_HEIGHT_RATIO,
    DEFAULT_CAPTION_SCALE,
    DetectorCache,
    LABEL_RED_EXTENDED,
    LABEL_RED_FLASH,
    PSEConfig,
    PartialDetectorCache,
    PartialMaskConfig,
    PartialVideoAnalysisResult,
    VideoAnalysisResult,
    ViewingConfig,
    analyze_frame_stream as analyze_partial_frame_stream,
    analyze_video,
    apply_partial_b2_dimming_bgr,
    build_frame_caption_schedule,
    build_mode_weight_masks,
    build_red_mode_weight_masks,
    compute_4metrics_union_from_result,
    mask_to_spans,
    overlay_hazard_caption_bgr,
)

B2Mode = Literal["base", "hold", "safe"]
RedMitigationMode = Literal["hybrid", "abs"]
VALID_B2_MODES: tuple[B2Mode, ...] = ("base", "hold", "safe")
VALID_RED_MODES: tuple[RedMitigationMode, ...] = ("hybrid", "abs")

# Default quantitative mitigation & spatiotemporal mask constants for B2 partial
RISK_CALIBRATED_THRESHOLD: float = 50.0
RISK_CALIBRATED_MAX: float = 100.0
ALPHA_AT_RISK_50: float = 0.50         # Linear multiplier at R_t = 50.0 (sRGB ~75.8%)
ALPHA_AT_RISK_100: float = 0.09        # Linear multiplier at R_t = 100.0 (200 -> 18 cd/m^2, sRGB ~36.7%)
TARGET_SAFE_LUM_CEIL_CD_M2: float = 18.0  # Physical luminance safety ceiling (< 20 cd/m^2) for L_max scaling
TARGET_SAFE_RED_EXCESS: float = 0.058  # < 0.0625 (#470000); quantizes to R <= 70 in 8-bit sRGB
TARGET_SAFE_RED_RATIO: float = 0.72    # < 0.80 saturation ratio threshold
MIN_ALPHA_FLOOR: float = 0.005
FADE_DURATION_SEC: float = 0.20
HALF_HOLD_DURATION_SEC: float = 0.50
TEMPORAL_WINDOW_SEC: float = 0.16
FEATHER_ANGLE_DEG: float = 0.40
SAFE_METRIC_THRESHOLD: float = 50.0    # for [0, 100] metrics (calibrated, ppr_pct, vfm_weibull)
SAFE_NOMURA_THRESHOLD: float = 0.50    # for raw risk_nomura e(t) in [0, 1]


def resolve_effective_alpha_at_100(
    alpha_at_100: float = ALPHA_AT_RISK_100,
    l_max_cd_m2: float = 200.0,
    *,
    scale_by_l_max: bool = True,
    safe_lum_ceil_cd_m2: float = TARGET_SAFE_LUM_CEIL_CD_M2,
) -> float:
    """Resolve `alpha_at_risk_100` scaled dynamically by display peak luminance `L_max` (`1-1a`)."""
    base_a100 = float(alpha_at_100)
    if not scale_by_l_max:
        return base_a100
    return min(base_a100, float(safe_lum_ceil_cd_m2) / max(float(l_max_cd_m2), 1e-6))


@dataclass(frozen=True, slots=True)
class B2PartialConfig:
    """Configuration parameters for B2 partial-screen channel-separated dimming."""

    mode: B2Mode = "base"
    red_mode: RedMitigationMode = "hybrid"
    alpha_at_risk_50: float = ALPHA_AT_RISK_50
    alpha_at_risk_100: float = ALPHA_AT_RISK_100
    scale_alpha_100_by_l_max: bool = True    # `(1-1a)` dynamic L_max ceiling linkage
    risk_threshold: float = RISK_CALIBRATED_THRESHOLD
    risk_max: float = RISK_CALIBRATED_MAX
    target_safe_red_excess: float = TARGET_SAFE_RED_EXCESS
    target_safe_red_ratio: float = TARGET_SAFE_RED_RATIO
    min_alpha_floor: float = MIN_ALPHA_FLOOR
    fade_duration_sec: float = FADE_DURATION_SEC
    half_hold_duration_sec: float = HALF_HOLD_DURATION_SEC
    temporal_window_sec: float = TEMPORAL_WINDOW_SEC
    feather_angle_deg: float = FEATHER_ANGLE_DEG
    borderless_guided: bool = True
    enhanced_spatial_pattern: bool = True    # `(4-a)` opt-in `(3-1..3-4a)` spatial detector in B2/partial
    enforce_zero_residual: bool = False      # `(2-a)` pure B2 proportional default
    show_hazard_caption: bool = True
    caption_height_ratio: float = DEFAULT_CAPTION_HEIGHT_RATIO
    caption_scale: float = DEFAULT_CAPTION_SCALE
    safe_metric_threshold: float = SAFE_METRIC_THRESHOLD
    safe_nomura_threshold: float = SAFE_NOMURA_THRESHOLD
    fourcc: str = "mp4v"

    def to_mask_config(self) -> PartialMaskConfig:
        """Build matching `PartialMaskConfig` for `vdcoreps`."""
        return PartialMaskConfig(
            temporal_window_sec=self.temporal_window_sec,
            temporal_fade_sec=self.fade_duration_sec,
            half_hold_duration_sec=self.half_hold_duration_sec,
            feather_angle_deg=self.feather_angle_deg,
            borderless_guided=self.borderless_guided,
            enhanced_spatial_pattern=self.enhanced_spatial_pattern,
            safe_metric_threshold=self.safe_metric_threshold,
            safe_nomura_threshold=self.safe_nomura_threshold,
        )


@dataclass(slots=True)
class B2PartialDimmingResult:
    """Complete before/after verification result from `dim_video_partial_b2`."""

    mode: B2Mode
    red_mode: RedMitigationMode
    config: B2PartialConfig
    input_video_path: str
    output_video_path: str
    orig_width: int
    orig_height: int
    analysis_width: int
    analysis_height: int
    fps: float
    total_frames: int
    before: PartialVideoAnalysisResult
    after: VideoAnalysisResult
    alpha_schedule: np.ndarray
    raw_alpha_target: np.ndarray
    beta_red_schedule: np.ndarray
    alpha_red_schedule: np.ndarray
    active_dimmed_area: np.ndarray
    mean_gb_linear_multiplier: np.ndarray
    mean_gb_srgb_multiplier: np.ndarray
    red_active_dimmed_area: np.ndarray
    mean_r_linear_multiplier: np.ndarray
    mean_r_srgb_multiplier: np.ndarray
    in_danger_mask: np.ndarray
    in_red_mask: np.ndarray
    extra_safe_runs: list[tuple[int, int]]
    over_4metrics_before: np.ndarray
    over_4metrics_after: np.ndarray
    l_max_cd_m2: float = 200.0
    effective_alpha_at_100: float = ALPHA_AT_RISK_100


def compute_4metrics_union_mask(
    res: VideoAnalysisResult,
    metric_threshold: float = SAFE_METRIC_THRESHOLD,
    nomura_threshold: float = SAFE_NOMURA_THRESHOLD,
) -> np.ndarray:
    """Return boolean frame mask `(calibrated >= 50) | (ppr >= 50) | (nomura >= 0.5) | (vfm >= 50)`."""
    df = res.frame_metrics_df
    if len(df) == 0:
        return np.zeros(0, dtype=bool)
    return (
        (df["risk_calibrated"].to_numpy(dtype=np.float64) >= metric_threshold)
        | (df["risk_ppr_pct"].to_numpy(dtype=np.float64) >= metric_threshold)
        | (df["risk_nomura"].to_numpy(dtype=np.float64) >= nomura_threshold)
        | (df["risk_vfm_weibull"].to_numpy(dtype=np.float64) >= metric_threshold)
    )


def compute_risk_proportional_alpha(
    risk_calibrated: np.ndarray,
    *,
    alpha_at_50: float = ALPHA_AT_RISK_50,
    alpha_at_100: float = ALPHA_AT_RISK_100,
    risk_threshold: float = RISK_CALIBRATED_THRESHOLD,
    risk_max: float = RISK_CALIBRATED_MAX,
) -> np.ndarray:
    """Compute Method B (`안 B-1`) linear brightness multiplier from `risk_calibrated`."""
    span = max(float(risk_max) - float(risk_threshold), 1e-6)
    r_vals = np.asarray(risk_calibrated, dtype=np.float64)
    s_norm = np.clip((r_vals - float(risk_threshold)) / span, 0.0, 1.0)
    return float(alpha_at_50) - s_norm * (float(alpha_at_50) - float(alpha_at_100))


def _stabilize_in_run_envelope(
    raw_targets: np.ndarray,
    runs: list[tuple[int, int]],
    fps: float,
    temporal_window_sec: float = 0.25,
) -> np.ndarray:
    """Sliding minimum envelope (`+-temporal_window_sec`) + odd-kernel moving average (`0.30s`) + re-clamp."""
    n_frames = len(raw_targets)
    stabilized = np.ones(n_frames, dtype=np.float64)
    win_radius = max(2, int(round(temporal_window_sec * fps)))
    k_base = max(3, int(round(0.30 * fps)) | 1)

    for s_run, e_run in runs:
        run_targets = raw_targets[s_run : e_run + 1]
        run_len = len(run_targets)
        env = np.empty(run_len, dtype=np.float64)
        for idx_local in range(run_len):
            w_s = max(0, idx_local - win_radius)
            w_e = min(run_len, idx_local + win_radius + 1)
            env[idx_local] = float(np.min(run_targets[w_s:w_e]))

        if run_len >= 3:
            k_smooth = min(run_len, k_base)
            if k_smooth % 2 == 0:
                k_smooth = max(3, k_smooth - 1) if k_smooth > 3 else 1
            if k_smooth >= 3:
                pad_s = k_smooth // 2
                padded = np.pad(env, (pad_s, pad_s), mode="edge")
                kernel = np.ones(k_smooth, dtype=np.float64) / float(k_smooth)
                smoothed_env = np.convolve(padded, kernel, mode="valid")[:run_len]
                final_run_val = np.minimum(smoothed_env, run_targets)
            else:
                final_run_val = env
        else:
            final_run_val = env

        stabilized[s_run : e_run + 1] = final_run_val

    return stabilized


def _apply_symmetric_cosine_fade(
    stabilized_schedule: np.ndarray,
    runs: list[tuple[int, int]],
    fps: float,
    fade_duration_sec: float,
) -> np.ndarray:
    """Base mode boundary transition: symmetric `0.20s` cosine Fade-in and Fade-out to `1.0`."""
    n_frames = len(stabilized_schedule)
    fade_frames = max(3, int(round(fade_duration_sec * fps))) if fade_duration_sec > 0.0 else 0
    out = np.copy(stabilized_schedule)
    if fade_frames <= 0:
        return out

    steps = np.arange(1, fade_frames + 1, dtype=np.float64)
    cosine_w = 0.5 * (1.0 - np.cos(math.pi * steps / float(fade_frames + 1)))

    for s_run, e_run in runs:
        start_val = float(stabilized_schedule[s_run])
        end_val = float(stabilized_schedule[e_run])

        for step_idx, w in enumerate(cosine_w, start=1):
            f_in = s_run - step_idx
            if f_in >= 0:
                ramp_in = start_val + (1.0 - start_val) * float(w)
                out[f_in] = min(out[f_in], ramp_in)

            f_out = e_run + step_idx
            if f_out < n_frames:
                ramp_out = end_val + (1.0 - end_val) * float(w)
                out[f_out] = min(out[f_out], ramp_out)

    return out


def _apply_half_hold_cosine_fade(
    stabilized_schedule: np.ndarray,
    runs: list[tuple[int, int]],
    fps: float,
    fade_duration_sec: float,
    half_hold_duration_sec: float,
) -> np.ndarray:
    """Hold/Safe mode boundary transition: 2-stage Half-Hold release + smooth anchor Fade-in."""
    n_frames = len(stabilized_schedule)
    fade_frames = max(3, int(round(fade_duration_sec * fps))) if fade_duration_sec > 0.0 else 0
    hold_frames = max(1, int(round(half_hold_duration_sec * fps)))
    out = np.copy(stabilized_schedule)
    if fade_frames <= 0:
        return out

    steps = np.arange(1, fade_frames + 1, dtype=np.float64)
    cosine_w = 0.5 * (1.0 - np.cos(math.pi * steps / float(fade_frames + 1)))

    for _, e_run in runs:
        end_val = float(stabilized_schedule[e_run])
        val_half = 0.5 * (end_val + 1.0)

        for step_idx, w in enumerate(cosine_w, start=1):
            f_idx = e_run + step_idx
            if f_idx >= n_frames:
                break
            ramp1 = end_val + (val_half - end_val) * float(w)
            out[f_idx] = min(out[f_idx], ramp1)

        hold_start = e_run + fade_frames + 1
        hold_end = e_run + fade_frames + hold_frames
        if hold_start < n_frames:
            sl_end = min(n_frames, hold_end + 1)
            out[hold_start:sl_end] = np.minimum(out[hold_start:sl_end], val_half)

        for step_idx, w in enumerate(cosine_w, start=1):
            f_idx = hold_end + step_idx
            if f_idx >= n_frames:
                break
            ramp2 = val_half + (1.0 - val_half) * float(w)
            out[f_idx] = min(out[f_idx], ramp2)

    for s_run, _ in runs:
        start_val = float(stabilized_schedule[s_run])
        anchor_idx = max(0, s_run - fade_frames - 1)
        entry_from_val = max(start_val, float(out[anchor_idx]))

        for step_idx, w in enumerate(cosine_w, start=1):
            f_idx = s_run - step_idx
            if f_idx < 0:
                break
            ramp_in = start_val + (entry_from_val - start_val) * float(w)
            out[f_idx] = min(out[f_idx], ramp_in)

    return out


def _smooth_runs_by_mode(
    raw_targets: np.ndarray,
    runs: list[tuple[int, int]],
    fps: float,
    cfg: B2PartialConfig,
) -> np.ndarray:
    """Stabilize in-interval targets and apply mode-specific (`base` vs `hold`/`safe`) boundary fade."""
    if not runs:
        return np.ones(len(raw_targets), dtype=np.float64)
    stabilized = _stabilize_in_run_envelope(
        raw_targets,
        runs,
        fps,
        temporal_window_sec=max(0.25, cfg.temporal_window_sec),
    )
    if cfg.mode == "base":
        return _apply_symmetric_cosine_fade(stabilized, runs, fps, cfg.fade_duration_sec)
    return _apply_half_hold_cosine_fade(
        stabilized, runs, fps, cfg.fade_duration_sec, cfg.half_hold_duration_sec
    )


def compute_alpha_and_red_schedule_partial_b2(
    res_before: PartialVideoAnalysisResult,
    cfg: B2PartialConfig = B2PartialConfig(),
    pse: PSEConfig = PSEConfig(),
    *,
    l_max_cd_m2: float | None = None,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[tuple[int, int]],
    np.ndarray,
]:
    """Compute per-frame `(alpha_sched, raw_alpha_target, beta_red_sched, alpha_red_sched)` for B2 partial.

    Uses `risk_calibrated`-proportional `alpha_t` (`Method B`) with `(1-1a)` dynamic `L_max` ceiling
    scaling `alpha_at_100_eff = min(alpha_at_100, 18.0 / L_max)`, and red-mask interior saturated-red
    statistics (`1-ii`: `red_mask_sat_red_excess`, `red_mask_red_escape_beta`) for `beta_R,t`.
    """
    if cfg.mode not in VALID_B2_MODES:
        raise ValueError(f"Unsupported B2 mode {cfg.mode!r}; expected one of {VALID_B2_MODES}.")
    if cfg.red_mode not in VALID_RED_MODES:
        raise ValueError(f"Unsupported red_mode {cfg.red_mode!r}; expected one of {VALID_RED_MODES}.")
    if res_before.physical_stats is None:
        raise ValueError("`res_before.physical_stats` is required (`collect_physical_stats=True`).")
    if not (0.0 < cfg.alpha_at_risk_100 <= cfg.alpha_at_risk_50 <= 1.0):
        raise ValueError(
            f"Expected `0 < alpha_at_risk_100 <= alpha_at_risk_50 <= 1`, got "
            f"({cfg.alpha_at_risk_50}, {cfg.alpha_at_risk_100})."
        )
    if cfg.risk_max <= cfg.risk_threshold:
        raise ValueError(
            f"Expected `risk_max > risk_threshold`, got ({cfg.risk_threshold}, {cfg.risk_max})."
        )

    eff_l_max = float(res_before.l_max_cd_m2 if l_max_cd_m2 is None else l_max_cd_m2)
    eff_alpha_100 = resolve_effective_alpha_at_100(
        cfg.alpha_at_risk_100,
        eff_l_max,
        scale_by_l_max=cfg.scale_alpha_100_by_l_max,
    )
    eff_alpha_50 = max(eff_alpha_100, float(cfg.alpha_at_risk_50))

    n_frames = res_before.total_frames
    if n_frames == 0:
        empty_f = np.zeros(0, dtype=np.float32)
        empty_b = np.zeros(0, dtype=bool)
        return empty_f, empty_f, empty_f, empty_f, empty_b, empty_b, [], empty_b

    df_before = res_before.frame_metrics_df
    stats = res_before.physical_stats
    risk_cal = df_before["risk_calibrated"].to_numpy(dtype=np.float64)
    sat_red_area = stats["sat_red_area"]

    in_interval_danger = df_before["is_dangerous"].to_numpy(dtype=bool)
    over_4metrics = compute_4metrics_union_from_result(
        res_before,
        metric_threshold=cfg.safe_metric_threshold,
        nomura_threshold=cfg.safe_nomura_threshold,
    )
    red_interval_mask = (
        res_before.frame_masks[LABEL_RED_FLASH] | res_before.frame_masks[LABEL_RED_EXTENDED]
    )

    if cfg.mode == "safe":
        in_danger_any = in_interval_danger | over_4metrics
        in_red_danger = red_interval_mask | (in_danger_any & (sat_red_area > pse.flash_area_min))
        extra_safe_runs = mask_to_spans(in_danger_any & (~in_interval_danger))
        red_exc_stat = stats.get("safe_red_mask_sat_red_excess", stats["q76_sat_red_excess"])
        red_esc_stat = stats.get("safe_red_mask_red_escape_beta", stats["q24_red_escape_beta"])
    else:
        in_danger_any = in_interval_danger.copy()
        in_red_danger = red_interval_mask.copy()
        extra_safe_runs = []
        red_exc_stat = stats.get("red_mask_sat_red_excess", stats["q76_sat_red_excess"])
        red_esc_stat = stats.get("red_mask_red_escape_beta", stats["q24_red_escape_beta"])

    # 1) Mask-core G,B & base luminance schedule `alpha_sched` via Method B (`risk_calibrated` proportional)
    alpha_target = np.ones(n_frames, dtype=np.float64)
    active_idx = np.flatnonzero(in_danger_any)
    if len(active_idx) > 0:
        req_alpha = compute_risk_proportional_alpha(
            risk_cal[active_idx],
            alpha_at_50=eff_alpha_50,
            alpha_at_100=eff_alpha_100,
            risk_threshold=cfg.risk_threshold,
            risk_max=cfg.risk_max,
        )
        alpha_target[active_idx] = np.clip(req_alpha, cfg.min_alpha_floor, 1.0)

    danger_runs = mask_to_spans(in_danger_any)
    alpha_sched_f64 = _smooth_runs_by_mode(alpha_target, danger_runs, res_before.fps, cfg)

    # 2) Red-mask-core additional multiplier `beta_red_schedule` (`Choice 2` + `1-ii` red-mask stats)
    red_ceiling_target = np.ones(n_frames, dtype=np.float64)
    red_idx = np.flatnonzero(in_red_danger)
    if len(red_idx) > 0:
        exc_all = red_exc_stat[red_idx].astype(np.float64)
        need_red_sub = exc_all > cfg.target_safe_red_excess
        if np.any(need_red_sub):
            sub_frames = red_idx[need_red_sub]
            cur_alpha = np.maximum(alpha_sched_f64[sub_frames], 1e-6)
            q_exc = np.maximum(exc_all[need_red_sub], 1e-6)

            gamma_abs = np.minimum(1.0, cfg.target_safe_red_excess / q_exc)

            if cfg.red_mode == "hybrid":
                esc_vals = red_esc_stat[sub_frames].astype(np.float64)
                if abs(cfg.target_safe_red_ratio - 0.72) > 1e-6:
                    r0 = float(np.clip(cfg.target_safe_red_ratio, 0.05, 0.79))
                    ratio_scale = (r0 / (1.0 - r0)) / (0.72 / 0.28)
                    esc_vals = np.where(esc_vals < 0.9999, np.clip(esc_vals * ratio_scale, 0.0, 1.0), 1.0)
                gamma_ratio = cur_alpha * esc_vals
                req_alpha_red = np.maximum(gamma_ratio, gamma_abs)
            else:
                req_alpha_red = gamma_abs

            red_ceiling_target[sub_frames] = np.clip(
                np.minimum(1.0, req_alpha_red), cfg.min_alpha_floor, 1.0
            )

    red_runs = mask_to_spans(in_red_danger)
    if not red_runs:
        beta_red_sched_f64 = np.ones(n_frames, dtype=np.float64)
    else:
        stabilized_red_ceiling = _stabilize_in_run_envelope(
            red_ceiling_target,
            red_runs,
            res_before.fps,
            temporal_window_sec=max(0.25, cfg.temporal_window_sec),
        )
        beta_red_in_run = np.ones(n_frames, dtype=np.float64)
        beta_red_in_run[red_idx] = np.clip(
            np.minimum(alpha_sched_f64[red_idx], stabilized_red_ceiling[red_idx])
            / np.maximum(alpha_sched_f64[red_idx], 1e-6),
            cfg.min_alpha_floor,
            1.0,
        )
        if cfg.mode == "base":
            beta_red_sched_f64 = _apply_symmetric_cosine_fade(
                beta_red_in_run, red_runs, res_before.fps, cfg.fade_duration_sec
            )
        else:
            beta_red_sched_f64 = _apply_half_hold_cosine_fade(
                beta_red_in_run,
                red_runs,
                res_before.fps,
                cfg.fade_duration_sec,
                cfg.half_hold_duration_sec,
            )

    alpha_sched = alpha_sched_f64.astype(np.float32)
    beta_red_sched = beta_red_sched_f64.astype(np.float32)
    alpha_red_sched = (alpha_sched_f64 * beta_red_sched_f64).astype(np.float32)

    return (
        alpha_sched,
        alpha_target.astype(np.float32),
        beta_red_sched,
        alpha_red_sched,
        in_danger_any,
        in_red_danger,
        extra_safe_runs,
        over_4metrics,
    )


def compute_b2_frame_multiplier_summaries(
    alpha_schedule: np.ndarray,
    weight_masks_u8: np.ndarray,
    beta_red_schedule: np.ndarray,
    red_weight_masks_u8: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute exact per-frame `(active_area, mean_gb_lin, mean_gb_srgb, red_area, mean_r_lin, mean_r_srgb)`."""
    n_frames = len(alpha_schedule)
    active_area = np.zeros(n_frames, dtype=np.float32)
    mean_gb_lin = np.ones(n_frames, dtype=np.float32)
    mean_gb_srgb = np.ones(n_frames, dtype=np.float32)
    red_area = np.zeros(n_frames, dtype=np.float32)
    mean_r_lin = np.ones(n_frames, dtype=np.float32)
    mean_r_srgb = np.ones(n_frames, dtype=np.float32)

    if n_frames == 0 or weight_masks_u8.size == 0:
        return active_area, mean_gb_lin, mean_gb_srgb, red_area, mean_r_lin, mean_r_srgb

    w_grid_256 = np.arange(256, dtype=np.float64) / 255.0
    n_px = float(weight_masks_u8.shape[1] * weight_masks_u8.shape[2])

    for t in range(n_frames):
        a_gb = float(np.clip(alpha_schedule[t], 0.0, 1.0))
        b_r = float(np.clip(beta_red_schedule[t], 0.0, 1.0))
        m_gb_u8 = weight_masks_u8[t]
        m_r_u8 = red_weight_masks_u8[t]

        has_gb = (a_gb < 0.9999) and (int(m_gb_u8.max()) > 0)
        has_r = (b_r < 0.9999) and (int(m_r_u8.max()) > 0)

        if has_gb:
            counts_gb = np.bincount(m_gb_u8.ravel(), minlength=256).astype(np.float64)
            mean_w_gb = float(np.dot(counts_gb, w_grid_256) / n_px)
            m_gb_vals = np.clip(1.0 - w_grid_256 * (1.0 - a_gb), 0.0, 1.0)
            srgb_gb_vals = np.power(m_gb_vals, 1.0 / 2.4)

            active_area[t] = mean_w_gb
            mean_gb_lin[t] = 1.0 - mean_w_gb * (1.0 - a_gb)
            mean_gb_srgb[t] = float(np.dot(counts_gb, srgb_gb_vals) / n_px)

        if not has_r:
            mean_r_lin[t] = mean_gb_lin[t]
            mean_r_srgb[t] = mean_gb_srgb[t]
            continue

        red_area[t] = float(m_r_u8.astype(np.float64).mean() / 255.0)
        if has_gb:
            m_gb_2d = 1.0 - (m_gb_u8.astype(np.float32) * (1.0 / 255.0)) * np.float32(1.0 - a_gb)
        else:
            m_gb_2d = np.ones_like(m_r_u8, dtype=np.float32)
        m_r_2d = np.clip(
            m_gb_2d * (1.0 - (m_r_u8.astype(np.float32) * (1.0 / 255.0)) * np.float32(1.0 - b_r)),
            0.0,
            1.0,
        )
        mean_r_lin[t] = float(m_r_2d.mean())
        mean_r_srgb[t] = float(np.power(np.maximum(m_r_2d, 1e-12), np.float32(1.0 / 2.4)).mean())

    return active_area, mean_gb_lin, mean_gb_srgb, red_area, mean_r_lin, mean_r_srgb


def dim_video_partial_b2(
    input_video_path: str | Path,
    output_video_path: str | Path,
    *,
    b2_config: B2PartialConfig | None = None,
    viewing: ViewingConfig | None = None,
    pse: PSEConfig | None = None,
    device: str | torch.device | None = None,
    partial_cache: PartialDetectorCache | None = None,
    verify_cache: DetectorCache | None = None,
    pass1_result: PartialVideoAnalysisResult | None = None,
) -> B2PartialDimmingResult:
    """Run 2-Pass B2 partial-screen channel-separated dimming at original resolution with verification."""
    in_path = Path(input_video_path)
    out_path = Path(output_video_path)
    cfg = b2_config or B2PartialConfig()
    view_cfg = viewing or ViewingConfig()
    pse_cfg = pse or PSEConfig()
    mask_cfg = cfg.to_mask_config()
    p_cache = partial_cache if partial_cache is not None else {}
    v_cache = verify_cache if verify_cache is not None else {}

    # Pass 1: Extract 7-category hazard intervals, 4 risk metrics, general & red core masks, & masked stats
    if (
        pass1_result is not None
        and pass1_result.physical_stats is not None
        and pass1_result.packed_hazards is not None
        and "red_mask_sat_red_excess" in pass1_result.physical_stats
        and pass1_result.mask_config.enhanced_spatial_pattern == mask_cfg.enhanced_spatial_pattern
        and abs(float(pass1_result.l_max_cd_m2) - float(view_cfg.l_max_cd_m2)) < 1e-4
    ):
        res_before = pass1_result
    else:
        res_before = analyze_video(
            in_path,
            viewing=view_cfg,
            pse=pse_cfg,
            mask_config=mask_cfg,
            device=device,
            collect_physical_stats=True,
            collect_masks=True,
            detector_cache=p_cache,
        )

    eff_alpha_100 = resolve_effective_alpha_at_100(
        cfg.alpha_at_risk_100,
        view_cfg.l_max_cd_m2,
        scale_by_l_max=cfg.scale_alpha_100_by_l_max,
    )
    (
        alpha_schedule,
        raw_alpha_target,
        beta_red_schedule,
        alpha_red_schedule,
        in_danger_mask,
        in_red_mask,
        extra_safe_runs,
        over4_before,
    ) = compute_alpha_and_red_schedule_partial_b2(
        res_before,
        cfg=cfg,
        pse=pse_cfg,
        l_max_cd_m2=view_cfg.l_max_cd_m2,
    )

    weight_masks_u8, _ = build_mode_weight_masks(
        res_before,
        mode=cfg.mode,
        hold_run_tails=True,
        mask_config=mask_cfg,
        device=device,
    )
    red_weight_masks_u8, _ = build_red_mode_weight_masks(
        res_before,
        mode=cfg.mode,
        hold_run_tails=True,
        pse=pse_cfg,
        mask_config=mask_cfg,
        device=device,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    orig_w = res_before.orig_width
    orig_h = res_before.orig_height
    fps = res_before.fps
    n_frames = res_before.total_frames
    timestamps_ms = res_before.frame_metrics_df["timestamp_ms"].to_numpy(dtype=np.float64)
    fourcc = cv2.VideoWriter_fourcc(*cfg.fourcc)
    sprite_cache: dict[tuple[str, int, int, float], CachedCaptionSprite | None] = {}

    def _encode_and_verify(
        sched_gb: np.ndarray,
        masks_gb_u8: np.ndarray,
        sched_beta_r: np.ndarray,
        masks_r_u8: np.ndarray,
    ) -> VideoAnalysisResult:
        if cfg.show_hazard_caption and n_frames > 0:
            eff_sched = np.minimum(sched_gb, sched_gb * sched_beta_r)
            eff_masks = np.maximum(masks_gb_u8, masks_r_u8)
            cap_texts, cap_opacities = build_frame_caption_schedule(
                res_before,
                eff_sched,
                eff_masks,
                in_danger_mask,
            )
        else:
            cap_texts = [""] * n_frames
            cap_opacities = np.zeros(n_frames, dtype=np.float32)

        cap = cv2.VideoCapture(str(in_path))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open input video for Pass 2 encoding: {in_path}")
        writer = cv2.VideoWriter(str(out_path), fourcc, fps, (orig_w, orig_h))
        if not writer.isOpened():
            cap.release()
            raise RuntimeError(f"Failed to open VideoWriter for output path: {out_path}")
        try:
            def _dimmed_stream() -> Iterable[tuple[int, float, np.ndarray]]:
                idx = 0
                dt_ms = 1000.0 / max(fps, 1e-6)
                while idx < n_frames:
                    ok, raw_bgr = cap.read()
                    if not ok or raw_bgr is None:
                        break
                    if raw_bgr.shape[1] != orig_w or raw_bgr.shape[0] != orig_h:
                        raw_bgr = cv2.resize(raw_bgr, (orig_w, orig_h), interpolation=cv2.INTER_AREA)

                    dim_bgr = apply_partial_b2_dimming_bgr(
                        raw_bgr,
                        float(sched_gb[idx]),
                        masks_gb_u8[idx],
                        float(sched_beta_r[idx]),
                        masks_r_u8[idx],
                    )
                    if cfg.show_hazard_caption and cap_texts[idx] and float(cap_opacities[idx]) > 1e-3:
                        out_bgr = overlay_hazard_caption_bgr(
                            dim_bgr,
                            cap_texts[idx],
                            float(cap_opacities[idx]),
                            sprite_cache,
                            height_ratio=cfg.caption_height_ratio,
                            scale=cfg.caption_scale,
                        )
                    else:
                        out_bgr = dim_bgr
                    writer.write(out_bgr)

                    t_ms = float(timestamps_ms[idx]) if idx < len(timestamps_ms) else (idx * dt_ms)
                    yield idx, t_ms, dim_bgr
                    idx += 1

            return analyze_partial_frame_stream(
                frame_iter=_dimmed_stream(),
                fps=fps,
                orig_width=orig_w,
                orig_height=orig_h,
                video_name=str(out_path),
                viewing=view_cfg,
                pse=pse_cfg,
                mask_config=mask_cfg,
                device=device,
                collect_physical_stats=True,
                collect_masks=False,
                detector_cache=v_cache if verify_cache is not None else p_cache,
            )
        finally:
            cap.release()
            writer.release()

    # Pass 2 (`2-a`: pure B2 proportional schedule by default; optional `(2-b)` reinforcement if enabled)
    res_after = _encode_and_verify(
        alpha_schedule,
        weight_masks_u8,
        beta_red_schedule,
        red_weight_masks_u8,
    )

    if cfg.enforce_zero_residual and n_frames > 0 and len(res_after.intervals) > 0:
        fade_f = max(3, int(round(cfg.fade_duration_sec * fps))) if cfg.fade_duration_sec > 0.0 else 3
        pad_f = max(fade_f, int(round(0.35 * fps)))
        steps = np.arange(1, fade_f + 1, dtype=np.float64)
        cos_out = 0.5 * (1.0 + np.cos(math.pi * steps / float(fade_f + 1)))

        reinforce_specs = (
            (0.32, 0.90, 0.85, 0.60),
            (0.18, 1.50, 1.05, 0.50),
            (0.12, 2.20, 1.25, 0.40),
        )
        for step_i, (r_knee, r_reach, r_post, a_scale) in enumerate(reinforce_specs, start=1):
            if len(res_after.intervals) == 0:
                break

            res_mask = np.zeros(n_frames, dtype=bool)
            has_res_red = False
            for rec in res_after.intervals:
                s_i = max(0, int(rec.start_frame) - pad_f)
                e_i = min(n_frames - 1, int(rec.end_frame) + pad_f)
                if e_i >= s_i:
                    res_mask[s_i : e_i + 1] = True
                if rec.label in (LABEL_RED_FLASH, LABEL_RED_EXTENDED):
                    has_res_red = True

            if not np.any(res_mask):
                break

            step_cfg = replace(
                mask_cfg,
                guided_knee=min(mask_cfg.guided_knee, r_knee),
                guided_reach_deg=max(mask_cfg.guided_reach_deg, r_reach),
                guided_post_sigma_deg=max(mask_cfg.guided_post_sigma_deg, r_post),
            )
            step_w_u8, _ = build_mode_weight_masks(
                res_before,
                mode=cfg.mode,
                hold_run_tails=True,
                mask_config=step_cfg,
                device=device,
            )
            step_r_u8, _ = (
                build_red_mode_weight_masks(
                    res_before,
                    mode=cfg.mode,
                    hold_run_tails=True,
                    pse=pse_cfg,
                    mask_config=step_cfg,
                    device=device,
                )
                if has_res_red
                else (red_weight_masks_u8, np.zeros(n_frames, dtype=np.float32))
            )

            res_spans = mask_to_spans(res_mask)
            alpha_next = alpha_schedule.astype(np.float64).copy()
            beta_next = beta_red_schedule.astype(np.float64).copy()
            for s_sp, e_sp in res_spans:
                weight_masks_u8[s_sp : e_sp + 1] = np.maximum(
                    weight_masks_u8[s_sp : e_sp + 1],
                    step_w_u8[s_sp : e_sp + 1],
                )
                scaled_seg = np.maximum(cfg.min_alpha_floor, alpha_next[s_sp : e_sp + 1] * a_scale)
                alpha_next[s_sp : e_sp + 1] = np.minimum(alpha_next[s_sp : e_sp + 1], scaled_seg)
                if has_res_red:
                    red_weight_masks_u8[s_sp : e_sp + 1] = np.maximum(
                        red_weight_masks_u8[s_sp : e_sp + 1],
                        step_r_u8[s_sp : e_sp + 1],
                    )
                    beta_next[s_sp : e_sp + 1] = np.maximum(
                        cfg.min_alpha_floor,
                        beta_next[s_sp : e_sp + 1] * 0.72,
                    )

                sa = float(alpha_next[s_sp])
                ea = float(alpha_next[e_sp])
                for d, cw in enumerate(cos_out, start=1):
                    if s_sp - d >= 0:
                        blend_w = np.rint(
                            (1.0 - cw) * weight_masks_u8[s_sp - d].astype(np.float32)
                            + cw * np.maximum(weight_masks_u8[s_sp - d], step_w_u8[s_sp - d]).astype(np.float32)
                        ).astype(np.uint8)
                        weight_masks_u8[s_sp - d] = np.maximum(weight_masks_u8[s_sp - d], blend_w)
                        ramp_in = sa + (float(alpha_next[s_sp - d]) - sa) * (1.0 - float(cw))
                        alpha_next[s_sp - d] = min(alpha_next[s_sp - d], ramp_in)
                    if e_sp + d < n_frames:
                        blend_w = np.rint(
                            (1.0 - cw) * weight_masks_u8[e_sp + d].astype(np.float32)
                            + cw * np.maximum(weight_masks_u8[e_sp + d], step_w_u8[e_sp + d]).astype(np.float32)
                        ).astype(np.uint8)
                        weight_masks_u8[e_sp + d] = np.maximum(weight_masks_u8[e_sp + d], blend_w)
                        ramp_out = ea + (float(alpha_next[e_sp + d]) - ea) * (1.0 - float(cw))
                        alpha_next[e_sp + d] = min(alpha_next[e_sp + d], ramp_out)

            alpha_schedule = alpha_next.astype(np.float32)
            beta_red_schedule = beta_next.astype(np.float32)
            alpha_red_schedule = (alpha_next * beta_next).astype(np.float32)
            res_after = _encode_and_verify(
                alpha_schedule,
                weight_masks_u8,
                beta_red_schedule,
                red_weight_masks_u8,
            )

    (
        active_dimmed_area,
        mean_gb_lin,
        mean_gb_srgb,
        red_active_area,
        mean_r_lin,
        mean_r_srgb,
    ) = compute_b2_frame_multiplier_summaries(
        alpha_schedule,
        weight_masks_u8,
        beta_red_schedule,
        red_weight_masks_u8,
    )

    over4_after = compute_4metrics_union_mask(
        res_after,
        metric_threshold=cfg.safe_metric_threshold,
        nomura_threshold=cfg.safe_nomura_threshold,
    )

    return B2PartialDimmingResult(
        mode=cfg.mode,
        red_mode=cfg.red_mode,
        config=cfg,
        input_video_path=str(in_path),
        output_video_path=str(out_path),
        orig_width=orig_w,
        orig_height=orig_h,
        analysis_width=res_before.analysis_width,
        analysis_height=res_before.analysis_height,
        fps=fps,
        total_frames=n_frames,
        before=res_before,
        after=res_after,
        alpha_schedule=alpha_schedule,
        raw_alpha_target=raw_alpha_target,
        beta_red_schedule=beta_red_schedule,
        alpha_red_schedule=alpha_red_schedule,
        active_dimmed_area=active_dimmed_area,
        mean_gb_linear_multiplier=mean_gb_lin,
        mean_gb_srgb_multiplier=mean_gb_srgb,
        red_active_dimmed_area=red_active_area,
        mean_r_linear_multiplier=mean_r_lin,
        mean_r_srgb_multiplier=mean_r_srgb,
        in_danger_mask=in_danger_mask,
        in_red_mask=in_red_mask,
        extra_safe_runs=extra_safe_runs,
        over_4metrics_before=over4_before,
        over_4metrics_after=over4_after,
        l_max_cd_m2=float(view_cfg.l_max_cd_m2),
        effective_alpha_at_100=float(eff_alpha_100),
    )
