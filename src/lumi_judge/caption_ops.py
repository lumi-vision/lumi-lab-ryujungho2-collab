"""Top-left Korean hazard-category stroke caption scheduling and rendering (`caption_ops.py`).

Implements the confirmed hazard caption specifications:
  - Timing (`1-b`): Active throughout the entire mask dimming span (`alpha_t < 1` and `W_t > 0`),
    displaying real-time per-frame hazard labels (`ALL_HAZARD_LABELS`) at full opacity during
    confirmed hazard frames and smoothly fading opacity along entry/exit ramps.
  - Format (`2`): `[위험 감지] <라벨1> · <라벨2>` (keeping `교집합` literally as `교집합`).
  - Style (`3-b`): Bright Korean text (`Malgun Gothic Bold`) with crisp black outline stroke and
    no background box, rendered via cached 2x-supersampled premultiplied-alpha sprites.
  - Verification separation (`4-a`): Applied strictly to the encoded `.mp4` output stream while keeping
    `Pass 2` closed-loop `vdcorefs` verification on pure dimmed frames.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .config import ALL_HAZARD_LABELS, VideoAnalysisResult, mask_to_spans

DEFAULT_CAPTION_PREFIX: str = "[위험 감지] "
DEFAULT_CAPTION_JOINER: str = " · "
DEFAULT_CAPTION_HEIGHT_RATIO: float = 0.040
DEFAULT_CAPTION_SCALE: float = 1.0

_FONT_CANDIDATES: tuple[str, ...] = (
    r"C:\Windows\Fonts\malgunbd.ttf",
    r"C:\Windows\Fonts\malgun.ttf",
    r"C:\Windows\Fonts\gulim.ttc",
)


@dataclass(slots=True)
class CachedCaptionSprite:
    """Premultiplied-BGR + alpha cropped sprite positioned at `(x0, y0)`."""

    x0: int
    y0: int
    premult_bgr_f32: np.ndarray  # (sh, sw, 3) float32 in [0, 255]
    alpha_f32: np.ndarray        # (sh, sw, 1) float32 in [0, 1]


def _resolve_korean_font_path() -> str | None:
    for cand in _FONT_CANDIDATES:
        if os.path.exists(cand):
            return cand
    win_fonts = Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts"
    for name in ("malgunbd.ttf", "malgun.ttf", " NanumGothicBold.ttf", "NanumGothic.ttf"):
        p = win_fonts / name.strip()
        if p.exists():
            return str(p)
    return None


def _load_font(font_px: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    font_path = _resolve_korean_font_path()
    if font_path is not None:
        return ImageFont.truetype(font_path, size=max(8, int(font_px)))
    return ImageFont.load_default()


def format_frame_hazard_caption(
    labels: list[str] | tuple[str, ...],
    *,
    prefix: str = DEFAULT_CAPTION_PREFIX,
    joiner: str = DEFAULT_CAPTION_JOINER,
) -> str:
    """Format active hazard labels for a single frame as `[위험 감지] 라벨1 · 라벨2`."""
    if not labels:
        return ""
    return f"{prefix}{joiner.join(labels)}"


def build_frame_caption_schedule(
    res_before: VideoAnalysisResult,
    alpha_schedule: np.ndarray,
    weight_masks_u8: np.ndarray | None,
    in_danger_mask: np.ndarray,
    *,
    prefix: str = DEFAULT_CAPTION_PREFIX,
    joiner: str = DEFAULT_CAPTION_JOINER,
    extra_safe_fallback_label: str = "연속 위험도 초과",
) -> tuple[list[str], np.ndarray]:
    """Compute per-frame `(caption_texts, caption_opacities)` synchronized with `alpha_schedule` (`1-b`).

    - On active hazard frames (`in_danger_mask[t] == True` where dimming is active):
      uses exact real-time frame hazard labels from `res_before.frame_masks` with opacity `1.0`.
    - On entry/exit fade frames (`in_danger_mask[t] == False` while `alpha_t < 0.9999` and `W_t > 0`):
      holds the adjacent boundary frame's label string and scales opacity smoothly in proportion to
      `(1 - alpha_t) / (1 - alpha_anchor)`.
    """
    n_frames = len(alpha_schedule)
    texts: list[str] = [""] * n_frames
    opacities = np.zeros(n_frames, dtype=np.float32)
    if n_frames == 0:
        return texts, opacities

    # Precompute active real-time label string on every frame in `in_danger_mask`
    raw_texts: list[str] = [""] * n_frames
    for t in np.flatnonzero(in_danger_mask):
        idx = int(t)
        active_lbls = [
            lbl
            for lbl in ALL_HAZARD_LABELS
            if (lbl in res_before.frame_masks and bool(res_before.frame_masks[lbl][idx]))
        ]
        if not active_lbls:
            active_lbls = [extra_safe_fallback_label]
        raw_texts[idx] = format_frame_hazard_caption(active_lbls, prefix=prefix, joiner=joiner)

    runs = mask_to_spans(in_danger_mask)
    if not runs:
        return texts, opacities

    run_starts = np.array([s for s, _ in runs], dtype=np.int32)
    run_ends = np.array([e for _, e in runs], dtype=np.int32)

    for t in range(n_frames):
        a_t = float(alpha_schedule[t])
        if a_t >= 0.9999:
            continue
        if weight_masks_u8 is not None and weight_masks_u8.size > 0:
            if int(weight_masks_u8[t].max()) == 0:
                continue

        if bool(in_danger_mask[t]):
            texts[t] = raw_texts[t]
            opacities[t] = 1.0
            continue

        # Boundary Fade-in / Fade-out transition frame outside `in_danger_mask`
        pos = int(np.searchsorted(run_starts, t))
        prev_e = int(run_ends[pos - 1]) if pos > 0 else -1
        next_s = int(run_starts[pos]) if pos < len(runs) else -1

        op_prev = 0.0
        if prev_e >= 0:
            denom_p = max(1.0 - float(alpha_schedule[prev_e]), 1e-6)
            op_prev = float(np.clip((1.0 - a_t) / denom_p, 0.0, 1.0))

        op_next = 0.0
        if next_s >= 0:
            denom_n = max(1.0 - float(alpha_schedule[next_s]), 1e-6)
            op_next = float(np.clip((1.0 - a_t) / denom_n, 0.0, 1.0))

        # Prefer closer run if one side is far away in time, otherwise track dominant fade envelope
        if prev_e >= 0 and next_s >= 0:
            dist_p = t - prev_e
            dist_n = next_s - t
            # Only consider a run boundary within 1.5s (~45-90f) fade/hold reach
            max_reach = max(10, int(round(1.5 * max(res_before.fps, 1.0))))
            if dist_p > max_reach:
                op_prev = 0.0
            if dist_n > max_reach:
                op_next = 0.0

        if op_prev <= 0.0 and op_next <= 0.0:
            # Fallback to nearest run anchor if dimming was extended by reinforcement pad
            if prev_e >= 0 and (next_s < 0 or (t - prev_e) <= (next_s - t)):
                anchor = prev_e
            else:
                anchor = next_s
            if anchor >= 0:
                denom_a = max(1.0 - float(alpha_schedule[anchor]), 1e-6)
                texts[t] = raw_texts[anchor]
                opacities[t] = float(np.clip((1.0 - a_t) / denom_a, 0.05, 1.0))
        elif op_prev >= op_next and prev_e >= 0:
            texts[t] = raw_texts[prev_e]
            opacities[t] = op_prev
        elif next_s >= 0:
            texts[t] = raw_texts[next_s]
            opacities[t] = op_next

    return texts, opacities


def _wrap_caption_lines(
    text: str,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    stroke_px: int,
    max_width_px: int,
    joiner: str = DEFAULT_CAPTION_JOINER,
) -> list[str]:
    """Wrap multi-label caption onto multiple lines at ` · ` only when exceeding `max_width_px`."""
    dummy = ImageDraw.Draw(Image.new("RGBA", (8, 8), (0, 0, 0, 0)))

    def _measure_w(s: str) -> int:
        bbox = dummy.textbbox((0, 0), s, font=font, stroke_width=stroke_px)
        return int(bbox[2] - bbox[0])

    if _measure_w(text) <= max_width_px or joiner not in text:
        return [text]

    parts = text.split(joiner)
    lines: list[str] = [parts[0]]
    cont_prefix = "  · "
    for tok in parts[1:]:
        cand = f"{lines[-1]}{joiner}{tok}"
        if _measure_w(cand) <= max_width_px:
            lines[-1] = cand
        else:
            lines.append(f"{cont_prefix}{tok}")
    return lines


def build_caption_sprite(
    text: str,
    frame_width: int,
    frame_height: int,
    *,
    height_ratio: float = DEFAULT_CAPTION_HEIGHT_RATIO,
    scale: float = DEFAULT_CAPTION_SCALE,
    text_rgb: tuple[int, int, int] = (248, 248, 248),
    stroke_rgb: tuple[int, int, int] = (0, 0, 0),
) -> CachedCaptionSprite | None:
    """Rasterize a 2x-supersampled Korean stroke caption sprite (`3-b`) for top-left placement."""
    if not text or frame_width <= 0 or frame_height <= 0:
        return None

    ss = 2  # 2x2 supersampling for crisp anti-aliased stroke + Hangul glyphs at all resolutions
    margin_x = max(8, int(round(frame_width * 0.020)))
    margin_y = max(8, int(round(frame_height * 0.024)))
    max_w_1x = max(64, frame_width - 2 * margin_x)

    min_font_1x = 13 if frame_height < 300 else 14
    font_1x = max(min_font_1x, int(round(frame_height * float(height_ratio) * float(scale))))

    # Ensure even wrapped lines fit horizontally inside `max_w_1x`
    while font_1x > 10:
        font_ss = _load_font(font_1x * ss)
        stroke_ss = max(3, int(round(font_1x * ss * 0.125)))
        lines = _wrap_caption_lines(text, font_ss, stroke_ss, max_w_1x * ss)
        dummy_draw = ImageDraw.Draw(Image.new("RGBA", (8, 8), (0, 0, 0, 0)))
        max_line_w_ss = max(
            int(dummy_draw.textbbox((0, 0), ln, font=font_ss, stroke_width=stroke_ss)[2])
            for ln in lines
        )
        if max_line_w_ss <= max_w_1x * ss or font_1x <= 11:
            break
        font_1x -= 1

    font_ss = _load_font(font_1x * ss)
    stroke_ss = max(3, int(round(font_1x * ss * 0.125)))
    lines = _wrap_caption_lines(text, font_ss, stroke_ss, max_w_1x * ss)

    dummy_draw = ImageDraw.Draw(Image.new("RGBA", (8, 8), (0, 0, 0, 0)))
    line_gap_ss = max(4, int(round(font_1x * ss * 0.22)))
    bboxes = [dummy_draw.textbbox((0, 0), ln, font=font_ss, stroke_width=stroke_ss) for ln in lines]

    pad_ss = stroke_ss + 4
    canvas_w_ss = max(int(b[2] - b[0]) for b in bboxes) + 2 * pad_ss
    line_heights_ss = [int(b[3] - b[1]) for b in bboxes]
    canvas_h_ss = sum(line_heights_ss) + line_gap_ss * max(0, len(lines) - 1) + 2 * pad_ss

    # Make canvas dimensions even for exact 2x2 box downsampling
    if canvas_w_ss % 2 == 1:
        canvas_w_ss += 1
    if canvas_h_ss % 2 == 1:
        canvas_h_ss += 1

    canvas = Image.new("RGBA", (canvas_w_ss, canvas_h_ss), (0, 0, 0, 0))
    draw = ImageDraw.Draw(canvas)

    cur_y = pad_ss
    for ln, b, lh in zip(lines, bboxes, line_heights_ss):
        draw_x = pad_ss - int(b[0])
        draw_y = cur_y - int(b[1])
        draw.text(
            (draw_x, draw_y),
            ln,
            font=font_ss,
            fill=(int(text_rgb[0]), int(text_rgb[1]), int(text_rgb[2]), 255),
            stroke_width=stroke_ss,
            stroke_fill=(int(stroke_rgb[0]), int(stroke_rgb[1]), int(stroke_rgb[2]), 255),
        )
        cur_y += lh + line_gap_ss

    rgba_ss = np.asarray(canvas, dtype=np.float32)  # (H_ss, W_ss, 4)
    alpha_ss = rgba_ss[..., 3:4] * (1.0 / 255.0)
    # Convert RGB -> BGR and premultiply BEFORE 2x2 area integration to prevent color fringing
    bgr_premult_ss = rgba_ss[..., [2, 1, 0]] * alpha_ss

    h_1x = canvas_h_ss // 2
    w_1x = canvas_w_ss // 2
    premult_bgr_1x = bgr_premult_ss.reshape(h_1x, 2, w_1x, 2, 3).mean(axis=(1, 3))
    alpha_1x = alpha_ss.reshape(h_1x, 2, w_1x, 2, 1).mean(axis=(1, 3))

    # Tightly crop to non-zero alpha bounding box and clamp within frame bounds
    nz_y, nz_x = np.nonzero(alpha_1x[..., 0] > (0.5 / 255.0))
    if len(nz_y) == 0:
        return None
    y_min, y_max = int(nz_y.min()), int(nz_y.max()) + 1
    x_min, x_max = int(nz_x.min()), int(nz_x.max()) + 1

    premult_crop = np.ascontiguousarray(premult_bgr_1x[y_min:y_max, x_min:x_max], dtype=np.float32)
    alpha_crop = np.ascontiguousarray(alpha_1x[y_min:y_max, x_min:x_max], dtype=np.float32)

    sh = min(premult_crop.shape[0], max(0, frame_height - margin_y))
    sw = min(premult_crop.shape[1], max(0, frame_width - margin_x))
    if sh <= 0 or sw <= 0:
        return None

    return CachedCaptionSprite(
        x0=margin_x,
        y0=margin_y,
        premult_bgr_f32=premult_crop[:sh, :sw],
        alpha_f32=alpha_crop[:sh, :sw],
    )


def overlay_hazard_caption_bgr(
    bgr_u8: np.ndarray,
    text: str,
    opacity: float,
    sprite_cache: dict[tuple[str, int, int, float], CachedCaptionSprite | None],
    *,
    height_ratio: float = DEFAULT_CAPTION_HEIGHT_RATIO,
    scale: float = DEFAULT_CAPTION_SCALE,
) -> np.ndarray:
    """Composite top-left Korean stroke caption onto a copy of `bgr_u8` (`4-a` non-mutating)."""
    op = float(np.clip(opacity, 0.0, 1.0))
    if not text or op <= 1e-3:
        return bgr_u8

    h, w = bgr_u8.shape[:2]
    key = (text, w, h, round(float(height_ratio) * float(scale), 5))
    if key not in sprite_cache:
        sprite_cache[key] = build_caption_sprite(
            text,
            frame_width=w,
            frame_height=h,
            height_ratio=height_ratio,
            scale=scale,
        )
    sprite = sprite_cache[key]
    if sprite is None:
        return bgr_u8

    out = bgr_u8.copy()
    sh, sw = sprite.alpha_f32.shape[:2]
    roi = out[sprite.y0 : sprite.y0 + sh, sprite.x0 : sprite.x0 + sw].astype(np.float32)
    eff_alpha = sprite.alpha_f32 * op
    blended = roi * (1.0 - eff_alpha) + (sprite.premult_bgr_f32 * op)
    out[sprite.y0 : sprite.y0 + sh, sprite.x0 : sprite.x0 + sw] = np.clip(
        np.rint(blended), 0.0, 255.0
    ).astype(np.uint8)
    return out


__all__ = [
    "DEFAULT_CAPTION_PREFIX",
    "DEFAULT_CAPTION_JOINER",
    "DEFAULT_CAPTION_HEIGHT_RATIO",
    "DEFAULT_CAPTION_SCALE",
    "CachedCaptionSprite",
    "format_frame_hazard_caption",
    "build_frame_caption_schedule",
    "build_caption_sprite",
    "overlay_hazard_caption_bgr",
]
