"""Controlled spatiotemporal synthesis for research; labels require independent analysis."""
from __future__ import annotations

import numpy as np


def render(clean: np.ndarray, fps: float, *, kind: str, frequency: float,
           center=(0.5, 0.5), radius=(0.25, 0.2), velocity=(0.0, 0.0),
           strength=0.5, cycles=8.0, pattern_speed=0.0, angle=0.0):
    """Return (synthesized BGR, time-varying support).

    Positions/radii are fractions of frame dimensions, velocity is fractions/s;
    cycles is cycles/frame width and pattern_speed is cycles/s. Flash frequency
    is cycles/s. Values are experimental settings, not risk classifications.
    """
    if clean.ndim != 4 or clean.shape[-1] != 3 or clean.dtype != np.uint8 or not len(clean):
        raise ValueError("clean must be nonempty T×H×W×3 uint8 BGR")
    if kind not in ("flash", "pattern"):
        raise ValueError("kind must be flash or pattern")
    vals = np.asarray([fps, frequency, *center, *radius, *velocity, strength, cycles, pattern_speed, angle], float)
    if not np.isfinite(vals).all() or fps <= 0 or not 0 <= frequency < fps / 2:
        raise ValueError("fps and frequency must be finite, with 0 <= frequency < fps/2")
    if min(radius) <= 0 or not 0 <= strength <= 1 or cycles <= 0:
        raise ValueError("radii and cycles must be positive; strength must be in [0,1]")
    t, h, w, _ = clean.shape
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    xx = (xx + 0.5) / w
    yy = (yy + 0.5) / h
    result = clean.copy()
    support = np.zeros((t, h, w), dtype=bool)
    for i in range(t):
        sec = i / fps
        dx = (xx - center[0] - velocity[0] * sec) / radius[0]
        dy = (yy - center[1] - velocity[1] * sec) / radius[1]
        mask = dx * dx + dy * dy <= 1
        support[i] = mask
        phase = 2 * np.pi * frequency * sec
        if kind == "flash":
            value = np.full((h, w), (1 + np.cos(phase)) / 2, dtype=np.float32)
        else:
            position = (xx * np.cos(angle) + yy * np.sin(angle)) * cycles
            value = (1 + np.cos(2 * np.pi * (position - pattern_speed * sec) + phase)) / 2
        alpha = (strength * mask)[..., None]
        result[i] = np.rint(clean[i].astype(np.float32) * (1 - alpha) + 255 * value[..., None] * alpha).clip(0, 255).astype(np.uint8)
    return result, support
