"""광과민성 연구용 영상 변형 데이터셋 생성기.

주의: 생성 데이터에는 광과민성 유발 가능성이 있는 신호가 포함될 수 있습니다.
동영상을 재생하거나 미리보기를 생성하지 않으며, 의학적 안전성 판정도 하지 않습니다.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

STIMULUS_TYPES = ("normal", "luminance", "red", "pattern")


@dataclass(frozen=True)
class DatasetConfig:
    frames: int = 16
    width: int = 128
    height: int = 96
    fps: int = 24
    seed: int = 42
    max_source_seconds: int = 60

    def validate(self) -> None:
        if self.frames < 4 or self.width < 32 or self.height < 32 or self.fps < 8 or self.max_source_seconds < 1:
            raise ValueError("프레임 수는 4 이상, 해상도는 32 이상, FPS는 8 이상이어야 합니다.")


def split_for_source(source_id: str, *, only_one: bool = False) -> str:
    """원본 단위로 분리해 파생 클립의 학습·평가 누출을 방지합니다."""
    if only_one:
        return "train"
    key = int(hashlib.sha256(source_id.encode("utf-8")).hexdigest()[:8], 16) % 100
    if key < 80:
        return "train"
    if key < 90:
        return "val"
    return "test"


def _read_frames(path: Path, cfg: DatasetConfig) -> np.ndarray:
    """입력 영상을 원본 FPS에 비례하여 추출하고 RGB 배열로 반환합니다."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"영상 파일을 열 수 없습니다: {path}")
    frames = []
    source_fps = cap.get(cv2.CAP_PROP_FPS)
    if not (np.isfinite(source_fps) and source_fps > 0):
        source_fps = cfg.fps
    # 원본의 앞부분 최대 max_source_seconds초만 읽어 메모리 사용량을 제한합니다.
    # VFR 영상에서는 OpenCV FPS가 부정확할 수 있으므로 별도 확인이 필요합니다.
    next_output_time = 0.0
    idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            source_time = idx / source_fps
            if source_time >= cfg.max_source_seconds:
                break
            if source_time + 1e-7 >= next_output_time:
                resized = cv2.resize(frame, (cfg.width, cfg.height), interpolation=cv2.INTER_AREA)
                frames.append(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
                next_output_time += 1.0 / cfg.fps
            idx += 1
    finally:
        cap.release()
    if not frames:
        raise ValueError(f"영상에서 프레임을 읽지 못했습니다: {path}")
    return np.stack(frames)


def _spatial_mask(cfg: DatasetConfig, rng: np.random.Generator) -> np.ndarray:
    """임의의 국소 직사각형 또는 타원 마스크를 만듭니다."""
    h, w = cfg.height, cfg.width
    cx, cy = int(rng.integers(w // 4, 3 * w // 4)), int(rng.integers(h // 4, 3 * h // 4))
    rx = int(rng.integers(max(5, w // 8), max(6, w // 3)))
    ry = int(rng.integers(max(5, h // 8), max(6, h // 3)))
    mask = np.zeros((h, w), dtype=np.uint8)
    shape = str(rng.choice(("ellipse", "rectangle", "multiple", "full"), p=(0.35, 0.35, 0.2, 0.1)))
    if shape == "ellipse":
        cv2.ellipse(mask, (cx, cy), (rx, ry), 0, 0, 360, 1, -1)
    elif shape == "rectangle":
        cv2.rectangle(mask, (max(0, cx - rx), max(0, cy - ry)),
                      (min(w - 1, cx + rx), min(h - 1, cy + ry)), 1, -1)
    elif shape == "multiple":
        cv2.ellipse(mask, (cx, cy), (rx // 2, ry // 2), 0, 0, 360, 1, -1)
        cv2.ellipse(mask, (min(w - 1, cx + rx), max(0, cy - ry)),
                    (max(3, rx // 3), max(3, ry // 3)), 0, 0, 360, 1, -1)
    else:
        mask[:] = 1
    return mask


def _activation(cfg: DatasetConfig, rng: np.random.Generator) -> tuple[np.ndarray, dict]:
    """제어 가능한 시간 파형을 생성합니다. 주파수만으로 위험을 판정하지 않습니다."""
    freq = float(rng.uniform(0.8, min(cfg.fps / 2 * 0.85, 10.0)))
    phase = float(rng.uniform(0, 2 * np.pi))
    waveform = str(rng.choice(("sine", "square", "triangle")))
    times = np.arange(cfg.frames, dtype=np.float32) / cfg.fps
    position = freq * times + phase / (2 * np.pi)
    if waveform == "square":
        activation = (np.sin(2 * np.pi * position) >= 0).astype(np.float32)
    elif waveform == "triangle":
        activation = 1 - np.abs(2 * (position - np.floor(position)) - 1)
    else:
        activation = (np.sin(2 * np.pi * position) + 1) / 2
    return activation.astype(np.float32), {
        "frequency_hz": freq, "phase_rad": phase, "waveform": waveform,
    }


def synthesize(clean: np.ndarray, cfg: DatasetConfig, kind: str,
               rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """원본, 변형본, 변형 허용 영역 마스크의 대응 관계를 유지합니다.

    출력: 변형 RGB uint8 [T,H,W,3], 변형 허용 ROI uint8 [T,H,W],
    시간 활성화 float32 [T], 합성 메타데이터.
    """
    if kind not in STIMULUS_TYPES:
        raise ValueError(f"지원되지 않는 합성 유형: {kind}")
    if clean.dtype != np.uint8 or clean.shape != (cfg.frames, cfg.height, cfg.width, 3):
        raise ValueError("원본 클립 형식은 [T,H,W,3] uint8 RGB여야 합니다.")
    if kind == "normal":
        return (clean.copy(), np.zeros(clean.shape[:3], np.uint8),
                np.zeros(cfg.frames, np.float32), {"intensity": 0.0})

    base_mask = _spatial_mask(cfg, rng)
    masks = np.repeat(base_mask[None], cfg.frames, axis=0)
    activation, profile = _activation(cfg, rng)
    intensity = float(rng.uniform(0.15, 0.85))
    rgb = clean.astype(np.float32) / 255.0
    modified = rgb.copy()
    yy, xx = np.mgrid[:cfg.height, :cfg.width]
    angle = float(rng.uniform(0, np.pi))
    stripe_size = float(rng.uniform(3.0, 15.0))
    spatial_wave = np.sin(2 * np.pi * (xx * np.cos(angle) + yy * np.sin(angle)) / stripe_size)
    # 마스크 바깥의 픽셀 값은 원본과 정확히 동일하게 유지합니다.
    for t in range(cfg.frames):
        amount = intensity * float(activation[t])
        if kind == "luminance":
            local = np.clip(rgb[t] * (1 - amount) + amount * 0.95, 0, 1)
        elif kind == "red":
            target_color = np.array([1.0, 0.04, 0.08], dtype=np.float32)
            local = rgb[t] * (1 - amount) + target_color * amount
        else:
            # 패턴은 방향·간격·시간 위상을 갖는 국소 대비 변화입니다.
            shift = t * float(profile["frequency_hz"]) / cfg.fps * np.pi
            pattern = np.sin(np.arcsin(np.clip(spatial_wave, -1, 1)) + shift)
            local = np.clip(rgb[t] + (pattern[..., None] * 0.5) * amount, 0, 1)
        changed = np.rint(local * 255).astype(np.uint8)
        modified[t] = rgb[t]
        # float 값 재양자화는 ROI 안에서만 이루어지게 합니다.
        modified[t] = np.where(base_mask[..., None] != 0, changed.astype(np.float32) / 255, rgb[t])
    output = np.rint(modified * 255).astype(np.uint8)
    output = np.where(masks[..., None] != 0, output, clean)
    profile.update({"intensity": intensity, "area_fraction": float(base_mask.mean()),
                    "stripe_period_px": stripe_size if kind == "pattern" else None,
                    "stripe_angle_rad": angle if kind == "pattern" else None})
    return output, masks, activation, profile


def _source_key(path: Path) -> str:
    """파일의 전체 내용 해시를 사용해 파일명·복사 위치가 달라도 동일 그룹으로 묶습니다."""
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()[:16]


def build_dataset(sources: list[Path], output: Path, *, samples_per_source: int = 20,
                  cfg: DatasetConfig | None = None) -> list[dict]:
    """동영상 파일에서 합성 학습셋과 기계 판독 가능한 라벨을 생성합니다."""
    cfg = cfg or DatasetConfig()
    cfg.validate()
    if samples_per_source < 1 or not sources:
        raise ValueError("원본 영상과 양수의 샘플 개수가 필요합니다.")
    sources = [Path(p) for p in sources]
    if not all(p.is_file() for p in sources):
        raise FileNotFoundError("원본 영상이 존재하는지 확인하세요.")
    rng = np.random.default_rng(cfg.seed)
    manifest = []
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for source in sources:
        key = _source_key(source)
        split = split_for_source(key, only_one=len(sources) == 1)
        decoded = _read_frames(source, cfg)
        destination = output / split
        destination.mkdir(exist_ok=True)
        for i in range(samples_per_source):
            # 각 클립은 원본에서 연속 추출합니다. 짧은 데모는 반복합니다.
            start = int(rng.integers(0, len(decoded)))
            index = (start + np.arange(cfg.frames)) % len(decoded)
            clean = decoded[index].copy()
            kind = STIMULUS_TYPES[i % len(STIMULUS_TYPES)]
            affected, mask, activation, params = synthesize(clean, cfg, kind, rng)
            name = f"{key}_{i:05d}"
            filename = destination / f"{name}.npz"
            np.savez_compressed(filename, clean=clean, affected=affected,
                                mask=mask, activation=activation)
            row = {
                "id": name, "source_id": key, "source_name": source.name,
                "split": split, "type": kind, "frames": cfg.frames,
                "height": cfg.height, "width": cfg.width, "fps": cfg.fps,
                "start_frame": start, "file": f"{split}/{name}.npz",
                "parameters": params, "risk_verified": False,
                "standard_result": "not_evaluated",
                "preview_safe": False,
            }
            (destination / f"{name}.json").write_text(
                json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8")
            manifest.append(row)
    with (output / "manifest.jsonl").open("w", encoding="utf-8") as f:
        for entry in manifest:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return manifest
