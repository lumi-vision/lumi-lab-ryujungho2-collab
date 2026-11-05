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
    # 1이면 클립 전체가 주파수 1개(v0.3 동작). 2 이상이면 주파수·세기가 바뀌는 구간 수.
    segments: int = 1
    # True면 IRIS 입력용 무손실 영상(FFV1/MKV)을 .npz 옆에 함께 저장합니다.
    export_video: bool = False

    def validate(self) -> None:
        if self.frames < 4 or self.width < 32 or self.height < 32 or self.fps < 8 or self.max_source_seconds < 1:
            raise ValueError("프레임 수는 4 이상, 해상도는 32 이상, FPS는 8 이상이어야 합니다.")
        if not 1 <= self.segments <= self.frames // 4:
            raise ValueError("구간 수는 1 이상, 프레임 수/4 이하여야 합니다(구간당 최소 4프레임).")


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


def _wave(position: np.ndarray, waveform: str) -> np.ndarray:
    """누적 주기 수(position)를 0~1 파형 값으로 바꿉니다."""
    if waveform == "square":
        return (np.sin(2 * np.pi * position) >= 0).astype(np.float32)
    if waveform == "triangle":
        return 1 - np.abs(2 * (position - np.floor(position)) - 1)
    return (np.sin(2 * np.pi * position) + 1) / 2


def _segment_bounds(cfg: DatasetConfig, rng: np.random.Generator) -> list[int]:
    """구간당 최소 4프레임을 보장하는 임의 경계 [0, ..., frames]."""
    spare = cfg.frames - 4 * cfg.segments
    cuts = np.sort(rng.integers(0, spare + 1, cfg.segments - 1))
    lengths = np.diff(np.concatenate(([0], cuts, [spare]))) + 4
    return [0, *np.cumsum(lengths).tolist()]


def _activation(cfg: DatasetConfig, rng: np.random.Generator) -> tuple[np.ndarray, dict, np.ndarray]:
    """제어 가능한 시간 파형을 생성합니다. 주파수만으로 위험을 판정하지 않습니다.

    반환: 활성화 [T], 메타데이터, 시작 시점부터의 누적 주기 수 [T](패턴 이동에 사용).
    `segments > 1`이면 구간마다 주파수와 진폭(0이면 휴지 구간)이 바뀌고,
    위상은 구간 경계에서 이어집니다.
    """
    max_freq = min(cfg.fps / 2 * 0.85, 10.0)
    if cfg.segments == 1:
        freq = float(rng.uniform(0.8, max_freq))
        phase = float(rng.uniform(0, 2 * np.pi))
        waveform = str(rng.choice(("sine", "square", "triangle")))
        times = np.arange(cfg.frames, dtype=np.float32) / cfg.fps
        activation = _wave(freq * times + phase / (2 * np.pi), waveform)
        # v0.3과 같은 연산 순서(t * f / fps)를 유지해 기존 시드의 결과를 바꾸지 않습니다.
        cycles = np.arange(cfg.frames) * freq / cfg.fps
        return activation.astype(np.float32), {
            "frequency_hz": freq, "phase_rad": phase, "waveform": waveform,
        }, cycles

    phase = float(rng.uniform(0, 2 * np.pi))
    waveform = str(rng.choice(("sine", "square", "triangle")))
    bounds = _segment_bounds(cfg, rng)
    freqs = np.empty(cfg.frames, dtype=np.float64)
    amps = np.empty(cfg.frames, dtype=np.float32)
    segments = []
    for start, end in zip(bounds[:-1], bounds[1:]):
        freq = float(rng.uniform(0.5, max_freq))
        # 휴지 구간을 섞어 통과·경고·실패 판정이 한 클립 안에서 바뀌게 합니다.
        amp = 0.0 if rng.random() < 0.2 else float(rng.uniform(0.3, 1.0))
        freqs[start:end] = freq
        amps[start:end] = amp
        segments.append({"start_frame": int(start), "end_frame": int(end),
                         "frequency_hz": freq, "amplitude": amp})
    # 프레임 t의 누적 주기 = sum_{k<t} f_k / fps  (구간 경계에서 위상 연속)
    cycles = np.concatenate(([0.0], np.cumsum(freqs[:-1]) / cfg.fps))
    activation = amps * _wave(cycles + phase / (2 * np.pi), waveform)
    mean_freq = float(np.mean([s["frequency_hz"] for s in segments]))
    return activation.astype(np.float32), {
        "frequency_hz": mean_freq, "phase_rad": phase, "waveform": waveform,
        "segments": segments,
    }, cycles


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
    activation, profile, cycles = _activation(cfg, rng)
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
            shift = float(cycles[t]) * np.pi
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
            video = None
            if cfg.export_video:
                from .lossless_video import write_lossless_video
                write_lossless_video(destination / f"{name}.mkv", affected, cfg.fps)
                video = f"{split}/{name}.mkv"
            row = {
                "id": name, "source_id": key, "source_name": source.name,
                "split": split, "type": kind, "frames": cfg.frames,
                "height": cfg.height, "width": cfg.width, "fps": cfg.fps,
                "start_frame": start, "file": f"{split}/{name}.npz",
                "parameters": params, "risk_verified": False,
                "standard_result": "not_evaluated",
                "preview_safe": False,
            }
            if video is not None:
                row["video"] = video
            (destination / f"{name}.json").write_text(
                json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8")
            manifest.append(row)
    with (output / "manifest.jsonl").open("w", encoding="utf-8") as f:
        for entry in manifest:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return manifest
