"""GAN 학습 파이프라인 시험용 합성 원본 영상 여러 개를 만듭니다(점멸·고대비 줄무늬 없음).

실제 영상을 확보하기 전까지 원본 단위 train/val/test 분할과 G의 일반화를 시험하기 위한 것입니다.
장면 종류(seed마다 다름): 천천히 이동하는 저주파 질감, 그라데이션 위의 도형, 느린 카메라 이동.
밝기 변화는 프레임당 매우 작게 제한합니다. 임상적 무해함을 인증하는 영상이 아닙니다.
"""
import argparse
from pathlib import Path

import cv2
import numpy as np


def smooth_texture(rng: np.random.Generator, h: int, w: int, scale: int) -> np.ndarray:
    """저주파 무작위 질감 [h,w,3] float 0~1 (큰 격자 잡음을 부드럽게 확대)."""
    small = rng.random((max(2, h // scale), max(2, w // scale), 3)).astype(np.float32)
    big = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
    return np.clip(cv2.GaussianBlur(big, (0, 0), scale / 2), 0, 1)


def make_scene(seed: int, frames: int, h: int, w: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    kind = seed % 3
    base_color = rng.uniform(0.25, 0.75, 3).astype(np.float32)
    out = np.empty((frames, h, w, 3), np.uint8)
    if kind == 0:  # 천천히 흐르는 질감
        tex = smooth_texture(rng, h * 2, w * 2, int(rng.integers(12, 40)))
        vx, vy = rng.uniform(-0.6, 0.6, 2)
        for t in range(frames):
            x0, y0 = int(w / 2 + vx * t) % w, int(h / 2 + vy * t) % h
            patch = tex[y0:y0 + h, x0:x0 + w]
            out[t] = np.rint(255 * np.clip(0.6 * base_color + 0.4 * patch, 0, 1))
    elif kind == 1:  # 그라데이션 + 움직이는 저대비 도형
        yy, xx = np.mgrid[:h, :w].astype(np.float32)
        angle = rng.uniform(0, 2 * np.pi)
        grad = (xx * np.cos(angle) + yy * np.sin(angle)) / max(h, w)
        bg = np.clip(base_color + 0.25 * (grad[..., None] - 0.5), 0, 1)
        shapes = [(rng.uniform(0, w), rng.uniform(0, h), rng.uniform(-0.8, 0.8), rng.uniform(-0.8, 0.8),
                   int(rng.integers(h // 10, h // 4)), rng.uniform(-0.12, 0.12, 3)) for _ in range(int(rng.integers(2, 5)))]
        for t in range(frames):
            img = bg.copy()
            for x, y, vx, vy, r, dc in shapes:
                c = (int((x + vx * t) % w), int((y + vy * t) % h))
                layer = np.zeros((h, w), np.float32)
                cv2.circle(layer, c, r, 1.0, -1, cv2.LINE_AA)
                img = np.clip(img + layer[..., None] * dc, 0, 1)
            out[t] = np.rint(255 * img)
    else:  # 큰 장면 위를 느리게 이동하는 카메라
        world = smooth_texture(rng, h * 3, w * 3, int(rng.integers(6, 20)))
        world = np.clip(0.5 * base_color + 0.5 * world, 0, 1)
        sx, sy = rng.uniform(0, w), rng.uniform(0, h)
        vx, vy = rng.uniform(-1.0, 1.0, 2)
        for t in range(frames):
            x0 = int(np.clip(sx + vx * t, 0, 2 * w - 1))
            y0 = int(np.clip(sy + vy * t, 0, 2 * h - 1))
            out[t] = np.rint(255 * world[y0:y0 + h, x0:x0 + w])
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="합성 원본 영상 생성(점멸 없음, 재생하지 않음)")
    parser.add_argument("--output", type=Path, required=True, help="원본 영상을 저장할 폴더")
    parser.add_argument("--count", type=int, default=30)
    parser.add_argument("--seconds", type=float, default=8.0)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=192)
    parser.add_argument("--seed", type=int, default=1000)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    frames = int(args.seconds * args.fps)
    for i in range(args.count):
        seed = args.seed + i
        clip = make_scene(seed, frames, args.height, args.width)
        # 프레임 간 평균 밝기 변화가 작은지 확인(점멸 없는 원본이어야 함)
        step = np.abs(np.diff(clip.reshape(frames, -1).mean(1).astype(np.float32))).max() if frames > 1 else 0
        if step > 2.0:
            raise RuntimeError(f"seed {seed}: 프레임 간 평균 밝기 변화 {step:.2f}가 큼")
        path = args.output / f"src_{seed:05d}.mkv"
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"FFV1"), args.fps, (args.width, args.height))
        if not writer.isOpened():
            raise RuntimeError("FFV1 인코더를 열 수 없습니다.")
        for frame in clip:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        writer.release()
    print(f"원본 {args.count}개 생성: {args.output} ({frames}프레임, {args.width}x{args.height})")


if __name__ == "__main__":
    main()
