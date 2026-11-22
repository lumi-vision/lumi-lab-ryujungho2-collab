"""LUMI 정상 움직임/객체 분할 테스트용 기하학 데이터.

낮은 대비의 단일 객체가 연속적으로 변형, 분열, 결합, 이동한다.
위험 자극 생성, 점멸, 반복 무늬, 외부 영상 합성은 포함하지 않는다.
object_mask는 객체 분할 정답이며 위험/보정 마스크가 아니다.
독립 검증 전에는 정상 라벨을 부여하지 않는다.
"""
import argparse
import json
from pathlib import Path

import numpy as np

FPS = 24
FRAMES = 192
HEIGHT = 108
WIDTH = 192


def make_clip(seed: int = 0) -> dict:
    """고정된 저대비 팔레트의 연속 기하학 변형과 객체 주석을 생성한다."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:HEIGHT, :WIDTH].astype(np.float32)
    xx = (xx - (WIDTH - 1) / 2) / HEIGHT
    yy = (yy - (HEIGHT - 1) / 2) / HEIGHT
    phase = rng.uniform(0, 2 * np.pi)
    direction = rng.choice([-1., 1.])
    frames = np.empty((FRAMES, HEIGHT, WIDTH, 3), dtype=np.uint8)
    masks = np.empty((FRAMES, HEIGHT, WIDTH), dtype=np.uint8)
    for i in range(FRAMES):
        u = i / (FRAMES - 1)
        # 부드러운 이동과 형태 변형. 색상은 시간에 따라 바꾸지 않는다.
        cx = direction * (-0.30 + 0.60 * u)
        cy = 0.09 * np.sin(2 * np.pi * u + phase)
        theta = phase + 0.65 * np.sin(2 * np.pi * u)
        c, s = np.cos(theta), np.sin(theta)
        x = c * (xx - cx) + s * (yy - cy)
        y = -s * (xx - cx) + c * (yy - cy)
        # 암시적 필드: 연결된 객체가 분리되고 다시 연결되는 위상 변화.
        sep = 0.035 + 0.285 * np.sin(np.pi * u) ** 2
        sx = 0.067 + 0.009 * np.sin(2 * np.pi * u + phase)
        sy = 0.095 + 0.018 * np.cos(2 * np.pi * u + phase)
        a = np.exp(-0.5 * (((x - sep / 2) / sx) ** 2 + (y / sy) ** 2))
        b = np.exp(-0.5 * (((x + sep / 2) / sx) ** 2 + (y / sy) ** 2))
        field = a + b
        masks[i] = field >= 0.65
        # 경계를 부드럽게 렌더링하고 낮은 대비의 고정 팔레트를 유지한다.
        blend = np.clip((field - 0.55) / 0.20, 0, 1)
        blend = blend * blend * (3 - 2 * blend)
        background = np.array([120, 124, 128], dtype=np.float32)
        foreground = np.array([136, 140, 144], dtype=np.float32)
        frames[i] = np.rint(background + blend[..., None] * (foreground - background)).astype(np.uint8)
    metadata = {
        'schema_version': 'geometry-1',
        'purpose': 'normal_motion_candidate_and_object_segmentation',
        'seed': int(seed),
        'annotation_source': 'analytic_geometry',
        'validation_status': 'pending_independent_review',
        'object_mask_semantics': 'implicit_field_contour_not_risk',
        'events': ['deformation', 'translation', 'split', 'merge'],
    }
    return {
        'input': frames,
        'clean': frames,
        'object_mask': masks,
        'fps': np.array(FPS, dtype=np.float32),
        'timestamps': np.arange(FRAMES, dtype=np.float64) / FPS,
        'layout': np.array('THWC'),
        'source_type': np.array('geometry_candidate'),
        'source_id': np.array(f'geometry_{seed:06d}'),
        'validation_status': np.array('pending_independent_review'),
        'metadata': np.array(json.dumps(metadata, ensure_ascii=False)),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=Path('geometry_candidates'))
    parser.add_argument('--count', type=int, default=1)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args(argv)
    if args.count < 1 or args.seed < 0:
        parser.error('count는 양수, seed는 0 이상이어야 합니다.')
    args.out.mkdir(parents=True, exist_ok=True)
    for seed in range(args.seed, args.seed + args.count):
        path = args.out / f'geometry_{seed:06d}.npz'
        clip = make_clip(seed)
        with path.open('xb') as output:
            np.savez_compressed(output, **clip)
        print(path)
    print('객체 마스크만 포함. 독립 검증 전 위험/안전 라벨을 부여하지 않습니다.')


if __name__ == '__main__':
    main()
