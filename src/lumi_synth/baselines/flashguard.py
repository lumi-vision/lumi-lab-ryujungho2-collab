"""FlashGuard에서 착안한 연구용 기준선. 원저자의 모델이나 안전 판정이 아니다."""
from __future__ import annotations

import time
import cv2
import numpy as np


def fit_logistic(values: np.ndarray, labels: np.ndarray, seed: int = 0) -> dict:
    """학습 분할의 특성과 이진 라벨로 로지스틱 계수를 학습한다."""
    from sklearn.linear_model import LogisticRegression
    x = np.asarray(values, dtype=np.float64).reshape(-1, 1)
    y = np.asarray(labels).reshape(-1)
    if len(x) != len(y) or len(x) < 2 or not np.isfinite(x).all() or set(y) != {0, 1}:
        raise ValueError('양쪽 클래스가 포함된 유한 특성과 이진 라벨이 필요합니다')
    model = LogisticRegression(random_state=seed, max_iter=1000).fit(x, y)
    return {'coefficient': float(model.coef_[0, 0]), 'intercept': float(model.intercept_[0]), 'seed': seed}


def features(frames: np.ndarray, fps: float) -> np.ndarray:
    """BGR 영상의 연속 프레임 CIELAB 변화율을 반환한다."""
    if frames.ndim != 4 or frames.shape[-1] != 3 or frames.dtype != np.uint8 or len(frames) < 2 or not np.isfinite(fps) or fps <= 0:
        raise ValueError('최소 2프레임의 uint8 BGR 영상과 양의 fps가 필요합니다')
    labs = [cv2.cvtColor(frame.astype(np.float32) / 255, cv2.COLOR_BGR2LAB) for frame in frames]
    delta = np.diff(np.stack(labs), axis=0)
    return (np.abs(delta[..., 0]) + np.linalg.norm(delta[..., 1:], axis=-1)) * fps


def analyze(frames: np.ndarray, fps: float, coefficient: float, intercept: float, threshold: float = 0.5, grid: tuple[int, int] = (50, 50)) -> dict:
    """외부에서 제공한 로지스틱 계수로 프레임별 공간 위험 점수를 산출한다."""
    if not np.isfinite([coefficient, intercept, threshold]).all() or not 0 <= threshold <= 1 or min(grid) < 1:
        raise ValueError('계수·문턱·격자 값 오류')
    t0 = time.perf_counter()
    change = features(frames, fps)
    h, w = frames.shape[1:3]
    ys = np.linspace(0, h - 1, grid[0]).astype(int)
    xs = np.linspace(0, w - 1, grid[1]).astype(int)
    sampled = change[:, ys[:, None], xs]
    logits = np.clip(coefficient * sampled + intercept, -60, 60)
    scores = 1 / (1 + np.exp(-logits))
    masks = np.zeros((len(frames), h, w), dtype=bool)
    masks[1:] = np.stack([cv2.resize((s >= threshold).astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool) for s in scores])
    return {'scores': scores, 'masks': masks, 'elapsed_seconds': time.perf_counter() - t0,
            'assumptions': '프레임별 즉시 점수, 균등격자 최근접 보간; 학습 계수는 외부 제공'}


def mitigate(frames: np.ndarray, masks: np.ndarray, darkening: float, color_alpha: float, window: int = 15) -> np.ndarray:
    """제공된 영역에 감광과 과거 평균 색상 오버레이를 적용한다."""
    if frames.ndim != 4 or frames.dtype != np.uint8 or frames.shape[-1] != 3 or masks.shape != (len(frames), *frames.shape[1:3]) or not 0 <= darkening <= 1 or not 0 <= color_alpha <= 1 or window < 1:
        raise ValueError('영상·마스크·보정 설정 오류')
    out = frames.copy()
    for i, frame in enumerate(frames):
        mean = frames[max(0, i - window + 1):i + 1].astype(np.float32).mean(axis=0)
        corrected = ((1 - color_alpha) * frame.astype(np.float32) + color_alpha * mean) * (1 - darkening)
        out[i][masks[i]] = np.rint(corrected[masks[i]]).clip(0, 255).astype(np.uint8)
    return out


def evaluate_binary(truth: np.ndarray, predictions: np.ndarray) -> dict:
    """외부 정답과 이진 예측을 비교하고 정의되지 않은 비율은 null로 반환한다."""
    truth, predictions = np.asarray(truth), np.asarray(predictions)
    if truth.shape != predictions.shape or truth.size == 0 or not np.isin(truth, [0, 1]).all() or not np.isin(predictions, [0, 1]).all():
        raise ValueError('동일 크기의 이진 정답과 예측이 필요합니다')
    tp = int(((truth == 1) & (predictions == 1)).sum())
    fn = int(((truth == 1) & (predictions == 0)).sum())
    tn = int(((truth == 0) & (predictions == 0)).sum())
    fp = int(((truth == 0) & (predictions == 1)).sum())
    return {'tp': tp, 'fn': fn, 'tn': tn, 'fp': fp, 'recall': tp / (tp + fn) if tp + fn else None,
            'specificity': tn / (tn + fp) if tn + fp else None}


def preservation(reference: np.ndarray, corrected: np.ndarray, mask: np.ndarray) -> dict:
    """원본이 있는 자료에 대해 전체와 마스크 밖 평균 절대 오차를 구한다."""
    if reference.shape != corrected.shape or reference.ndim != 4 or reference.shape[-1] != 3 or mask.shape != reference.shape[:-1]:
        raise ValueError('원본·보정·마스크 크기 오류')
    diff = np.abs(reference.astype(np.float32) - corrected.astype(np.float32))
    return {'mae': float(diff.mean()), 'outside_mae': float(diff[~mask].mean()) if (~mask).any() else None}
