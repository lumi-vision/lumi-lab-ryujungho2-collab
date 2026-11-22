"""기존 세 탐지기의 출력을 공통 형식으로 변환하는 연구용 인터페이스."""
from __future__ import annotations
import time
import os
import sys
from pathlib import Path
import cv2
import numpy as np
import psutil
import torch
from sklearn.metrics import roc_auc_score
from lumi_synth import hazcls
from lumi_synth.baselines import flashguard

TYPES = ('lum', 'red', 'pattern')


def team_assessment(frames, fps, device):
    """로컬 평가와 MCP가 같은 Partial B2 판정기와 관측 설정을 사용한다."""
    sibling = Path(__file__).resolve().parents[4] / 'lumi-lab-rockrovks'
    engine = (Path(os.environ.get('LUMI_ENGINE_DIR', sibling)) / 'Partial' / 'B2').resolve()
    if not (engine / 'vdcoreps/pattern_motion.py').is_file():
        raise RuntimeError('LUMI_ENGINE_DIR에 동적 면적 규칙이 적용된 팀원 저장소를 지정하세요.')
    if str(engine) not in sys.path:
        sys.path.insert(0, str(engine))
    from partial_coreops_1112 import B2PartialConfig
    from vdcoreps import (analyze_frames, LABEL_LUM_FLASH, LABEL_INTERSECTION, LABEL_SCENE_TRANSITION,
                          LABEL_RED_FLASH, LABEL_RED_EXTENDED, LABEL_LUM_EXTENDED, LABEL_STATIC_PATTERN)
    from vdcoreps import pattern_motion
    if Path(pattern_motion.__file__).resolve().parent.parent != engine:
        raise RuntimeError('다른 B2 엔진이 이미 로드되어 있습니다. 별도 프로세스에서 실행하세요.')
    result = analyze_frames(list(frames), fps, mask_config=B2PartialConfig().to_mask_config(), device=device)
    active = set(result.active_labels)
    review = bool(result.frame_metrics_df['pattern_review_required'].any())
    decisions = {'lum': bool(active.intersection((LABEL_LUM_FLASH, LABEL_INTERSECTION, LABEL_SCENE_TRANSITION, LABEL_LUM_EXTENDED))),
                 'red': bool(active.intersection((LABEL_RED_FLASH, LABEL_RED_EXTENDED))),
                 'pattern': True if LABEL_STATIC_PATTERN in active else (None if review else False)}
    return decisions, result.is_dangerous, review, pattern_motion.PROFILE


def infer(frames: np.ndarray, fps: float, model: str, *, coefficient=None, intercept=None, threshold=0.5, checkpoint=None, device='cpu') -> dict:
    """영상의 모델별 출력을 분석하고 처리 시간과 메모리 측정치를 기록한다."""
    if frames.dtype != np.uint8 or frames.ndim != 4 or frames.shape[-1] != 3 or len(frames) < 2 or not np.isfinite(fps) or fps <= 0:
        raise ValueError('BGR 영상과 양의 fps가 필요합니다')
    if not 0 <= threshold <= 1:
        raise ValueError('문턱은 0~1이어야 합니다')
    proc = psutil.Process()
    before = proc.memory_info().rss
    if device.startswith('cuda'):
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    details = {}
    if model == 'B0':
        decisions, detected, review, profile = team_assessment(frames, fps, device)
        details = {'detector': profile, 'pattern_review_required': review}
        scores = {key: None for key in TYPES}
        any_score = None
        decision = True if detected else (None if review else False)
    elif model == 'B1':
        if coefficient is None or intercept is None:
            raise ValueError('B1은 학습 자료에서 적합된 계수가 필요합니다')
        result = flashguard.analyze(frames, fps, coefficient, intercept, threshold)
        any_score = float(result['scores'].max())
        decision = any_score >= threshold
        scores = {key: None for key in TYPES}
        decisions = {key: None for key in TYPES}
    elif model == 'B2':
        if checkpoint is None or len(frames) != 48 or fps != 24:
            raise ValueError('B2는 체크포인트와 24fps 48프레임을 요구합니다')
        state = torch.load(checkpoint, map_location=device, weights_only=False)
        net = hazcls.HazNet(width=int(state['config']['width'])).to(device).eval()
        net.load_state_dict(state['model'])
        temporal = np.stack([cv2.resize(frame, (hazcls.TW, hazcls.TH), interpolation=cv2.INTER_AREA) for frame in frames])
        spatial = np.stack([cv2.resize(frames[i], (hazcls.PW, hazcls.PH), interpolation=cv2.INTER_AREA if frames.shape[2] >= hazcls.PW else cv2.INTER_LINEAR) for i in hazcls.PAT_IDX])
        with torch.no_grad():
            result = net(torch.from_numpy(temporal[None]).to(device), torch.from_numpy(spatial[None]).to(device))
            vals = torch.sigmoid(result['logit'])[0].cpu().numpy()
        scores = {key: float(val) for key, val in zip(TYPES, vals)}
        decisions = {key: val >= threshold for key, val in scores.items()}
        any_score = max(scores.values())
        decision = any(decisions.values())
    else:
        raise ValueError('지원하지 않는 모델')
    if device.startswith('cuda'):
        torch.cuda.synchronize(device)
    mem = proc.memory_info()
    decisions['any'] = decision
    scores['any'] = any_score
    return {'model': model, **details, 'scores': scores, 'decisions': decisions, 'any_score': any_score, 'any_decision': decision,
            'seconds': time.perf_counter() - started, 'rss_before': before, 'rss_after': mem.rss,
            'process_peak_rss': getattr(mem, 'peak_wset', None),
            'gpu_peak_allocated': torch.cuda.max_memory_allocated(device) if device.startswith('cuda') else None}


def metrics(labels: list[int], decisions: list[bool], scores: list[float] | None = None) -> dict:
    """외부 라벨에 대한 지표를 계산하며 정의되지 않은 값은 null로 둔다."""
    y, p = np.asarray(labels), np.asarray(decisions)
    if not y.size or y.shape != p.shape or not np.isin(y, [0, 1]).all() or not np.isin(p, [0, 1]).all():
        raise ValueError('같은 길이의 이진 라벨과 예측이 필요합니다')
    tp = int(((y == 1) & (p == 1)).sum()); fp = int(((y == 0) & (p == 1)).sum())
    fn = int(((y == 1) & (p == 0)).sum()); tn = int(((y == 0) & (p == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None
    auc = None
    if scores is not None:
        s = np.asarray(scores, dtype=float)
        if s.shape != y.shape or not np.isfinite(s).all():
            raise ValueError('유한한 점수가 각 라벨에 필요합니다')
        if len(np.unique(y)) == 2:
            auc = float(roc_auc_score(y, s))
    return {'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn, 'precision': precision,
            'recall': recall, 'f1': f1, 'fnr': fn / (tp + fn) if tp + fn else None, 'auroc': auc}


def evaluate_records(records: list[dict]) -> dict:
    """이용 조건·라벨 근거가 확인된 최종 시험 결과만 공통 지표로 집계한다."""
    if not records:
        raise ValueError('평가할 결과가 없습니다')
    for record in records:
        if not record.get('rights_confirmed') or not record.get('label_evidence') or record.get('role') != 'final_test' or record.get('selection_used') is not False:
            raise ValueError('권한·라벨 근거·독립 최종 시험 조건이 확인되지 않았습니다')
    report = {}
    for key in ('any', *TYPES):
        relevant = [r for r in records if r.get('labels', {}).get(key) is not None and r.get('decisions', {}).get(key) is not None]
        if not relevant:
            report[key] = None
            continue
        scores = [r.get('scores', {}).get(key) for r in relevant]
        report[key] = metrics([r['labels'][key] for r in relevant], [r['decisions'][key] for r in relevant],
                              scores if all(s is not None for s in scores) else None)
    return report
