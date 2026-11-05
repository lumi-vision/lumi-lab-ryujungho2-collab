"""정지 이미지 1장에서 계산할 수 있는 PSE 관련 공간 특징 (image_discriminator v2).

v1(`image_discriminator_1.ipynb`) 대비 변경점
1. 줄무늬 쌍 개수와 패턴 면적을 **같은 영역**에서 계산한다. 주사선마다 규칙적인
   명암 줄무늬 구간을 찾고, 5쌍 이상인 구간만 패턴 면적에 넣는다.
2. 줄무늬 쌍 계산의 1쌍 누락(`경계 수 // 2`)을 고쳤다: 줄무늬 수를 직접 센다.
3. 고대비·고휘도·적색은 정지 화면 위험 기준이 아니므로(표준에서는 프레임 간
   '변화'의 조건) 판정하지 않고 **특징값**으로만 출력한다.
4. '확률'이라는 이름을 쓰지 않는다. 시피질 참고값은 시청 기하(화면 크기·거리)를
   입력으로 받으며 임상 확률이 아니다.
5. 원본 해상도로 분석한다(4K 초과만 축소하고 기록).

이 모듈의 출력은 연구용 스크리닝 값이며 ITU-R BT.1702 / WCAG 적합성 판정이나
임상적 안전성 판단이 아니다.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

PEAK_WHITE_CD = 200.0          # SDR 흰색 가정 (cd/m²)
DARK_LIMIT_CD = 160.0          # ITU-R BT.1702 명암 대비 규칙의 분기점
MIN_PAIRS = 5                  # 규칙적 패턴: 명암 쌍 5개 이상
SAME_POLARITY_RATIO = 1.75     # 같은 극성(밝은-밝은, 어두운-어두운) 줄무늬 폭 허용 비
LINE_COHERENCE = 0.6           # 인접 주사선 5개 중 이 비율 이상에서 함께 검출돼야 인정
MAX_SIDE = 3840


# ---------------------------------------------------------------- 입력·휘도
def load_image_rgb(path: str | Path) -> np.ndarray:
    """한글 경로를 지원하는 이미지 로더. (H, W, 3) uint8 RGB."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"입력 파일을 찾을 수 없습니다: {path}")
    bgr = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"이미지를 읽을 수 없습니다(정지 이미지 전용): {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def srgb_to_linear(rgb_u8: np.ndarray) -> np.ndarray:
    """IEC 61966-2-1 sRGB 감마 복원. 결과 [0, 1] float32."""
    c = rgb_u8.astype(np.float32) / 255.0
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4).astype(np.float32)


def luminance_cd(linear_rgb: np.ndarray, peak_white: float = PEAK_WHITE_CD) -> np.ndarray:
    """선형 RGB → 휘도(cd/m²). 표시 장치 최대 밝기를 `peak_white`로 가정한다."""
    y = 0.2126 * linear_rgb[..., 0] + 0.7152 * linear_rgb[..., 1] + 0.0722 * linear_rgb[..., 2]
    return (peak_white * y).astype(np.float32)


def is_contrast_pair(high: float, low: float) -> bool:
    """ITU-R BT.1702 명암 대비 규칙: 어두운 쪽 <160이면 ΔL≥20, 아니면 Michelson>1/17."""
    if low < DARK_LIMIT_CD:
        return high - low >= 20.0
    return (high - low) / (high + low) > 1.0 / 17.0


# ---------------------------------------------------------------- 1D 줄무늬
def _alternating_extrema(p: np.ndarray) -> list[int]:
    """인접한 두 극값이 항상 명암 대비 규칙을 만족하는 교대 극값의 인덱스."""
    n = len(p)
    if n < 2:
        return []
    out: list[int] = []
    hi = lo = 0
    direction = 0      # 0: 미정, 1: 최댓값 추적 중, -1: 최솟값 추적 중
    cand = 0
    for i in range(1, n):
        v = p[i]
        if direction == 0:
            if v > p[hi]:
                hi = i
            if v < p[lo]:
                lo = i
            if is_contrast_pair(p[hi], p[lo]):
                if hi < lo:
                    out.append(hi); direction = -1; cand = lo
                else:
                    out.append(lo); direction = 1; cand = hi
        elif direction == 1:
            if v > p[cand]:
                cand = i
            elif is_contrast_pair(p[cand], v):
                out.append(cand); direction = -1; cand = i
        else:
            if v < p[cand]:
                cand = i
            elif is_contrast_pair(v, p[cand]):
                out.append(cand); direction = 1; cand = i
    if direction != 0:
        out.append(cand)
    return out


def _edge_between(p: np.ndarray, a: int, b: int) -> int:
    """극값 a→b 사이에서 중간 밝기를 처음 넘는 위치."""
    mid = (p[a] + p[b]) / 2.0
    seg = p[a:b + 1]
    hits = np.flatnonzero(seg <= mid) if p[a] > p[b] else np.flatnonzero(seg >= mid)
    return a + int(hits[0]) if hits.size else b


def _plateau_start(p: np.ndarray, i0: int, mid: float, high: bool) -> int:
    side = p[:i0 + 1] >= mid if high else p[:i0 + 1] <= mid
    off = np.flatnonzero(~side)
    return int(off[-1]) + 1 if off.size else 0


def _plateau_end(p: np.ndarray, i1: int, mid: float, high: bool) -> int:
    side = p[i1:] >= mid if high else p[i1:] <= mid
    off = np.flatnonzero(~side)
    return i1 + (int(off[0]) if off.size else len(side))


@dataclass(frozen=True)
class StripeRun:
    start: int      # 첫 줄무늬 시작 픽셀
    end: int        # 마지막 줄무늬 끝 픽셀(미포함)
    stripes: int    # 밝은/어두운 줄무늬 개수

    @property
    def pairs(self) -> int:
        return self.stripes // 2


def stripe_runs(profile: np.ndarray) -> list[StripeRun]:
    """1D 휘도 프로파일에서 폭이 규칙적인 명암 줄무늬 구간을 찾는다.

    - 줄무늬 = 대비 규칙을 만족하는 교대 극값 하나를 중심으로 한 띠
    - 같은 극성 줄무늬(k와 k-2)의 폭 비가 `SAME_POLARITY_RATIO` 이하이면 같은 구간
      (흰 바탕 위 가는 검은 선처럼 명/암 폭이 달라도 허용)
    - 영상 경계에서 잘린 끝 줄무늬는 더 좁은 경우만 허용, 넓은 바탕은 줄무늬로 세지 않음
    """
    p = np.asarray(profile, dtype=np.float32)
    ext = _alternating_extrema(p)
    if len(ext) < 2:
        return []
    edges = [_edge_between(p, ext[k], ext[k + 1]) for k in range(len(ext) - 1)]
    n = len(p)
    first_high = p[ext[0]] > p[ext[1]]
    mid0 = (p[ext[0]] + p[ext[1]]) / 2.0
    midn = (p[ext[-2]] + p[ext[-1]]) / 2.0
    last_high = p[ext[-1]] > p[ext[-2]]
    lefts = [_plateau_start(p, ext[0], mid0, first_high)] + edges
    rights = edges + [_plateau_end(p, ext[-1], midn, last_high)]
    widths = [max(1, r - l) for l, r in zip(lefts, rights)]
    truncated = {0: lefts[0] == 0, len(widths) - 1: rights[-1] >= n}

    def consistent(k: int) -> bool:
        a, b = widths[k], widths[k - 2]
        if truncated.get(k) and a <= b:
            return True
        if truncated.get(k - 2) and b <= a:
            return True
        return max(a, b) / min(a, b) <= SAME_POLARITY_RATIO

    runs: list[StripeRun] = []
    start = 0
    for k in range(2, len(widths) + 1):
        if k == len(widths) or not consistent(k):
            # k-2와 k가 어긋나면 구간 [start, k-1]을 닫고 k-1부터 새 구간
            end_k = k - 1
            if end_k - start + 1 >= 2:
                # 보수적 계수: 구간을 둘러싼 넓은 바탕(폭 불일치로 제외된 줄무늬)도
                # 경계 줄무늬 1개로 센다. 흰 바탕 위 패턴의 끝 흰 줄무늬가 바탕과
                # 합쳐져 1쌍 적게 세는 것을 막는다(과소 계수보다 과대 계수를 택함).
                # 부작용: 줄무늬와 다른 제3의 밝기 바탕(예: 회색)은 실제 경계 줄무늬로
                # 볼 수 있어 1쌍 많게 셀 수 있다(회색 바탕 4쌍 → 5쌍).
                bounds = int(start > 0) + int(end_k < len(widths) - 1)
                runs.append(StripeRun(lefts[start], rights[end_k], end_k - start + 1 + bounds))
            start = k - 1
    return runs


def count_stripe_pairs(profile: np.ndarray) -> int:
    """프로파일 안의 가장 긴 규칙 구간의 명암 쌍 수."""
    return max((r.pairs for r in stripe_runs(profile)), default=0)


# ---------------------------------------------------------------- 2D 패턴
def _scan_mask(lum: np.ndarray, step: int, min_pairs: int):
    """행 방향 주사선마다 min_pairs 이상인 줄무늬 구간을 표시.

    반환: (표본행 × W) 마스크, 표시한 구간 목록 [(표본행 i, StripeRun)], 단일 주사선 최대 쌍 수
    """
    rows = range(0, lum.shape[0], step)
    marks = np.zeros((len(rows), lum.shape[1]), dtype=bool)
    marked: list[tuple[int, StripeRun]] = []
    single_best = 0
    for i, y in enumerate(rows):
        for run in stripe_runs(lum[y]):
            single_best = max(single_best, run.pairs)
            if run.pairs >= min_pairs:
                marks[i, run.start:run.end] = True
                marked.append((i, run))
    return marks, marked, single_best


def _coherent_best(coherent: np.ndarray, marked: list[tuple[int, StripeRun]]) -> int:
    """일관성 필터를 통과한 영역(구간의 절반 이상)에 속한 구간 중 최대 쌍 수."""
    return max((run.pairs for i, run in marked
                if coherent[i, run.start:run.end].mean() >= 0.5), default=0)


def _coherent(marks: np.ndarray, window: int = 5) -> np.ndarray:
    """인접 주사선에서도 함께 검출된 위치만 남긴다(글자·잡음의 우연한 일치 제거)."""
    if marks.shape[0] < window:
        return marks
    pad = window // 2
    padded = np.pad(marks.astype(np.float32), ((pad, pad), (0, 0)))
    c = np.cumsum(padded, axis=0)
    c = np.vstack([np.zeros((1, c.shape[1]), np.float32), c])
    mean = (c[window:] - c[:-window]) / window
    return marks & (mean >= LINE_COHERENCE)


def _expand(marks: np.ndarray, step: int, length: int) -> np.ndarray:
    return np.repeat(marks, step, axis=0)[:length]


@dataclass
class PatternResult:
    max_pairs: int              # 패턴 영역(일관성 통과)에 속한 줄무늬 구간의 최대 쌍 수
    max_pairs_single_line: int  # 진단용: 필터 전 단일 주사선 최대값 (글자·잡음 포함)
    area_fraction: float
    mask: np.ndarray            # analyzed_shape 크기 bool — 5쌍 이상 줄무늬가 검출된 픽셀
    analyzed_shape: tuple[int, int]


def detect_regular_pattern(lum: np.ndarray, *, min_pairs: int = MIN_PAIRS,
                           max_lines: int = 270) -> PatternResult:
    """가로·세로 주사선으로 규칙적 명암 줄무늬 영역을 찾는다.

    가로 주사선은 세로·사선·동심원·방사형 줄무늬를, 세로 주사선은 가로 줄무늬를 잡는다.
    주사선 간격(step)이 커지는 큰 영상에서는 높이(폭)가 약 3×step 미만인 패턴이
    일관성 필터에서 제외될 수 있다.
    """
    h, w = lum.shape
    step_r = max(1, math.ceil(h / max_lines))
    step_c = max(1, math.ceil(w / max_lines))
    row_marks, row_runs, single_r = _scan_mask(lum, step_r, min_pairs)
    col_marks, col_runs, single_c = _scan_mask(lum.T, step_c, min_pairs)
    row_ok, col_ok = _coherent(row_marks), _coherent(col_marks)
    mask = _expand(row_ok, step_r, h) | _expand(col_ok, step_c, w).T
    best = max(_coherent_best(row_ok, row_runs), _coherent_best(col_ok, col_runs))
    return PatternResult(best, max(single_r, single_c), float(mask.mean()), mask, (h, w))


# ---------------------------------------------------------------- 특징값
def saturated_red_fraction(linear_rgb: np.ndarray) -> float:
    """WCAG 포화 적색 정의 R/(R+G+B) ≥ 0.8 (선형 RGB). 판정이 아닌 면적 특징."""
    s = linear_rgb.sum(axis=-1)
    red = linear_rgb[..., 0]
    ratio = np.divide(red, s, out=np.zeros_like(red), where=s > 1e-4)
    return float(((ratio >= 0.8) & (red > 0.01)).mean())


def luminance_features(lum: np.ndarray) -> dict[str, float]:
    p5, p50, p95 = np.percentile(lum, [5, 50, 95])
    return {"mean_cd": float(lum.mean()), "p05_cd": float(p5), "p50_cd": float(p50),
            "p95_cd": float(p95), "bright_area_ge160": float((lum >= DARK_LIMIT_CD).mean())}


# ---------------------------------------------------------------- 시청 기하 참고값
@dataclass(frozen=True)
class ViewingGeometry:
    diagonal_inch: float = 27.0
    aspect: float = 16 / 9
    distance_cm: float = 60.0

    def screen_cm(self) -> tuple[float, float]:
        d = self.diagonal_inch * 2.54
        h = d / math.sqrt(1 + self.aspect ** 2)
        return h * self.aspect, h


def pattern_visual_angle_deg(area_fraction: float, geometry: ViewingGeometry) -> float:
    """패턴 면적을 같은 넓이의 원으로 보고 그 지름이 이루는 시각도.

    가정: Binnie 관계식의 θ를 '자극의 등가 원 지름 시각도'로 해석했다.
    Jordan & Vanderheiden(2024) 부록 A 원문과 대조가 필요하다.
    """
    if area_fraction <= 0:
        return 0.0
    w, h = geometry.screen_cm()
    diameter = 2.0 * math.sqrt(area_fraction * w * h / math.pi)
    return math.degrees(2.0 * math.atan(diameter / (2.0 * geometry.distance_cm)))


def binnie_reference(theta_deg: float) -> dict[str, float]:
    """Q = 100(1 - e^(-0.0574θ)), 비율 = 0.021Q - 0.1827 (Q∈[8.75, 56.37]에서 0→1).

    광과민 환자 중 '패턴 자극'에 뇌파 이상 반응을 보이는 비율의 참고 곡선이다.
    발작 확률이 아니며 고휘도·적색 자극에는 적용하지 않는다.
    """
    q = 100.0 * (1.0 - math.exp(-0.0574 * theta_deg))
    prop = 0.0 if q <= 8.75 else 1.0 if q >= 56.37 else 0.021 * q - 0.1827
    return {"theta_deg": theta_deg, "Q": q, "binnie_reference": float(min(max(prop, 0.0), 1.0))}


# ---------------------------------------------------------------- 통합
def analyze_image(rgb_u8: np.ndarray, *, area_thresholds: tuple[float, ...] = (0.25, 0.40),
                  geometry: ViewingGeometry | None = None,
                  peak_white: float = PEAK_WHITE_CD) -> dict:
    """정지 이미지 1장의 패턴 스크리닝 + 공간 특징. 결과에 영상(픽셀)은 포함하지 않는다(mask 제외)."""
    geometry = geometry or ViewingGeometry()
    resized = False
    h, w = rgb_u8.shape[:2]
    if max(h, w) > MAX_SIDE:
        scale = MAX_SIDE / max(h, w)
        size = (max(1, round(w * scale)), max(1, round(h * scale)))
        rgb_u8 = cv2.resize(rgb_u8, size, interpolation=cv2.INTER_AREA)
        resized = True
    lin = srgb_to_linear(rgb_u8)
    lum = luminance_cd(lin, peak_white)
    pattern = detect_regular_pattern(lum)
    mask = pattern.mask
    if resized:  # 마스크를 입력 크기로 되돌려 원본과 겹쳐 볼 수 있게 한다
        mask = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
    screening = {f"pattern_over_{int(round(t * 100))}pct": bool(
        pattern.max_pairs >= MIN_PAIRS and pattern.area_fraction > t) for t in area_thresholds}
    return {
        "analysis": "lumi.static_features v2 (research screening, not conformance)",
        "input_shape": (h, w), "analyzed_shape": pattern.analyzed_shape, "resized": resized,
        "pattern": {"max_pairs": pattern.max_pairs,
                    "max_pairs_single_line": pattern.max_pairs_single_line,
                    "area_fraction": pattern.area_fraction,
                    **screening},
        "features": {**luminance_features(lum),
                     "saturated_red_area": saturated_red_fraction(lin)},
        "reference": binnie_reference(pattern_visual_angle_deg(pattern.area_fraction, geometry)),
        "geometry": {"diagonal_inch": geometry.diagonal_inch, "aspect": geometry.aspect,
                     "distance_cm": geometry.distance_cm, "peak_white_cd": peak_white},
        "mask": mask,               # 입력 이미지와 같은 (H, W)
    }
