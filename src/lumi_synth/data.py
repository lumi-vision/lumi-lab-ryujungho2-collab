"""깨끗한 바탕 클립 만들기와 데이터셋 저장 (오라클·GAN 공통 재료).

흐름
    원본 영화(CC BY) ─► 일정 간격으로 후보 구간 ─► 작은 해상도로 읽기 ─► 판정 규칙으로 '깨끗함' 확인
                     ─► 통과한 클립만 바탕 Y 로 사용

'깨끗함'의 기준 (docs/CRITERIA.md 4-1, docs/DATA_PLAN.md §2)
    - 화면 전이 0번 (ITU 는 3번까지 안전이지만, 바탕은 0번만 허용 → 나중에 측정되는 깜빡임은 모두 우리가 넣은 것)
    - 평균 밝기 ≥ 20 cd/m² (너무 어두운 장면·크레딧 제외)
    - 장면 전환 없음 (연속 프레임 평균 절대 차이가 큰 구간 제외)

분할(split)은 '원본 영화 단위'로 한다: 같은 영화의 클립이 학습과 시험에 동시에 들어가면 장면을 외워서
점수가 부풀 수 있기 때문이다(데이터 누수). 기본값: Big Buck Bunny = 학습/검증(시간 구간으로 나눔),
Elephants Dream = 시험.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from lumi_synth import measure

MIN_MEAN_CD = 20.0          # 너무 어두운 장면 제외 (cd/m²)
SCENE_CUT_MAD = 40.0        # 연속 프레임 평균 절대 차이(0~255)가 이보다 크면 장면 전환으로 봄


@dataclass(frozen=True)
class ClipSpec:
    """클립 규격 = '트랙'. 두 트랙을 같은 코드로 돌린다(docs/DATA_PLAN.md §7).

    - 노트북 트랙(기본): 128×72, 24 fps(원본 그대로), 48프레임(2초) — CPU 로 GAN 학습 가능한 크기
    - 집 PC 트랙(PLAN.md): 1920×1080, 30 fps(24→30 프레임 복제), 150프레임(5초)
    """
    width: int = 128
    height: int = 72
    frames: int = 48
    fps: float = 24.0

    @property
    def tag(self) -> str:
        """산출물 이름에 붙이는 트랙 표시. 예: '128x72_24fps', '1920x1080_30fps'."""
        return f"{self.width}x{self.height}_{self.fps:g}fps"


def duplicate_schedule(n_out: int, src_fps: float, dst_fps: float) -> np.ndarray:
    """출력 프레임 i 에 쓸 원본 프레임 번호 = floor(i · src/dst). dst ≥ src 일 때만 쓴다.

    24→30 fps 면 원본 4장마다 1장을 한 번 더 넣는다(PLAN.md 와 같은 방식). 프레임을 섞거나 새로 만들지 않으므로
    새로운 밝기 변화(깜빡임)가 생기지 않는다 — 같은 프레임을 반복하면 그 순간 변화는 0 이다.
    """
    return np.floor(np.arange(n_out) * (src_fps / dst_fps) + 1e-9).astype(int)


def read_clip(path: Path, start_frame: int, spec: ClipSpec, src_fps: float | None = None) -> np.ndarray | None:
    """start_frame 부터 읽어 spec 규격 (T,H,W,3) uint8 BGR 로. 모자라면 None.

    - 축소는 INTER_AREA(면적 평균): 깜빡임 판정은 면적 비율이 중요하므로 픽셀을 건너뛰지 않고 평균한다.
      원본과 해상도가 같으면 그대로 둔다.
    - src_fps < spec.fps − 1 이면 duplicate_schedule 로 프레임을 복제해 spec.fps 로 맞춘다.
    """
    convert = src_fps is not None and spec.fps - src_fps > 1.0
    idx = duplicate_schedule(spec.frames, src_fps, spec.fps) if convert else np.arange(spec.frames)
    need = int(idx[-1]) + 1
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"영상을 열 수 없습니다: {path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    src = []
    for _ in range(need):
        ok, f = cap.read()
        if not ok:
            break
        if f.shape[1] != spec.width or f.shape[0] != spec.height:
            f = cv2.resize(f, (spec.width, spec.height), interpolation=cv2.INTER_AREA)
        src.append(f)
    cap.release()
    if len(src) < need:
        return None
    return np.stack([src[i] for i in idx])


def is_clean(clip: np.ndarray, fps: float) -> tuple[bool, dict]:
    """바탕으로 쓸 수 있는지 판정하고, 이유를 함께 돌려준다."""
    r = measure.analyze(clip, fps)
    mad = np.abs(np.diff(clip.astype(np.int16), axis=0)).mean(axis=(1, 2, 3))
    info = {"screen_transitions": int(len(r["events"])), "mean_cd": float(r["mean_cd"].mean()),
            "max_frame_diff": float(mad.max()), "peak_area": r["peak_area"]}
    ok = (info["screen_transitions"] == 0 and info["mean_cd"] >= MIN_MEAN_CD
          and info["max_frame_diff"] < SCENE_CUT_MAD)
    return ok, info


def scan_clean_clips(path: Path, spec: ClipSpec, *, start_sec: float, end_sec: float,
                     step_sec: float, limit: int) -> list[dict]:
    """[start_sec, end_sec) 구간에서 step_sec 간격으로 후보를 읽어, 깨끗한 클립을 최대 limit 개 모은다.

    클립끼리 겹치지 않도록 step_sec ≥ 클립 길이로 쓰는 것을 권장한다.
    """
    cap = cv2.VideoCapture(str(path))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or spec.fps
    cap.release()
    # 허용 오차 1 fps: Elephants Dream 은 컨테이너가 23.39 fps 로 보고하지만 명목 24 fps 영화다.
    # 판정의 1초 창(W = round(fps))이 23 ↔ 24 프레임으로 4% 달라질 뿐이라 spec.fps(24)를 그대로 쓴다.
    # spec.fps 가 더 크면(24→30) 프레임 복제로 올린다. 더 작으면(프레임 버리기) 하지 않는다.
    if src_fps - spec.fps > 1.0:
        raise ValueError(f"원본 fps {src_fps} 가 spec.fps {spec.fps} 보다 큽니다. 프레임을 버리는 변환은 하지 않습니다.")
    kept, sec = [], start_sec
    while sec < end_sec and len(kept) < limit:
        start = int(round(sec * src_fps))
        clip = read_clip(path, start, spec, src_fps)
        if clip is None:
            break
        ok, info = is_clean(clip, spec.fps)
        if ok:
            kept.append({"clip": clip, "source": path.name, "start_frame": start, "check": info})
        sec += step_sec
    return kept


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def save_sample(folder: Path, sample_id: str, arrays: dict, meta: dict) -> dict:
    """배열은 npz(압축, 무손실), 메타데이터는 manifest 한 줄(JSON)로 저장한다. 영상 파일로 만들지 않는다
    (위험 자극을 실수로 재생하지 않도록 — 영상이 필요하면 FFV1 무손실로 따로 만든다)."""
    folder.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(folder / f"{sample_id}.npz", **arrays)
    row = {"id": sample_id, "file": f"{folder.name}/{sample_id}.npz", **meta}
    return json.loads(json.dumps(row, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
