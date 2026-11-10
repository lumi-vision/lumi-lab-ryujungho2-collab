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
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from lumi_synth import measure

MIN_MEAN_CD = 20.0          # 너무 어두운 장면 제외 (cd/m²)
SCENE_CUT_MAD = 40.0        # 연속 프레임 평균 절대 차이(0~255)가 이보다 크면 장면 전환으로 봄


@dataclass(frozen=True)
class ClipSpec:
    """클립 규격 = '트랙'. 판정 해상도 = 학습 해상도 = 768×432 로 통일(사용자 결정 2026-11-09, docs/RESOLUTION_ENV_REPORT.md).

    - 기본(통일 트랙): 768×432, 24 fps, 48프레임(2초). 768×432 는 16:9, 16 으로 나누어지고(생성기 축소 단계),
      시청 조건 56 cm·15인치에서 1도당 23.3 픽셀 → 최고 11.6 cpd 라 줄무늬 1~10 cpd 를 표현·판정할 수 있다.
    - 128×72 (예비 실험, 2026-11-09 까지): 깜빡임 전용. 줄무늬는 판정 불가(checks/rules/resolution_sensitivity.md).
    """
    width: int = 768
    height: int = 432
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


def content_box(src_w: int, src_h: int, spec: ClipSpec) -> tuple[int, int, int, int]:
    """원본(src_w×src_h)을 비율 그대로 spec 캔버스(16:9 화면) 안에 넣을 때의 영상 영역 (x0, y0, w, h).

    실제 플레이어처럼 '긴 쪽을 화면에 맞추고 남는 쪽은 검은 여백'으로 둔다(레터박스·필러박스).
      배율 s = min(W/src_w, H/src_h), 영상 크기 = round(src·s), 가운데 정렬.
    s > 1 (원본이 작아 키워야 함)이면 ValueError — 확대 금지.
    16:9 원본은 여백 0 (반올림으로 1픽셀 이하 차이는 꽉 채움).
    """
    s = min(spec.width / src_w, spec.height / src_h)
    if s > 1.0 + 1e-9:
        raise ValueError(f"원본 {src_w}×{src_h} 를 {spec.width}×{spec.height} 화면에 넣으려면 {s:.2f}배 확대해야 합니다(확대 금지)")
    w = min(spec.width, int(round(src_w * s)))
    h = min(spec.height, int(round(src_h * s)))
    if spec.width - w <= 1:
        w = spec.width
    if spec.height - h <= 1:
        h = spec.height
    return (spec.width - w) // 2, (spec.height - h) // 2, w, h


def letterbox(frame_bgr: np.ndarray, spec: ClipSpec) -> np.ndarray:
    """프레임 한 장을 비율을 지켜 spec 캔버스에 넣는다(남는 곳은 검정 0).

    왜 (2026-11-09, 팀원 판정기 점검에서 발견):
      - 판정 규칙의 면적은 '화면에 표시된 면적' 대비다(ITU). 세로 9:16 영상은 16:9 화면의 32%만 차지하므로
        영상 프레임 기준 80% 깜빡임 = 화면 기준 약 26%. 여백을 넣으면 면적이 자동으로 화면 기준이 된다.
      - 줄무늬 cpd 의 1도당 픽셀(measure.pixels_per_degree)은 16:9 화면을 가정한다. 비율을 무시하고 늘리면
        가로·세로 1도당 픽셀이 달라진다(세로 영상: 가로 12.3 vs 실제 37.8 — cpd 3배 오차).
      - 검은 여백은 밝기가 변하지 않으므로 깜빡임 전이가 생기지 않는다.
    """
    x0, y0, w, h = content_box(frame_bgr.shape[1], frame_bgr.shape[0], spec)
    if (w, h) == (spec.width, spec.height):
        if frame_bgr.shape[1] == w and frame_bgr.shape[0] == h:
            return frame_bgr
        return cv2.resize(frame_bgr, (w, h), interpolation=cv2.INTER_AREA)
    out = np.zeros((spec.height, spec.width, 3), np.uint8)
    out[y0:y0 + h, x0:x0 + w] = cv2.resize(frame_bgr, (w, h), interpolation=cv2.INTER_AREA)
    return out


# =============================================================================================
# 영상 디코드: YCbCr → BGR 변환의 행렬(BT.601/709/2020)과 range(limited/full)를 영상 표시대로 (2026-11-10)
#
# 왜: cv2.VideoCapture(FFmpeg 백엔드)는 영상의 color_range·colorspace 표시를 보지 않는다.
#     - 행렬은 항상 BT.601, range 는 픽셀 형식이 yuvj* 일 때만 full, 그 밖에는 항상 limited 로 펼친다.
#     - 그래서 BT.709 로 저장된 HD 영상은 유채색 휘도가 최대 +39/−15 cd/m² 틀어지고(순적색 42.6 → 34.7),
#       포화 적색·Δu′v′ 판정도 바뀐다. 측정: docs/research/opencv_decode.md, scripts/decode_colorspace_probe.py
# 어떻게: 영상을 열 때 PyAV 로 표시를 읽어 '읽기 경로'를 영상 단위로 고른다.
#     - cv2 가 맞게 읽는 영상(RGB 계열, 또는 BT.601 + cv2 가 고를 range 와 같은 range)은 지금처럼 cv2 로 읽는다
#       → 이미 검증한 결과(pse-test-media RGB, BT.601 SD 원본, 기존 오라클 데이터)가 비트 단위로 그대로다.
#     - 그 밖(BT.709, BT.2020, full range 인데 yuvj 가 아님 등)은 PyAV 로 읽고 행렬·range 를 명시해 변환한다.
#     - 한 영상은 처음부터 끝까지 한 경로로만 읽는다(두 라이브러리의 프레임 번호 계산이 다를 수 있음).
# 표시가 없을 때 (mpv video/csputils.c mp_csp_guess_colorspace, libplacebo pl_color_system_guess_ycbcr 와 같은 규칙)
#     - 행렬: 가로 ≥ 1280 또는 세로 > 576 이면 BT.709, 아니면 BT.601
#     - range: YCbCr 은 limited (libplacebo pl_color_levels_guess, mpv 도 unknown → limited). yuvj* 는 full.
# =============================================================================================
_RGB_PIX_PREFIXES = ("rgb", "bgr", "gbr", "argb", "abgr", "0rgb", "0bgr", "rgba", "bgra", "pal8")
_GRAY_PIX_PREFIXES = ("gray", "ya")
# AVColorSpace → 행렬. 0(RGB 항등)·2(미지정)는 '표시 없음'으로 보고 해상도 규칙을 쓴다.
_MATRIX_BY_AVCOL = {1: "bt709", 5: "bt601", 6: "bt601", 9: "bt2020", 10: "bt2020", 7: "smpte240m", 4: "fcc"}
_UNSUPPORTED_AVCOL = {8: "YCgCo", 11: "SMPTE2085", 12: "chroma-derived NCL", 13: "chroma-derived CL", 14: "ICtCp"}
# AVColorTransferCharacteristic 중 HDR: 판정 규칙(sRGB EOTF, 최대 200 cd/m²)으로 잴 수 없다.
_HDR_TRC = {16: "PQ (SMPTE ST 2084)", 18: "HLG (ARIB STD-B67)"}


@dataclass(frozen=True)
class DecodeInfo:
    """영상 하나의 색 표시와 읽기 경로. manifest 에 그대로 남긴다(as_dict)."""
    pix_fmt: str
    width: int
    height: int
    matrix: str            # 'bt601' | 'bt709' | 'bt2020' | 'smpte240m' | 'fcc' | 'rgb' | 'gray'
    color_range: str       # 'limited' | 'full'
    matrix_tagged: bool    # False 면 해상도 규칙으로 추정한 값
    range_tagged: bool
    color_trc: int         # AVColorTransferCharacteristic 값 (기록용, HDR 이면 decode_info 가 거부)
    route: str             # 'opencv' | 'pyav'

    def as_dict(self) -> dict:
        return dict(self.__dict__)


def decode_info(path: Path) -> DecodeInfo:
    """PyAV 로 첫 프레임의 색 표시를 읽고 읽기 경로를 정한다(경로·수정 시각·크기 기준으로 캐시).

    HDR(PQ/HLG) 전달 특성이나 지원하지 않는 행렬(YCgCo, ICtCp 등)이면 ValueError — 조용히 틀리게 재지 않는다.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"영상을 열 수 없습니다: {p}")
    s = p.stat()
    return _decode_info_cached(str(p.resolve()), s.st_mtime_ns, s.st_size)


@lru_cache(maxsize=64)
def _decode_info_cached(path: str, _mtime: int, _size: int) -> DecodeInfo:
    import av   # 지연 import

    with av.open(path) as c:
        st = c.streams.video[0]
        fr = next(c.decode(st), None)
        if fr is None:
            raise ValueError(f"영상에 디코드할 프레임이 없습니다: {path}")
        pix = fr.format.name
        w, h = fr.width, fr.height
        cs, cr = int(fr.colorspace), int(fr.color_range)
        trc = int(getattr(fr, "color_trc", 2) or 2)
    if trc in _HDR_TRC:
        raise ValueError(f"HDR 영상({_HDR_TRC[trc]})은 지원하지 않습니다: 판정 규칙은 SDR(sRGB, 최대 200 cd/m²) 기준. {path}")
    if cs in _UNSUPPORTED_AVCOL and not pix.startswith(_RGB_PIX_PREFIXES + _GRAY_PIX_PREFIXES):
        raise ValueError(f"지원하지 않는 색 행렬({_UNSUPPORTED_AVCOL[cs]})입니다: {path}")
    range_tagged = cr in (1, 2)
    if pix.startswith(_RGB_PIX_PREFIXES):
        return DecodeInfo(pix, w, h, "rgb", "full", True, range_tagged, trc, "opencv")
    if pix.startswith(_GRAY_PIX_PREFIXES):          # 색차 없음 → 행렬 무관. 두 경로 결과가 같음(리뷰에서 확인)
        return DecodeInfo(pix, w, h, "gray", "limited" if cr == 1 else "full", True, range_tagged, trc, "opencv")
    matrix_tagged = cs in _MATRIX_BY_AVCOL
    matrix = _MATRIX_BY_AVCOL[cs] if matrix_tagged else ("bt709" if (w >= 1280 or h > 576) else "bt601")
    yuvj = pix.startswith("yuvj")
    color_range = "full" if (cr == 2 or (not range_tagged and yuvj)) else "limited"
    cv2_range = "full" if yuvj else "limited"            # cv2 가 실제로 쓰는 range (실험으로 확인)
    route = "opencv" if (matrix == "bt601" and color_range == cv2_range) else "pyav"
    return DecodeInfo(pix, w, h, matrix, color_range, matrix_tagged, range_tagged, trc, route)


def iter_frames_bgr(path: Path, start_frame: int = 0, count: int | None = None, *, route: str | None = None):
    """영상에서 BGR uint8 프레임을 start_frame 부터 count 장(None 이면 끝까지) 차례로 돌려준다.

    route 를 주지 않으면 decode_info() 가 정한 경로를 쓴다('opencv'/'pyav' 강제는 테스트용).
    프레임 번호의 뜻: PyAV 경로는 '파일 처음부터 표시 순서로 센 번호'(간격이 고르지 않은 파일에서도 정확, 테스트로 확인).
      cv2 경로는 cv2 의 CAP_PROP_POS_FRAMES 를 그대로 쓴다. cv2 는 시각 × 평균 fps 로 번호를 정하므로 간격이 고르지
      않은 파일(Elephants Dream)에서는 차례로 센 번호와 1프레임 안팎 다를 수 있다 — 기존 데이터와 같은 결과를 지키려고
      바꾸지 않음(리뷰 2026-11-10).
    """
    info = decode_info(path)
    route = route or info.route
    if route == "opencv":
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise FileNotFoundError(f"영상을 열 수 없습니다: {path}")
        try:
            if start_frame:
                cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            n = 0
            while count is None or n < count:
                ok, f = cap.read()
                if not ok:
                    return
                yield f
                n += 1
        finally:
            cap.release()
        return
    yield from _iter_pyav(path, start_frame, count, info)


def _iter_pyav(path: Path, start_frame: int, count: int | None, info: DecodeInfo):
    """PyAV 경로. 프레임 번호 = '파일 처음부터 표시 순서로 센 번호'.

    시각(pts) × fps 로 번호를 계산하지 않는다: 실제 파일은 프레임 간격이 고르지 않을 수 있다(Elephants Dream:
    첫 간격 11프레임, 1초 공백 등 35곳 — 평균 fps 23.39 vs 공칭 24). 대신 패킷 pts 를 정렬한 표
    (n번째 프레임의 pts, _frame_pts_table)로 목표 pts 를 찾고, 그 앞 키프레임으로 이동한 뒤 목표 pts 까지 건너뛴다.
    착지한 프레임이 목표보다 뒤면(이동이 넘침) 더 앞으로 다시 이동하고, 그래도 안 되거나 pts 표를 만들 수 없으면
    처음부터 센다.
    """
    import av
    from av.video.reformatter import ColorRange, Colorspace

    src_cs = {"bt601": Colorspace.ITU601, "bt709": Colorspace.ITU709, "bt2020": Colorspace.BT2020,
              "smpte240m": Colorspace.SMPTE240M, "fcc": Colorspace.FCC}.get(info.matrix)
    src_cr = ColorRange.JPEG if info.color_range == "full" else ColorRange.MPEG

    def convert(fr):
        if src_cs is None:                                   # RGB·회색: 행렬 변환 없음
            return fr.to_ndarray(format="bgr24")
        return fr.reformat(format="bgr24", src_colorspace=src_cs, src_color_range=src_cr,
                           dst_color_range=ColorRange.JPEG).to_ndarray()

    table = _frame_pts_table(path) if start_frame > 0 else None
    if table is not None and start_frame >= len(table):
        return
    with av.open(str(path)) as c:
        st = c.streams.video[0]
        st.thread_type = "AUTO"
        frames = None
        sought = False
        if table is not None:
            target = int(table[start_frame])
            seek_to = target
            for step in range(6):                            # 넘치면 더 앞 표 위치로 (24·2^k 프레임 앞)
                c.seek(seek_to, stream=st, backward=True, any_frame=False)
                sought = True
                dec = c.decode(st)
                land = next(dec, None)
                if land is None or land.pts is None:
                    break
                if land.pts <= target:
                    frames = _skip_until_pts(land, dec, target)
                    break
                seek_to = int(table[max(0, start_frame - 24 * (2 ** step))])
            if frames is not None:
                first = next(frames, None)                   # 목표 pts 와 정확히 같은 프레임이어야 한다
                frames = _chain(first, frames) if (first is not None and first.pts == target) else None
        if frames is None:                                   # 처음부터 센다
            if sought:
                c.seek(0, stream=st, backward=True, any_frame=False)
            frames = _skip_count(c.decode(st), start_frame)
        n = 0
        for fr in frames:
            yield convert(fr)
            n += 1
            if count is not None and n >= count:
                return


def _skip_until_pts(first, rest, target):
    """first, rest… 중 pts ≥ target 인 첫 프레임부터 돌려준다. 목표 전의 pts 없는 프레임은 건너뛴다
    (그 첫 프레임의 pts 가 target 과 다르면 호출한 쪽이 '처음부터 세기'로 바꾼다)."""
    started = False
    for fr in _chain(first, rest):
        if not started:
            if fr.pts is None or fr.pts < target:
                continue
            started = True
        yield fr


def _skip_count(frames, k):
    for i, fr in enumerate(frames):
        if i >= k:
            yield fr


def _chain(first, rest):
    yield first
    yield from rest


def _frame_pts_table(path: Path):
    """n번째(표시 순서) 프레임의 pts 배열. 패킷만 읽고 디코드하지 않는다.

    디코더가 내보내지 않는 패킷(MP4/MOV 편집 목록으로 잘린 앞부분, is_discard)은 뺀다 — 리뷰 2026-11-10:
    빼지 않으면 잘린 개수만큼 앞 프레임으로 이동했다. pts 없는 패킷이나 같은 pts 가 둘 이상이면 None(처음부터 세기).
    """
    p = Path(path)
    s = p.stat()
    return _frame_pts_table_cached(str(p.resolve()), s.st_mtime_ns, s.st_size)


@lru_cache(maxsize=16)
def _frame_pts_table_cached(path: str, _mtime: int, _size: int):
    import av

    pts = []
    with av.open(path) as c:
        st = c.streams.video[0]
        for pkt in c.demux(st):
            if pkt.size == 0 or pkt.is_discard:               # 끝 표시용 빈 패킷, 편집 목록으로 버려지는 패킷
                continue
            if pkt.pts is None:
                return None
            pts.append(pkt.pts)
    table = np.sort(np.asarray(pts, dtype=np.int64))
    if len(table) > 1 and (np.diff(table) == 0).any():
        return None
    return table


def read_clip(path: Path, start_frame: int, spec: ClipSpec, src_fps: float | None = None) -> np.ndarray | None:
    """start_frame 부터 읽어 spec 규격 (T,H,W,3) uint8 BGR 로. 모자라면 None.

    - 디코드: 영상의 행렬·range 표시대로 변환(iter_frames_bgr, 위 설명). BT.601 SD·RGB 영상은 이전과 같은 cv2 경로.
    - 비율 유지 + 검은 여백(letterbox): 16:9 가 아닌 영상도 화면에 표시되는 모습 그대로 넣는다.
    - 축소는 INTER_AREA(면적 평균): 깜빡임 판정은 면적 비율이 중요하므로 픽셀을 건너뛰지 않고 평균한다.
      원본과 해상도가 같으면 그대로 둔다. 확대가 필요하면 ValueError(확대 금지).
    - src_fps < spec.fps − 1 이면 duplicate_schedule 로 프레임을 복제해 spec.fps 로 맞춘다.
    한계: 여백이 있는 클립에 오라클이 마스크를 놓으면 여백에도 빛이 더해질 수 있다(현재 원본은 모두 16:9 라 여백 없음).
    """
    convert = src_fps is not None and spec.fps - src_fps > 1.0
    idx = duplicate_schedule(spec.frames, src_fps, spec.fps) if convert else np.arange(spec.frames)
    need = int(idx[-1]) + 1
    src = []
    for f in iter_frames_bgr(path, start_frame, need):
        try:
            # 확대 금지: 원본보다 크게 만들면 없는 세부(가는 줄·작은 점)를 지어내지 못하고 흐려진 영상만 생긴다.
            # (2026-11-09: 768×432 통일 때 시험 분할 Elephants Dream 이 480×270 이라 조용히 확대될 뻔함)
            f = letterbox(f, spec)
        except ValueError as e:
            raise ValueError(f"{e}: {path}") from None
        src.append(f)
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
    decode = decode_info(path).as_dict()       # 어떤 행렬·range·경로로 읽었는지 manifest 에 남긴다
    kept, sec = [], start_sec
    while sec < end_sec and len(kept) < limit:
        start = int(round(sec * src_fps))
        clip = read_clip(path, start, spec, src_fps)
        if clip is None:
            break
        ok, info = is_clean(clip, spec.fps)
        if ok:
            kept.append({"clip": clip, "source": path.name, "start_frame": start, "check": info, "decode": decode})
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
