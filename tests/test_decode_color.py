"""영상 디코드의 색 행렬·range 처리 (data.decode_info / iter_frames_bgr / read_clip), 2026-11-10.

배경: cv2.VideoCapture 는 영상의 BT.709·full range 표시를 무시하고 항상 BT.601·limited 로 변환한다
(docs/research/opencv_decode.md). 표시대로 읽는지, 그리고 cv2 가 맞게 읽던 영상은 이전과 비트 단위로 같은지 확인한다.
테스트 영상은 모두 정지 회색 또는 단색이며, 프레임 번호 확인용 영상만 프레임마다 회색 밝기가 조금씩 올라간다(깜빡임 없음).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

av = pytest.importorskip("av")
cv2 = pytest.importorskip("cv2")

from lumi_synth.data import ClipSpec, decode_info, iter_frames_bgr, read_clip  # noqa: E402

YUV_COLOR = (120, 90, 180)          # limited range Y, Cb, Cr — 채도 높은 주황


def _theory_rgb(kr: float, kb: float) -> np.ndarray:
    y, u, v = YUV_COLOR
    yy, cb, cr = (y - 16) / 219, (u - 128) / 224, (v - 128) / 224
    r = yy + 2 * (1 - kr) * cr
    b = yy + 2 * (1 - kb) * cb
    g = (yy - kr * r - kb * b) / (1 - kr - kb)
    return np.clip(np.round(np.array([r, g, b]) * 255), 0, 255)


def _h264(path, w, h, frames_yuv, *, matrix=None, color_range=None, keyint=None):
    """frames_yuv: [(Y평면, U평면, V평면), …] 를 무손실(qp 0) H.264 로 저장."""
    with av.open(str(path), "w") as c:
        st = c.add_stream("libx264", rate=24)
        st.width, st.height, st.pix_fmt = w, h, "yuv420p"
        params = []
        if matrix:
            params.append(f"colormatrix={matrix}")
        if keyint:
            params.append(f"keyint={keyint}:min-keyint=1:scenecut=0")
        st.options = {"qp": "0", **({"x264-params": ":".join(params)} if params else {})}
        if color_range:
            st.codec_context.color_range = color_range
        for planes in frames_yuv:
            fr = av.VideoFrame(w, h, "yuv420p")
            for k, arr in enumerate(planes):
                ls = fr.planes[k].line_size
                fr.planes[k].update(np.pad(arr, ((0, 0), (0, ls - arr.shape[1]))).tobytes())
            if color_range:
                fr.color_range = color_range
            for pkt in st.encode(fr):
                c.mux(pkt)
        for pkt in st.encode():
            c.mux(pkt)
    return path


def _solid(w, h, n, yuv=YUV_COLOR):
    y, u, v = yuv
    return [(np.full((h, w), y, np.uint8), np.full((h // 2, w // 2), u, np.uint8),
             np.full((h // 2, w // 2), v, np.uint8))] * n


def _cv2_first(path):
    cap = cv2.VideoCapture(str(path))
    ok, f = cap.read()
    cap.release()
    assert ok
    return f


def test_bt709_tagged_video_decodes_with_bt709(tmp_path):
    p = _h264(tmp_path / "bt709.mp4", 32, 32, _solid(32, 32, 3), matrix="bt709")
    info = decode_info(p)
    assert (info.matrix, info.matrix_tagged, info.color_range, info.route) == ("bt709", True, "limited", "pyav")
    rgb = next(iter_frames_bgr(p))[16, 16][::-1].astype(int)
    assert np.abs(rgb - _theory_rgb(0.2126, 0.0722)).max() <= 2          # BT.709 이론값 (214, 101, 41)
    cv2_rgb = _cv2_first(p)[16, 16][::-1].astype(int)
    assert np.abs(cv2_rgb - _theory_rgb(0.299, 0.114)).max() <= 2       # cv2 는 BT.601 로 읽음 (문제의 재현)


def test_full_range_tagged_rawvideo_keeps_levels(tmp_path):
    """full range 표시(Y4M XCOLORRANGE=FULL)인데 픽셀 형식이 yuv420p → cv2 는 limited 로 펼치는 경우."""
    w, h, ys = 96, 32, [0, 16, 64, 128, 235, 255]
    bw = w // len(ys)
    y = np.zeros((h, w), np.uint8)
    for i, v in enumerate(ys):
        y[:, i * bw:(i + 1) * bw] = v
    uv = np.full((h // 2, w // 2), 128, np.uint8)
    p = tmp_path / "full.y4m"
    with open(p, "wb") as f:
        f.write(f"YUV4MPEG2 W{w} H{h} F24:1 Ip A1:1 C420jpeg XCOLORRANGE=FULL\n".encode())
        for _ in range(3):
            f.write(b"FRAME\n" + y.tobytes() + uv.tobytes() + uv.tobytes())
    info = decode_info(p)
    assert (info.color_range, info.range_tagged, info.route) == ("full", True, "pyav")
    g = next(iter_frames_bgr(p))[h // 2, [i * bw + bw // 2 for i in range(len(ys))], 1].astype(int)
    assert np.abs(g - ys).max() <= 1                                     # full: G = Y
    g_cv2 = _cv2_first(p)[h // 2, [i * bw + bw // 2 for i in range(len(ys))], 1].astype(int)
    assert g_cv2[1] == 0 and g_cv2[4] == 255                             # cv2: 16→0, 235→255 (limited 로 펼침)


def test_bt601_sd_video_keeps_opencv_path_bit_exact(tmp_path):
    """이미 검증한 결과가 그대로인지: BT.601 limited SD 영상은 cv2 경로, 결과가 cv2 직접 읽기와 비트 단위로 같다."""
    p = _h264(tmp_path / "bt601.mp4", 64, 36, _solid(64, 36, 4), matrix="smpte170m")
    info = decode_info(p)
    assert (info.matrix, info.route) == ("bt601", "opencv")
    assert np.array_equal(next(iter_frames_bgr(p)), _cv2_first(p))
    clip = read_clip(p, 0, ClipSpec(64, 36, 4, 24.0))
    assert clip.shape == (4, 36, 64, 3) and np.array_equal(clip[0], _cv2_first(p))


def test_untagged_matrix_guessed_by_resolution(tmp_path):
    """표시가 없으면 mpv·libplacebo 규칙: 가로 ≥ 1280 또는 세로 > 576 → BT.709, 아니면 BT.601."""
    sd = decode_info(_h264(tmp_path / "sd.mp4", 64, 36, _solid(64, 36, 2)))
    hd = decode_info(_h264(tmp_path / "hd.mp4", 1280, 720, _solid(1280, 720, 2)))
    assert (sd.matrix, sd.matrix_tagged, sd.route) == ("bt601", False, "opencv")
    assert (hd.matrix, hd.matrix_tagged, hd.route) == ("bt709", False, "pyav")
    rgb = next(iter_frames_bgr(tmp_path / "hd.mp4"))[360, 640][::-1].astype(int)
    assert np.abs(rgb - _theory_rgb(0.2126, 0.0722)).max() <= 2


@pytest.mark.parametrize("start", [0, 5, 9, 14])
def test_pyav_seek_returns_same_frame_index_as_opencv(tmp_path, start):
    """두 경로의 프레임 번호가 같은지: 프레임 i 의 회색 Y = 40 + 8i (무채색이라 행렬과 무관), 키프레임 간격 4."""
    w, h, n = 64, 36, 20
    frames = [(np.full((h, w), 40 + 8 * i, np.uint8), np.full((h // 2, w // 2), 128, np.uint8),
               np.full((h // 2, w // 2), 128, np.uint8)) for i in range(n)]
    p = _h264(tmp_path / "ramp.mp4", w, h, frames, keyint=4)
    a = np.stack(list(iter_frames_bgr(p, start, 3, route="opencv")))
    b = np.stack(list(iter_frames_bgr(p, start, 3, route="pyav")))
    assert a.shape == b.shape == (3, h, w, 3)
    assert np.abs(a.astype(int) - b.astype(int)).max() <= 1
    expected = np.clip(np.round((40 + 8 * (start + np.arange(3)) - 16) * 255 / 219), 0, 255)
    assert np.abs(b[:, 0, 0, 1].astype(int) - expected).max() <= 1


ED = Path(__file__).resolve().parents[1] / "data/raw/elephantsdream-480-h264-st-aac.mov"


@pytest.mark.skipif(not ED.exists(), reason="data/raw 에 Elephants Dream 없음")
def test_pyav_seek_matches_sequential_count_on_real_file():
    """평균 fps(23.39)와 공칭 fps(24)가 다른 실제 파일: 이동 후 프레임 = 처음부터 차례로 센 프레임 (리뷰 2026-11-10).
    (이전 구현은 평균 fps 로 번호를 매겨 10곳 중 8곳에서 1프레임 밀렸다.) 프레임 내용은 수치로만 비교한다."""
    starts = [100, 300, 497, 500, 503, 700, 900, 1000, 1100, 1250]
    seq = {}
    want = set(starts)
    for i, f in enumerate(iter_frames_bgr(ED, 0, max(starts) + 1, route="pyav")):
        if i in want:
            seq[i] = f
    for s in starts:
        got = next(iter_frames_bgr(ED, s, 1, route="pyav"))
        assert np.array_equal(got, seq[s]), f"start_frame={s}"


def test_pyav_seek_with_edit_list_discarded_packets(tmp_path):
    """앞 3프레임이 편집 목록으로 잘린 MP4(패킷 pts/dts 를 3프레임 앞당겨 다시 묶음): 디코더는 37장만 내보낸다.
    pts 표에 버려지는 패킷이 들어가면 이동이 3프레임 앞으로 떨어진다 (재리뷰 2026-11-10에서 재현된 HIGH)."""
    w, h, n = 64, 36, 40
    frames = [(np.full((h, w), 20 + 5 * i, np.uint8), np.full((h // 2, w // 2), 128, np.uint8),
               np.full((h // 2, w // 2), 128, np.uint8)) for i in range(n)]
    src = _h264(tmp_path / "ramp40.mp4", w, h, frames, matrix="bt709", keyint=8)
    dst = tmp_path / "shifted.mp4"
    with av.open(str(src)) as ic, av.open(str(dst), "w") as oc:
        ist = ic.streams.video[0]
        ost = oc.add_stream_from_template(ist)
        shift = None
        for pkt in ic.demux(ist):
            if pkt.size == 0:
                continue
            if shift is None:
                shift = 3 * int(round(1 / (24 * float(ist.time_base))))   # 3프레임
            pkt.pts -= shift
            pkt.dts -= shift
            pkt.stream = ost
            oc.mux(pkt)
    seq = list(iter_frames_bgr(dst, 0, None, route="pyav"))
    assert decode_info(dst).route == "pyav" and len(seq) < n
    for s in [1, 3, 9, 17, 25, len(seq) - 1]:
        got = next(iter_frames_bgr(dst, s, 1, route="pyav"))
        assert np.array_equal(got, seq[s]), f"start_frame={s}"
    assert list(iter_frames_bgr(dst, len(seq), 1, route="pyav")) == []


def test_hdr_transfer_is_rejected(tmp_path):
    p = tmp_path / "pq.mp4"
    with av.open(str(p), "w") as c:
        st = c.add_stream("libx264", rate=24)
        st.width, st.height, st.pix_fmt = 32, 32, "yuv420p"
        st.options = {"qp": "0", "x264-params": "colormatrix=bt2020nc:colorprim=bt2020:transfer=smpte2084"}
        for planes in _solid(32, 32, 2):
            fr = av.VideoFrame(32, 32, "yuv420p")
            for k, arr in enumerate(planes):
                ls = fr.planes[k].line_size
                fr.planes[k].update(np.pad(arr, ((0, 0), (0, ls - arr.shape[1]))).tobytes())
            for pkt in st.encode(fr):
                c.mux(pkt)
        for pkt in st.encode():
            c.mux(pkt)
    with pytest.raises(ValueError, match="HDR"):
        decode_info(p)


def test_bt601_full_range_yuvj_stays_on_opencv(tmp_path):
    """BT.601 + full range 가 yuvj 로 디코드되면 cv2 도 full 로 맞게 읽으므로 cv2 경로(이전과 같음)."""
    w, h = 64, 36
    y = np.full((h, w), 200, np.uint8)
    uv = np.full((h // 2, w // 2), 128, np.uint8)
    p = _h264(tmp_path / "full601.mp4", w, h, [(y, uv, uv)] * 2, matrix="smpte170m", color_range=2)
    info = decode_info(p)
    assert info.pix_fmt.startswith("yuvj") and (info.color_range, info.route) == ("full", "opencv")
    assert abs(int(next(iter_frames_bgr(p))[10, 10, 1]) - 200) <= 1


def test_bt709_decode_changes_luminance_verdict(tmp_path):
    """판정까지 확인: 검정 ↔ 빨강(190,0,0) 초당 4번 교대(BT.709 표시, 64×36 — 파일은 tmp, 수치로만).
    올바른 디코드는 ΔL ≈ 21 cd/m² 로 위험, cv2(BT.601) 디코드는 ≈ 17.5 로 20 미만이라 안전 → 위험을 놓친다."""
    w, h, n = 64, 36, 48

    def ycc(rgb):
        r, g, b = (np.array(rgb, float) / 255).tolist()
        y = 0.2126 * r + 0.7152 * g + 0.0722 * b
        Y, U, V = (round(16 + 219 * y), round(128 + 224 * (b - y) / 1.8556), round(128 + 224 * (r - y) / 1.5748))
        return (np.full((h, w), Y, np.uint8), np.full((h // 2, w // 2), U, np.uint8), np.full((h // 2, w // 2), V, np.uint8))
    black, red = ycc([0, 0, 0]), ycc([190, 0, 0])
    p = _h264(tmp_path / "flash709.mp4", w, h, [red if (i // 3) % 2 else black for i in range(n)], matrix="bt709")
    old = np.stack(list(iter_frames_bgr(p, route="opencv")))
    new = np.stack(list(iter_frames_bgr(p)))
    from lumi_synth import measure
    l_old = float(measure.luminance_cd(old[3:4])[0, 18, 32])
    l_new = float(measure.luminance_cd(new[3:4])[0, 18, 32])
    assert l_old < 20.0 <= l_new
    assert not measure.analyze(old, 24.0)["dangerous"]
    assert measure.analyze(new, 24.0)["dangerous"]


def test_iter_frames_count_and_end(tmp_path):
    p = _h264(tmp_path / "short.mp4", 32, 32, _solid(32, 32, 5), matrix="bt709")
    assert len(list(iter_frames_bgr(p))) == 5
    assert len(list(iter_frames_bgr(p, 3))) == 2
    assert len(list(iter_frames_bgr(p, 1, 2))) == 2
