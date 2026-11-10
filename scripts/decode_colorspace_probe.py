"""OpenCV(cv2.VideoCapture, FFmpeg 백엔드)가 YUV → BGR 변환에서 range(limited/full)와 행렬(BT.601/709)을
어떻게 처리하는지 실험하고, data/ 아래 원본 영상의 색 메타데이터를 출력한다. 결과: docs/research/opencv_decode.md

왜: 우리 판정기(measure.luminance_cd)는 cv2 가 돌려준 8비트 BGR 을 sRGB 로 보고 휘도를 계산한다.
    cv2 가 range·행렬을 잘못 고르면 휘도·적색 판정이 조용히 틀린다. 코드에는 range·행렬 처리 부분이 없다.

방법 (모두 정지 화면, 깜빡임 없음 — 위험 자극 아님)
  1) range: Y = 0,16,64,128,235,255 회색 띠를 limited / full 표시로 저장(Y4M 태그, H.264 color_range) → cv2 로 읽기
  2) 행렬: 같은 YUV 색 하나를 BT.601 / BT.709 표시로 H.264 저장 → cv2 로 읽어 이론값과 비교
  3) 영향: RGB 격자 17³ 을 BT.709 로 인코딩 → BT.601 로 디코드했을 때 휘도(sRGB×200 cd/m²)·포화 적색 판정 변화
  4) data/ 아래 영상의 color_range·matrix 메타데이터 (PyAV)

실행: .venv\\Scripts\\python.exe scripts/decode_colorspace_probe.py   (PyAV 필요: pip install av)
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import av
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RANGE = {0: "unspecified", 1: "tv(limited)", 2: "pc(full)"}
MATRIX = {1: "bt709", 2: "unspecified", 5: "bt470bg(601)", 6: "smpte170m(601)", 9: "bt2020nc"}
YS = [0, 16, 64, 128, 235, 255]


def _bands(w, h):
    y = np.zeros((h, w), np.uint8)
    bw = w // len(YS)
    for i, v in enumerate(YS):
        y[:, i * bw:(i + 1) * bw] = v
    return y, bw


def _cv2_g(path, h, bw):
    cap = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG)
    ok, f = cap.read()
    cap.release()
    return [int(f[h // 2, i * bw + bw // 2, 1]) for i in range(len(YS))] if ok else None


def _meta(path):
    with av.open(str(path)) as c:
        st = c.streams.video[0]
        fr = next(c.decode(st))
        return (f"{st.codec_context.name} {fr.width}x{fr.height} pix={st.codec_context.pix_fmt} "
                f"range={RANGE.get(int(fr.color_range), fr.color_range)} matrix={MATRIX.get(int(fr.colorspace), fr.colorspace)}")


def _h264(path, w, h, planes, color_range=None, matrix=None):
    with av.open(str(path), "w") as c:
        st = c.add_stream("libx264", rate=24)
        st.width, st.height, st.pix_fmt = w, h, "yuv420p"
        opts = {"qp": "0"}
        if matrix:
            opts["x264-params"] = f"colormatrix={matrix}"
        st.options = opts
        if color_range:
            st.codec_context.color_range = color_range
        for _ in range(3):
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


def probe_range(tmp: Path):
    print("== 1) range ==")
    w, h = 96, 32
    y, bw = _bands(w, h)
    uv = np.full((h // 2, w // 2), 128, np.uint8)
    for name, tag in [("y4m_none", ""), ("y4m_limited", "XCOLORRANGE=LIMITED"), ("y4m_full", "XCOLORRANGE=FULL")]:
        p = tmp / f"{name}.y4m"
        with open(p, "wb") as f:
            f.write(f"YUV4MPEG2 W{w} H{h} F24:1 Ip A1:1 C420jpeg{(' ' + tag) if tag else ''}\n".encode())
            for _ in range(3):
                f.write(b"FRAME\n" + y.tobytes() + uv.tobytes() + uv.tobytes())
        print(f"  {name:12s} [{_meta(p)}]  Y{YS} -> cv2 G{_cv2_g(p, h, bw)}")
    for name, rng in [("h264_limited", 1), ("h264_full", 2)]:
        p = tmp / f"{name}.mp4"
        _h264(p, w, h, [y, uv, uv], color_range=rng)
        print(f"  {name:12s} [{_meta(p)}]  Y{YS} -> cv2 G{_cv2_g(p, h, bw)}")
    print("  limited 해석 이론값 G", [int(np.clip(round((v - 16) * 255 / 219), 0, 255)) for v in YS], "/ full 해석 이론값 G = Y")


def _dec(Y, U, V, kr, kb):
    kg = 1 - kr - kb
    y = (np.asarray(Y, float) - 16) / 219
    cb = (np.asarray(U, float) - 128) / 224
    cr = (np.asarray(V, float) - 128) / 224
    r = y + 2 * (1 - kr) * cr
    b = y + 2 * (1 - kb) * cb
    g = (y - kr * r - kb * b) / kg
    return np.clip(np.round(np.stack([r, g, b], -1) * 255), 0, 255) / 255


def _enc(rgb, kr, kb):
    kg = 1 - kr - kb
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    y = kr * r + kg * g + kb * b
    return (np.round(16 + 219 * y), np.round(128 + 224 * (b - y) / (2 * (1 - kb))),
            np.round(128 + 224 * (r - y) / (2 * (1 - kr))))


def probe_matrix(tmp: Path):
    print("== 2) 행렬 ==")
    w = h = 32
    Y, U, V = 120, 90, 180
    t601 = (_dec(Y, U, V, 0.299, 0.114) * 255).round().astype(int).tolist()
    t709 = (_dec(Y, U, V, 0.2126, 0.0722) * 255).round().astype(int).tolist()
    print(f"  YUV=({Y},{U},{V}) 이론 RGB: BT.601 {t601}, BT.709 {t709}")
    for m in ("smpte170m", "bt709"):
        p = tmp / f"mat_{m}.mp4"
        _h264(p, w, h, [np.full((h, w), Y, np.uint8), np.full((h // 2, w // 2), U, np.uint8),
                        np.full((h // 2, w // 2), V, np.uint8)], matrix=m)
        cap = cv2.VideoCapture(str(p), cv2.CAP_FFMPEG)
        ok, f = cap.read()
        cap.release()
        b, g, r = (int(x) for x in f[h // 2, w // 2])
        print(f"  표시 {m:10s} [{_meta(p)}] -> cv2 RGB [{r}, {g}, {b}]")


def impact():
    print("== 3) BT.709 영상을 BT.601 로 디코드할 때의 영향 (RGB 격자 17^3) ==")
    g = np.linspace(0, 1, 17)
    rgb = np.stack(np.meshgrid(g, g, g, indexing="ij"), -1).reshape(-1, 3)
    Y, U, V = _enc(rgb, 0.2126, 0.0722)
    right, wrong = _dec(Y, U, V, 0.2126, 0.0722), _dec(Y, U, V, 0.299, 0.114)

    def lum(x):
        lin = np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)
        return 200 * (lin @ np.array([0.2126, 0.7152, 0.0722])), lin
    Lr, linr = lum(right)
    Lw, linw = lum(wrong)
    d = Lw - Lr
    gray = np.ptp(rgb, axis=1) == 0
    print(f"  휘도 차 cd/m²: 최대 {d.max():+.2f}, 최소 {d.min():+.2f}, 평균 |차| {np.abs(d).mean():.2f}, 무채색 최대 |차| {np.abs(d[gray]).max():.2f}")
    for name, c in [("순적색", [1, 0, 0]), ("순녹색", [0, 1, 0]), ("순청색", [0, 0, 1]), ("흰색", [1, 1, 1])]:
        i = int(np.argmin(np.abs(rgb - c).sum(1)))
        print(f"    {name}: 올바른 디코드 {Lr[i]:6.1f}, cv2 {Lw[i]:6.1f} cd/m²")

    def sat(lin):
        s = lin.sum(1)
        return (s > 0) & (lin[:, 0] >= 0.8 * s)
    print(f"  포화 적색(선형 R/(R+G+B) ≥ 0.8) 판정이 바뀌는 색: {int((sat(linr) != sat(linw)).sum())}/{len(rgb)}")


def sources():
    print("== 4) data/ 아래 영상 메타데이터 ==")
    data = ROOT / "data"
    vids = sorted(p for p in data.rglob("*") if p.suffix.lower() in {".mov", ".mp4", ".m4v", ".mkv", ".avi", ".webm"})
    if not vids:
        print("  (영상 없음)")
    for p in vids:
        try:
            print(f"  {p.relative_to(data)}: {_meta(p)}")
        except Exception as e:  # noqa: BLE001 — 메타데이터 출력용
            print(f"  {p.relative_to(data)}: 읽기 실패 {e}")


if __name__ == "__main__":
    print("cv2", cv2.__version__, "/ av", av.__version__)
    with tempfile.TemporaryDirectory() as d:
        probe_range(Path(d))
        probe_matrix(Path(d))
    impact()
    sources()
