"""0단계: 환경을 점검하고 원본 영상(Big Buck Bunny 1080p)을 준비한 뒤 checks/step0/README.md를 만든다.

하는 일
1. 파이썬과 패키지 버전 확인
2. 원본 영상이 없으면 Blender 공식 서버에서 zip을 받아 풀기 (약 725 MB)
3. 원본 영상의 SHA256이 기록해 둔 값과 같은지 확인
4. OpenCV로 원본 영상을 열 수 있는지 확인 (1920×1080이어야 함)
5. 무손실 저장(FFV1 .mkv)이 정말 무손실인지 확인: 무작위 1080p 프레임을 저장했다 다시 읽어 픽셀이 100% 같은지
6. 공간 패턴 재료(pse-test-media)가 있는지 확인

사용: python scripts\\step0_check.py
"""
import hashlib
import platform
import sys
import tempfile
import urllib.request
import zipfile
from importlib.metadata import version
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
VIDEO = ROOT / "data/sources/big_buck_bunny_1080p/big_buck_bunny_1080p_h264.mov"
VIDEO_URL = "https://download.blender.org/peach/bigbuckbunny_movies/big_buck_bunny_1080p_h264.mov.zip"
EXPECTED_SHA256 = "dc2146a2b1172def56730143ad80cd1825b7fad15f1fc9c23a4e7d01a741ac11"
PATTERNS = ROOT.parent / "tools/pse-test-media/area_patterns/25pct_screen"
TEMPORAL = ROOT.parent / "tools/pse-test-media/lum_flash_patterns/rel_lum"
OUT = ROOT / "checks/step0/README.md"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download_video() -> str:
    """원본 영상이 없을 때만 받는다. 무엇을 했는지 한 줄로 돌려준다."""
    if VIDEO.exists():
        return "이미 있음"
    VIDEO.parent.mkdir(parents=True, exist_ok=True)
    zpath = VIDEO.parent / (VIDEO.name + ".zip")
    print(f"원본 영상 받는 중: {VIDEO_URL}")
    urllib.request.urlretrieve(VIDEO_URL, zpath)
    with zipfile.ZipFile(zpath) as z:
        names = z.namelist()
        if names != [VIDEO.name]:
            raise RuntimeError(f"zip 안 파일이 예상과 다름: {names}")
        z.extract(VIDEO.name, VIDEO.parent)
    zpath.unlink()
    return "공식 서버에서 받아 풀었음"


def probe(path: Path) -> dict:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return {"ok": False}
    info = {
        "ok": True,
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
    ok, _ = cap.read()
    info["first_frame_read"] = ok
    cap.release()
    return info


def lossless_roundtrip() -> tuple[bool, str]:
    """무작위 1080p 프레임 5장을 FFV1로 저장하고 다시 읽어 완전히 같은지 본다."""
    rng = np.random.default_rng(0)
    frames = rng.integers(0, 256, size=(5, 1080, 1920, 3), dtype=np.uint8)
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "roundtrip.mkv"
        w = cv2.VideoWriter(str(p), cv2.VideoWriter_fourcc(*"FFV1"), 30, (1920, 1080))
        if not w.isOpened():
            return False, "FFV1 쓰기 장치를 열 수 없음"
        for f in frames:
            w.write(f)
        w.release()
        cap = cv2.VideoCapture(str(p))
        back = []
        while True:
            ok, f = cap.read()
            if not ok:
                break
            back.append(f)
        cap.release()
    if len(back) != len(frames):
        return False, f"프레임 수 다름: 저장 {len(frames)}, 읽음 {len(back)}"
    diff = int(np.abs(np.stack(back).astype(int) - frames.astype(int)).max())
    return diff == 0, f"1920×1080 프레임 {len(back)}장, 최대 픽셀 차이 {diff}"


def main() -> int:
    rows = []  # (항목, 결과, 통과 여부)

    rows.append(("파이썬", f"{platform.python_version()} ({sys.executable})", True))
    for pkg in ["numpy", "opencv-python", "matplotlib", "pytest", "lumi-synth"]:
        try:
            rows.append((f"패키지 {pkg}", version(pkg), True))
        except Exception:
            rows.append((f"패키지 {pkg}", "설치 안 됨", False))

    try:
        rows.append(("원본 영상 준비", download_video(), True))
    except Exception as e:
        rows.append(("원본 영상 준비", f"실패: {e}", False))

    if VIDEO.exists():
        got = sha256(VIDEO)
        rows.append(("원본 영상 SHA256", f"`{got[:16]}…` (기대값 `{EXPECTED_SHA256[:16]}…`)", got == EXPECTED_SHA256))
        info = probe(VIDEO)
        if info["ok"]:
            secs = info["frames"] / info["fps"] if info["fps"] else 0
            size_ok = (info["width"], info["height"]) == (1920, 1080)
            rows.append(("원본 영상 열기",
                         f"{info['width']}×{info['height']}, {info['fps']:.2f} fps, "
                         f"{info['frames']}프레임 (약 {secs / 60:.1f}분)",
                         info["first_frame_read"] and size_ok))
        else:
            rows.append(("원본 영상 열기", "열 수 없음", False))

    ok, msg = lossless_roundtrip()
    rows.append(("무손실 저장(FFV1) 시험", msg, ok))

    n_pat = len(list(PATTERNS.glob("*.png")))
    rows.append(("공간 패턴 25pct_screen", f"PNG {n_pat}개 (기대 16개: 모양 8종 × f/a)", n_pat == 16))
    n_tmp = len(list(TEMPORAL.glob("*.csv")))
    rows.append(("시간 패턴 rel_lum", f"CSV {n_tmp}개", n_tmp > 0))

    all_ok = all(r[2] for r in rows)
    lines = [
        "# 0단계 점검표: 환경과 재료 준비",
        "",
        f"**결과: {'✅ 통과' if all_ok else '❌ 실패'}** ({sum(r[2] for r in rows)}/{len(rows)} 항목 통과)",
        "",
        "| 항목 | 결과 | 통과 |",
        "|---|---|---|",
        *[f"| {a} | {b} | {'✅' if c else '❌'} |" for a, b, c in rows],
        "",
        "## 항목 설명",
        "",
        "- **원본 영상**: Big Buck Bunny 1080p (CC BY 3.0, © Blender Foundation). "
        "공간 패턴이 1920×1080으로 만들어져 있어서 같은 해상도가 필요합니다.",
        "- **SHA256**: 파일의 지문입니다. 기록해 둔 값과 같으면 공식 원본과 한 바이트도 다르지 않다는 뜻입니다.",
        "- **무손실 저장 시험**: 합성 영상을 FFV1 형식으로 저장합니다. 저장했다가 다시 읽었을 때 픽셀 차이가 0이어야 깜빡임이 압축 때문에 뭉개지지 않습니다.",
        "- **공간·시간 패턴**: `C:\\dev\\lumi-vision\\tools\\pse-test-media` (BSD-3-Clause)",
        "",
        "만든 명령: `python scripts\\step0_check.py`",
        "",
    ]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
