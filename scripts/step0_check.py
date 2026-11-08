"""0단계 점검: 환경과 원본 영상이 준비됐는지 확인하고 checks/step0/README.md를 만든다.

확인 항목
1. 파이썬과 패키지 버전
2. 원본 영상의 SHA256이 DATA_SOURCES.md에 적힌 값과 같은지
3. OpenCV로 원본 영상을 열 수 있는지 (해상도, fps, 길이)
4. 무손실 저장(FFV1 .mkv)이 정말 무손실인지: 무작위 프레임을 저장했다 다시 읽어 픽셀이 100% 같은지

사용: .venv\\Scripts\\python scripts\\step0_check.py
"""
import hashlib
import platform
import sys
import tempfile
from importlib.metadata import version
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
VIDEO = ROOT / "data/sources/big_buck_bunny/BigBuckBunny_640x360.m4v"
EXPECTED_SHA256 = "738e2f999860553d056dd79c952f58f63cbb73892a57c72342ce9e5330d9d2d7"
OUT = ROOT / "checks/step0/README.md"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


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
    """무작위 프레임 10장을 FFV1로 저장하고 다시 읽어 완전히 같은지 본다."""
    rng = np.random.default_rng(0)
    frames = rng.integers(0, 256, size=(10, 96, 128, 3), dtype=np.uint8)
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "roundtrip.mkv"
        w = cv2.VideoWriter(str(p), cv2.VideoWriter_fourcc(*"FFV1"), 24, (128, 96))
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
    return diff == 0, f"프레임 {len(back)}장, 최대 픽셀 차이 {diff}"


def main() -> int:
    rows = []  # (항목, 결과, 통과 여부)

    rows.append(("파이썬", f"{platform.python_version()} ({sys.executable})", True))
    for pkg in ["numpy", "opencv-python", "matplotlib", "pytest"]:
        try:
            rows.append((f"패키지 {pkg}", version(pkg), True))
        except Exception:
            rows.append((f"패키지 {pkg}", "설치 안 됨", False))

    if VIDEO.exists():
        got = sha256(VIDEO)
        rows.append(("원본 영상 SHA256", f"`{got[:16]}…` (기대값 `{EXPECTED_SHA256[:16]}…`)", got == EXPECTED_SHA256))
        info = probe(VIDEO)
        if info["ok"]:
            secs = info["frames"] / info["fps"] if info["fps"] else 0
            rows.append(("원본 영상 열기",
                         f"{info['width']}×{info['height']}, {info['fps']:.2f} fps, "
                         f"{info['frames']}프레임 (약 {secs / 60:.1f}분)",
                         info["first_frame_read"]))
        else:
            rows.append(("원본 영상 열기", "열 수 없음", False))
    else:
        rows.append(("원본 영상", f"파일 없음: {VIDEO}", False))

    ok, msg = lossless_roundtrip()
    rows.append(("무손실 저장(FFV1) 시험", msg, ok))

    all_ok = all(r[2] for r in rows)
    lines = [
        "# 0단계 점검표: 환경과 원본 영상 준비",
        "",
        f"**결과: {'✅ 통과' if all_ok else '❌ 실패'}** ({sum(r[2] for r in rows)}/{len(rows)} 항목 통과)",
        "",
        "| 항목 | 결과 | 통과 |",
        "|---|---|---|",
        *[f"| {a} | {b} | {'✅' if c else '❌'} |" for a, b, c in rows],
        "",
        "## 항목 설명",
        "",
        "- **SHA256**: 파일의 지문입니다. `DATA_SOURCES.md`에 적어 둔 값과 같으면 공식 원본과 한 바이트도 다르지 않다는 뜻입니다.",
        "- **무손실 저장 시험**: 앞으로 합성 영상을 FFV1 형식으로 저장합니다. 저장했다가 다시 읽었을 때 픽셀 차이가 0이어야 깜빡임이 압축 때문에 뭉개지지 않습니다.",
        "",
        "만든 명령: `.venv\\Scripts\\python scripts\\step0_check.py`",
        "",
    ]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
