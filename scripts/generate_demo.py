"""기본 동작 검증을 위해 점멸 없이 천천히 움직이는 짧은 데모 영상을 생성한다."""
from pathlib import Path
import argparse

import cv2
import numpy as np


def generate(path: Path, fps=24, duration=3):
    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (320, 180))
    if not writer.isOpened():
        raise RuntimeError("MJPG 코덱으로 데모 AVI 영상을 생성할 수 없습니다")
    try:
        for i in range(int(fps * duration)):
            img = np.empty((180, 320, 3), dtype=np.uint8)
            img[:] = (160, 153, 146)
            # 천천히 움직이는 저대비 도형으로 구성하며 점멸·줄무늬를 사용하지 않는다.
            x = 70 + int(i * 0.6)
            cv2.circle(img, (x, 90), 24, (174, 168, 164), -1, cv2.LINE_AA)
            writer.write(img)
    finally:
        writer.release()
    print(f"데모 영상 생성 완료: {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="sample/lumi_demo.avi")
    opts = parser.parse_args()
    generate(Path(opts.output))
