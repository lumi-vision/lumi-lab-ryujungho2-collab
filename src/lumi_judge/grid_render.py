"""격자 판정기 G — 근거 그림: 대표 정지 프레임 위에 판정에 쓰인 격자 칸을 칠한다 (사람·에이전트 확인용).

- 정지 프레임 한 장만 쓴다(영상·연속 프레임은 저장하지 않음 — 광과민 안전 원칙).
- 칸 색: 휘도 깜빡임 = 빨강 테두리, 적색 깜빡임 = 자홍, 정지 패턴 = 주황, 변화 패턴 = 노랑, 흐름(면제) = 초록 점선 느낌의 얇은 테두리.
- 칸은 채우지 않고 테두리 + 반투명 덮개(원본 내용을 가리지 않도록).
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

COLORS = {"휘도 플래시": (40, 40, 230), "적색 플래시": (200, 40, 200), "휘도 연장": (40, 120, 230), "적색 연장": (160, 80, 220),
          "정적 공간 패턴": (0, 150, 255), "변화 공간 패턴": (0, 230, 255), "흐름(면제)": (60, 200, 60)}
LEGEND_EN = {"휘도 플래시": "luminance flash", "적색 플래시": "red flash", "휘도 연장": "extended flash", "적색 연장": "extended red",
             "정적 공간 패턴": "static pattern", "변화 공간 패턴": "changing pattern", "흐름(면제)": "flow (exempt)"}


def draw(frame_bgr: np.ndarray, layers: list[tuple[str, list, tuple]], title: str, out: Path, min_w: int = 768) -> str:
    """layers: [(라벨, 칸 목록 [[r,c],...], (rows, cols))]. 글자는 OpenCV 기본 글꼴 한계로 영문 범례 + 파일명에 시각."""
    img = frame_bgr.copy()
    if img.shape[1] < min_w:
        s = min_w / img.shape[1]
        img = cv2.resize(img, (min_w, int(round(img.shape[0] * s))), interpolation=cv2.INTER_NEAREST)
    h, w = img.shape[:2]
    over = img.copy()
    for lab, cells, (rows, cols) in layers:
        col = COLORS.get(lab, (255, 255, 255))
        ch, cw = h / rows, w / cols
        for r, c in cells:
            p0 = (int(c * cw), int(r * ch)); p1 = (int((c + 1) * cw) - 1, int((r + 1) * ch) - 1)
            cv2.rectangle(over, p0, p1, col, -1)
            cv2.rectangle(img, p0, p1, col, 1 if lab == "흐름(면제)" else 2)
    img = cv2.addWeighted(over, 0.25, img, 0.75, 0)
    bar = np.full((22 * (len(layers) + 1) + 6, w, 3), 24, np.uint8)
    cv2.putText(bar, title, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (235, 235, 235), 1, cv2.LINE_AA)
    for i, (lab, cells, _) in enumerate(layers, 1):
        cv2.rectangle(bar, (8, 8 + 22 * i), (22, 22 + 22 * i), COLORS.get(lab, (255, 255, 255)), -1)
        cv2.putText(bar, f"{LEGEND_EN.get(lab, lab)}  ({len(cells)} cells)", (30, 20 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (235, 235, 235), 1, cv2.LINE_AA)
    out.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(".png", np.vstack([img, bar]))
    out.write_bytes(buf.tobytes())
    return str(out)
