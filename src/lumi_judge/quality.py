"""영상 품질·크기 프로파일 — 규격 문턱(20 cd/m²·25 %·40 %·6 쌍)은 고정하고, '재는 방식'만 영상에 맞춘다 (2026-11-26).

원칙: 문턱이 화질에 따라 바뀌면 같은 패턴이 화질에 따라 위험/안전이 갈려 규격이 무너진다. 그래서 바꾸는 것은
  ① 격자 칸 크기(시각도) ② 판정 가능 범위의 명시 ③ 압축 블록 인공물 의심 표시 — 셋뿐.

  cell_deg      = max(TARGET_CELL_DEG, MIN_CELL_PX / ppd)
                  위쪽 한계: 반지름 r° 동심원의 이웃 칸 방향 변화 ≈ 칸/r rad ≤ 30°(0.52) → 칸 ≤ 0.52 r.
                  위험으로 판정돼야 할 가장 작은 동심원(3 cpd 에서 반지름 안 6 쌍 ≈ 2°) → 칸 ≲ 1° (계산 — 실험 확인 전).
                  아래쪽 한계: 칸 안 픽셀이 적으면 후보 비율·방향이 흔들림 → 최소 8 px (구현 선택).
  cpd_max_judgeable = 0.45 × ppd   (나이퀴스트 ppd/2 의 0.9 — 필터 묶음 상한과 같음)
  flash_max_countable = fps / 2    (프레임마다 반전해도 초당 fps/2 회가 최대)
  blockiness    = 8 px 격자 경계에서의 이웃 픽셀 밝기 차 평균 ÷ 그 밖 위치의 평균 (1 ≈ 블록 없음; 압축이 심하면 커짐)
  noise_cd      = 3×3 중앙값 필터 잔차의 MAD×1.4826 (cd/m²)
  palette       = 프레임 고유 색 수 (GIF 는 ≤ 256)
"""
from __future__ import annotations

import numpy as np
import torch

from .grid_flash import luminance_cd
from .grid_pattern import VIEW_DEFAULT, field_of_view, pixels_per_degree

TARGET_CELL_DEG = 1.0
MIN_CELL_PX = 8.0
BLOCK = 8
BLOCKY_RATIO = 1.3          # 이보다 크면 '블록 인공물 뚜렷' (구현 선택 — 강건성 실험에서 확인)


@torch.no_grad()
def profile(frames_bgr: np.ndarray, fps: float, view=VIEW_DEFAULT, n_sample: int = 6) -> dict:
    T, h, w = frames_bgr.shape[:3]
    ppd = pixels_per_degree(w, h, view=view)
    fov_w, fov_h = field_of_view(w, h, view)
    cell_deg = max(TARGET_CELL_DEG, MIN_CELL_PX / ppd)
    idx = np.linspace(0, T - 1, min(n_sample, T)).round().astype(int)
    lum = luminance_cd(torch.from_numpy(np.ascontiguousarray(frames_bgr[idx])).cuda())     # [n,h,w]
    # 잡음: 3×3 중앙값 잔차
    pad = torch.nn.functional.pad(lum[:, None], (1, 1, 1, 1), mode="replicate")
    med = pad.unfold(2, 3, 1).unfold(3, 3, 1).reshape(len(idx), 1, h, w, 9).median(-1).values[:, 0]
    res = (lum - med).abs().flatten()
    noise = float(1.4826 * res.median())
    # 블록 인공물: 8 px 경계의 가로·세로 이웃 차 / 나머지
    dx = (lum[:, :, 1:] - lum[:, :, :-1]).abs().mean((0, 1))           # [w-1]
    dy = (lum[:, 1:, :] - lum[:, :-1, :]).abs().mean((0, 2))           # [h-1]
    def ratio(d):
        if len(d) < 2 * BLOCK:
            return 1.0
        k = torch.arange(len(d), device=d.device)
        on = (k % BLOCK) == BLOCK - 1
        return float(d[on].mean() / d[~on].mean().clamp_min(1e-6))
    blocky = max(ratio(dx), ratio(dy))
    pal = int(len(np.unique(frames_bgr[idx[0]].reshape(-1, 3), axis=0)))
    return {"w": w, "h": h, "fps": float(fps), "view": list(view), "fov_deg": [round(fov_w, 2), round(fov_h, 2)], "ppd": round(ppd, 2),
            "cell_deg": round(cell_deg, 3), "cell_px": round(cell_deg * ppd, 1),
            "cpd_max_judgeable": round(0.45 * ppd, 2), "flash_max_countable": round(fps / 2.0, 2),
            "noise_cd": round(noise, 2), "blockiness": round(blocky, 3), "blocky": blocky > BLOCKY_RATIO, "palette": pal}


def limits_sentence(p: dict) -> str:
    s = (f"판정 가능 범위: 줄무늬 {p['cpd_max_judgeable']:.1f} cpd 이하(해상도 {p['w']}×{p['h']}, 1° = {p['ppd']:.0f} px) · "
         f"깜빡임 초당 {p['flash_max_countable']:.0f}회까지 셀 수 있음({p['fps']:.0f} fps) · 격자 칸 {p['cell_deg']:.2f}° ({p['cell_px']:.0f} px)")
    if p["blocky"]:
        s += f" · 압축 블록 인공물 뚜렷(지수 {p['blockiness']:.2f}) — 8 px 주기·축 정렬 줄무늬는 의심 표시"
    if p["palette"] <= 256:
        s += f" · 색 {p['palette']}개(GIF 팔레트) — 디더링 무늬 주의"
    return s
