"""시연용 위험 영상 9개를 만든다: 휘도 깜빡임 3, 적색 깜빡임 2, 줄무늬 3, 원본 1.

⚠️ 결과 영상(data/hazard_demo/*.mkv)은 광과민성 발작을 일으킬 수 있는 자극입니다. 재생하지 마세요.

바탕: Big Buck Bunny 1080p 80초부터 2초를 960×540으로 줄이고 24→30 fps(프레임 복제)로 바꿈.
      960×540 은 판정기 둘 다 처리하기 적당한 크기(팀원 판별기는 긴 변 720으로 다시 줄여 판정).
합성: 영역 안만 바꾸고 영역 밖은 원본 그대로.
  - 휘도 깜빡임: 선형 빛 더하기 X = min(1, Y + δ·w(t)), δ = 0.3 (흰색 200 cd/m² 기준 약 60 cd/m²), 사각파
  - 적색 깜빡임: 영역을 pse-test-media 색 쌍으로 교대 (휘도가 같아 휘도 깜빡임은 없음)
        위험: (205,78,78) ↔ (31,134,134), Δu′v′ = 0.2046 / 색 차이 미달: (205,78,78) ↔ (60,132,132), Δu′v′ = 0.1948
  - 줄무늬: 영역을 세로 사각파 줄무늬(회색 30 ↔ 200, 2.6 ↔ 115.5 cd/m²)로 덮음. 주기 12 px ≈ 2.4 cpd (56 cm, 15인치)

사용: python scripts\\make_demo_hazards.py
"""
import json
from pathlib import Path

import cv2
import numpy as np

from lumi_synth import measure

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data/sources/big_buck_bunny_1080p/big_buck_bunny_1080p_h264.mov"
OUT = ROOT / "data/hazard_demo"
W, H, FPS, N = 960, 540, 30.0, 60
START_SEC = 80


def srgb_to_lin(u8):
    c = u8.astype(np.float32) / 255.0
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def lin_to_srgb(lin):
    c = np.clip(lin, 0, 1)
    s = np.where(c <= 0.0031308, c * 12.92, 1.055 * c ** (1 / 2.4) - 0.055)
    return np.clip(np.round(s * 255), 0, 255).astype(np.uint8)


def background() -> np.ndarray:
    cap = cv2.VideoCapture(str(SRC))
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(START_SEC * src_fps)))
    need = int(np.ceil(N * src_fps / FPS)) + 1
    frames = []
    for _ in range(need):
        ok, f = cap.read()
        if not ok:
            break
        frames.append(cv2.resize(f, (W, H), interpolation=cv2.INTER_AREA))
    cap.release()
    idx = [int(i * src_fps / FPS) for i in range(N)]       # 24 → 30 fps: 4장마다 1장 복제
    return np.stack([frames[i] for i in idx])


def rect_mask(area: float) -> np.ndarray:
    """화면 가운데, 16:9 비율 사각형으로 면적 area."""
    m = np.zeros((H, W), bool)
    rw, rh = int(round(W * np.sqrt(area))), int(round(H * np.sqrt(area)))
    y0, x0 = (H - rh) // 2, (W - rw) // 2
    m[y0:y0 + rh, x0:x0 + rw] = True
    return m


def left_mask(area: float) -> np.ndarray:
    m = np.zeros((H, W), bool)
    m[:, :int(round(W * area))] = True
    return m


def square_wave(hz: float) -> np.ndarray:
    t = np.arange(N) / FPS
    return ((t * hz * 2).astype(int) % 2 == 1).astype(np.float32)


def lum_flash(bg, mask, hz, delta=0.3):
    lin = srgb_to_lin(bg)
    w = square_wave(hz)[:, None, None, None]
    out = np.minimum(1.0, lin + delta * w * mask[None, :, :, None])
    return lin_to_srgb(out)


def red_flash(bg, mask, hz, rgb_a, rgb_b):
    out = bg.copy()
    w = square_wave(hz)
    for t in range(N):
        r, g, b = rgb_b if w[t] else rgb_a
        out[t][mask] = (b, g, r)                            # OpenCV 는 BGR
    return out


def stripes(bg, mask, motion, period=12, hz=5.0, speed=3.0):
    out = bg.copy()
    x = np.arange(W)
    for t in range(N):
        off = speed * t if motion == "drift" else 0.0
        bright = ((x - off) % period) < period / 2
        if motion == "reverse" and int(t * hz * 2 / FPS) % 2 == 1:
            bright = ~bright
        row = np.where(bright, 200, 30).astype(np.uint8)
        img = np.broadcast_to(row[None, :, None], (H, W, 3))
        out[t][mask] = img[mask]
    return out


def write(path: Path, frames: np.ndarray) -> None:
    wr = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"FFV1"), FPS, (W, H))
    for f in frames:
        wr.write(f)
    wr.release()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    bg = background()
    ppd = measure.pixels_per_degree(W, H, 0.0)
    specs = [
        ("v00_clean", "원본(합성 없음)", "안전", lambda: bg, None),
        ("v01_lum_rect40_5hz", "휘도 깜빡임: 가운데 40%, 5 Hz, +60 cd/m²", "위험(휘도)", lambda: lum_flash(bg, rect_mask(0.40), 5), rect_mask(0.40)),
        ("v02_lum_rect20_5hz", "휘도 깜빡임: 가운데 20%, 5 Hz (면적 미달)", "안전", lambda: lum_flash(bg, rect_mask(0.20), 5), rect_mask(0.20)),
        ("v03_lum_rect40_2hz", "휘도 깜빡임: 가운데 40%, 2 Hz (횟수 미달)", "안전", lambda: lum_flash(bg, rect_mask(0.40), 2), rect_mask(0.40)),
        ("v04_red_rect40_5hz", "적색 깜빡임: 40%, 5 Hz, Δu′v′ 0.2046", "위험(적색)", lambda: red_flash(bg, rect_mask(0.40), 5, (205, 78, 78), (31, 134, 134)), rect_mask(0.40)),
        ("v05_red_d_rect40_5hz", "적색 깜빡임: 40%, 5 Hz, Δu′v′ 0.1948 (색 차이 미달)", "안전", lambda: red_flash(bg, rect_mask(0.40), 5, (205, 78, 78), (60, 132, 132)), rect_mask(0.40)),
        ("v06_stripe_static50", f"줄무늬 정지: 왼쪽 50%, {ppd / 12:.1f} cpd", "위험(줄무늬, > 40%)", lambda: stripes(bg, left_mask(0.50), "static"), left_mask(0.50)),
        ("v07_stripe_reverse30", f"줄무늬 명암 반전 5 Hz: 왼쪽 30%, {ppd / 12:.1f} cpd", "위험(줄무늬, > 25%)", lambda: stripes(bg, left_mask(0.30), "reverse"), left_mask(0.30)),
        ("v08_stripe_drift50", f"줄무늬 한 방향 흐름 3 px/프레임: 왼쪽 50%, {ppd / 12:.1f} cpd", "제외(ITU: 매끄러운 흐름)", lambda: stripes(bg, left_mask(0.50), "drift"), left_mask(0.50)),
    ]
    manifest = []
    for name, desc, expect, make, mask in specs:
        frames = make()
        path = OUT / f"{name}.mkv"
        write(path, frames)
        if mask is not None:
            cv2.imwrite(str(OUT / f"{name}_mask.png"), mask.astype(np.uint8) * 255)
        manifest.append({"name": name, "file": path.name, "설명": desc, "설계상 기대": expect,
                         "mask_area": float(mask.mean()) if mask is not None else 0.0})
        print(name, desc)
    meta = {"width": W, "height": H, "fps": FPS, "frames": N, "source": "BBB 1080p 80초~", "ppd": ppd,
            "videos": manifest}
    (OUT / "manifest.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
