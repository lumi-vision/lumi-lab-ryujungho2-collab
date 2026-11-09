"""규칙 검증: 우리 판정기(measure.analyze)가 pse-test-media의 ITU 기대값과 얼마나 일치하는지 잰다.

⚠️ 이 스크립트가 만드는 프레임에는 광과민성 발작을 일으킬 수 있는 깜빡임이 들어 있다.
   그래서 프레임을 화면에 띄우거나 영상/이미지 파일로 저장하지 않고, 메모리에서 만들어 바로 판정기에 넣는다.

하는 일
1. pse-test-media의 video_creation/<세트>/*.json 을 읽는다.
   (각 JSON = 공간 패턴 PNG + 시간·색 패턴 CSV + 배경색 + 기대 판정 expected_result)
2. pse-test-media의 영상 생성 과정(generate_frames.py → generate_video.py)을 메모리에서 똑같이 재현한다.
   - generate_frames.py: 프레임마다 PNG의 불투명 픽셀을 CSV 색으로 다시 칠하고(recolor_image),
     여러 패턴을 알파 합성한 뒤 배경색 위에 얹는다(merge_images).
   - generate_video.py: 그 PNG를 cv2.imread(=BGR, 알파 버림)로 읽고, 첫 프레임을 padding 장 반복한 뒤
     모든 프레임을 FFV1(무손실)로 저장한다. padding 기본값은 video_config.json 의 10.
   → FFV1은 무손실이라 '저장 후 다시 읽은 프레임' = '메모리에서 만든 BGR 프레임'. (--ffv1-check 로 직접 확인 가능)
3. measure.analyze(frames, fps) 로 판정하고, JSON 의 expected_result["itu_r1702_4"] 와 비교한다.
4. 결과 요약을 출력하고 checks/rules/README.md (한국어)를 쓴다.

빨강 깜빡임이 들어간 세트는 건너뛴다: 우리 판정기는 휘도(밝기)만 보므로, ITU 의 '빨강 깜빡임' 조건으로
실패하는 영상은 원리상 맞힐 수 없다. 건너뛸지는 CSV/배경색에 회색이 아닌 색(r,g,b 가 서로 다름)이
있는지로 자동 판단하고, 건너뛴 세트와 이유를 README 에 적는다.

사용 예:
  .venv\\Scripts\\python.exe scripts\\validate_rules_pse.py --pse-root <pse-test-media 경로>
  .venv\\Scripts\\python.exe scripts\\validate_rules_pse.py --pse-root <경로> --sets all-luminance
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))            # 설치본이 아니라 '지금 작업 폴더의' measure.py 를 쓰기 위해
from lumi_synth import measure                    # noqa: E402

DEFAULT_SETS = ["broadcast_30fps_01", "30fps_alternating_01"]
OUT = ROOT / "checks/rules/README.md"


# =====================================================================================
# 1. pse-test-media 프레임 재현 (generate_frames.py / generate_video.py 와 같은 계산)
# =====================================================================================
def load_spatial(png_path: Path) -> np.ndarray:
    """공간 패턴 PNG 를 RGBA 배열(H×W×4 uint8)로 읽는다. (recolor_image 의 img.convert('RGBA')와 같음)"""
    with Image.open(png_path) as img:
        return np.array(img.convert("RGBA"))


def recolor(spatial_rgba: np.ndarray, r: int, g: int, b: int, a: int) -> Image.Image:
    """generate_frames.recolor_image 와 같은 일: 알파가 0이 아닌(불투명) 픽셀을 모두 (r,g,b,a)로 칠한다."""
    data = spatial_rgba.copy()
    mask = data[:, :, 3] != 0
    data[mask] = [r, g, b, a]
    return Image.fromarray(data)


def merge(images: list[Image.Image], bg_color: tuple | None) -> Image.Image | None:
    """generate_frames.merge_images 와 같은 일: 패턴을 차례로 알파 합성한 뒤 배경색 위에 얹는다."""
    if not images:
        return None
    merged = images[0].copy()
    for img in images[1:]:
        merged = Image.alpha_composite(merged, img)
    if bg_color is not None:
        bg = Image.new("RGBA", merged.size, bg_color)
        merged = Image.alpha_composite(bg, merged)
    return merged


def read_temporal(csv_path: Path) -> list[tuple[int, int, int, int]]:
    """시간·색 패턴 CSV → 프레임별 (r, g, b, a) 목록."""
    with open(csv_path, newline="") as f:
        return [(int(row["r"]), int(row["g"]), int(row["b"]), int(row["a"])) for row in csv.DictReader(f)]


def build_frames(json_path: Path, padding: int, scale: int = 1) -> tuple[np.ndarray, float]:
    """JSON 하나 → (프레임 T×H×W×3 uint8 BGR, fps). pse-test-media 가 만든 영상의 픽셀과 같다.

    scale > 1 이면 가로세로를 1/scale 로 줄인다(속도용). 면적이 바뀌어 판정이 뒤집힐 수 있으므로 기본은 1(원본).
    """
    spec = json.loads(json_path.read_text())
    jdir = json_path.parent
    bg = tuple(map(int, spec["bgcolor"].strip("()").split(","))) if spec.get("bgcolor") else None
    patterns = [(load_spatial(jdir / p["spatial"]), read_temporal(jdir / p["temporal_color"])) for p in spec["pattern"]]
    n_rows = max(len(colors) for _, colors in patterns)    # 가장 긴 CSV 길이 = 프레임 수

    cache: dict[tuple, np.ndarray] = {}   # 같은 색 조합의 프레임은 한 번만 합성 (generate_frames 의 해시 중복 제거와 같은 효과)
    frames = []
    for i in range(n_rows):
        # CSV 가 짧아 i 행이 없는 패턴은 그 프레임에서 빠진다 (원본 코드와 같음)
        key = tuple(colors[i] if i < len(colors) else None for _, colors in patterns)
        if key not in cache:
            imgs = [recolor(sp, *c) for (sp, _), c in zip(patterns, key) if c is not None]
            merged = merge(imgs, bg)
            if merged is None:
                continue
            rgba = np.array(merged)
            # cv2.imread(PNG RGBA) 는 알파를 버리고 BGR 로 읽는다 → RGB 순서만 뒤집으면 같다.
            bgr = np.ascontiguousarray(rgba[:, :, 2::-1])
            if scale > 1:
                h, w = bgr.shape[:2]
                bgr = cv2.resize(bgr, (w // scale, h // scale), interpolation=cv2.INTER_AREA)
            cache[key] = bgr
        frames.append(cache[key])
    # generate_video.py: 첫 프레임을 padding 장 먼저 쓴다.
    frames = [frames[0]] * padding + frames
    return np.stack(frames), float(spec["framerate"])


def ffv1_roundtrip(frames: np.ndarray, fps: float, tmp_dir: Path) -> np.ndarray:
    """(선택) 실제로 FFV1 .avi 로 저장했다가 다시 읽는다. 원본 도구와 같은 코덱 경로가 무손실인지 확인용."""
    tmp_dir.mkdir(parents=True, exist_ok=True)
    path = tmp_dir / "roundtrip.avi"
    h, w = frames.shape[1:3]
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"FFV1"), fps, (w, h))
    for f in frames:
        vw.write(f)
    vw.release()
    cap = cv2.VideoCapture(str(path))
    out = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        out.append(f)
    cap.release()
    path.unlink(missing_ok=True)       # 깜빡임 영상 파일을 남기지 않는다
    return np.stack(out)


# =====================================================================================
# 2. 메모리 절약: measure.luminance_cd 를 '프레임 묶음 단위'로 나눠 부르기
# =====================================================================================
# luminance_cd 는 (T,H,W,3) 전체를 float32 로 바꾸며 중간 배열을 여러 개 만든다.
# 1920×1080×44프레임이면 중간 배열 하나가 약 1.1 GB 라서 RAM 이 부족할 수 있다.
# luminance_cd 는 픽셀마다 따로 계산하는 함수(원소별 연산)이므로, 프레임을 몇 장씩 나눠 계산해 이어 붙여도
# 결과가 비트 단위로 같다. 그래서 measure.py 는 고치지 않고, 모듈 안의 이름만 이 래퍼로 바꿔 끼운다.
# (analyze() 와 criteria_report() 는 실행할 때 모듈의 luminance_cd 이름을 찾으므로 래퍼가 쓰인다.)
_ORIGINAL_LUMINANCE = measure.luminance_cd


def _chunked_luminance(frames_bgr: np.ndarray, chunk: int = 4) -> np.ndarray:
    out = np.empty(frames_bgr.shape[:-1], dtype=np.float32)
    for i in range(0, len(frames_bgr), chunk):
        out[i:i + chunk] = _ORIGINAL_LUMINANCE(frames_bgr[i:i + chunk])
    return out


def install_chunked_luminance() -> None:
    """래퍼를 끼우기 전에, 작은 무작위 영상에서 원래 함수와 결과가 완전히 같은지 먼저 확인한다."""
    rng = np.random.default_rng(0)
    probe = rng.integers(0, 256, size=(11, 7, 5, 3), dtype=np.uint8)
    a, b = _ORIGINAL_LUMINANCE(probe), _chunked_luminance(probe)
    assert a.dtype == b.dtype and np.array_equal(a, b), "나눠 계산한 휘도가 원래 함수와 다릅니다"
    measure.luminance_cd = _chunked_luminance


# =====================================================================================
# 3. 세트 분류와 기대값 읽기
# =====================================================================================
def is_gray(rgba: tuple) -> bool:
    r, g, b = rgba[:3]
    return r == g == b


def set_has_color(set_dir: Path) -> bool:
    """세트 안 어떤 프레임 색/배경색에라도 회색이 아닌 색(r,g,b 다름)이 있으면 True (= 빨강 깜빡임 포함)."""
    for jp in sorted(set_dir.glob("*.json")):
        spec = json.loads(jp.read_text())
        if spec.get("bgcolor") and not is_gray(tuple(map(int, spec["bgcolor"].strip("()").split(",")))):
            return True
        for p in spec["pattern"]:
            if any(c[3] != 0 and not is_gray(c) for c in read_temporal(jp.parent / p["temporal_color"])):
                return True
    return False


def norm_expected(value: str | None) -> str | None:
    """'pass', 'fail', 'pass - small area' 같은 값을 pass/fail 로 정리한다."""
    if value is None:
        return None
    v = value.strip().lower()
    return "fail" if v.startswith("fail") else "pass" if v.startswith("pass") else None


def listing(set_dir: Path) -> dict[str, dict]:
    """세트의 목록 CSV(<세트>.csv: filename, pass, dimension). dimension = 설계상 '무너뜨린 조건'."""
    p = set_dir / f"{set_dir.name}.csv"
    if not p.exists():
        return {}
    with open(p, newline="") as f:
        return {row["filename"]: row for row in csv.DictReader(f)}


# =====================================================================================
# 4. 불일치 원인 조사용 보조 값
# =====================================================================================
def color_cd(c: tuple) -> float:
    """색 하나(r,g,b)의 휘도(cd/m²). 판정기와 같은 함수(luminance_cd)로 잰다."""
    return float(_ORIGINAL_LUMINANCE(np.array([[[c[2], c[1], c[0]]]], np.uint8))[0, 0])


def diagnose(frames: np.ndarray, fps: float, json_path: Path) -> dict:
    """불일치 영상에 대해 원인을 찾을 재료를 모은다.

    - criteria: 조건별 판정표 (measure.criteria_report: 밝기/픽셀 횟수/면적/최종)
    - geometry: 패턴마다 불투명 면적, 프레임별 휘도 순서(cd/m², 투명이면 배경 휘도에 '(배경)' 표시)
    - overlap : 패턴이 2개면 두 영역이 겹치는 면적 (겹친 픽셀은 두 깜빡임을 모두 겪는다)
    """
    spec = json.loads(json_path.read_text())
    bg = tuple(map(int, spec["bgcolor"].strip("()").split(","))) if spec.get("bgcolor") else None
    bg_cd = color_cd(bg) if bg else None
    geo, masks = [], []
    for p in spec["pattern"]:
        sp = load_spatial(json_path.parent / p["spatial"])
        colors = read_temporal(json_path.parent / p["temporal_color"])
        seq = [f"{color_cd(c):.1f}" if c[3] != 0 else f"({bg_cd:.1f})" for c in colors]
        masks.append(sp[:, :, 3] != 0)
        geo.append({"spatial": p["spatial"], "temporal": p["temporal_color"],
                    "opaque_area": float(masks[-1].mean()), "seq": seq})
    overlap = float((masks[0] & masks[1]).mean()) if len(masks) == 2 else None
    return {"criteria": measure.criteria_report(frames, fps), "geometry": geo, "overlap": overlap, "bg_cd": bg_cd}


# 사람이 조사한 불일치 원인 (2026-11-09, pse-test-media ef78542 / measure.py SHA256 79675481…).
# 판정기나 pse-test-media 가 바뀌면 다시 조사해야 한다. 여기에 없는 불일치는 보고서에 '조사 필요'로 나온다.
CAUSE_MULTISTEP = "A. 여러 프레임에 나뉜 밝기 변화 (밝기 조건)"
CAUSE_AGGREGATE = "B. 서로 다른 영역의 엇박자 깜빡임을 하나로 합쳐 셈 (영역·동기 조건)"
CAUSE_LABEL = "C. 기대값(JSON) 자체가 의심됨"
NOTES: dict[tuple[str, str], tuple[str, str]] = {
    ("broadcast_30fps_01", "f001y033"): (CAUSE_MULTISTEP,
        "y033 은 기대 fail 인 f033 과 깜빡임 하나만 다르다: 11→13프레임 하강이 87.9→68.4→49.2 cd/m² "
        "(한 프레임 걸음 19.5, 19.2 — 둘 다 20 미만, 합계 38.7). pse-test-media 는 이 깜빡임을 '밝기 문턱 미달'로 "
        "설계했다(파일명 y = 깜빡임 하나만 문턱 아래). 우리 pixel_transitions 는 '직전 극값'에서 재므로 두 프레임(67 ms)에 "
        "걸친 38.7 cd/m² 하강을 전이 1번으로 센다 → 3.5번/s → 위험. f033 에서는 같은 하강이 90.2→69.3→49.2 (20.9, 20.1)."),
    ("broadcast_30fps_01", "f005y036"): (CAUSE_MULTISTEP,
        "y033 과 같은 구조: 하강 146.1→127.5→107.9 cd/m² (걸음 18.6, 19.6 / 합계 38.2). 걸음마다 재면 문턱 미달, "
        "극값 기준으로 재면 전이."),
    ("broadcast_30fps_01", "f006l034"): (CAUSE_MULTISTEP,
        "l = 모든 깜빡임이 밝기 문턱 미달로 설계됨. 깜빡임마다 한쪽(오르막 또는 내리막)이 두 걸음 68.4↔87.9↔107.9 "
        "(19.5, 20.0 미만 — 정확히는 19.97)으로 나뉘어 있다. 합계 39.5 cd/m² 라서 우리 판정기는 매번 전이로 센다."),
    ("broadcast_30fps_inf01", "f003y038_f005cs035"): (CAUSE_MULTISTEP,
        "목표 패턴 y038 (f003 영역)의 하강 184.3→164.6→146.1 (걸음 19.7, 18.5 / 합계 38.2, 어두운 쪽 146.1 < 160). "
        "픽셀 기준 횟수가 이미 3.5번이라 두 영역을 합친 효과가 아니라 A 원인."),
    ("broadcast_30fps_inf01", "f002c015_f001cs012"): (CAUSE_AGGREGATE,
        "두 영역(f002·f001, 각 25.19%, 겹침 0%)이 각각 3번/s 로 깜빡인다(따로 판정하면 둘 다 3.0번/s, 안전). "
        "co012 의 깜빡임 하나(18~19프레임)만 c015 와 엇박자. 우리 screen_transitions 는 화면 전체에서 25% 넘는 "
        "전이를 한 줄의 상태 기계로 세므로, 엇박자 깜빡임 하나가 더해져 4번/s → 위험. pse 기대값은 영역마다 따로 센다."),
    ("broadcast_30fps_inf02", "f002tc03_f004ics03"): (CAUSE_AGGREGATE,
        "tc03(f002)·ics03(f004) 각각 3번/s, 2프레임 이상 엇박자(따로 판정: 3.0, 3.0). 우리는 두 줄을 합쳐 6번/s. "
        "두 영역이 겹치는 5.74% 픽셀은 실제로 6번/s 를 겪지만(픽셀 기준 6번) 면적이 25% 미만이다."),
    ("broadcast_30fps_inf02", "f006tc03_f001ico03"): (CAUSE_AGGREGATE,
        "tc03(f006)·ico03(f001) — ico03 은 깜빡임 하나만 엇박자. 합쳐 세서 4번/s. 겹침 9.26% (< 25%)."),
    ("broadcast_30fps_inf02", "f007ty03_f006ics03"): (CAUSE_AGGREGATE,
        "ty03(f007)은 깜빡임 하나가 문턱 미달(20.4→38.9→20.0: 18.5, 18.9)이라 혼자서는 3번/s 이하, "
        "ics03(f006)도 3번/s. 엇박자 두 영역을 합쳐 5.5번/s. 겹침 10.19% (< 25%)."),
    ("broadcast_30fps_inf02", "f001tf03_f003ico03"): (CAUSE_LABEL,
        "JSON 은 모든 표준 pass 인데, 같은 세트의 목록 CSV(broadcast_30fps_inf02.csv)는 pass=FALSE, dimension=FAIL 이다. "
        "tf03 = '실패하는 목표 패턴'(lum_flash_patterns/30fps_broadcast_interference/README.md). tf03 만 f001 영역에 "
        "따로 판정해도 3.5번/s (25.19%) → 위험. 즉 우리 결과는 패턴 설계·목록 CSV 와 일치하고 JSON 과만 다르다. "
        "(우리 판정의 4.5번/s 에는 B 원인으로 ico03 이 더해짐.) 이 불일치는 판정기 오류로 보기 어렵지만 숨기지 않고 센다."),
}


def label_conflicts(vc: Path, sets: list[str]) -> list[tuple]:
    """목록 CSV 의 pass 열과 JSON 의 itu_r1702_4 가 다른 영상. (CSV pass 열은 세트의 '대상 표준' 기준이라
    trace24·wcag 세트에서는 원래 다를 수 있다 — video_creation/README.md 의 표준 표 참고)"""
    out = []
    for s in sets:
        for name, row in listing(vc / s).items():
            if "pass" not in row:
                continue
            exp = norm_expected(json.loads((vc / s / f"{name}.json").read_text())["expected_result"].get("itu_r1702_4"))
            if (row["pass"].upper() == "TRUE") != (exp == "pass"):
                out.append((s, name, row["pass"], row.get("dimension", ""), exp))
    return out


# =====================================================================================
# 5. 실행과 보고서
# =====================================================================================
def git_head(path: Path) -> str:
    try:
        return subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:
        return "(알 수 없음)"


def git_dirty(path: Path, rel: str) -> bool:
    r = subprocess.run(["git", "-C", str(path), "status", "--porcelain", "--", rel], capture_output=True, text=True)
    return bool(r.stdout.strip())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pse-root", required=True, type=Path, help="pse-test-media 저장소 경로")
    ap.add_argument("--sets", nargs="+", default=DEFAULT_SETS,
                    help="video_creation 아래 세트 이름들. 'all-luminance' = 빨강이 없는 모든 세트")
    ap.add_argument("--scale", type=int, default=1, help="가로세로 축소 배수(속도용). 기본 1 = 원본 해상도(권장)")
    ap.add_argument("--padding", type=int, default=None, help="앞쪽 반복 프레임 수. 기본 = video_config.json 값")
    ap.add_argument("--ffv1-check", type=int, default=0,
                    help="처음 N개 영상은 FFV1 저장→읽기 후 프레임이 같은지도 확인 (임시 폴더, 파일은 바로 지움)")
    ap.add_argument("--tmp-dir", type=Path, default=Path(tempfile.gettempdir()),
                    help="--ffv1-check 용 임시 폴더 위치 (저장소 밖으로 둘 것)")
    ap.add_argument("--save-results", type=Path, default=None, help="(선택) 영상별 원시 결과 JSON 경로")
    ap.add_argument("--out", type=Path, default=OUT, help="보고서 경로")
    args = ap.parse_args()

    pse = args.pse_root.resolve()
    vc = pse / "video_creation"
    config = json.loads((pse / "video_config.json").read_text())
    padding = config.get("padding", 0) if args.padding is None else args.padding

    # 세트 고르기: 빨강이 들어간 세트는 건너뛰고 이유를 기록
    all_sets = sorted(d.name for d in vc.iterdir() if d.is_dir())
    wanted = all_sets if args.sets == ["all-luminance"] else args.sets
    sets, skipped = [], []
    for s in all_sets:
        has_color = set_has_color(vc / s)
        if s in wanted and not has_color:
            sets.append(s)
        elif s in wanted and has_color:
            skipped.append((s, "회색이 아닌 색(빨강) 프레임이 있음 — 휘도 전용 판정기로는 빨강 깜빡임 조건을 볼 수 없음"))
        elif has_color:
            skipped.append((s, "빨강 포함 (요청 세트 아님)"))
    unknown = [s for s in wanted if s not in all_sets]
    if unknown:
        sys.exit(f"없는 세트: {unknown}")

    install_chunked_luminance()
    rows, diags = [], {}
    t_all = time.time()
    n_done = 0
    for s in sets:
        meta = listing(vc / s)
        for jp in sorted((vc / s).glob("*.json")):
            t0 = time.time()
            spec = json.loads(jp.read_text())
            frames, fps = build_frames(jp, padding, args.scale)
            if n_done < args.ffv1_check:
                with tempfile.TemporaryDirectory(dir=args.tmp_dir) as td:
                    back = ffv1_roundtrip(frames, fps, Path(td))
                assert back.shape == frames.shape and np.array_equal(back, frames), f"FFV1 왕복 불일치: {jp.name}"
            r = measure.analyze(frames, fps)
            exp_raw = spec.get("expected_result", {}).get("itu_r1702_4")
            exp = norm_expected(exp_raw)
            ours = "fail" if r["dangerous"] else "pass"
            row = {
                "set": s, "name": jp.stem, "fps": fps, "frames": len(frames),
                "size": f"{frames.shape[2]}x{frames.shape[1]}",
                "expected_raw": exp_raw, "expected": exp,
                "ofcom": norm_expected(spec.get("expected_result", {}).get("ofcom2017")),
                "dimension": meta.get(jp.stem, {}).get("dimension", ""),
                "ours": ours, "max_flashes": r["max_flashes"], "peak_area": r["peak_area"],
                "n_events": len(r["events"]), "agree": exp == ours, "sec": time.time() - t0,
            }
            if not row["agree"]:
                diags[(s, jp.stem)] = diagnose(frames, fps, jp)
            rows.append(row)
            n_done += 1
            print(f"[{n_done}] {s}/{jp.stem}: 기대 {exp_raw} / 우리 {ours} "
                  f"(깜빡임 {r['max_flashes']:g}/s, 면적 {r['peak_area']:.4%}) {'일치' if row['agree'] else '불일치'} "
                  f"{row['sec']:.1f}s", flush=True)
            del frames, r
    runtime = time.time() - t_all

    # ---- 요약 출력 ----
    agree = sum(r["agree"] for r in rows)
    cm = {(e, o): sum(1 for r in rows if r["expected"] == e and r["ours"] == o)
          for e in ("fail", "pass") for o in ("fail", "pass")}
    print(f"\n일치 {agree}/{len(rows)}  (실행 {runtime:.0f}초)")
    print(f"기대 fail: 우리 fail {cm['fail', 'fail']}, 우리 pass {cm['fail', 'pass']}")
    print(f"기대 pass: 우리 fail {cm['pass', 'fail']}, 우리 pass {cm['pass', 'pass']}")
    for s, why in skipped:
        print(f"건너뜀 {s}: {why}")

    if args.save_results:      # (선택) 원시 결과를 JSON 으로 저장 — 저장소 밖 경로를 권장
        args.save_results.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    write_report(args, pse, padding, sets, skipped, rows, diags, cm, agree, runtime)
    print(f"보고서: {args.out}")


def write_report(args, pse, padding, sets, skipped, rows, diags, cm, agree, runtime) -> None:
    measure_path = ROOT / "src/lumi_synth/measure.py"
    measure_sha = hashlib.sha256(measure_path.read_bytes()).hexdigest()
    dirty = git_dirty(ROOT, "src/lumi_synth/measure.py")
    L = []
    L.append("# 규칙 검증: 우리 판정기 vs pse-test-media ITU 기대값\n")
    core = [r for r in rows if r["set"] in DEFAULT_SETS]
    L.append(f"**결과: ITU-R BT.1702 기대값과 {agree}/{len(rows)} 일치**"
             + (f" (기본 2세트 {sum(r['agree'] for r in core)}/{len(core)})" if core and len(core) != len(rows) else "") + "\n")
    L.append("> ⚠️ 시험 영상은 광과민성 발작 위험 패턴입니다. 프레임은 메모리에서만 만들어 판정기에 넣었고, "
             "화면에 띄우거나 파일로 남기지 않았습니다.\n")
    L.append("## 무엇을 시험했나\n")
    L.append("- 판정기: `src/lumi_synth/measure.py` 의 `analyze(frames_bgr, fps)` → `dangerous` (1초 깜빡임 > 3번)")
    L.append("- 정답: pse-test-media `video_creation/<세트>/*.json` 의 `expected_result[\"itu_r1702_4\"]` (pass/fail)")
    L.append("- 프레임: pse-test-media `generate_frames.py` + `generate_video.py` 와 같은 계산을 메모리에서 재현 "
             f"(PNG 불투명 픽셀 재칠 → 알파 합성 → 배경 → BGR, 앞쪽 padding {padding}프레임 = `video_config.json`). "
             "원본 도구는 FFV1(무손실)로 저장하므로 픽셀이 같습니다"
             + (f" (`--ffv1-check {args.ffv1_check}`: 처음 {args.ffv1_check}개 영상은 실제 FFV1 저장→읽기 후 픽셀 100% 같음을 확인)." if args.ffv1_check else "."))
    L.append(f"- 해상도: {'원본 1920×1080 (축소 없음)' if args.scale == 1 else f'1/{args.scale} 축소 — 면적 판정이 달라질 수 있음'}. "
             "docs/BASICS.md·PLAN.md 에 적힌 대로 640×360 으로 줄이면 안전 패턴 a003 이 25.05%가 되어 위험으로 뒤집히므로 원본에서 잽니다.")
    L.append("- 메모리: `luminance_cd` 를 프레임 4장씩 나눠 부르는 래퍼로 바꿔 끼웠습니다(원소별 계산이라 결과가 비트 단위로 같음, "
             "실행 시작 때 원래 함수와 같은지 assert). measure.py 자체는 고치지 않았습니다.\n")
    L.append("## 버전\n")
    L.append(f"- pse-test-media: `{git_head(pse)}`")
    L.append(f"- 이 저장소 HEAD: `{git_head(ROOT)}`")
    L.append(f"- measure.py SHA256: `{measure_sha}`"
             + (" — ⚠️ **커밋되지 않은 수정이 있는 measure.py 로 실행함** (HEAD 의 measure.py 와 다름)" if dirty else ""))
    L.append(f"- 실행 시간: {runtime:.0f}초 ({len(rows)}개 영상, CPU)\n")
    # 혼동표: (가) 기본 2세트(ITU·Ofcom 용으로 설계된 세트)만, (나) 이번에 돌린 전체
    groups = [("기본 2세트 (" + ", ".join(DEFAULT_SETS) + ")", [r for r in rows if r["set"] in DEFAULT_SETS]),
              ("이번 실행 전체", rows)]
    for title, rs in groups:
        if not rs:
            continue
        g = {(e, o): sum(1 for r in rs if r["expected"] == e and r["ours"] == o)
             for e in ("fail", "pass") for o in ("fail", "pass")}
        L.append(f"## 혼동표 — {title} (행 = 기대값, 열 = 우리 판정)\n")
        L.append("| | 우리 fail (위험) | 우리 pass (안전) |")
        L.append("|---|---|---|")
        L.append(f"| **기대 fail** | {g['fail', 'fail']} | {g['fail', 'pass']} |")
        L.append(f"| **기대 pass** | {g['pass', 'fail']} | {g['pass', 'pass']} |\n")
        L.append(f"일치 **{sum(r['agree'] for r in rs)}/{len(rs)}**\n")
    L.append("### 세트별\n")
    L.append("| 세트 | 영상 수 | 일치 | 기대 fail 수 |")
    L.append("|---|---|---|---|")
    for s in sets:
        rs = [r for r in rows if r["set"] == s]
        L.append(f"| {s} | {len(rs)} | {sum(r['agree'] for r in rs)}/{len(rs)} | {sum(r['expected'] == 'fail' for r in rs)} |")
    L.append("")
    if skipped:
        L.append("### 건너뛴 세트\n")
        L.append("| 세트 | 이유 |")
        L.append("|---|---|")
        for s, why in skipped:
            L.append(f"| {s} | {why} |")
        L.append("")
    # 일치했지만 '아무 깜빡임도 못 본' 영상: 일치가 판정기의 정확성 증거가 되기 어렵다
    zero = [r for r in rows if r["n_events"] == 0]
    L.append("### 일치 해석 주의\n")
    L.append(f"- 우리 판정기가 화면 전이를 한 번도 세지 않은 영상: {len(zero)}개 "
             f"(그중 기대 pass {sum(r['expected'] == 'pass' for r in zero)}개). trace24·WCAG 용 세트처럼 깜빡이는 영역이 "
             "화면의 25%보다 훨씬 작거나(416×416 나 341×256 의 25%), 배경과 패턴의 밝기 차이가 20 cd/m² 미만인 경우다. "
             "이런 영상의 일치는 '면적이 작거나 밝기 차이가 작으면 안전'만 확인한다.")
    fail_sets = sorted({r["set"] for r in rows if r["expected"] == "fail"})
    L.append(f"- 기대 fail 영상이 있는 세트: {', '.join(fail_sets) or '없음'} (나머지 세트의 ITU 기대값은 모두 pass).")
    if core:
        L.append(f"- measure.py 모듈 설명(D2)에는 '초기화 전 판정기(같은 정의)가 pse-test-media ITU 기대값 56/56과 일치'라고 "
                 f"적혀 있다. 현재 measure.py 로 같은 기본 2세트를 재면 **{sum(r['agree'] for r in core)}/{len(core)}** 이다. "
                 "그 문장은 지금 판정기로는 재현되지 않는다.")
    L.append("")
    conflicts = label_conflicts(pse / "video_creation", sets)
    L.append("### pse-test-media 기대값 메모\n")
    L.append("- `itu_r1702_4` 를 정답으로 썼다. 키 이름의 `_4` 가 BT.1702 의 어느 개정판(-0…-3)을 뜻하는지 저장소에 "
             "설명이 없다(video_creation/README.md 는 'Recommendation ITU-R BT.1702' 링크만 둠). 확인 못 함.")
    L.append("- 기대값의 'pass - small area' 는 pass 로 셌다.")
    L.append("- 세트 목록 CSV 의 `pass` 열은 그 세트의 **대상 표준** 기준이다(예: trace24 세트는 trace24). "
             "그래서 ITU 기대값과 다른 행이 있는 것은 정상이다. 다만 ITU 가 대상 표준인 세트에서 다른 행은 기대값이 의심스럽다.")
    if conflicts:
        L.append("\n| 세트 | 영상 | 목록 CSV pass | dimension | JSON itu_r1702_4 |")
        L.append("|---|---|---|---|---|")
        for s, name, p, dim, exp in conflicts:
            L.append(f"| {s} | {name} | {p} | {dim} | {exp} |")
    L.append("")
    L.append("## 불일치 분석\n")
    if not diags:
        L.append("불일치 없음.\n")
    else:
        # 원인별 개수 요약 (NOTES 에 없는 불일치는 '조사 필요')
        causes: dict[str, list[str]] = {}
        for key in diags:
            causes.setdefault(NOTES.get(key, ("조사 필요 (새 불일치)", ""))[0], []).append(f"{key[0]}/{key[1]}")
        L.append("| 원인 | 개수 | 영상 |")
        L.append("|---|---|---|")
        for cause, names in sorted(causes.items()):
            L.append(f"| {cause} | {len(names)} | {', '.join(names)} |")
        L.append("")
        L.append("- **A. 여러 프레임에 나뉜 밝기 변화**: 우리 판정기는 픽셀마다 '직전 전이 이후의 극값'에서 변화량을 잰다"
                 "(measure.pixel_transitions, 의도된 설계). pse-test-media 의 y/l 패턴은 한 깜빡임의 오르막이나 내리막을 "
                 "20 cd/m² 미만 걸음 두 개로 나눠 '문턱 미달'로 설계했다. 걸음 합계(약 38~40 cd/m²)는 문턱의 두 배라서, "
                 "우리 판정은 위험이다. 즉 ITU 의 '20 cd/m² 이상 변화'를 한 프레임 사이로 재느냐(pse 기대값), "
                 "여러 프레임 누적으로 재느냐(우리)의 정의 차이다.")
        L.append("- **B. 엇박자 두 영역 합산**: 우리 screen_transitions 는 화면 어디서든 같은 방향 25% 초과 전이가 있으면 "
                 "하나의 화면 상태 기계에 넣는다. 그래서 각자 3번/s 인 두 영역이 2프레임 이상 엇갈려 깜빡이면 합쳐서 4~6번/s 가 된다. "
                 "pse 기대값은 영역별(또는 동시에 일어난 것만 면적 합산)로 센다. 겹친 픽셀만 실제로 더 많이 깜빡이지만 그 면적은 25% 미만.")
        L.append("- **C. 기대값 의심**: JSON 과 목록 CSV 가 서로 다르고, 패턴 설계로는 우리 결과가 맞는 경우.\n")
    for (s, name), d in diags.items():
        r = next(x for x in rows if x["set"] == s and x["name"] == name)
        L.append(f"### {s}/{name} — 기대 {r['expected_raw']}, 우리 {r['ours']}\n")
        cause, note = NOTES.get((s, name), ("조사 필요 (새 불일치)", "아직 조사하지 않은 불일치입니다."))
        L.append(f"- **원인: {cause}** — {note}")
        L.append(f"- 목록 CSV 의 설계 조건(dimension): `{r['dimension'] or '-'}`, 배경 휘도 {d['bg_cd']:.1f} cd/m²")
        for g in d["geometry"]:
            L.append(f"- 패턴 `{g['spatial'].split('/')[-1]}` (불투명 면적 {g['opaque_area']:.4%}) × "
                     f"`{'/'.join(g['temporal'].split('/')[-2:])}` 프레임별 휘도(cd/m², 괄호 = 투명이라 배경): "
                     f"{' '.join(g['seq'])}")
        if d["overlap"] is not None:
            L.append(f"- 두 영역의 겹침 면적: {d['overlap']:.4%}")
        L.append("\n| 조건 | 측정값 | 기준 | 넘음 |")
        L.append("|---|---|---|---|")
        for c in d["criteria"]:
            L.append(f"| {c['조건']} | {c['측정값']} | {c['기준']} | {'예' if c['넘음'] else '아니오'} |")
        L.append("")
    L.append("## 영상별 결과\n")
    L.append("| 세트 | 영상 | 설계 조건 | 기대(ITU) | 우리 | 1초 최대 깜빡임 | 최대 면적 | 화면 전이 수 | 일치 |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        L.append(f"| {r['set']} | {r['name']} | {r['dimension'] or '-'} | {r['expected_raw']} | {r['ours']} | "
                 f"{r['max_flashes']:g} | {r['peak_area']:.4%} | {r['n_events']} | {'✅' if r['agree'] else '❌'} |")
    L.append("")
    L.append("## 다시 만들기\n")
    L.append("```")
    L.append(f".venv\\Scripts\\python.exe scripts\\validate_rules_pse.py --pse-root <pse-test-media 경로> --sets {' '.join(args.sets)}"
             + (f" --ffv1-check {args.ffv1_check}" if args.ffv1_check else "") + (f" --scale {args.scale}" if args.scale != 1 else ""))
    L.append("```")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
