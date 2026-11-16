"""위험 영상 생성기 묶음 — 플레이어의 '생성' 기능(scripts/make_playlist.py --serve)이 쓴다.

⚠️ 만든 영상은 광과민성 위험 자극이다. 이 PC 에서만 보고, 공유·발표 화면·녹화에 쓰지 않는다.

종류(kind)
    gan     학습한 생성기(체크포인트)가 만든 것. 체크포인트마다 종류가 하나다(config.hazard: lum 휘도 · red 적색 · stripe 줄무늬).
            같은 조건에서 z 를 바꿔 여러 개를 만든다
    lum     규칙 기반 휘도 깜빡임 (oracle.render_flicker: 선형 빛 더하기)
    red     규칙 기반 적색 깜빡임 (oracle.red_pair + render_red: 같은 휘도의 빨강 ↔ 파랑 쪽 색)
    stripe  규칙 기반 줄무늬 (oracle.render_stripes: 정지 · 반전 · 흐름)
    mix     위 넷을 무작위로 섞는다
GAN 은 종류별로 따로 학습한다(gan.py §3-2, 2026-11-14). 2026-11-14 이전 체크포인트는 모두 휘도 깜빡임이다.

의도(intent)
    hazard  위험이 되도록 만든 것
    near    조건 하나만 기준 아래로 둔 것 — 휘도·적색: 횟수(초당 1~2.5회), 줄무늬: 면적(20~35%)
의도는 정답이 아니다. 판정은 플레이어가 measure 로 따로 잰다(의도대로 됐는지 보려는 것).

조건을 비워 두면 seed 에서 무작위로 뽑는다. 뽑은 값은 모두 레시피에 적혀 같은 영상을 다시 만들 수 있다.
"""
from __future__ import annotations

import functools
import hashlib
import json
import time
import zipfile
from pathlib import Path

import numpy as np

from lumi_synth import measure, oracle

ROOT = Path(__file__).resolve().parents[2]
TRACK = ROOT / "data" / "oracle_768x432_24fps"
GAN_DIR = ROOT / "data" / "gan"
KINDS = ("gan", "lum", "red", "stripe")
KIND_KO = {"gan": "GAN", "lum": "규칙 휘도", "red": "규칙 적색", "stripe": "규칙 줄무늬", "mix": "섞음"}
HAZARD_KO = {"lum": "휘도", "red": "적색", "stripe": "줄무늬"}
INTENTS = ("hazard", "near")
SHAPES = ("ellipse", "rectangle", "blobs", "full")
SPLITS = ("val", "test", "train")


# ---- 바탕 -----------------------------------------------------------------------------------
def _npy_shape(npz_path: Path, key: str) -> tuple:
    """npz 안 배열의 모양만 읽는다(압축을 풀지 않음 — train 바탕은 풀면 2.9 GB)."""
    with zipfile.ZipFile(npz_path) as z, z.open(key + ".npy") as f:
        major, _ = np.lib.format.read_magic(f)
        read = np.lib.format.read_array_header_1_0 if major == 1 else np.lib.format.read_array_header_2_0
        return read(f)[0]


def background_counts(track: Path = TRACK) -> dict:
    """분할별 바탕 클립 수 {split: n} (파일이 없는 분할은 빠짐)."""
    out = {}
    for s in SPLITS:
        p = track / f"backgrounds_{s}.npz"
        if p.exists():
            out[s] = int(_npy_shape(p, "clips")[0])
    return out


@functools.lru_cache(maxsize=2)
def load_backgrounds(split: str, track: Path = TRACK) -> tuple[np.ndarray, float]:
    """backgrounds_<split>.npz → (클립 N×T×H×W×3 uint8 BGR, fps). 한 번 읽으면 기억해 둔다."""
    with np.load(track / f"backgrounds_{split}.npz") as z:
        return z["clips"], float(z["fps"])


# ---- GAN 체크포인트 ---------------------------------------------------------------------------
def list_checkpoints(gan_dir: Path = GAN_DIR) -> list[dict]:
    """data/gan/*.pt 목록(새것부터). 판정 규칙 기록이 없는 체크포인트는 2026-11-14 이전 규칙으로 학습한 것일 수 있다."""
    import torch
    out = []
    for p in sorted(gan_dir.glob("*.pt"), key=lambda q: q.stat().st_mtime, reverse=True):
        try:
            cfg = torch.load(p, map_location="cpu", weights_only=False)["config"]
        except Exception as e:  # noqa: BLE001 — 깨진 파일 하나가 목록을 막지 않게
            out.append({"name": p.name, "error": f"{type(e).__name__}: {e}"})
            continue
        rule = cfg.get("judge_rule")
        out.append({"name": p.name, "hazard": cfg.get("hazard", "lum"), "steps": cfg.get("steps"), "lambda_div": cfg.get("lambda_div"), "seed": cfg.get("seed"),
                    "train_hw": cfg.get("train_hw"), "fps": cfg.get("fps"), "soft_judge": cfg.get("soft_judge_version"),
                    "judge_rule": rule, "made": time.strftime("%Y-%m-%d %H:%M", time.localtime(p.stat().st_mtime))})
    return out


@functools.lru_cache(maxsize=4)
def _load_generator(path: str, mtime_ns: int):
    import torch
    from lumi_synth.gan import make_generator
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    g = make_generator(cfg.get("hazard", "lum"), int(cfg["width"]), int(cfg["zdim"]), float(cfg["fps"]),
                       red_palette=cfg.get("red_palette", False), lum_smooth=bool(cfg.get("lum_smooth", False)),
                       stripe_cond=cfg.get("stripe_cond", "mlp"))
    g.load_state_dict(ck["generator"])
    return g.eval(), cfg


def checkpoint_hazard(checkpoint: Path) -> str:
    """체크포인트가 만드는 위험 종류 (lum | red | stripe). 기록이 없는 옛 체크포인트는 휘도."""
    return _load_generator(str(checkpoint), checkpoint.stat().st_mtime_ns)[1].get("hazard", "lum")


def gan_render(checkpoint: Path, clean: np.ndarray, mask: np.ndarray, target: float, z_seed: int, fps: float) -> dict:
    """G(Y, M, c, z) → {'flicker': uint8 BGR, 'hazard': 종류, 종류별 요약}. z 는 z_seed 로 정해진다.

    요약 — 휘도: mean_light_cd(마스크 안 평균 추가 빛), 적색: mean_red(빨강으로 바꾼 정도 평균),
    줄무늬: cpd · angle(도) · speed_dps · hi_cd(G 가 고른 줄무늬 조건) · mean_cover(덮은 정도 평균)."""
    import torch
    g, cfg = _load_generator(str(checkpoint), checkpoint.stat().st_mtime_ns)
    hw = tuple(cfg.get("train_hw") or ())
    if hw and hw != clean.shape[1:3]:
        raise ValueError(f"해상도 불일치: G 학습 {hw[1]}×{hw[0]} ≠ 바탕 {clean.shape[2]}×{clean.shape[1]}")
    if abs(float(cfg["fps"]) - fps) > 1e-6:
        raise ValueError(f"fps 불일치: G 학습 {cfg['fps']} ≠ 바탕 {fps}")
    # 다른 학습이 GPU를 사용 중이면 추론을 강제로 얹지 않는다.
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        free_bytes, _ = torch.cuda.mem_get_info()
        use_cuda = free_bytes >= max(4 * 1024 ** 3, clean.nbytes * 64)
    dev = torch.device("cuda" if use_cuda else "cpu")
    g = g.to(dev)
    z = torch.randn(1, int(cfg["zdim"]), generator=torch.Generator().manual_seed(int(z_seed))).to(dev)
    with torch.no_grad():
        y = torch.from_numpy(clean[None]).to(dev).permute(0, 4, 1, 2, 3).float() / 255.0
        m = torch.from_numpy(mask[None, None, None]).to(dev).float()
        outs = g(y, m, torch.tensor([float(target)], device=dev), z)
        out, field = outs[0], outs[1]
        x = (out[0].permute(1, 2, 3, 0) * 255.0).round().clamp(0, 255).byte().cpu().numpy()
        mean_in = float((field[0, 0] * m[0, 0, 0]).sum() / max(1.0, float(m.sum()) * field.shape[2]))
        hazard = cfg.get("hazard", "lum")
        # 합성 비중 α(T×H×W): 원본 대신 합성한 것이 차지한 몫(손실 마스크 M_loss 의 근거). 휘도는 빛을 더하기만
        # 하므로 None(render 가 포화 픽셀로 따로 계산). 적색 v2+ 는 '빨강 아닌 상태'의 물듦 κ 까지 합친다.
        if hazard == "red" and len(outs) > 2:
            kappa = outs[2][:, 3][:, None, None, None, None]
            alpha = m * (1 - (1 - kappa) * (1 - field))
        elif hazard in ("red", "stripe"):
            alpha = m * field
        else:
            alpha = None
        alpha_np = None if alpha is None else alpha[0, 0].float().clamp(0, 1).cpu().numpy()
        info: dict = {"hazard": hazard, "inference_device": dev.type}
        if hazard == "lum":
            info["mean_light_cd"] = round(mean_in * measure.WHITE_CD, 1)
        elif hazard == "red":
            info["mean_red"] = round(mean_in, 3)
            if len(outs) > 2:                                       # 적색 v2: G 가 고른 색 조건 (gan.red_palette)
                info["palette"] = [round(float(v), 3) for v in outs[2][0]]
        else:
            cpd, theta, speed, hi = (float(v) for v in outs[2][0][:4])
            if outs[2].shape[1] > 4:                                # 줄무늬 v2: 반전 Hz · 움직임
                info.update(reversal_hz=round(float(outs[2][0][4]), 2),
                            motion=("static", "reversing", "drifting")[int(outs[2][0][7])])
            if outs[2].shape[1] >= 13:                              # 줄무늬 v5: 무늬 종류
                from lumi_synth.gan import PATTERN_SHAPES
                info["pattern"] = PATTERN_SHAPES[int(outs[2][0][9])]
            info.update(cpd=round(cpd, 2), angle=round(float(np.degrees(theta)), 1), speed_dps=round(speed, 2),
                        hi_cd=round(hi * measure.WHITE_CD, 1), mean_cover=round(mean_in, 3))
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    return {"flicker": x, "alpha": alpha_np, **info}


# ---- 레시피 -----------------------------------------------------------------------------------
def sample_recipe(kind: str, intent: str, seed: int, **fixed) -> dict:
    """조건을 seed 에서 뽑아 레시피를 만든다. fixed 에 준 값(None 이 아닌 것)은 그대로 쓴다.

    공통: split, bg, shape, area, center, aspect.  gan: checkpoint, z_seed.
    lum: freq_hz, waveform, delta_cd.  red: freq_hz, sat_ratio, delta_uv.
    stripe: cpd, angle, motion, hz(반전), speed_dps(흐름), lo_cd, hi_cd.
    """
    rng = np.random.default_rng(seed)
    if kind == "mix":
        kind = str(rng.choice(KINDS if fixed.get("checkpoint") else KINDS[1:]))
    if intent == "both":
        intent = str(rng.choice(INTENTS))
    if intent not in INTENTS:
        raise ValueError(f"intent 는 {INTENTS + ('both',)} 중 하나: {intent}")
    if kind not in KINDS:
        raise ValueError(f"kind 는 {KINDS + ('mix',)} 중 하나: {kind}")
    base = oracle.sample_params("hazard", rng)                       # 면적 0.32~0.70, 모양·중심·비율
    r = {"kind": kind, "intent": intent, "seed": int(seed), "shape": base.shape, "area": base.area,
         "center": list(base.center), "aspect": base.aspect}
    slow = float(rng.uniform(1.0, 2.5))                              # 근처 안전: 초당 1~2.5회 (기준 3회 초과)
    if kind == "gan":
        r.update(z_seed=int(rng.integers(0, 2 ** 31 - 1)))
        if fixed.get("checkpoint"):                                  # 체크포인트가 만드는 종류를 레시피에 적어 둔다
            r["gan_hazard"] = checkpoint_hazard(GAN_DIR / fixed["checkpoint"])
            if r["gan_hazard"] == "stripe" and r["shape"] != "full":  # 줄무늬 G 는 영역 45~85% 에서 학습했다(train_gan.sample_masks)
                r["area"] = float(rng.uniform(0.45, 0.85))
    elif kind == "lum":
        r.update(freq_hz=base.freq_hz if intent == "hazard" else slow, waveform=base.waveform, delta_cd=base.delta_cd)
    elif kind == "red":
        r.update(freq_hz=float(rng.uniform(4.0, 8.0)) if intent == "hazard" else slow,
                 sat_ratio=float(rng.uniform(0.82, 0.95)), delta_uv=float(rng.uniform(0.23, 0.32)))
    else:
        motion = str(rng.choice(oracle.STRIPE_MOTIONS))
        r.update(cpd=float(np.exp(rng.uniform(np.log(1.5), np.log(8.0)))), angle=float(rng.choice([0, 30, 45, 90, 135])),
                 motion=motion, hz=float(rng.uniform(3.0, 8.0)), speed_dps=float(rng.uniform(2.0, 6.0)),
                 lo_cd=10.0, hi_cd=float(rng.uniform(80.0, 140.0)),
                 area=float(rng.uniform(0.45, 0.85)) if intent == "hazard" else float(rng.uniform(0.20, 0.35)))
        if r["shape"] == "full":                                     # 줄무늬 면적은 의도가 정한다
            r["shape"] = "rectangle"
    if kind == "stripe" and fixed.get("freq_hz") is not None:        # 줄무늬에서 주파수 = 명암 반전 주파수
        fixed = {**fixed, "hz": fixed["freq_hz"]}
    for k, v in fixed.items():                                       # 이 종류에 없는 조건(예: 휘도에 cpd)은 무시
        if v is not None and (k in r or k in ("split", "bg", "source", "start_sec", "seconds", "max_side") or (k == "checkpoint" and kind == "gan")):
            r[k] = v
    if r["shape"] == "full":
        r["area"] = 1.0
    return r


def validate(recipe: dict, clean: np.ndarray, fps: float) -> None:
    """잘못된 조건·표본화 불가능한 주파수를 합성 전에 거부한다."""
    if clean.dtype != np.uint8 or clean.ndim != 4 or clean.shape[-1] != 3 or min(clean.shape[:3]) < 2:
        raise ValueError("원본은 T×H×W×3 uint8 BGR (각 축 2 이상)이어야 합니다")
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("fps 는 양의 유한수여야 합니다")
    if recipe.get("kind") not in KINDS or recipe.get("intent") not in INTENTS:
        raise ValueError("알 수 없는 kind 또는 intent")
    if recipe.get("shape") not in SHAPES:
        raise ValueError("알 수 없는 영역 모양")
    for key in ("area", "aspect"):
        if not np.isfinite(recipe[key]) or recipe[key] <= 0:
            raise ValueError(f"{key} 는 양의 유한수여야 합니다")
    if recipe["area"] > 1:
        raise ValueError("area 는 1 이하여야 합니다")
    center = np.asarray(recipe["center"])
    if center.shape != (2,) or not np.isfinite(center).all() or ((center < 0) | (center > 1)).any():
        raise ValueError("center 는 0~1 좌표 두 개여야 합니다")
    kind = recipe["kind"]
    freq = recipe.get("freq_hz") if kind in ("lum", "red") else recipe.get("hz") if kind == "stripe" and recipe.get("motion") == "reversing" else None
    if freq is not None and (not np.isfinite(freq) or not 0 < freq < fps / 2):
        raise ValueError(f"주파수는 0 초과, fps/2 ({fps / 2:g} Hz) 미만이어야 합니다")
    for key in ("delta_cd", "cpd", "speed_dps", "lo_cd", "hi_cd", "delta_uv", "sat_ratio"):
        if key in recipe and (not np.isfinite(recipe[key]) or recipe[key] < 0):
            raise ValueError(f"{key} 는 0 이상의 유한수여야 합니다")
    if kind == "lum" and recipe["waveform"] not in ("square", "sine", "triangle"):
        raise ValueError("알 수 없는 파형")
    if kind == "red" and not 0 < recipe["sat_ratio"] <= 1:
        raise ValueError("sat_ratio 는 0 초과 1 이하여야 합니다")
    if kind == "stripe":
        if recipe["motion"] not in oracle.STRIPE_MOTIONS or recipe["cpd"] <= 0:
            raise ValueError("줄무늬 움직임 또는 cpd 오류")
        if not np.isfinite(recipe["angle"]) or not recipe["lo_cd"] < recipe["hi_cd"] <= measure.WHITE_CD:
            raise ValueError("줄무늬 각도 또는 휘도 범위 오류")


def render(recipe: dict, clean: np.ndarray, fps: float) -> dict:
    """레시피 + 깨끗한 바탕(T×H×W×3 uint8 BGR) → {'flicker', 'mask', 'expected', 'detail'}."""
    validate(recipe, clean, fps)
    t_len, h, w = clean.shape[:3]
    r = recipe
    mask = oracle.region_mask(r["shape"], r["area"], h, w, center=r["center"], aspect=r["aspect"], seed=r["seed"])
    detail: dict = {"mask_area": float(mask.mean())}
    detail["clean_sha256"] = hashlib.sha256(clean.tobytes()).hexdigest()
    kind = r["kind"]
    if kind == "gan":
        detail["checkpoint_sha256"] = hashlib.sha256((GAN_DIR / r["checkpoint"]).read_bytes()).hexdigest()
        res = gan_render(GAN_DIR / r["checkpoint"], clean, mask, 1.0 if r["intent"] == "hazard" else 0.0, r["z_seed"], fps)
        out = res.pop("flicker")
        alpha = res.pop("alpha")
        detail.update({f"gan_{k}" if k != "hazard" else "gan_hazard": v for k, v in res.items()})
    elif kind == "lum":
        wv = oracle.wave(r["waveform"], r["freq_hz"], t_len, fps)
        out = oracle.render_flicker(clean, mask, wv, r["delta_cd"])
        alpha = None
    elif kind == "red":
        pair = oracle.red_pair(r["sat_ratio"], r["delta_uv"])
        out = oracle.render_red(clean, mask, oracle.wave("square", r["freq_hz"], t_len, fps), pair["a_rgb"], pair["b_rgb"])
        alpha = np.broadcast_to(mask, (t_len, h, w)).astype(np.float32)        # 평평한 두 색으로 덮음 = 원본 전부 대체
        detail.update(a_rgb=list(pair["a_rgb"]), b_rgb=list(pair["b_rgb"]), delta_uv_measured=round(pair["delta_uv"], 3))
    elif kind == "stripe":
        out = oracle.render_stripes(clean, mask, fps, r["cpd"], r["angle"], r["lo_cd"], r["hi_cd"], r["motion"],
                                    r["hz"], r["speed_dps"])
        alpha = np.broadcast_to(mask, (t_len, h, w)).astype(np.float32)        # 줄무늬로 덮음 = 원본 전부 대체
        period = oracle.stripe_period_px(w, h, r["cpd"], r["angle"])
        detail["pairs_in_mask"] = round(oracle.stripe_pairs_in_mask(mask, period, r["angle"]), 1)
    else:
        raise ValueError(f"kind: {kind}")
    if alpha is None:                                                         # 빛 더하기: 255 에 닿은 채널만 정보 손실
        alpha = clip_alpha(out, clean)
    return {"flicker": out, "mask": mask, "expected": r["intent"] == "hazard", "detail": detail, "alpha": alpha}


def clip_alpha(out: np.ndarray, clean: np.ndarray) -> np.ndarray:
    """빛을 더한 합성의 정보 손실: 바뀐 픽셀에서 255 로 포화된 채널의 비율(0, 1/3, 2/3, 1) — T×H×W float32."""
    sat = (out == 255) & (clean < 255)
    return sat.mean(axis=-1, dtype=np.float32)


def describe(recipe: dict) -> str:
    """목록에 보일 한 줄 설명."""
    r = recipe
    what = {"gan": f"{HAZARD_KO.get(r.get('gan_hazard', 'lum'), '?')} · z {r.get('z_seed')}",
            "lum": f"{r.get('freq_hz', 0):.1f} Hz {r.get('waveform')} +{r.get('delta_cd', 0):.0f} cd/m²",
            "red": f"{r.get('freq_hz', 0):.1f} Hz Δu′v′ {r.get('delta_uv', 0):.2f}",
            "stripe": f"{r.get('cpd', 0):.1f} cpd {r.get('angle', 0):.0f}° {r.get('motion')}"}[r["kind"]]
    background = Path(r["source"]).name if r.get("source") else f"{r['split']} {r['bg']:03d}"
    return (f"{KIND_KO[r['kind']]} · {'위험' if r['intent'] == 'hazard' else '근처 안전'} · {what} · "
            f"{r['shape']} {r['area']:.0%} · {background}")


def make(recipe: dict, out_dir: Path, *, clean: np.ndarray | None = None, fps: float | None = None) -> dict:
    """레시피대로 만들어 out_dir/<id>.npz 로 저장하고 {'path', 'flicker', 'mask', 'fps', 'expected', 'detail'} 를 돌려준다.

    npz v2: clean, flicker, mask_allow, mask_changed(T,H,W), fps, intent_hazard, recipe.
    clean 은 합성 전 원본이며 안전 판정이나 PLAN 의 B 정답을 의미하지 않는다.
    위험 라벨(hazard_*)은 만들지 않는다. 의도와 실제 판정은 별개다.
    """
    if clean is None:
        if recipe.get("source"):
            clean, fps = load_source(recipe)
        else:
            clips, fps = load_backgrounds(recipe["split"])
            bg = int(recipe["bg"])
            if not 0 <= bg < len(clips):
                raise ValueError("바탕 번호가 범위를 벗어났습니다")
            clean = clips[bg]
    if fps is None:
        raise ValueError("원본 클립과 함께 fps 를 지정해야 합니다")
    res = render(recipe, clean, fps)
    out_dir.mkdir(parents=True, exist_ok=True)
    sample_id = str(recipe["id"])
    if not sample_id or Path(sample_id).name != sample_id or "/" in sample_id or "\\" in sample_id:
        raise ValueError("id 는 경로가 아닌 파일 이름이어야 합니다")
    path = out_dir / f"{sample_id}.npz"
    exp = res["expected"]
    changed = np.any(res["flicker"] != clean, axis=-1)
    with path.open("xb") as f:
        np.savez_compressed(f, schema_version=np.int32(2), clean=clean, flicker=res["flicker"],
                            mask_allow=res["mask"], mask_changed=changed, fps=np.float64(fps),
                            intent_hazard=np.bool_(exp),
                            recipe=np.array(json.dumps({**recipe, **res["detail"]}, ensure_ascii=False)))
    return {"path": path, "fps": fps, "clean": clean, "mask_changed": changed, **res}


def load_source(recipe: dict) -> tuple[np.ndarray, float]:
    """외부 영상 구간을 원래 fps 로 읽는다. 공간 축소만 허용하며 fps 는 바꾸지 않는다."""
    import cv2
    import av
    from lumi_synth.data import decode_info, iter_frames_bgr

    path = Path(recipe["source"]).resolve()
    decode_info(path)  # 지원하지 않는 색 공간·HDR 은 기존 디코더 규칙대로 거부
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        rate = stream.average_rate or stream.guessed_rate
        if rate is None:
            raise ValueError("원본 fps 를 읽을 수 없습니다")
        fps = float(rate)
    start, seconds = float(recipe.get("start_sec", 0)), float(recipe.get("seconds", 2))
    max_side = int(recipe.get("max_side", 768))
    if not np.isfinite([start, seconds, fps]).all() or start < 0 or not 0 < seconds <= 10 or fps <= 0 or max_side < 2:
        raise ValueError("구간은 시작 ≥ 0, 길이 0~10초, fps > 0, max_side ≥ 2 이어야 합니다")
    frames = []
    for frame in iter_frames_bgr(path, int(round(start * fps)), max(2, int(round(seconds * fps)))):
        h, w = frame.shape[:2]
        if max(h, w) > max_side:
            scale = max_side / max(h, w)
            frame = cv2.resize(frame, (max(2, round(w * scale)), max(2, round(h * scale))), interpolation=cv2.INTER_AREA)
        frames.append(frame)
    if len(frames) < 2:
        raise ValueError("지정 구간에 프레임이 2장 이상 필요합니다")
    return np.stack(frames), fps
