"""G v2를 영상 하나에 직접 적용해 보는 시험 스크립트.

⚠ 출력 영상에는 광과민성 발작을 일으킬 수 있는 깜빡임이 들어 있다. 이 스크립트는 영상을 자동으로 재생하지 않는다.
  볼 때는 작은 창, 밝은 방, 한쪽 눈 가리기(pse-test-media 권고)를 지키고, 가능하면 플레이어의 프레임 단위 이동(N/P)으로 본다.
  광과민성이 있거나 의심되면 보지 않는다.

사용 예:
  # 검증용 배경(학습에 쓰지 않은 실제 안전 GIF) 중 하나로
  python scripts/try_gan_v2.py --bg-index 0
  # 내 영상으로 (깜빡임 없는 영상만)
  python scripts/try_gan_v2.py --video D:/clip.mp4 --roi 0.2,0.2,0.8,0.8 --seed 3

출력(<out>/): side_by_side_FLASHING.mp4 (원본 | 문턱 근처 목표 | 문턱 넘기 목표, 4배 확대),
              near_FLASHING.mkv, fail_FLASHING.mkv (무손실 원본 크기), input.mkv, result.json
보는 법: python -m lumi preview <out>/side_by_side_FLASHING.mp4 --allow-preview
          (일시정지로 열림 · SPACE 재생/정지 · N/P 한 프레임 · Q 종료)
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prep_real_clips import load_clip  # noqa: E402

from lumi.gan_v2 import TARGET_FAIL, TARGET_NEAR, GeneratorV2  # noqa: E402
from lumi.judge import judge_luminance_flash  # noqa: E402
from lumi.lossless_video import write_lossless_video  # noqa: E402


def roi_mask(spec: str, h: int, w: int) -> np.ndarray:
    m = np.zeros((h, w), np.uint8)
    if spec == "full":
        m[:] = 1
    elif spec == "center":
        m[h // 6: h - h // 6, w // 6: w - w // 6] = 1          # 화면의 약 44%
    else:
        x0, y0, x1, y1 = (float(v) for v in spec.split(","))
        m[int(y0 * h): int(y1 * h), int(x0 * w): int(x1 * w)] = 1
    return m


def panel(clip: np.ndarray, title: str, sub: str, scale: int) -> np.ndarray:
    t, h, w, _ = clip.shape
    big = np.stack([cv2.resize(f, (w * scale, h * scale), interpolation=cv2.INTER_NEAREST) for f in clip])
    bar = np.full((t, 56, w * scale, 3), 24, np.uint8)
    for i in range(t):
        cv2.putText(bar[i], title, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240, 240, 240), 2, cv2.LINE_AA)
        cv2.putText(bar[i], sub, (8, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 200, 255), 1, cv2.LINE_AA)
    return np.concatenate([bar, big], axis=1)


def main() -> None:
    ap = argparse.ArgumentParser(description="G v2 직접 시험 (재생은 사용자가 직접)")
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--video", type=Path, help="내 영상 파일(깜빡임 없는 영상)")
    src.add_argument("--bg-index", type=int, default=0, help="검증용 실제 배경 클립 번호(0~203)")
    ap.add_argument("--data", type=Path, default=Path("../_work/datasets/real_bg_v1"))
    ap.add_argument("--checkpoint", type=Path, default=Path("../_work/checkpoints/g_v2_e2b.pt"))
    ap.add_argument("--roi", default="center", help="center | full | x0,y0,x1,y1 (0~1 비율)")
    ap.add_argument("--seed", type=int, default=0, help="z 시드(바꾸면 깜빡임 방식이 달라짐)")
    ap.add_argument("--repeat", type=int, default=3, help="32프레임 클립을 몇 번 이어 붙일지(보기 편하게)")
    ap.add_argument("--scale", type=int, default=4)
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = ck["config"]
    fps, frames, h, w = cfg["fps"], cfg["frames"], cfg["height"], cfg["width_px"]
    G = GeneratorV2(cfg["width"], cfg["zdim"], frames, fps, cfg.get("output", "residual")).to(device).eval()
    G.load_state_dict(ck["generator"])

    if args.video:
        clean, _ = load_clip(args.video, fps, frames, w, h)
        if clean is None:
            raise SystemExit("영상을 읽지 못했습니다.")
        name = args.video.stem
    else:
        val = np.load(args.data / "val.npy", mmap_mode="r")
        clean = np.asarray(val[args.bg_index])
        srcs = [json.loads(l) for l in open(args.data / "manifest.jsonl", encoding="utf-8")]
        name = next(s["source"] for s in srcs if s["split"] == "val" and s["index"] == args.bg_index)
    j_in = judge_luminance_flash(clean, fps)
    if j_in.fail:
        print("경고: 입력 영상이 이미 판정상 깜빡임 실패입니다. G가 만든 것과 구분되지 않습니다.")

    mask = roi_mask(args.roi, h, w)
    out_dir = args.output or Path("../_work/trials") / f"{datetime.now():%Y%m%d_%H%M%S}_{name}"
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    z = torch.randn(1, cfg["zdim"], device=device)
    x = torch.from_numpy(clean).to(device).permute(3, 0, 1, 2).float()[None] / 255
    m = torch.from_numpy(mask).to(device).float()[None, None, None].expand(1, 1, frames, h, w)
    results, clips = {}, {}
    for key, tv, label in (("near", TARGET_NEAR, "문턱 근처 목표"), ("fail", TARGET_FAIL, "문턱 넘기 목표")):
        with torch.no_grad():
            o = G(x, m, torch.tensor([tv], device=device), z)
        clip = np.rint(o[0].permute(1, 2, 3, 0).float().cpu().numpy() * 255).astype(np.uint8)
        clip = np.where(mask[None, :, :, None] > 0, clip, clean)
        j = judge_luminance_flash(clip, fps)
        results[key] = {"max_transitions_1s": j.max_transitions_1s, "fail": j.fail,
                        "transition_frames": list(j.transition_frames), "max_area": round(j.max_area, 3)}
        clips[key] = clip
        write_lossless_video(out_dir / f"{key}_FLASHING.mkv", np.concatenate([clip] * args.repeat), fps)
    write_lossless_video(out_dir / "input.mkv", np.concatenate([clean] * args.repeat), fps)

    def sub(r):
        return f"judge: {'FAIL' if r['fail'] else 'pass'} | {r['max_transitions_1s']} transitions/1s"
    rep = lambda c: np.concatenate([c] * args.repeat)
    sbs = np.concatenate([
        panel(rep(clean), "input (no flash)", f"judge: pass | {j_in.max_transitions_1s} transitions/1s", args.scale),
        panel(rep(clips["near"]), "target: near threshold", sub(results["near"]), args.scale),
        panel(rep(clips["fail"]), "target: over threshold", sub(results["fail"]), args.scale),
    ], axis=2)
    vw = cv2.VideoWriter(str(out_dir / "side_by_side_FLASHING.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps,
                         (sbs.shape[2], sbs.shape[1]))
    for f in sbs:
        vw.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    vw.release()
    info = {"source": name, "roi": args.roi, "roi_area": round(float(mask.mean()), 3), "seed": args.seed,
            "checkpoint": str(args.checkpoint), "fps": fps, "frames": frames, "repeat": args.repeat,
            "input_judge": {"max_transitions_1s": j_in.max_transitions_1s, "fail": j_in.fail}, "outputs": results}
    (out_dir / "result.json").write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"원본: {name} · ROI {args.roi} (면적 {mask.mean():.0%}) · z 시드 {args.seed}")
    for key, label in (("near", "문턱 근처 목표"), ("fail", "문턱 넘기 목표")):
        r = results[key]
        print(f"  {label}: 판정 {'실패' if r['fail'] else '통과'}, 1초 최대 전이 {r['max_transitions_1s']}회")
    print(f"출력 폴더: {out_dir.resolve()}")
    print("⚠ *_FLASHING 파일은 깜빡임이 있다. 자동 재생하지 않았다. 볼 때:")
    print(f"  python -m lumi preview \"{(out_dir / 'side_by_side_FLASHING.mp4').resolve()}\" --allow-preview")
    print("  (일시정지로 열림 · SPACE 재생/정지 · N/P 한 프레임 · Q 종료 · 작은 창·밝은 방·한쪽 눈 가리기 권장)")


if __name__ == "__main__":
    main()
