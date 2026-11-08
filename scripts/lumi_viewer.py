"""LUMI G v2 뷰어 — 영상 고르기 → G 적용 → 미리보기를 한 번에.

실행:  ..\\_work\\.venv\\Scripts\\python.exe scripts\\lumi_viewer.py   (저장소 폴더에서)
       또는 이 파일을 더블클릭(venv가 ..\\_work\\.venv에 있으면 자동으로 그 파이썬으로 다시 실행)

⚠ 결과 화면에는 광과민성 발작을 일으킬 수 있는 깜빡임이 나온다. 시작 전에 확인 창이 뜬다.
  작은 창·밝은 방·한쪽 눈 가리기를 권장하고, 광과민성이 있거나 의심되면 보지 않는다.

창 안 키:  SPACE 재생/정지 · N / P 한 프레임 앞/뒤 · Z 다른 깜빡임(z) · R ROI 바꾸기
           O 다른 영상 열기 · S 저장(_work/trials) · Q 또는 ESC 종료
"""
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
VENV_PY = REPO.parent / "_work" / ".venv" / "Scripts" / "python.exe"

# venv가 아닌 파이썬(예: 더블클릭 시 시스템 파이썬)으로 실행됐으면 venv 파이썬으로 다시 실행한다.
# (venv의 python.exe는 기본 파이썬을 가리키는 런처라 경로 대신 sys.prefix로 판별한다)
if VENV_PY.exists() and Path(sys.prefix).resolve() != VENV_PY.parents[1].resolve():
    sys.exit(subprocess.call([str(VENV_PY), str(Path(__file__).resolve()), *sys.argv[1:]]))

sys.path.insert(0, str(REPO / "src"))

import json  # noqa: E402
from datetime import datetime  # noqa: E402
import tkinter as tk  # noqa: E402
from tkinter import filedialog, messagebox  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from lumi.gan_v2 import TARGET_FAIL, TARGET_NEAR, GeneratorV2  # noqa: E402
from lumi.judge import judge_luminance_flash  # noqa: E402

CHECKPOINT = REPO.parent / "_work" / "checkpoints" / "g_v2_e2b.pt"
TRIALS = REPO.parent / "_work" / "trials"
ROIS = [("center", (1 / 6, 1 / 6, 5 / 6, 5 / 6)), ("full", (0, 0, 1, 1)),
        ("left", (0.0, 0.1, 0.6, 0.9)), ("right", (0.4, 0.1, 1.0, 0.9)), ("top", (0.1, 0.0, 0.9, 0.6))]
SCALE, REPEAT, WINDOW = 3, 3, "LUMI G v2 viewer"


def ask_video() -> Path | None:
    root = tk.Tk(); root.withdraw(); root.attributes("-topmost", True)
    p = filedialog.askopenfilename(title="깜빡임이 없는 영상을 고르세요",
                                   filetypes=[("영상", "*.mp4 *.avi *.mov *.mkv *.gif *.webm"), ("모든 파일", "*.*")])
    root.destroy()
    return Path(p) if p else None


def confirm_warning() -> bool:
    root = tk.Tk(); root.withdraw(); root.attributes("-topmost", True)
    ok = messagebox.askyesno("광과민성 경고",
                             "결과 화면에는 발작을 일으킬 수 있는 깜빡임이 나옵니다.\n\n"
                             "· 광과민성이 있거나 의심되면 보지 마세요.\n"
                             "· 작은 창, 밝은 방, 한쪽 눈 가리기를 권장합니다.\n"
                             "· 화면은 일시정지 상태로 열립니다(N/P로 한 프레임씩).\n\n계속할까요?")
    root.destroy()
    return ok


def load_clip(path: Path, fps: float, frames: int, w: int, h: int) -> np.ndarray | None:
    """실제 시각 기준으로 fps에 재샘플, 짧으면 반복, 가운데 4:3 자르기 후 w×h로."""
    cap = cv2.VideoCapture(str(path))
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 10.0
    src = []
    while len(src) < 2000:
        ok, f = cap.read()
        if not ok:
            break
        src.append(f)
    cap.release()
    if len(src) < 2:
        return None
    h0, w0 = src[0].shape[:2]
    cw, ch = (h0 * 4 // 3, h0) if w0 * 3 > h0 * 4 else (w0, w0 * 3 // 4)
    x0, y0 = (w0 - cw) // 2, (h0 - ch) // 2
    dur = len(src) / fps_in
    out = []
    for i in range(frames):
        f = src[min(len(src) - 1, int(((i / fps) % dur) * fps_in))]
        f = cv2.resize(f[y0:y0 + ch, x0:x0 + cw], (w, h), interpolation=cv2.INTER_AREA)
        out.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    return np.stack(out)


class Viewer:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ck = torch.load(CHECKPOINT, map_location=self.device, weights_only=False)
        c = ck["config"]
        self.fps, self.frames, self.h, self.w, self.zdim = c["fps"], c["frames"], c["height"], c["width_px"], c["zdim"]
        self.G = GeneratorV2(c["width"], c["zdim"], self.frames, self.fps, c.get("output", "residual")).to(self.device).eval()
        self.G.load_state_dict(ck["generator"])
        self.roi_i, self.seed, self.name, self.clean = 0, 0, "", None

    def mask(self):
        x0, y0, x1, y1 = ROIS[self.roi_i][1]
        m = np.zeros((self.h, self.w), np.uint8)
        m[int(y0 * self.h): int(y1 * self.h), int(x0 * self.w): int(x1 * self.w)] = 1
        return m

    def generate(self):
        m = self.mask()
        torch.manual_seed(self.seed)
        z = torch.randn(1, self.zdim, device=self.device)
        x = torch.from_numpy(self.clean).to(self.device).permute(3, 0, 1, 2).float()[None] / 255
        mt = torch.from_numpy(m).to(self.device).float()[None, None, None].expand(1, 1, self.frames, self.h, self.w)
        self.outs, self.judges = {}, {}
        for key, tv in (("near", TARGET_NEAR), ("fail", TARGET_FAIL)):
            with torch.no_grad():
                o = self.G(x, mt, torch.tensor([tv], device=self.device), z)
            clip = np.rint(o[0].permute(1, 2, 3, 0).float().cpu().numpy() * 255).astype(np.uint8)
            self.outs[key] = np.where(m[None, :, :, None] > 0, clip, self.clean)
            self.judges[key] = judge_luminance_flash(self.outs[key], self.fps)
        self.judges["input"] = judge_luminance_flash(self.clean, self.fps)
        self.frames_bgr = self.compose()

    def compose(self):
        def label(j):
            return f"{'FAIL' if j.fail else 'pass'} | {j.max_transitions_1s} transitions/1s"
        panels = [("input", self.clean, label(self.judges["input"])),
                  ("target: near threshold", self.outs["near"], label(self.judges["near"])),
                  ("target: over threshold", self.outs["fail"], label(self.judges["fail"]))]
        out = []
        for i in range(self.frames * REPEAT):
            row = []
            for title, clip, sub in panels:
                f = cv2.resize(clip[i % self.frames], (self.w * SCALE, self.h * SCALE), interpolation=cv2.INTER_NEAREST)
                bar = np.full((50, self.w * SCALE, 3), 24, np.uint8)
                cv2.putText(bar, title, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 240, 240), 1, cv2.LINE_AA)
                cv2.putText(bar, sub, (6, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 150), 1, cv2.LINE_AA)
                row.append(np.concatenate([bar, f], 0))
            frame = np.concatenate(row, 1)
            foot = np.full((28, frame.shape[1], 3), 24, np.uint8)
            cv2.putText(foot, f"{self.name} | ROI {ROIS[self.roi_i][0]} ({self.mask().mean():.0%}) | z seed {self.seed} | "
                              "SPACE play  N/P step  Z new z  R ROI  O open  S save  Q quit",
                        (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
            out.append(cv2.cvtColor(np.concatenate([frame, foot], 0), cv2.COLOR_RGB2BGR))
        return out

    def open(self, path: Path) -> bool:
        clip = load_clip(path, self.fps, self.frames, self.w, self.h)
        if clip is None:
            messagebox.showerror("오류", f"영상을 읽지 못했습니다:\n{path}")
            return False
        self.clean, self.name = clip, path.name
        self.generate()
        return True

    def save(self):
        d = TRIALS / f"{datetime.now():%Y%m%d_%H%M%S}_{Path(self.name).stem}"
        d.mkdir(parents=True, exist_ok=True)
        vw = cv2.VideoWriter(str(d / "side_by_side_FLASHING.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), self.fps,
                             self.frames_bgr[0].shape[1::-1])
        for f in self.frames_bgr:
            vw.write(f)
        vw.release()
        info = {"source": self.name, "roi": ROIS[self.roi_i][0], "seed": self.seed,
                **{k: {"fail": j.fail, "max_transitions_1s": j.max_transitions_1s} for k, j in self.judges.items()}}
        (d / "result.json").write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")
        return d

    def run(self):
        cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
        i, playing, msg = 0, False, ""
        while True:
            f = self.frames_bgr[i].copy()
            state = "PLAY" if playing else "PAUSED"
            cv2.putText(f, f"{state}  frame {i % self.frames + 1}/{self.frames}  {msg}", (6, f.shape[0] - 34),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 255, 120), 1, cv2.LINE_AA)
            cv2.imshow(WINDOW, f)
            key = cv2.waitKey(int(1000 / self.fps) if playing else 50) & 0xFF
            if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1 or key in (ord("q"), 27):
                break
            if key == ord(" "):
                playing = not playing
            elif key == ord("n"):
                playing, i = False, (i + 1) % len(self.frames_bgr)
            elif key == ord("p"):
                playing, i = False, (i - 1) % len(self.frames_bgr)
            elif key == ord("z"):
                self.seed += 1; self.generate(); playing, i, msg = False, 0, "new z"
            elif key == ord("r"):
                self.roi_i = (self.roi_i + 1) % len(ROIS); self.generate(); playing, i, msg = False, 0, "new ROI"
            elif key == ord("o"):
                p = ask_video()
                if p and self.open(p):
                    playing, i, msg = False, 0, "opened"
            elif key == ord("s"):
                msg = f"saved: {self.save().name}"
            elif playing:
                i = (i + 1) % len(self.frames_bgr)
        cv2.destroyAllWindows()


def main():
    if not CHECKPOINT.exists():
        raise SystemExit(f"체크포인트가 없습니다: {CHECKPOINT}")
    if not confirm_warning():
        return
    v = Viewer()
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else ask_video()
    if path is None or not v.open(path):
        return
    v.run()


if __name__ == "__main__":
    main()
