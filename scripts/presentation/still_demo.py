"""발표용 단일 프레임 G. 실행파일에는 예시 원본과 CPU 추론 모델을 함께 넣는다."""

import argparse
import json
import sys
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import av
import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw, ImageFont, ImageOps, ImageTk


SIZE = (768, 432)
PAPER = "#f5f3ec"
INK = "#162f31"
TEAL = "#13695e"
EXAMPLE = "Big Buck Bunny · 원본 프레임 720 (30초)"
CREDIT = "© Blender Foundation | peach.blender.org | CC BY 3.0 · 원본 발췌 및 국소 밝기 변환"


def load_scene(path):
    """이미지 한 장 또는 영상의 첫 표시 프레임을 원래 비율로 맞춘다."""
    if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}:
        with Image.open(path) as image:
            source = ImageOps.exif_transpose(image).convert("RGB")
    else:
        with av.open(str(path)) as container:
            stream = next(iter(container.streams.video), None)
            frame = next(container.decode(stream), None) if stream is not None else None
            if frame is None:
                raise ValueError("표시할 영상 프레임이 없습니다.")
            if frame.format.name not in {"yuv420p", "yuvj420p", "yuv422p", "yuv444p", "rgb24", "bgr24"} or int(frame.color_trc or 0) in (16, 18):
                raise ValueError("이 발표판은 8비트 SDR 원본을 지원합니다.")
            source = frame.to_image()
    fitted = ImageOps.contain(source, SIZE, Image.Resampling.LANCZOS)
    left, top = (SIZE[0] - fitted.width) // 2, (SIZE[1] - fitted.height) // 2
    fitted_frame = Image.new("RGB", SIZE)
    fitted_frame.paste(fitted, (left, top))
    return fitted_frame, (left, top, left + fitted.width, top + fitted.height)


def predict(session, source, bounds, center):
    """실제 G 출력의 양자화 결과와 원본을 비교한다. 여백은 마스크에서 제외한다."""
    left, top, right, bottom = bounds
    x, y = center
    rx, ry = max(1, (right-left) // 6), max(1, (bottom-top) // 5)
    rows, cols = np.ogrid[:SIZE[1], :SIZE[0]]
    region = ((cols-x)/rx)**2 + ((rows-y)/ry)**2 <= 1
    region &= (cols >= left) & (cols < right) & (rows >= top) & (rows < bottom)
    rgb = np.asarray(source)
    frame = np.ascontiguousarray(rgb[:, :, ::-1].transpose(2, 0, 1)[None], dtype=np.float32) / 255
    generated, = session.run(["output"], {"frame": frame, "mask": region[None, None].astype(np.float32)})
    if not np.isfinite(generated).all():
        raise ValueError("G 출력에 유효하지 않은 수가 있습니다.")
    result = np.rint(255 * generated.clip(0, 1))[0, ::-1].transpose(1, 2, 0).astype(np.uint8)
    changed = np.any(rgb != result, axis=2)
    counts = {"roi": int(region.sum()), "inside": int((changed & region).sum()),
              "outside": int((changed & ~region).sum())}
    marked = source.copy()
    # 실제 마스크의 내부 경계만 표시해 영상 끝과 여백에서도 추론 범위와 일치시킨다.
    padded = np.pad(region, 1)
    edge = region & ~(padded[:-2, 1:-1] & padded[2:, 1:-1] & padded[1:-1, :-2] & padded[1:-1, 2:])
    marked.paste((63, 232, 188), mask=Image.fromarray(edge))
    return marked, Image.fromarray(result), counts


class Demo:
    def __init__(self, root, assets):
        self.root, self.assets = root, assets
        root.title("LUMI G | 국소 합성 시연")
        root.configure(bg=PAPER)
        root.geometry("1080x630")
        options = ort.SessionOptions()
        options.intra_op_num_threads = 4
        self.session = ort.InferenceSession(str(assets / "still.onnx"), options, providers=["CPUExecutionProvider"])
        self.main = tk.Frame(root, bg=PAPER, padx=24, pady=16)
        self.main.pack(fill="both", expand=True)
        self.label("LUMI   /   GENERATOR", size=11, color=TEAL, bold=True).pack(anchor="w")
        self.label("원본의 어느 부분이 바뀌었나?", size=23, bold=True).pack(anchor="w")
        bar = tk.Frame(self.main, bg=PAPER)
        bar.pack(fill="x", pady=(8, 12))
        ttk.Button(bar, text="원본 파일 선택…", command=self.choose).pack(side="left")
        ttk.Button(bar, text="BBB 예시 열기", command=self.example).pack(side="left", padx=6)
        self.show_changes = tk.BooleanVar(value=False)
        ttk.Checkbutton(bar, text="변경 픽셀 강조", variable=self.show_changes,
                        command=self.redraw).pack(side="left", padx=8)
        ttk.Button(bar, text="발표용 PNG 저장…", command=self.save).pack(side="right")
        self.scene = self.label("", size=12, bold=True)
        self.scene.pack(anchor="w")
        self.label("왼쪽 원본을 클릭하면 그 위치에서 휘도 G를 다시 실행합니다.", size=10).pack(anchor="w")
        pair = tk.Frame(self.main, bg=PAPER)
        pair.pack(fill="both", expand=True, pady=(10, 4))
        pair.rowconfigure(1, weight=1)
        self.panels = []
        for index, title in enumerate(("원본  ·  입력 마스크", "G 출력  ·  정지 프레임")):
            pair.columnconfigure(index, weight=1, uniform="pair")
            heading = self.label(title, master=pair, size=11, bold=True)
            heading.grid(row=0, column=index, sticky="w")
            if index == 1:
                self.output_heading = heading
            canvas = tk.Canvas(pair, bg=INK, highlightthickness=0, width=220, height=125)
            canvas.grid(row=1, column=index, sticky="nsew", padx=(0, 4) if index == 0 else (4, 0))
            canvas.bind("<Configure>", lambda event: self.redraw())
            self.panels.append(canvas)
        self.panels[0].bind("<Button-1>", self.click)
        self.stats = self.label("", size=14, color=TEAL, bold=True)
        self.stats.pack(anchor="w", pady=(6, 2))
        self.label("사용자가 위치를 지정합니다. 변경량은 8비트 RGB에서 원본과 값이 다른 픽셀 수입니다.", size=10).pack(anchor="w")
        self.credit = self.label("", size=9)
        self.credit.pack(anchor="w")
        self.example()

    def label(self, value, *, master=None, size, color=INK, bold=False):
        return tk.Label(master or self.main, text=value, bg=PAPER, fg=color, anchor="w",
                        font=("Malgun Gothic", size, "bold" if bold else "normal"))

    def example(self):
        self.load(self.assets / "example.png", EXAMPLE, CREDIT)

    def load(self, path, name, credit):
        source, bounds = load_scene(path)
        result = predict(self.session, source, bounds, (SIZE[0]//2, SIZE[1]//2))
        self.source, self.bounds = source, bounds
        self.scene.config(text=name)
        self.credit.config(text=credit)
        self.show(result)

    def choose(self):
        chosen = filedialog.askopenfilename(title="원본 한 장 또는 영상 파일 선택",
                    filetypes=[("영상 및 이미지", "*.mp4 *.mov *.mkv *.webm *.png *.jpg *.jpeg *.bmp"), ("전체", "*.*")])
        if chosen:
            self.root.config(cursor="watch")
            self.root.update_idletasks()
            try:
                path = Path(chosen)
                self.load(path, path.name, "사용자 선택 파일 · 단일 이미지 또는 영상의 첫 표시 프레임")
            except Exception as error:
                messagebox.showerror("원본 열기 실패", str(error))
            finally:
                self.root.config(cursor="")

    def show(self, result):
        self.marked, self.result, self.counts = result
        changed = np.any(np.asarray(self.source) != np.asarray(self.result), axis=2)
        self.overlay = ImageOps.grayscale(self.source).convert("RGB")
        self.overlay.paste((0, 222, 169), mask=Image.fromarray(changed))
        self.stats.config(text=f"마스크 안 변경   {self.counts['inside']:,} / {self.counts['roi']:,} px"
                          f"       마스크 밖 변경   {self.counts['outside']:,} px")
        self.redraw()

    def redraw(self):
        if not hasattr(self, "marked"):
            return
        self.photos = []
        self.output_heading.configure(text="원본 대비 변경 픽셀  ·  초록색" if self.show_changes.get() else "G 출력  ·  정지 프레임")
        right = self.overlay if self.show_changes.get() else self.result
        for panel, image in zip(self.panels, (self.marked, right)):
            width, height = max(2, panel.winfo_width()), max(2, panel.winfo_height())
            display = ImageOps.contain(image, (width, height), Image.Resampling.LANCZOS)
            left, top = (width-display.width)//2, (height-display.height)//2
            photo = ImageTk.PhotoImage(display)
            self.photos.append(photo)
            panel.delete("all")
            panel.create_image(left, top, anchor="nw", image=photo)
            if panel is self.panels[0]:
                self.viewport = left, top, display.width, display.height

    def click(self, event):
        left, top, width, height = self.viewport
        x, y = (event.x-left)*SIZE[0]//width, (event.y-top)*SIZE[1]//height
        x0, y0, x1, y1 = self.bounds
        if x0 <= x < x1 and y0 <= y < y1:
            self.show(predict(self.session, self.source, self.bounds, (x, y)))

    def save(self):
        path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("정지 화면", "*.png")])
        if path:
            try:
                self.save_panel(Path(path))
            except OSError as error:
                messagebox.showerror("저장 실패", str(error))

    def save_panel(self, path):
        page = Image.new("RGB", (1680, 630), PAPER)
        draw = ImageDraw.Draw(page)
        def line(x, y, value, size, color=INK):
            draw.text((x, y), value, fill=color, font=ImageFont.truetype("malgun.ttf", size))
        line(24, 15, "LUMI  /  국소 합성 G", 38)
        line(24, 78, self.scene.cget("text"), 22)
        panels = [(self.marked, "원본과 입력 마스크"), (self.result, "학습된 휘도 G의 단일 프레임 출력"),
                  (self.overlay, "원본 대비 변경 픽셀 · 초록색")]
        for index, (image, title) in enumerate(panels):
            x = 24 + 550 * index
            line(x, 130, title, 21)
            page.paste(image.resize((536, 302), Image.Resampling.LANCZOS), (x, 172))
        line(24, 492, self.stats.cget("text"), 30, TEAL)
        line(24, 553, "변경 픽셀은 768 × 432 해상도에서 8비트 RGB 값으로 계산했습니다.", 18)
        line(24, 587, self.credit.cget("text"), 17)
        page.save(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets", type=Path, default=Path(getattr(sys, "_MEIPASS", Path(__file__).parent)))
    parser.add_argument("--verify", type=Path, help="화면을 열지 않고 기본 시연과 영역 변경을 검사할 새 폴더")
    args = parser.parse_args()
    root = tk.Tk()
    if args.verify:
        root.withdraw()
    app = Demo(root, args.assets)
    if args.verify:
        args.verify.mkdir(parents=True, exist_ok=False)
        app.save_panel(args.verify / "LUMI_G.png")
        original = app.result.copy()
        first = app.counts
        # 창 크기에 맞춘 좌표를 실제 클릭 처리 경로에 전달한다.
        from types import SimpleNamespace
        app.viewport = (0, 0, *SIZE)
        app.click(SimpleNamespace(x=240, y=180))
        if not all(row["inside"] > 0 and row["outside"] == 0 for row in (first, app.counts)):
            raise ValueError("국소 합성 검사가 실패했습니다.")
        if np.array_equal(np.asarray(original), np.asarray(app.result)):
            raise ValueError("위치를 옮긴 뒤 출력이 동일합니다.")
        (args.verify / "verification.json").write_text(json.dumps([first, app.counts], indent=2), encoding="utf-8")
        root.destroy()
    else:
        root.mainloop()


if __name__ == "__main__":
    main()
