"""기존 BBB 검증 쌍의 전체 변경 마스크를 대조하고 가운데 한 프레임만 발표용으로 저장한다."""

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def main():
    root = Path(__file__).resolve().parents[2]
    report = root / "checks/gan/generator_pipeline_2026-11-14.json"
    record = json.loads(report.read_text(encoding="utf-8"))[0]
    pair = root / "data/generator_review_20261114" / Path(record["pair"]).name
    with np.load(pair, allow_pickle=False) as sample:
        source, result, allow = sample["clean"], sample["flicker"], sample["mask_allow"]
        changed = np.any(source != result, axis=-1)
        if not np.array_equal(changed, sample["mask_changed"]):
            raise ValueError("저장된 변경 마스크와 재계산 결과가 다릅니다.")
        outside = int((changed & ~allow).sum())
        altered = int(changed.any(axis=(1, 2)).sum())
        index = len(source) // 2
        originals = (source[index], result[index])
        fps = float(sample["fps"])
    page = Image.new("RGB", (1600, 740), "#f5f3ec")
    draw = ImageDraw.Draw(page)
    def label(x, y, value, size):
        draw.text((x, y), value, font=ImageFont.truetype("malgun.ttf", size), fill="#162f31")
    label(24, 16, "기존 G가 영화 클립에 적용한 국소 변화", 38)
    label(24, 84, f"Big Buck Bunny · 기존 검증 바탕 0 · {len(source)}프레임 중 한 장 (인덱스 {index}) · {fps:g} fps", 23)
    for x, name, frame in zip((24, 808), ("원본", "학습된 휘도 G 출력"), originals):
        label(x, 132, name, 22)
        page.paste(Image.fromarray(frame[:, :, ::-1]), (x, 176))
    label(24, 627, f"전체 클립 재검사   |   변화가 있는 프레임 {altered} / {len(source)}장   ·   마스크 밖 변경 {outside:,}픽셀", 28)
    label(24, 695, "© Blender Foundation | peach.blender.org | CC BY 3.0 · 원본 발췌 및 국소 밝기 변환", 17)
    page.save(root / "docs/G_기존_클립_검증.png")


if __name__ == "__main__":
    main()
