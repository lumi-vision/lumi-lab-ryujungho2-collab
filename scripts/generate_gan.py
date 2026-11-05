"""훈련된 GAN으로 새로운 합성 후보를 만들되 학습 분할 밖으로 내보내지 않습니다."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from lumi.device import describe, resolve_device
from lumi.gan_synthesis import OracleDataset, load_generator


def main() -> None:
    parser = argparse.ArgumentParser(description="LUMI GAN 조건부 데이터 증강(영상 자동 재생 없음)")
    parser.add_argument("--dataset", type=Path, default=Path("datasets/oracle"))
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/lumi_cgan.pt"))
    parser.add_argument("--output", type=Path, default=Path("datasets/gan"))
    parser.add_argument("--per-clip", type=int, default=1)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    args = parser.parse_args()
    if args.per_clip < 1:
        parser.error("클립별 생성 개수는 1 이상이어야 합니다.")
    device = resolve_device(args.device)
    print(f"생성 장치: {describe(device)}")
    model = load_generator(args.checkpoint, device)
    dataset = OracleDataset(args.dataset, split="train")
    folder = args.output / "train"
    folder.mkdir(parents=True, exist_ok=True)
    manifest = []
    with torch.inference_mode():
        for item in dataset:
            for i in range(args.per_clip):
                clean = item["clean"].unsqueeze(0).to(device)
                mask = item["mask"].unsqueeze(0).to(device)
                activation = item["activation"].unsqueeze(0).to(device)
                mode = item["mode"].unsqueeze(0).to(device)
                out = model(clean, mask, activation, mode)[0]
                augmented = np.rint(out.permute(1, 2, 3, 0).cpu().numpy() * 255).astype(np.uint8)
                source_clean = np.rint(item["clean"].permute(1, 2, 3, 0).numpy() * 255).astype(np.uint8)
                source_mask = item["mask"][0].numpy().astype(np.uint8)
                augmented = np.where(source_mask[..., None] != 0, augmented, source_clean)
                name = f"{item['id']}_gan_{i:02d}"
                out_file = folder / f"{name}.npz"
                np.savez_compressed(out_file, clean=source_clean, affected=augmented,
                                    mask=source_mask, activation=item["activation"][0, :, 0, 0].numpy())
                row = {"id": name, "source_id": item["source_id"], "split": "train",
                       "type": item["type"], "file": f"train/{name}.npz",
                       "generator": "conditional_patchgan", "risk_verified": False,
                       "standard_result": "not_evaluated", "preview_safe": False}
                (folder / f"{name}.json").write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8")
                manifest.append(row)
    with (args.output / "manifest.jsonl").open("w", encoding="utf-8") as handle:
        for row in manifest:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"GAN 데이터 증강 완료: {len(manifest)}개 (train 분할만 생성)")
    print("주의: 생성 결과의 실제 위험성·기준 위반 여부는 별도의 도구로 검증해야 합니다.")


if __name__ == "__main__":
    main()
