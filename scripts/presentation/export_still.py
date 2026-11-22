"""기존 휘도 G에서 한 프레임 추론만 고정하여 발표용 가중치를 내보낸다."""

import argparse
import hashlib
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from PIL import Image

from lumi_synth.data import ClipSpec, read_clip
from lumi_synth.gan import LightGenerator


ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT = ROOT / "data/gan/w_div2.0_2k_s0.pt"
EXPECTED_HASH = "6f61f747e78b92efff2e3922aff084e16c475f6f5d0e7bd7950e097a097b01da"


class StillView(torch.nn.Module):
    """시간축 길이를 1로 고정한 기존 휘도 G의 발표용 입출력."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, frame, mask):
        source = frame.unsqueeze(2)
        result, _ = self.model(source, mask.unsqueeze(2), source.new_zeros(1),
                               source.new_zeros(1, self.model.zdim))
        return result.squeeze(2)


def export_assets(output):
    output.mkdir(parents=True, exist_ok=True)
    if hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest() != EXPECTED_HASH:
        raise ValueError("등록된 체크포인트와 파일 해시가 다릅니다.")
    checkpoint = torch.load(CHECKPOINT, weights_only=True, map_location="cpu")
    model = LightGenerator(width=checkpoint["config"]["width"], zdim=checkpoint["config"]["zdim"])
    model.load_state_dict(checkpoint["generator"], strict=True)
    view = StillView(model).eval()
    movie = ROOT / "data/sources/big_buck_bunny_1080p/big_buck_bunny_1080p_h264.mov"
    clip = read_clip(movie, 720, ClipSpec(frames=1))
    if clip is None:
        raise ValueError("원본에서 720번째 프레임을 읽지 못했습니다.")
    Image.fromarray(clip[0, :, :, ::-1]).save(output / "example.png")
    frame = torch.from_numpy(clip.copy()).permute(0, 3, 1, 2).float() / 255
    mask = torch.zeros_like(frame[:, :1])
    mask[:, :, 124:308, 280:488] = 1
    path = output / "still.onnx"
    with torch.inference_mode():
        torch.onnx.export(view, (frame, mask), str(path), input_names=["frame", "mask"],
                          output_names=["output"], opset_version=18, dynamo=False)
        session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        for offset in (0, 160):
            region = mask.roll(offset, dims=3)
            expected = view(frame, region).numpy()
            actual, = session.run(None, {"frame": frame.numpy(), "mask": region.numpy()})
            error = np.abs(expected - actual)
            np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=0)
            print(f"영역 이동 {offset}px, PyTorch와 내보낸 모델의 최대 절대 오차 {error.max():.8f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    export_assets(parser.parse_args().output)
