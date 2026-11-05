"""CUDA가 없는 환경에서도 작은 GAN의 순전파와 학습 단계를 검증합니다."""

import pytest


def test_gan_shapes_and_mask_identity():
    torch = pytest.importorskip("torch")
    from lumi.gan_synthesis import ConditionalGenerator, ConditionalDiscriminator, mode_tensor
    torch.set_num_threads(2)
    gen = ConditionalGenerator(width=8)
    disc = ConditionalDiscriminator(width=8)
    clean = torch.rand(1, 3, 4, 32, 32)
    mask = torch.zeros(1, 1, 4, 32, 32)
    mask[:, :, :, 5:19, 6:20] = 1
    activation = torch.ones_like(mask)
    mode = mode_tensor(["luminance"])
    result = gen(clean, mask, activation, mode)
    assert result.shape == clean.shape
    assert torch.allclose(result * (1 - mask), clean * (1 - mask))
    assert disc(clean, result, mask, activation, mode).shape[0:2] == (1, 1)


def test_single_gan_training_step():
    torch = pytest.importorskip("torch")
    from lumi.gan_synthesis import ConditionalGenerator, ConditionalDiscriminator, mode_tensor, gan_step
    torch.set_num_threads(2)
    generator = ConditionalGenerator(width=8)
    discriminator = ConditionalDiscriminator(width=8)
    clean = torch.rand(1, 3, 4, 32, 32)
    mask = torch.zeros(1, 1, 4, 32, 32)
    mask[:, :, :, 8:24, 8:24] = 1
    batch = {"clean": clean, "affected": torch.clamp(clean + mask * 0.1, 0, 1),
             "mask": mask, "activation": torch.ones_like(mask),
             "mode": mode_tensor(["luminance"])}
    result = gan_step(generator, discriminator, batch,
                      torch.optim.Adam(generator.parameters(), lr=1e-4),
                      torch.optim.Adam(discriminator.parameters(), lr=1e-4), device=torch.device("cpu"))
    assert all(np_is_finite(v) for v in result.values())


def np_is_finite(value):
    return value == value and abs(value) < 1e9


def _tiny_batch(torch, device="cpu"):
    from lumi.gan_synthesis import mode_tensor
    clean = torch.rand(1, 3, 4, 32, 32)
    mask = torch.zeros(1, 1, 4, 32, 32)
    mask[:, :, :, 8:24, 8:24] = 1
    return {"clean": clean, "affected": torch.clamp(clean + mask * 0.1, 0, 1),
            "mask": mask, "activation": torch.ones_like(mask),
            "mode": mode_tensor(["luminance"])}


def test_resolve_device_auto_and_missing_cuda():
    torch = pytest.importorskip("torch")
    from lumi.device import resolve_device
    assert resolve_device("cpu").type == "cpu"
    assert resolve_device("auto").type == ("cuda" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        with pytest.raises(RuntimeError):
            resolve_device("cuda")


@pytest.mark.parametrize("device_name", ["cpu", "cuda"])
def test_bf16_gan_step(device_name):
    torch = pytest.importorskip("torch")
    if device_name == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA 없음 — 집 GPU PC에서 실행됨")
    from lumi.gan_synthesis import ConditionalGenerator, ConditionalDiscriminator, gan_step
    device = torch.device(device_name)
    generator = ConditionalGenerator(width=8).to(device)
    discriminator = ConditionalDiscriminator(width=8).to(device)
    result = gan_step(generator, discriminator, _tiny_batch(torch),
                      torch.optim.Adam(generator.parameters(), lr=1e-4),
                      torch.optim.Adam(discriminator.parameters(), lr=1e-4),
                      device=device, amp_dtype=torch.bfloat16)
    assert all(np_is_finite(v) for v in result.values())
    # bf16 순전파여도 파라미터는 fp32로 유지되어야 한다.
    assert all(p.dtype == torch.float32 for p in generator.parameters())
