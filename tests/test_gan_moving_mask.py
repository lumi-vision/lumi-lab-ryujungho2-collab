import pytest
import torch

from lumi_synth.gan import make_generator, moving_mask, spectral_diversity


def test_mask_moves_without_wrap_and_zero_velocity_matches():
    mask = torch.zeros(2, 1, 1, 16, 20)
    mask[:, :, :, 4:12, 4:10] = 1
    out = moving_mask(mask, 6, 10, torch.tensor([[0., 0.], [0.4, 0.]]))
    assert out.shape == (2, 1, 6, 16, 20)
    assert torch.allclose(out[0, :, 0], out[0, :, -1], atol=1e-6)
    assert not torch.allclose(out[1, :, 0], out[1, :, -1])
    assert out.min() >= 0 and out.max() <= 1


@pytest.mark.parametrize('hazard,zdim,options', [
    ('lum', 8, {}), ('red', 12, {'red_palette': 'z2'}),
    ('stripe', 14, {'stripe_cond': 'z3'}),
])
def test_generator_accepts_dynamic_mask_and_has_finite_gradients(hazard, zdim, options):
    torch.manual_seed(1)
    clean = torch.rand(1, 3, 8, 16, 24)
    mask = torch.zeros(1, 1, 1, 16, 24)
    mask[..., 3:13, 4:18] = 1
    mask = moving_mask(mask, 8, 24, torch.tensor([[0.3, 0.]]))
    g = make_generator(hazard, width=4, zdim=zdim, fps=24, **options)
    z1, z2 = torch.randn(1, zdim), torch.randn(1, zdim)
    target = torch.ones(1)
    a = g(clean, mask, target, z1)
    b = g(clean, mask, target, z2)
    assert torch.equal(a[0][(mask == 0).expand_as(clean)], clean[(mask == 0).expand_as(clean)])
    loss = a[0].mean() + spectral_diversity(a[1], b[1], mask, z1, z2)
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in g.parameters() if p.grad is not None)
