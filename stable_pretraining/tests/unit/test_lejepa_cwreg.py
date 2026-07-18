"""Synthetic tests comparing LeJEPA SIGReg and CWReg regularizers."""

import pytest
import torch

from stable_pretraining.methods.lejepa import CWReg, SlicedEppsPulley

pytestmark = pytest.mark.unit


def _sigreg(x: torch.Tensor, num_slices: int = 256) -> torch.Tensor:
    return SlicedEppsPulley(num_slices=num_slices, n_points=17)(x)


def _cwreg(x: torch.Tensor) -> torch.Tensor:
    return CWReg()(x)


def _synthetic_batch(seed: int, n: int = 64, d: int = 8) -> torch.Tensor:
    return torch.randn(n, d, generator=torch.Generator().manual_seed(seed))


def _rank(values: torch.Tensor) -> torch.Tensor:
    order = torch.argsort(values)
    ranks = torch.empty_like(order, dtype=torch.float)
    ranks[order] = torch.arange(values.numel(), dtype=torch.float)
    return ranks


def _spearman(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    rx = _rank(x)
    ry = _rank(y)
    return torch.corrcoef(torch.stack([rx, ry]))[0, 1]


def _paired_regularizer_losses() -> tuple[torch.Tensor, torch.Tensor]:
    sigreg_losses = []
    cwreg_losses = []
    for seed in range(20):
        x = _synthetic_batch(seed)
        if seed % 4 == 1:
            x = x + 0.2 * seed / 4
        elif seed % 4 == 2:
            x = x * (1 + 0.1 * seed)
        elif seed % 4 == 3:
            x = torch.cat([x[:32] - 0.1 * seed, x[32:] + 0.1 * seed])

        sigreg_losses.append(_sigreg(x))
        cwreg_losses.append(_cwreg(x))

    return torch.stack(sigreg_losses), torch.stack(cwreg_losses)


@pytest.mark.parametrize("regularizer", [_sigreg, _cwreg])
def test_lejepa_gaussian_regularizers_backward(regularizer) -> None:
    x = _synthetic_batch(0).requires_grad_(True)

    loss = regularizer(x)
    loss.backward()

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert x.grad.norm() > 0


@pytest.mark.parametrize("regularizer", [_sigreg, _cwreg])
def test_lejepa_gaussian_regularizers_order_obvious_deviations(regularizer) -> None:
    gaussian = _synthetic_batch(1)
    shifted = gaussian + 0.75
    scaled = gaussian * 2.0
    mixture = torch.cat([gaussian[:32] - 1.0, gaussian[32:] + 1.0])
    g = torch.Generator().manual_seed(2)
    normal = torch.randn(gaussian.shape, generator=g)
    chi2 = torch.randn(*gaussian.shape, 3, generator=g).square().sum(-1)
    heavy_tailed = normal / torch.sqrt(chi2 / 3)

    gaussian_loss = regularizer(gaussian)

    assert regularizer(shifted) > gaussian_loss * 2
    assert regularizer(scaled) > gaussian_loss * 2
    assert regularizer(mixture) > gaussian_loss * 2
    assert regularizer(heavy_tailed) > gaussian_loss * 1.5


def test_cwreg_and_sigreg_rank_correlate_on_synthetic_batches() -> None:
    sigreg_losses, cwreg_losses = _paired_regularizer_losses()

    assert _spearman(sigreg_losses, cwreg_losses) > 0.99


def test_cwreg_and_sigreg_have_rough_scalar_calibration() -> None:
    sigreg_losses, cwreg_losses = _paired_regularizer_losses()
    alpha = (sigreg_losses[:10] * cwreg_losses[:10]).sum() / (
        sigreg_losses[:10].square().sum()
    )

    calibrated = alpha * sigreg_losses[10:]
    relative_error = (calibrated - cwreg_losses[10:]).abs() / cwreg_losses[
        10:
    ].clamp_min(1e-8)

    assert alpha > 0
    assert relative_error.median() < 0.45
    assert relative_error.max() < 0.7


def test_cwreg_and_sigreg_gradients_are_aligned() -> None:
    x = _synthetic_batch(123).requires_grad_(True)
    sigreg_loss = _sigreg(x, num_slices=512)
    sigreg_grad = torch.autograd.grad(sigreg_loss, x, retain_graph=True)[0]

    cwreg_loss = _cwreg(x)
    cwreg_grad = torch.autograd.grad(cwreg_loss, x)[0]

    cosine = torch.nn.functional.cosine_similarity(
        sigreg_grad.flatten(), cwreg_grad.flatten(), dim=0
    )

    assert cosine > 0.8
