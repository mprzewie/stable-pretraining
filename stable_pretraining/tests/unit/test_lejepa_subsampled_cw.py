"""Tests for the pair-subsampled production CW regularizer."""

import pytest
import torch

from stable_pretraining.methods.lejepa import (
    CWReg,
    LeJEPA,
    PermutationSubsampledCWReg,
    SubsampledCWReg,
)


pytestmark = pytest.mark.unit


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_subsampled_cw_shape_device_dtype_and_backward(dtype: torch.dtype) -> None:
    x = torch.randn(12, 7, dtype=dtype, requires_grad=True)
    regularizer = SubsampledCWReg(gamma=0.5, pairs_per_sample=4, seed=3)

    loss = regularizer(x)
    gradient = torch.autograd.grad(loss, x)[0]

    assert loss.shape == ()
    assert loss.device == x.device
    assert loss.dtype == x.dtype
    assert gradient.shape == x.shape
    assert gradient.device == x.device
    assert gradient.dtype == x.dtype
    assert torch.isfinite(gradient).all()
    assert gradient.norm() > 0


def test_subsampled_cw_fixed_seed_reproduces_loss_and_gradient() -> None:
    x1 = torch.randn(
        16, 5, generator=torch.Generator().manual_seed(1)
    ).requires_grad_(True)
    x2 = x1.detach().clone().requires_grad_(True)
    first = SubsampledCWReg(gamma=0.5, num_pairs=73, seed=19)
    second = SubsampledCWReg(gamma=0.5, num_pairs=73, seed=19)

    loss1 = first(x1)
    loss2 = second(x2)
    grad1 = torch.autograd.grad(loss1, x1)[0]
    grad2 = torch.autograd.grad(loss2, x2)[0]

    torch.testing.assert_close(loss1, loss2, rtol=0, atol=0)
    torch.testing.assert_close(grad1, grad2, rtol=0, atol=0)


def test_subsampled_cw_monte_carlo_loss_is_unbiased() -> None:
    x = torch.randn(
        7,
        4,
        dtype=torch.float64,
        generator=torch.Generator().manual_seed(4),
    )
    expected = CWReg(gamma=0.5)(x)
    estimates = []
    regularizer = SubsampledCWReg(gamma=0.5, num_pairs=512)
    for seed in range(128):
        generator = torch.Generator().manual_seed(seed)
        estimates.append(regularizer(x, generator=generator))

    actual = torch.stack(estimates).mean()
    torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.002)


def test_subsampled_cw_monte_carlo_gradient_is_unbiased() -> None:
    base = torch.randn(
        6,
        3,
        dtype=torch.float64,
        generator=torch.Generator().manual_seed(8),
    )
    full_x = base.clone().requires_grad_(True)
    full_gradient = torch.autograd.grad(CWReg(gamma=0.5)(full_x), full_x)[0]

    gradients = []
    regularizer = SubsampledCWReg(gamma=0.5, num_pairs=128)
    for seed in range(256):
        x = base.clone().requires_grad_(True)
        generator = torch.Generator().manual_seed(seed)
        gradient = torch.autograd.grad(regularizer(x, generator=generator), x)[0]
        gradients.append(gradient)

    mean_gradient = torch.stack(gradients).mean(dim=0)
    relative_error = (mean_gradient - full_gradient).norm() / full_gradient.norm()
    cosine = torch.nn.functional.cosine_similarity(
        mean_gradient.flatten(), full_gradient.flatten(), dim=0
    )

    assert relative_error < 0.03
    assert cosine > 0.999


def test_subsampled_cw_does_not_call_pairwise_matrix_ops(monkeypatch) -> None:
    def forbidden(*args, **kwargs):
        raise AssertionError("pairwise matrix operation called")

    monkeypatch.setattr(torch, "cdist", forbidden)
    monkeypatch.setattr(torch, "pdist", forbidden)
    x = torch.randn(33, 9, requires_grad=True)
    regularizer = SubsampledCWReg(
        gamma=0.5,
        pairs_per_sample=5,
        pair_chunk_size=17,
        seed=2,
    )

    loss = regularizer(x)
    loss.backward()

    assert x.grad is not None


def test_subsampled_cw_supports_two_samples() -> None:
    x = torch.randn(2, 5, dtype=torch.float64, requires_grad=True)
    regularizer = SubsampledCWReg(gamma=0.5, pairs_per_sample=3, seed=0)

    loss = regularizer(x)
    gradient = torch.autograd.grad(loss, x)[0]

    assert torch.isfinite(loss)
    assert torch.isfinite(gradient).all()


def test_subsampled_cw_rejects_single_sample() -> None:
    with pytest.raises(ValueError, match="at least two"):
        SubsampledCWReg(gamma=0.5)(torch.randn(1, 4))


def test_lejepa_configures_subsampled_cw(monkeypatch) -> None:
    backbone = torch.nn.Identity()
    backbone.num_features = 8
    monkeypatch.setattr(
        "stable_pretraining.methods.lejepa.timm.create_model",
        lambda *args, **kwargs: backbone,
    )

    model = LeJEPA(
        encoder_name="unused",
        projector=torch.nn.Identity(),
        sigreg="subsampled_cw",
        subsampled_cw_pairs_per_sample=7,
        subsampled_cw_seed=11,
    )

    assert isinstance(model.sigreg, SubsampledCWReg)
    assert model.sigreg.pairs_per_sample == 7
    assert model.sigreg.seed == 11


def test_permutation_subsampled_cw_balances_incident_degrees() -> None:
    sample_count = 17
    rounds = 9
    left, right = PermutationSubsampledCWReg._matching_indices(
        sample_count,
        rounds,
        torch.device("cpu"),
        torch.Generator().manual_seed(12),
    )

    assert left.numel() == sample_count * rounds
    assert torch.all(left != right)
    torch.testing.assert_close(
        torch.bincount(left, minlength=sample_count),
        torch.full((sample_count,), rounds),
    )
    torch.testing.assert_close(
        torch.bincount(right, minlength=sample_count),
        torch.full((sample_count,), rounds),
    )


def test_permutation_subsampled_cw_builds_one_permutation_per_forward(
    monkeypatch,
) -> None:
    original_randperm = torch.randperm
    calls = 0

    def counted_randperm(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original_randperm(*args, **kwargs)

    monkeypatch.setattr(torch, "randperm", counted_randperm)
    regularizer = PermutationSubsampledCWReg(
        gamma=0.5,
        pairs_per_sample=9,
        pair_chunk_size=17,
        seed=4,
    )
    regularizer(torch.randn(13, 5))

    assert calls == 1


def test_permutation_subsampled_cw_fixed_seed_reproduces_loss_and_gradient() -> None:
    base = torch.randn(13, 6, generator=torch.Generator().manual_seed(3))
    first_x = base.clone().requires_grad_(True)
    second_x = base.clone().requires_grad_(True)
    first = PermutationSubsampledCWReg(
        gamma=0.5, pairs_per_sample=5, seed=21
    )
    second = PermutationSubsampledCWReg(
        gamma=0.5, pairs_per_sample=5, seed=21
    )

    first_loss = first(first_x)
    second_loss = second(second_x)
    first_gradient = torch.autograd.grad(first_loss, first_x)[0]
    second_gradient = torch.autograd.grad(second_loss, second_x)[0]

    torch.testing.assert_close(first_loss, second_loss, rtol=0, atol=0)
    torch.testing.assert_close(first_gradient, second_gradient, rtol=0, atol=0)


def test_permutation_subsampled_cw_monte_carlo_is_unbiased() -> None:
    base = torch.randn(
        7,
        4,
        dtype=torch.float64,
        generator=torch.Generator().manual_seed(14),
    )
    full_x = base.clone().requires_grad_(True)
    full_loss = CWReg(gamma=0.5)(full_x)
    full_gradient = torch.autograd.grad(full_loss, full_x)[0]

    losses = []
    gradients = []
    regularizer = PermutationSubsampledCWReg(
        gamma=0.5, pairs_per_sample=4
    )
    for seed in range(256):
        x = base.clone().requires_grad_(True)
        loss = regularizer(x, generator=torch.Generator().manual_seed(seed))
        losses.append(loss.detach())
        gradients.append(torch.autograd.grad(loss, x)[0])

    mean_loss = torch.stack(losses).mean()
    mean_gradient = torch.stack(gradients).mean(dim=0)
    relative_gradient_error = (
        (mean_gradient - full_gradient).norm() / full_gradient.norm()
    )

    torch.testing.assert_close(mean_loss, full_loss, rtol=0.02, atol=0.002)
    assert relative_gradient_error < 0.03


def test_lejepa_configures_permutation_subsampled_cw(monkeypatch) -> None:
    backbone = torch.nn.Identity()
    backbone.num_features = 8
    monkeypatch.setattr(
        "stable_pretraining.methods.lejepa.timm.create_model",
        lambda *args, **kwargs: backbone,
    )

    model = LeJEPA(
        encoder_name="unused",
        projector=torch.nn.Identity(),
        sigreg="permutation_subsampled_cw",
        subsampled_cw_pairs_per_sample=6,
        subsampled_cw_seed=13,
    )

    assert isinstance(model.sigreg, PermutationSubsampledCWReg)
    assert model.sigreg.pairs_per_sample == 6
    assert model.sigreg.seed == 13
