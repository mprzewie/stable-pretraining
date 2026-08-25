"""Tests for the LaPHorn SwAV assignment variant."""

import pytest
import torch

from stable_pretraining.losses import LaphornSwAVLoss

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("shape", [(8, 5), (5, 8)])
def test_laphorn_assignments_have_swav_marginals(shape: tuple[int, int]) -> None:
    batch_size, n_prototypes = shape
    scores = torch.randn(shape, generator=torch.Generator().manual_seed(7))

    assignments = LaphornSwAVLoss(laphorn_h=0.3).assignments(scores)

    assert assignments.shape == scores.shape
    assert torch.all(assignments >= 0)
    torch.testing.assert_close(
        assignments.sum(dim=1), torch.ones(batch_size), atol=2e-5, rtol=2e-5
    )
    torch.testing.assert_close(
        assignments.sum(dim=0),
        torch.full((n_prototypes,), batch_size / n_prototypes),
        atol=2e-5,
        rtol=2e-5,
    )


def test_laphorn_swav_loss_is_finite_and_backpropagates() -> None:
    batch_size, projection_dim, n_prototypes = 6, 4, 5
    proj1 = torch.randn(batch_size, projection_dim, requires_grad=True)
    proj2 = torch.randn(batch_size, projection_dim, requires_grad=True)
    prototypes = torch.nn.Linear(projection_dim, n_prototypes, bias=False)

    loss = LaphornSwAVLoss(laphorn_h=0.3)(proj1, proj2, prototypes)
    loss.backward()

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert proj1.grad is not None
    assert proj2.grad is not None
    assert prototypes.weight.grad is not None


def test_laphorn_width_arguments_are_mutually_exclusive() -> None:
    with pytest.raises(ValueError, match="at most one"):
        LaphornSwAVLoss(laphorn_h=0.3, laphorn_log_h=-1.0)
