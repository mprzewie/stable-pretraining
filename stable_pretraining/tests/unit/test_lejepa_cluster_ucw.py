"""Algebraic tests for ClusterUCWReg.

The explicit reference below does not use ``cw_normality``. It reconstructs
cw-torch's normality objective from its three analytic terms and computes the
data-data term only over pairs from different image groups.
"""

from __future__ import annotations

import math

import pytest
import torch

from cw_torch.metric import cw_normality_scale_factor

from stable_pretraining.methods.lejepa import (
    ClusterUCWReg,
    ReferenceClusterUCWReg,
)

pytestmark = pytest.mark.unit


def explicit_cluster_u_cw(
    x: torch.Tensor,
    gamma: float | torch.Tensor = 0.5,
) -> torch.Tensor:
    """Direct pairwise cluster-U CW objective for input ``[V, B, D]``.

    This is intentionally independent of ``cw_normality``. It mirrors the
    formula used by cw-torch, except that the data-data term averages only
    kernel interactions whose rows originate from different image groups.
    The data-Gaussian and Gaussian-Gaussian terms are unchanged and still use
    every view.
    """
    if x.ndim != 3:
        raise ValueError(f"Expected [V, B, D], got {tuple(x.shape)}.")

    num_views, num_groups, feature_dim = x.shape
    if num_groups < 2:
        raise ValueError("At least two independent image groups are required.")

    gamma_t = torch.as_tensor(gamma, device=x.device, dtype=x.dtype)
    k_dim = x.new_tensor(1.0 / (2.0 * feature_dim - 3.0))

    # Put all views from one source image next to each other:
    # [V, B, D] -> [B, V, D] -> [B*V, D].
    flat = x.permute(1, 0, 2).reshape(num_groups * num_views, feature_dim)
    group_ids = torch.arange(num_groups, device=x.device).repeat_interleave(
        num_views
    )

    # cw-torch data-data kernel:
    # k(z, z') = 1 / sqrt(gamma + ||z-z'||^2 / (2D-3)).
    squared_distances = torch.cdist(flat, flat).square()
    data_kernel = torch.rsqrt(gamma_t + k_dim * squared_distances)

    # Exactly B(B-1)V^2 ordered interactions. This excludes both diagonal
    # entries and all distinct-view pairs from the same source image.
    between_group_mask = group_ids[:, None] != group_ids[None, :]
    data_data = data_kernel[between_group_mask].mean()

    # Exact cw-torch sample-to-standard-Gaussian term, over every view.
    squared_norms = flat.square().sum(dim=1)
    data_target = torch.rsqrt(gamma_t + 0.5 + k_dim * squared_norms).mean()

    # Exact cw-torch standard-Gaussian self term.
    target_target = torch.rsqrt(1.0 + gamma_t)

    cw_no_outer_scale = data_data + target_target - 2.0 * data_target
    cw = x.new_tensor(cw_normality_scale_factor) * cw_no_outer_scale

    # Match CWReg / ClusterUCWReg's external convention exactly.
    total_samples = num_groups * num_views
    return 2.0 * math.pi * total_samples * cw


@pytest.mark.parametrize(
    ("num_views", "num_groups", "feature_dim"),
    [
        (1, 2, 3),
        (2, 3, 5),
        (4, 5, 7),
    ],
)
def test_cluster_u_matches_explicit_pairwise_value(
    num_views: int,
    num_groups: int,
    feature_dim: int,
) -> None:
    torch.manual_seed(1234 + num_views + num_groups + feature_dim)
    x = torch.randn(
        num_views,
        num_groups,
        feature_dim,
        dtype=torch.float64,
    )

    actual = ClusterUCWReg(gamma=0.5)(x)
    expected = explicit_cluster_u_cw(x, gamma=0.5)

    torch.testing.assert_close(actual, expected, rtol=1e-9, atol=1e-10)


def test_cluster_u_matches_explicit_pairwise_gradient() -> None:
    """The black-box decomposition must also preserve the exact gradient."""
    torch.manual_seed(2026)
    x_actual = torch.randn(3, 4, 6, dtype=torch.float64, requires_grad=True)
    x_expected = x_actual.detach().clone().requires_grad_(True)

    actual = ClusterUCWReg(gamma=0.5)(x_actual)
    expected = explicit_cluster_u_cw(x_expected, gamma=0.5)

    grad_actual = torch.autograd.grad(actual, x_actual)[0]
    grad_expected = torch.autograd.grad(expected, x_expected)[0]

    torch.testing.assert_close(actual, expected, rtol=1e-9, atol=1e-10)
    torch.testing.assert_close(
        grad_actual,
        grad_expected,
        rtol=1e-8,
        atol=1e-9,
    )


@pytest.mark.parametrize(
    ("num_views", "num_groups", "feature_dim"),
    [
        (1, 2, 3),
        (2, 3, 5),
        (4, 5, 7),
    ],
)
def test_optimized_cluster_u_matches_reference_value(
    num_views: int,
    num_groups: int,
    feature_dim: int,
) -> None:
    torch.manual_seed(4321 + num_views + num_groups + feature_dim)
    x = torch.randn(num_views, num_groups, feature_dim, dtype=torch.float64)

    actual = ClusterUCWReg(gamma=0.5)(x)
    expected = ReferenceClusterUCWReg(gamma=0.5)(x)

    torch.testing.assert_close(actual, expected, rtol=1e-9, atol=1e-10)


def test_optimized_cluster_u_matches_reference_gradient() -> None:
    torch.manual_seed(2028)
    x_actual = torch.randn(4, 5, 7, dtype=torch.float64, requires_grad=True)
    x_expected = x_actual.detach().clone().requires_grad_(True)

    actual = ClusterUCWReg(gamma=0.5)(x_actual)
    expected = ReferenceClusterUCWReg(gamma=0.5)(x_expected)

    grad_actual = torch.autograd.grad(actual, x_actual)[0]
    grad_expected = torch.autograd.grad(expected, x_expected)[0]

    torch.testing.assert_close(actual, expected, rtol=1e-9, atol=1e-10)
    torch.testing.assert_close(grad_actual, grad_expected, rtol=1e-8, atol=1e-9)


def test_optimized_cluster_u_matches_reference_in_float32() -> None:
    torch.manual_seed(2029)
    x = torch.randn(8, 16, 32)

    actual = ClusterUCWReg(gamma=0.5)(x)
    expected = ReferenceClusterUCWReg(gamma=0.5)(x)

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_cluster_u_is_invariant_to_group_and_view_permutations() -> None:
    """Only group membership matters, not the order of groups or views."""
    torch.manual_seed(7)
    x = torch.randn(3, 5, 4, dtype=torch.float64)
    regularizer = ClusterUCWReg(gamma=0.5)

    reference = regularizer(x)
    permuted_views = regularizer(x[torch.tensor([2, 0, 1])])
    permuted_groups = regularizer(x[:, torch.tensor([4, 1, 3, 0, 2])])

    torch.testing.assert_close(reference, permuted_views, rtol=1e-9, atol=1e-10)
    torch.testing.assert_close(reference, permuted_groups, rtol=1e-9, atol=1e-10)


def test_explicit_mask_contains_only_between_group_pairs() -> None:
    """Sanity-check the combinatorics behind the explicit estimator."""
    num_views = 3
    num_groups = 4
    group_ids = torch.arange(num_groups).repeat_interleave(num_views)
    mask = group_ids[:, None] != group_ids[None, :]

    assert mask.sum().item() == num_groups * (num_groups - 1) * num_views**2

    # Every same-group block, including its diagonal, must be excluded.
    for group in range(num_groups):
        idx = (group_ids == group).nonzero(as_tuple=True)[0]
        assert not mask[idx[:, None], idx[None, :]].any()
