"""Tests for the random-cluster U-CW control regularizer."""

import math

import pytest
import torch

from cw_torch.metric import cw_normality

from stable_pretraining.methods.lejepa import RandomClusterUCWReg

pytestmark = pytest.mark.unit


def _explicit_random_cluster_u_cw(
    x: torch.Tensor,
    pseudo_groups: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    num_views, num_groups, feature_dim = x.shape
    gamma_t = torch.as_tensor(gamma, device=x.device, dtype=x.dtype)
    flat = x.permute(1, 0, 2).reshape(
        num_groups * num_views,
        feature_dim,
    )
    pooled_cw = cw_normality(flat, gamma_t)
    pseudo_within_cw = torch.stack(
        [cw_normality(group_views, gamma_t) for group_views in pseudo_groups]
    ).mean()
    random_cluster_u_cw = (
        num_groups * pooled_cw - pseudo_within_cw
    ) / (num_groups - 1)
    return 2.0 * math.pi * num_groups * num_views * random_cluster_u_cw


def test_random_groups_preserve_views_and_mix_source_images() -> None:
    num_views = 3
    num_groups = 5
    view_ids = torch.arange(num_views)[:, None].expand(num_views, num_groups)
    source_ids = torch.arange(num_groups)[None, :].expand(num_views, num_groups)
    x = torch.stack([view_ids, source_ids], dim=-1).float().requires_grad_(True)

    pseudo = RandomClusterUCWReg(seed=7)._make_random_groups(x)

    expected_views = torch.arange(
        num_views,
        dtype=x.dtype,
    ).expand(num_groups, num_views)
    torch.testing.assert_close(pseudo[..., 0], expected_views)
    for sources in pseudo[..., 1]:
        assert sources.unique().numel() == num_views
    assert pseudo.requires_grad


def test_random_groups_are_reproducible_across_steps() -> None:
    x = torch.randn(3, 5, 4)
    first = RandomClusterUCWReg(seed=11)
    second = RandomClusterUCWReg(seed=11)

    torch.testing.assert_close(
        first._make_random_groups(x),
        second._make_random_groups(x),
    )
    torch.testing.assert_close(
        first._make_random_groups(x),
        second._make_random_groups(x),
    )
    assert first.global_step.item() == second.global_step.item() == 2


def test_random_cluster_matches_explicit_value_and_gradient() -> None:
    torch.manual_seed(2027)
    x_actual = torch.randn(3, 5, 7, dtype=torch.float64, requires_grad=True)
    x_expected = x_actual.detach().clone().requires_grad_(True)

    actual = RandomClusterUCWReg(gamma=0.5, seed=19)(x_actual)
    reference = RandomClusterUCWReg(gamma=0.5, seed=19)
    pseudo_groups = reference._make_random_groups(x_expected)
    expected = _explicit_random_cluster_u_cw(x_expected, pseudo_groups, gamma=0.5)

    grad_actual = torch.autograd.grad(actual, x_actual)[0]
    grad_expected = torch.autograd.grad(expected, x_expected)[0]

    torch.testing.assert_close(actual, expected, rtol=1e-9, atol=1e-10)
    torch.testing.assert_close(grad_actual, grad_expected, rtol=1e-8, atol=1e-9)


def test_random_cluster_rejects_more_views_than_groups() -> None:
    with pytest.raises(ValueError, match="V=4 > B=3"):
        RandomClusterUCWReg()(torch.randn(4, 3, 2))
