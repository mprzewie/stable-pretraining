"""Tests for grouped multi-view CW gradient analysis helpers."""

import math

import pytest
import torch

from assets.cli.analyze_multiview_cw_gradients import (
    EPS,
    CWReg,
    analyze_batch,
    cw_component_losses,
    flatten_view_major,
    grad,
    pair_masks,
)

pytestmark = pytest.mark.unit


def _sample_z(batch_size: int = 4, num_views: int = 3, dim: int = 5) -> torch.Tensor:
    generator = torch.Generator().manual_seed(123)
    return torch.randn(batch_size, num_views, dim, generator=generator)


def test_pair_masks_distinguish_diagonal_within_and_between() -> None:
    batch_size, num_views, num_global = 3, 4, 2
    masks = pair_masks(batch_size, num_views, num_global)
    n = batch_size * num_views

    assert masks["diagonal"].sum().item() == n
    assert masks["within"].sum().item() == batch_size * num_views * (num_views - 1)
    assert masks["between"].sum().item() == batch_size * (batch_size - 1) * num_views**2
    assert (
        masks["diagonal"].to(torch.int)
        + masks["within"].to(torch.int)
        + masks["between"].to(torch.int)
    ).eq(1).all()


def test_within_pair_type_masks_partition_within_pairs() -> None:
    masks = pair_masks(batch_size=5, num_views=6, num_global=2)

    combined = (
        masks["within_gg"].to(torch.int)
        + masks["within_gl"].to(torch.int)
        + masks["within_ll"].to(torch.int)
    )
    assert torch.equal(combined.bool(), masks["within"])
    assert masks["within_gg"].sum().item() == 5 * 2 * 1
    assert masks["within_gl"].sum().item() == 5 * 2 * 2 * 4
    assert masks["within_ll"].sum().item() == 5 * 4 * 3


def test_decomposed_losses_sum_to_production_cw() -> None:
    z = _sample_z().requires_grad_(True)
    losses = cw_component_losses(z, gamma=0.5, num_global=2)
    production = CWReg(gamma=0.5)(flatten_view_major(z))

    assert torch.allclose(
        losses["pair_full"],
        losses["within"] + losses["between"] + losses["diagonal"],
        atol=1e-6,
        rtol=1e-6,
    )
    assert torch.allclose(losses["full"], production, atol=1e-5, rtol=1e-5)


def test_decomposed_gradients_sum_to_production_cw_gradient() -> None:
    z = _sample_z().requires_grad_(True)
    losses = cw_component_losses(z, gamma=0.5, num_global=2)
    production = CWReg(gamma=0.5)(flatten_view_major(z))

    g_parts = (
        grad(losses["within"], z)
        + grad(losses["between"], z)
        + grad(losses["diagonal"], z)
        + grad(losses["target"], z)
    )
    g_prod = grad(production, z, retain_graph=False)

    assert torch.allclose(g_parts, g_prod, atol=1e-5, rtol=1e-5)


def test_group_aware_estimator_uses_between_instance_denominator() -> None:
    z = _sample_z(batch_size=3, num_views=2, dim=4)
    losses = cw_component_losses(z, gamma=0.5, num_global=2)
    full_denominator = (3 * 2) ** 2
    group_denominator = 3 * (3 - 1) * 2**2

    # Same between numerator and CW outer scale; only the denominator differs.
    expected = losses["between"] * full_denominator / group_denominator
    assert torch.allclose(losses["group_pair"], expected, atol=1e-6, rtol=1e-6)


def test_group_aware_estimator_is_group_and_view_permutation_invariant() -> None:
    z = _sample_z(batch_size=5, num_views=4, dim=3)
    base = cw_component_losses(z, gamma=0.5, num_global=2)["group_full"]

    group_perm = torch.tensor([2, 4, 0, 3, 1])
    view_perm = torch.tensor([1, 0, 3, 2])
    permuted = z.index_select(0, group_perm).index_select(1, view_perm)
    value = cw_component_losses(permuted, gamma=0.5, num_global=2)["group_full"]

    assert torch.allclose(base, value, atol=1e-5, rtol=1e-5)


def test_fixed_representations_give_deterministic_outputs() -> None:
    z = _sample_z(batch_size=4, num_views=4, dim=4)

    first = analyze_batch(z, gamma=0.5, num_global=2)
    second = analyze_batch(z, gamma=0.5, num_global=2)

    for key in [
        "cw_loss_full",
        "group_cw_loss",
        "within_fraction_of_cw_gradient",
        "cosine_within_alignment",
    ]:
        assert math.isclose(first[key], second[key], rel_tol=0.0, abs_tol=EPS)


def test_nearly_identical_positive_views_have_repulsive_within_gradient() -> None:
    z = torch.zeros(2, 2, 3)
    z[0, 0, 0] = -0.01
    z[0, 1, 0] = 0.01
    z[1, 0, 0] = 1.0
    z[1, 1, 0] = 1.1
    z.requires_grad_(True)

    within = cw_component_losses(z, gamma=0.5, num_global=2)["within"]
    g_within = grad(within, z, retain_graph=False)
    before = (z[0, 0] - z[0, 1]).square().sum()
    after = ((z - 0.1 * g_within)[0, 0] - (z - 0.1 * g_within)[0, 1]).square().sum()

    assert after > before


def test_tiny_cpu_end_to_end_batch_smoke() -> None:
    z = _sample_z(batch_size=3, num_views=4, dim=4)
    row = analyze_batch(z, gamma=0.5, num_global=2, finite_step=1e-4)

    assert row["gradient_reconstruction_error"] < 1e-5
    assert row["production_cw_gradient_error"] < 1e-5
    assert row["within_type_gradient_reconstruction_error"] < 1e-5
    assert math.isfinite(row["finite_step_within_measured_delta_alignment"])
