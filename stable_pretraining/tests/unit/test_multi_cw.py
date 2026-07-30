from itertools import combinations
import math

import pytest
import torch
from torch import nn

from stable_pretraining.methods.joint_cw import JointCWLoss
from stable_pretraining.methods.multi_cw import (
    MultiViewBlockCWLoss,
    MultiViewCWDecomposition,
)

pytestmark = pytest.mark.unit


def _covariance(
    n_global: int,
    n_local: int,
    rho_gg: float,
    rho_gl: float,
    rho_ll: float,
) -> torch.Tensor:
    view_count = n_global + n_local
    covariance = torch.full(
        (view_count, view_count),
        rho_ll,
        dtype=torch.float64,
    )
    covariance[:n_global, :n_global] = rho_gg
    covariance[:n_global, n_global:] = rho_gl
    covariance[n_global:, :n_global] = rho_gl
    covariance.fill_diagonal_(1.0)
    return covariance


def _joint_pair_whitener(rho: float) -> torch.Tensor:
    return torch.tensor(
        [
            [1.0, 1.0],
            [1.0, -1.0],
        ],
        dtype=torch.float64,
    ) / torch.tensor(
        [
            [math.sqrt(2.0 * (1.0 + rho))],
            [math.sqrt(2.0 * (1.0 - rho))],
        ],
        dtype=torch.float64,
    )


def test_covariance_matches_block_eigenspaces() -> None:
    n_global, n_local = 3, 4
    rho_gg, rho_gl, rho_ll = 0.4, 0.2, 0.3
    decomposition = MultiViewCWDecomposition(
        n_global=n_global,
        n_local=n_local,
        rho_gg=rho_gg,
        rho_gl=rho_gl,
        rho_ll=rho_ll,
    )
    covariance = _covariance(
        n_global,
        n_local,
        rho_gg,
        rho_gl,
        rho_ll,
    )

    assert torch.allclose(
        decomposition.q_global @ decomposition.q_global.T,
        torch.eye(n_global - 1, dtype=torch.float64),
    )
    assert torch.allclose(
        decomposition.q_local @ decomposition.q_local.T,
        torch.eye(n_local - 1, dtype=torch.float64),
    )
    assert torch.allclose(
        decomposition.q_global.sum(dim=1),
        torch.zeros(n_global - 1, dtype=torch.float64),
    )
    assert torch.allclose(
        decomposition.q_local.sum(dim=1),
        torch.zeros(n_local - 1, dtype=torch.float64),
    )

    expected_eigenvalues = torch.cat(
        [
            torch.full(
                (n_global - 1,),
                1.0 - rho_gg,
                dtype=torch.float64,
            ),
            torch.full(
                (n_local - 1,),
                1.0 - rho_ll,
                dtype=torch.float64,
            ),
            decomposition.collective_eigenvalues,
        ]
    )
    assert torch.allclose(
        torch.linalg.eigvalsh(covariance),
        expected_eigenvalues.sort().values,
    )


def test_g2_l6_joint_and_multi_cw_have_same_gaussian_covariance_target() -> None:
    """Every JointCW pair and the full MultiCW basis whiten the same A."""
    n_global, n_local = 2, 6
    rho_gg, rho_gl, rho_ll = 0.88, 0.72, 0.61
    covariance = _covariance(
        n_global,
        n_local,
        rho_gg,
        rho_gl,
        rho_ll,
    )
    decomposition = MultiViewCWDecomposition(
        n_global=n_global,
        n_local=n_local,
        rho_gg=rho_gg,
        rho_gl=rho_gl,
        rho_ll=rho_ll,
    )

    collective_coordinates = torch.zeros(2, 8, dtype=torch.float64)
    collective_coordinates[0, :n_global] = 1.0 / math.sqrt(n_global)
    collective_coordinates[1, n_global:] = 1.0 / math.sqrt(n_local)
    collective_whitener = (
        torch.diag(decomposition.collective_eigenvalues.rsqrt())
        @ decomposition.collective_eigenvectors.T
        @ collective_coordinates
    )
    global_whitener = torch.zeros(n_global - 1, 8, dtype=torch.float64)
    global_whitener[:, :n_global] = decomposition.q_global / math.sqrt(1.0 - rho_gg)
    local_whitener = torch.zeros(n_local - 1, 8, dtype=torch.float64)
    local_whitener[:, n_global:] = decomposition.q_local / math.sqrt(1.0 - rho_ll)
    multi_whitener = torch.cat(
        [collective_whitener, global_whitener, local_whitener],
        dim=0,
    )

    assert torch.allclose(
        multi_whitener @ covariance @ multi_whitener.T,
        torch.eye(8, dtype=torch.float64),
        atol=1e-12,
        rtol=1e-12,
    )

    for first, second in combinations(range(8), 2):
        if second < n_global:
            rho = rho_gg
        elif first < n_global:
            rho = rho_gl
        else:
            rho = rho_ll
        pair_covariance = covariance[torch.tensor([first, second])][
            :, torch.tensor([first, second])
        ]
        joint_whitener = _joint_pair_whitener(rho)
        assert torch.allclose(
            joint_whitener @ pair_covariance @ joint_whitener.T,
            torch.eye(2, dtype=torch.float64),
            atol=1e-12,
            rtol=1e-12,
        )


def test_g2_l6_synthetic_gaussian_whitens_for_multi_and_every_joint_pair() -> None:
    torch.manual_seed(5)
    n_global, n_local = 2, 6
    rho_gg, rho_gl, rho_ll = 0.88, 0.72, 0.61
    covariance = _covariance(
        n_global,
        n_local,
        rho_gg,
        rho_gl,
        rho_ll,
    )
    sample_count = 30_000
    standard = torch.randn(sample_count, 8, 1, dtype=torch.float64)
    correlated = torch.einsum(
        "vw,bwd->bvd",
        torch.linalg.cholesky(covariance),
        standard,
    )

    multi_modes = MultiViewCWDecomposition()(correlated)["full"].squeeze(-1)
    multi_centered = multi_modes - multi_modes.mean(dim=0)
    multi_covariance = multi_centered.T @ multi_centered / sample_count
    assert torch.allclose(
        multi_covariance,
        torch.eye(8, dtype=torch.float64),
        atol=0.035,
        rtol=0.0,
    )

    for first, second in combinations(range(8), 2):
        if second < n_global:
            rho = rho_gg
        elif first < n_global:
            rho = rho_gl
        else:
            rho = rho_ll
        pair = correlated[:, [first, second], 0]
        joint_modes = pair @ _joint_pair_whitener(rho).T
        joint_centered = joint_modes - joint_modes.mean(dim=0)
        joint_covariance = joint_centered.T @ joint_centered / sample_count
        assert torch.allclose(
            joint_covariance,
            torch.eye(2, dtype=torch.float64),
            atol=0.035,
            rtol=0.0,
        )


def test_synthetic_gaussian_is_whitened() -> None:
    torch.manual_seed(0)
    covariance = _covariance(2, 6, 0.88, 0.72, 0.61)
    sample_count, feature_dim = 30_000, 2
    standard = torch.randn(
        sample_count,
        8,
        feature_dim,
        dtype=torch.float64,
    )
    correlated = torch.einsum(
        "vw,bwd->bvd",
        torch.linalg.cholesky(covariance),
        standard,
    )

    whitened = MultiViewCWDecomposition()(correlated)["full"].flatten(1)
    centered = whitened - whitened.mean(dim=0)
    empirical_covariance = centered.T @ centered / sample_count

    assert whitened.mean(dim=0).abs().max() < 0.025
    assert torch.allclose(
        empirical_covariance,
        torch.eye(16, dtype=torch.float64),
        atol=0.035,
        rtol=0.0,
    )


def test_block_whitening_matches_direct_whitening_distances() -> None:
    torch.manual_seed(1)
    covariance = _covariance(2, 6, 0.88, 0.72, 0.61)
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    direct_whitener = eigenvectors @ torch.diag(eigenvalues.rsqrt()) @ eigenvectors.T
    sample = torch.randn(32, 8, 3, dtype=torch.float64)

    block = MultiViewCWDecomposition()(sample)["full"].flatten(1)
    direct = torch.einsum(
        "vw,bwd->bvd",
        direct_whitener,
        sample,
    ).flatten(1)

    assert torch.allclose(
        torch.pdist(block),
        torch.pdist(direct),
        atol=1e-10,
        rtol=1e-10,
    )


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"rho_gg": 1.0}, "rho_gg"),
        (
            {"rho_gg": 0.2, "rho_gl": 0.9, "rho_ll": 0.2},
            "not positive definite",
        ),
        ({"rho_gg": 1.0 - 1e-9}, "Global residual eigenvalue"),
        ({"eps": 0.0}, "eps"),
    ],
)
def test_decomposition_rejects_invalid_covariance(
    kwargs: dict[str, float],
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        MultiViewCWDecomposition(**kwargs)


def test_decomposition_rejects_malformed_inputs() -> None:
    decomposition = MultiViewCWDecomposition()

    with pytest.raises(ValueError, match="Expected 8 views"):
        decomposition(torch.randn(4, 7, 3))
    with pytest.raises(ValueError, match="at least two"):
        decomposition(torch.randn(1, 8, 3))
    with pytest.raises(TypeError, match="floating-point"):
        decomposition(torch.ones(4, 8, 3, dtype=torch.int64))
    with pytest.raises(ValueError, match="at least one view"):
        decomposition([])
    with pytest.raises(RuntimeError):
        decomposition([torch.randn(4, 3), torch.randn(5, 3)])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"gamma": 0.0},
        {"w_joint": -0.1, "w_collective_major": 0.35},
        {"w_joint": 0.5},
        {"variant": "full_joint", "w_joint": 1.0},
    ],
)
def test_loss_rejects_invalid_configuration(
    kwargs: dict[str, float | str],
) -> None:
    with pytest.raises(ValueError):
        MultiViewBlockCWLoss(**kwargs)


@pytest.mark.parametrize(
    "variant",
    ["canonical", "blockwise", "coarse", "full_joint"],
)
def test_loss_variants_backpropagate_to_every_view(variant: str) -> None:
    torch.manual_seed(2)
    views = [torch.randn(12, 4, requires_grad=True) for _ in range(8)]
    loss, diagnostics = MultiViewBlockCWLoss(variant=variant)(views)

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert all(not value.requires_grad for value in diagnostics.values())

    loss.backward()
    for view in views:
        assert view.grad is not None
        assert torch.isfinite(view.grad).all()


@pytest.mark.parametrize(
    ("variant", "expected"),
    [
        ("full_joint", 1.0),
        ("coarse", 1.9),
        ("blockwise", 1.77),
        ("canonical", 2.12),
    ],
)
def test_variant_weights_are_applied_once(
    variant: str,
    expected: float,
) -> None:
    class FakeDecomposition(nn.Module):
        def forward(self, z: torch.Tensor) -> dict[str, torch.Tensor]:
            def block(value: float) -> torch.Tensor:
                return z.new_full((2, 1, 1), value)

            return {
                "full": block(1.0),
                "collective": block(2.0),
                "collective_major": block(3.0),
                "collective_minor": block(4.0),
                "global_residual": block(5.0),
                "local_residual": block(6.0),
                "residual": block(7.0),
            }

    objective = MultiViewBlockCWLoss(variant=variant)
    objective.decomposition = FakeDecomposition()
    objective._cw = lambda block: block[0, 0, 0]

    loss, _ = objective(torch.zeros(2, 8, 1))
    assert loss == pytest.approx(expected)


@pytest.mark.parametrize(
    "variant",
    ["canonical", "blockwise", "coarse", "full_joint"],
)
def test_loss_is_invariant_to_within_group_permutations(
    variant: str,
) -> None:
    torch.manual_seed(3)
    sample = torch.randn(16, 8, 5)
    permutation = torch.tensor([1, 0, 4, 2, 7, 3, 6, 5])
    objective = MultiViewBlockCWLoss(variant=variant)

    original, _ = objective(sample)
    permuted, _ = objective(sample[:, permutation])

    assert torch.allclose(original, permuted, atol=2e-6, rtol=2e-6)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_low_precision_inputs_are_promoted(dtype: torch.dtype) -> None:
    sample = torch.randn(8, 8, 4).to(dtype)
    decomposition = MultiViewCWDecomposition()
    blocks = decomposition(sample)

    assert blocks["full"].dtype == torch.float32
    loss, _ = MultiViewBlockCWLoss()(sample)
    assert loss.dtype == torch.float32
    assert torch.isfinite(loss)


@pytest.mark.parametrize(
    ("beta", "w_plus"),
    [(0.0, 0.0), (0.4, 0.15), (1.0, 0.5)],
)
def test_two_view_canonical_multi_cw_equals_joint_cw(
    beta: float,
    w_plus: float,
) -> None:
    """For one view per group, both losses use the same plus/minus modes."""
    torch.manual_seed(4)
    rho = 0.7
    joint_z1 = torch.randn(16, 5, dtype=torch.float64, requires_grad=True)
    joint_z2 = torch.randn(16, 5, dtype=torch.float64, requires_grad=True)
    multi_z1 = joint_z1.detach().clone().requires_grad_()
    multi_z2 = joint_z2.detach().clone().requires_grad_()

    joint_objective = JointCWLoss(
        gamma=0.5,
        rho=rho,
        beta=beta,
        w_plus=w_plus,
    )
    multi_objective = MultiViewBlockCWLoss(
        gamma=0.5,
        variant="canonical",
        n_global=1,
        n_local=1,
        rho_gg=0.0,
        rho_gl=rho,
        rho_ll=0.0,
        w_joint=1.0 - beta,
        w_collective_major=w_plus,
        w_collective_minor=beta - w_plus,
        w_global_residual=0.0,
        w_local_residual=0.0,
    )

    joint_loss, joint_diagnostics = joint_objective(joint_z1, joint_z2)
    multi_loss, multi_diagnostics = multi_objective(
        torch.stack([multi_z1, multi_z2], dim=1)
    )

    assert torch.allclose(joint_loss, multi_loss, atol=1e-10, rtol=1e-10)
    assert torch.allclose(
        joint_diagnostics["cw_joint"],
        multi_diagnostics["cw/joint"],
        atol=1e-10,
        rtol=1e-10,
    )
    assert torch.allclose(
        joint_diagnostics["cw_plus"],
        multi_diagnostics["cw/collective_major"],
        atol=1e-10,
        rtol=1e-10,
    )
    assert torch.allclose(
        joint_diagnostics["cw_minus"],
        multi_diagnostics["cw/collective_minor"],
        atol=1e-10,
        rtol=1e-10,
    )

    joint_loss.backward()
    multi_loss.backward()
    assert torch.allclose(
        joint_z1.grad,
        multi_z1.grad,
        atol=1e-10,
        rtol=1e-10,
    )
    assert torch.allclose(
        joint_z2.grad,
        multi_z2.grad,
        atol=1e-10,
        rtol=1e-10,
    )
