"""Tests for observational joint-structure diagnostics."""

import math

import pytest
import torch

from assets.cli.analyze_multiview_cw_gradients import (
    all_pairs_sample_and_stats,
    construct_pairs,
    covariance,
    covariance_estimate,
    covariance_from_sufficient_stats,
    cross_covariance,
    deterministic_projection,
    joint_stability_rows,
    joint_structure_all_pairs_chunked_from_image_split,
    joint_structure_estimators,
    joint_structure_metrics,
    joint_structure_metrics_from_image_split,
    matched_null_rows,
    positive_pair_indices,
    resolve_fixed_projection_dim,
    shared_row_subsample,
    should_run_joint_null,
    sufficient_statistics,
    split_calibration_evaluation,
    unscaled_cw_normality_score,
    CWReg,
    NULL_CALIBRATION_CACHE,
)

pytestmark = pytest.mark.unit


def _correlated_gaussian(n: int = 4096, d: int = 8, rho: float = 0.7) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(123)
    x = torch.randn(n, d, generator=generator)
    eps = torch.randn(n, d, generator=generator)
    y = rho * x + math.sqrt(1.0 - rho * rho) * eps
    return x, y


def _metrics(z1: torch.Tensor, z2: torch.Tensor, shuffled: torch.Tensor | None = None) -> dict[str, float]:
    if shuffled is None:
        shuffled = z2.roll(1, dims=0)
    return joint_structure_metrics(
        z1,
        z2,
        shuffled,
        gamma=0.5,
        covariance_eps=1e-5,
        projection_diagnostics=8,
        seed=0,
        fixed_rhos=[0.0, 0.5, 0.9],
        covariance_estimator="oas",
        calibration_fraction=0.5,
        num_null_repetitions=0,
        stats_chunk_size=64,
        gaussianity_max_samples=1024,
    )


def test_positive_and_shuffled_pair_construction() -> None:
    z = torch.arange(3 * 4 * 2, dtype=torch.float32).view(3, 4, 2)

    z1, z2, pairs = construct_pairs(
        z, "global_local", num_global=2, shuffled=False, estimator="all_pairs_clustered"
    )
    _, z2_shuffled, _ = construct_pairs(
        z, "global_local", num_global=2, shuffled=True, seed=0, estimator="all_pairs_clustered"
    )

    assert pairs == [(0, 2), (0, 3), (1, 2), (1, 3), (2, 0), (2, 1), (3, 0), (3, 1)]
    assert z1.shape == z2.shape == (24, 2)
    assert not torch.equal(z2, z2_shuffled)


def test_pair_type_separation() -> None:
    assert positive_pair_indices(5, 2, "global_global") == [(0, 1), (1, 0)]
    assert positive_pair_indices(5, 2, "global_local") == [
        (0, 2),
        (0, 3),
        (0, 4),
        (1, 2),
        (1, 3),
        (1, 4),
        (2, 0),
        (2, 1),
        (3, 0),
        (3, 1),
        (4, 0),
        (4, 1),
    ]
    assert positive_pair_indices(5, 2, "local_local") == [
        (2, 3),
        (2, 4),
        (3, 2),
        (3, 4),
        (4, 2),
        (4, 3),
    ]


def test_covariance_and_cross_covariance_on_synthetic_data() -> None:
    x = torch.tensor([[1.0, 0.0], [3.0, 2.0], [5.0, 4.0]])
    y = 2.0 * x

    cov_x = covariance(x)
    cov_xy = cross_covariance(x, y)

    assert torch.allclose(cov_xy, 2.0 * cov_x)


def test_recover_known_rho_from_correlated_gaussian() -> None:
    z1, z2 = _correlated_gaussian(rho=0.65)
    row = _metrics(z1, z2)

    assert abs(row["rho_hat"] - 0.65) < 0.04
    assert row["scalar_model_relative_residual"] < 0.12


def test_anisotropic_dependence_has_high_scalar_residual() -> None:
    generator = torch.Generator().manual_seed(5)
    n, d = 4096, 8
    x = torch.randn(n, d, generator=generator)
    rhos = torch.linspace(0.05, 0.95, d)
    eps = torch.randn(n, d, generator=generator)
    y = rhos * x + torch.sqrt(1.0 - rhos.square()) * eps

    row = _metrics(x, y)

    assert row["scalar_model_relative_residual"] > 0.25
    assert row["explained_diagonal"] > row["explained_rho_i"]


def test_z_plus_and_z_minus_variances_under_known_rho() -> None:
    rho = 0.4
    z1, z2 = _correlated_gaussian(rho=rho)
    row = _metrics(z1, z2)

    assert abs(row["rho_hat"] - rho) < 0.05
    assert row["plus_minus_variance_ratio"] > 1.0


def test_true_shuffled_statistics_on_synthetic_pairs() -> None:
    z1, z2 = _correlated_gaussian(rho=0.9)
    shuffled = z2.roll(997, dims=0)
    row = _metrics(z1, z2, shuffled)

    assert row["mean_pair_cosine"] > row["shuffled_mean_pair_cosine"] + 0.5
    assert row["mean_pair_distance"] < row["shuffled_mean_pair_distance"]
    assert row["energy_auc"] > 0.95


def test_deterministic_outputs_for_fixed_inputs() -> None:
    z1, z2 = _correlated_gaussian(n=1024, d=6, rho=0.5)

    first = _metrics(z1, z2)
    second = _metrics(z1, z2)

    for key in ["rho_hat", "scalar_model_relative_residual", "energy_auc"]:
        assert first[key] == second[key]


def test_tiny_cpu_joint_structure_smoke() -> None:
    z1, z2 = _correlated_gaussian(n=32, d=4, rho=0.5)
    row = _metrics(z1, z2)

    assert math.isfinite(row["rho_hat"])
    assert math.isfinite(row["unscaled_cw_normality_plus"])
    assert math.isfinite(row["projection_kurtosis_minus"])


def test_balanced_pair_construction_one_pair_per_image_and_deterministic() -> None:
    z = torch.arange(6 * 5 * 2, dtype=torch.float32).view(6, 5, 2)
    first = construct_pairs(
        z,
        "global_local",
        num_global=2,
        estimator="balanced_one_pair_per_image",
        seed=3,
    )
    second = construct_pairs(
        z,
        "global_local",
        num_global=2,
        estimator="balanced_one_pair_per_image",
        seed=3,
    )

    assert first[0].shape == (6, 2)
    assert first[2] == second[2]
    assert torch.equal(first[0], second[0])
    assert len(set(first[2])) > 1


def test_heldout_split_is_disjoint() -> None:
    cal, eva = split_calibration_evaluation(101, 0.5, 7, torch.device("cpu"))

    assert set(cal.tolist()).isdisjoint(set(eva.tolist()))
    assert len(cal) + len(eva) == 101


def test_gaussianity_score_is_not_production_scaled() -> None:
    z = torch.randn(64, 6, generator=torch.Generator().manual_seed(9))
    unscaled = unscaled_cw_normality_score(z, gamma=0.5)
    production = float(CWReg(gamma=0.5)(z))

    assert production > unscaled * 100


def test_full_subset_estimate_independent_of_stats_chunk_size_convention() -> None:
    z1, z2 = _correlated_gaussian(n=512, d=8, rho=0.55)
    first = joint_structure_metrics(
        z1,
        z2,
        z2.roll(1, dims=0),
        gamma=0.5,
        covariance_eps=1e-5,
        projection_diagnostics=8,
        seed=0,
        fixed_rhos=[0.0],
        covariance_estimator="oas",
        calibration_fraction=0.5,
        num_null_repetitions=0,
        stats_chunk_size=32,
    )
    second = joint_structure_metrics(
        z1,
        z2,
        z2.roll(1, dims=0),
        gamma=0.5,
        covariance_eps=1e-5,
        projection_diagnostics=8,
        seed=0,
        fixed_rhos=[0.0],
        covariance_estimator="oas",
        calibration_fraction=0.5,
        num_null_repetitions=0,
        stats_chunk_size=512,
    )

    assert first["rho_hat"] == second["rho_hat"]


def test_chunked_sufficient_statistics_match_unchunked() -> None:
    z1, z2 = _correlated_gaussian(n=513, d=7, rho=0.35)
    full = sufficient_statistics(z1, z2, chunk_size=999)
    chunked = sufficient_statistics(z1, z2, chunk_size=32)

    for block in ["x", "y", "xy"]:
        assert torch.allclose(
            covariance_from_sufficient_stats(full, which=block, denominator="unbiased"),
            covariance_from_sufficient_stats(chunked, which=block, denominator="unbiased"),
            atol=1e-6,
            rtol=1e-6,
        )


def test_custom_oas_matches_sklearn() -> None:
    sklearn_covariance = pytest.importorskip("sklearn.covariance")
    x = torch.randn(257, 11, generator=torch.Generator().manual_seed(17), dtype=torch.float64)

    ours, shrinkage, _ = covariance_estimate(x, method="oas", eps=0.0, chunk_size=37)
    ref = sklearn_covariance.OAS(store_precision=False, assume_centered=False).fit(x.numpy())

    assert abs(shrinkage - float(ref.shrinkage_)) < 1e-6
    assert torch.allclose(ours, torch.from_numpy(ref.covariance_), atol=1e-8, rtol=1e-8)


def test_image_level_split_constructs_pairs_after_split_for_both_estimators() -> None:
    z = torch.randn(12, 5, 3, generator=torch.Generator().manual_seed(23))
    cal_ids, eval_ids = split_calibration_evaluation(12, 0.5, 5, torch.device("cpu"))

    for estimator in ["balanced_one_pair_per_image", "all_pairs_clustered"]:
        row, pair_obs, _ = joint_structure_metrics_from_image_split(
            z.index_select(0, cal_ids),
            z.index_select(0, eval_ids),
            "global_local",
            2,
            gamma=0.5,
            covariance_eps=1e-5,
            projection_diagnostics=4,
            seed=0,
            fixed_rhos=[0.0],
            covariance_estimator="ridge",
            estimator=estimator,
            num_null_repetitions=0,
            stats_chunk_size=3,
        )
        assert set(cal_ids.tolist()).isdisjoint(set(eval_ids.tolist()))
        assert row["calibration_source_image_count"] == len(cal_ids)
        assert row["evaluation_source_image_count"] == len(eval_ids)
        assert pair_obs == row["evaluation_pair_observations"]


def test_whitened_scalar_rho_common_difference_are_primary() -> None:
    z1, z2 = _correlated_gaussian(n=4096, d=8, rho=0.6)
    scale = torch.linspace(0.5, 2.0, 8)
    row = _metrics(z1 * scale, z2 * scale)

    assert row["plus_isotropy_error"] < row["raw_plus_isotropy_error"]
    assert row["minus_isotropy_error"] < row["raw_minus_isotropy_error"]
    assert row["plus_minus_crosscov_norm"] < 0.08


def test_exact_duplicate_null_and_energy_are_disabled() -> None:
    z = torch.randn(256, 6, generator=torch.Generator().manual_seed(29))
    row = joint_structure_metrics(
        z,
        z.clone(),
        z.roll(1, dims=0),
        gamma=0.5,
        covariance_eps=1e-5,
        projection_diagnostics=4,
        seed=0,
        fixed_rhos=[0.0],
        covariance_estimator="oas",
        calibration_fraction=0.5,
        num_null_repetitions=3,
        stats_chunk_size=32,
        pair_type="same_view_duplicate",
    )

    assert math.isnan(row["energy_auc"])
    assert math.isnan(row["rho_used_for_energy"])
    assert row["rho_was_clipped"]
    assert row["scalar_model_relative_residual_observed_to_null_median"] < 2.0


def test_gaussianity_diagnostics_are_capped_and_deterministic() -> None:
    z1, z2 = _correlated_gaussian(n=4096, d=8, rho=0.5)
    first = joint_structure_metrics(
        z1,
        z2,
        z2.roll(1, dims=0),
        gamma=0.5,
        covariance_eps=1e-5,
        projection_diagnostics=4,
        seed=11,
        fixed_rhos=[0.0],
        covariance_estimator="oas",
        calibration_fraction=0.5,
        num_null_repetitions=0,
        stats_chunk_size=128,
        gaussianity_max_samples=128,
    )
    second = joint_structure_metrics(
        z1,
        z2,
        z2.roll(1, dims=0),
        gamma=0.5,
        covariance_eps=1e-5,
        projection_diagnostics=4,
        seed=11,
        fixed_rhos=[0.0],
        covariance_estimator="oas",
        calibration_fraction=0.5,
        num_null_repetitions=0,
        stats_chunk_size=128,
        gaussianity_max_samples=128,
    )

    assert first["gaussianity_num_samples"] == 128
    assert first["gaussianity_was_subsampled"]
    assert first["unscaled_cw_normality_plus"] == second["unscaled_cw_normality_plus"]


def test_fixed_projection_dimension_and_matrix_across_image_counts() -> None:
    q = resolve_fixed_projection_dim(
        6,
        image_counts=[10, 20, 40],
        available_images=40,
        calibration_fraction=0.5,
        original_dim=8,
    )
    first = deterministic_projection(8, q, 123, torch.device("cpu"), torch.float32)
    second = deterministic_projection(8, q, 123, torch.device("cpu"), torch.float32)

    assert q == 4
    assert torch.equal(first, second)


def test_null_cache_key_respects_repetitions_gamma_epsilon_and_rho() -> None:
    NULL_CALIBRATION_CACHE.clear()
    observed = {
        "rho_hat": 0.25,
        "scalar_model_relative_residual": 0.1,
        "relative_offdiagonal_energy": 0.1,
        "antisymmetric_energy": 0.1,
        "singular_value_dispersion": 0.1,
        "plus_isotropy_error": 0.1,
        "minus_isotropy_error": 0.1,
        "plus_minus_crosscov_norm": 0.1,
        "unscaled_cw_normality_plus": 0.1,
        "unscaled_cw_normality_minus": 0.1,
    }
    kwargs = dict(
        observed=observed,
        n_cal=64,
        n_eval=64,
        dim=4,
        repetitions=2,
        gamma=0.5,
        covariance_eps=1e-5,
        covariance_estimator="oas",
        projection_diagnostics=4,
        fixed_rhos=[0.0],
        seed=0,
        stats_chunk_size=16,
        estimator="balanced_one_pair_per_image",
        same_view_duplicate=False,
        gaussianity_max_samples=32,
    )

    assert not matched_null_rows(**kwargs)[2]
    assert matched_null_rows(**kwargs)[2]
    changed_reps = dict(kwargs, repetitions=3)
    changed_gamma = dict(kwargs, gamma=0.6)
    changed_eps = dict(kwargs, covariance_eps=2e-5)
    changed_rho = dict(kwargs, observed=dict(observed, rho_hat=0.25002))
    assert not matched_null_rows(**changed_reps)[2]
    assert not matched_null_rows(**changed_gamma)[2]
    assert not matched_null_rows(**changed_eps)[2]
    assert not matched_null_rows(**changed_rho)[2]


def test_stability_rows_do_not_mix_projected_and_full_space() -> None:
    rows = [
        {
            "estimator": "balanced_one_pair_per_image",
            "analysis_space": "projected",
            "pair_type": "global_local",
        },
        {
            "estimator": "balanced_one_pair_per_image",
            "analysis_space": "full",
            "pair_type": "global_local",
        },
        {
            "estimator": "all_pairs_clustered",
            "analysis_space": "projected",
            "pair_type": "global_local",
        },
    ]

    selected = joint_stability_rows(rows, "projected")

    assert selected == [rows[0]]


def test_chunked_all_pairs_sufficient_statistics_match_materialized_reference() -> None:
    z = torch.randn(9, 5, 4, generator=torch.Generator().manual_seed(31))
    z1, z2, _ = construct_pairs(z, "local_local", 2, estimator="all_pairs_clustered")
    ref = sufficient_statistics(z1, z2, chunk_size=z1.shape[0])
    chunked = sufficient_statistics(z1, z2, chunk_size=7)

    assert torch.allclose(
        covariance_from_sufficient_stats(ref, which="xy", denominator="unbiased"),
        covariance_from_sufficient_stats(chunked, which="xy", denominator="unbiased"),
        atol=1e-6,
        rtol=1e-6,
    )


def test_default_mode_does_not_select_all_pairs_estimator() -> None:
    assert joint_structure_estimators(False) == ["balanced_one_pair_per_image"]
    assert joint_structure_estimators(True) == ["balanced_one_pair_per_image", "all_pairs_clustered"]


def test_representative_null_scope_only_runs_on_first_partition_and_aug() -> None:
    common = dict(
        estimator="balanced_one_pair_per_image",
        analysis_space="projected",
        null_full_space=False,
        null_scope="representative",
        pair_type="global_local",
    )

    assert should_run_joint_null(partition_id=0, augmentation_id=0, **common)
    assert not should_run_joint_null(partition_id=1, augmentation_id=0, **common)
    assert not should_run_joint_null(partition_id=0, augmentation_id=1, **common)
    assert should_run_joint_null(partition_id=0, augmentation_id=0, **dict(common, pair_type="same_view_duplicate"))
    assert not should_run_joint_null(
        partition_id=0,
        augmentation_id=0,
        **dict(common, estimator="all_pairs_clustered"),
    )


def test_null_cache_reuses_across_observational_partitions_with_fixed_null_seed() -> None:
    NULL_CALIBRATION_CACHE.clear()
    observed = {
        "rho_hat": 0.35,
        "scalar_model_relative_residual": 0.1,
        "relative_offdiagonal_energy": 0.1,
        "antisymmetric_energy": 0.1,
        "singular_value_dispersion": 0.1,
        "plus_isotropy_error": 0.1,
        "minus_isotropy_error": 0.1,
        "plus_minus_crosscov_norm": 0.1,
        "unscaled_cw_normality_plus": 0.1,
        "unscaled_cw_normality_minus": 0.1,
    }
    kwargs = dict(
        observed=observed,
        n_cal=64,
        n_eval=64,
        dim=4,
        repetitions=2,
        gamma=0.5,
        covariance_eps=1e-5,
        covariance_estimator="oas",
        projection_diagnostics=4,
        fixed_rhos=[0.0],
        seed=123,
        stats_chunk_size=16,
        estimator="balanced_one_pair_per_image",
        same_view_duplicate=False,
        gaussianity_max_samples=32,
    )

    assert not matched_null_rows(**kwargs)[2]
    assert matched_null_rows(**kwargs)[2]


def test_null_cache_does_not_substitute_isotropic_for_anisotropic_marginals() -> None:
    NULL_CALIBRATION_CACHE.clear()
    base = {
        "rho_hat": 0.2,
        "scalar_model_relative_residual": 0.1,
        "relative_offdiagonal_energy": 0.1,
        "antisymmetric_energy": 0.1,
        "singular_value_dispersion": 0.1,
        "plus_isotropy_error": 0.1,
        "minus_isotropy_error": 0.1,
        "plus_minus_crosscov_norm": 0.1,
        "unscaled_cw_normality_plus": 0.1,
        "unscaled_cw_normality_minus": 0.1,
        "target_fit_cw_plus": 0.1,
        "target_fit_cw_minus": 0.1,
        "shape_cw_plus": 0.1,
        "shape_cw_minus": 0.1,
    }
    kwargs = dict(
        n_cal=64,
        n_eval=64,
        dim=4,
        repetitions=1,
        gamma=0.5,
        covariance_eps=1e-5,
        covariance_estimator="oas",
        projection_diagnostics=4,
        fixed_rhos=[0.0],
        seed=123,
        stats_chunk_size=16,
        estimator="balanced_one_pair_per_image",
        same_view_duplicate=False,
        gaussianity_max_samples=32,
    )
    isotropic = dict(base, _c11_matrix=torch.eye(4), _c22_matrix=torch.eye(4))
    anisotropic = dict(base, _c11_matrix=torch.diag(torch.tensor([0.2, 1.0, 2.0, 4.0])), _c22_matrix=torch.eye(4))

    assert not matched_null_rows(observed=isotropic, **kwargs)[2]
    assert not matched_null_rows(observed=anisotropic, **kwargs)[2]


def test_target_fit_and_shape_cw_separate_anisotropy_from_shape() -> None:
    z1, z2 = _correlated_gaussian(n=4096, d=8, rho=0.4)
    scale = torch.linspace(0.2, 2.5, 8)
    row = _metrics(z1 * scale, z2 * scale)

    assert row["target_fit_cw_plus"] > row["shape_cw_plus"]
    assert row["target_fit_cw_minus"] > row["shape_cw_minus"]


def test_shape_cw_detects_isotropic_nongaussian_data() -> None:
    generator = torch.Generator().manual_seed(91)
    x = torch.randn(4096, 8, generator=generator)
    heavy = (torch.rand(4096, 8, generator=generator) < 0.08).float()
    heavy = heavy * torch.randn(4096, 8, generator=generator) * 5.0
    x = x / x.std(dim=0, unbiased=False)
    heavy = heavy / heavy.std(dim=0, unbiased=False)
    gaussian = _metrics(x, x.roll(1, dims=0))
    nongaussian = _metrics(heavy, heavy.roll(1, dims=0))

    assert nongaussian["shape_cw_plus"] > gaussian["shape_cw_plus"]


def test_shared_gaussianity_subsample_uses_identical_indices() -> None:
    tensors = [torch.arange(100).float().view(100, 1) + offset for offset in range(6)]

    sampled, count, was_subsampled, idx = shared_row_subsample(tensors, 17, seed=44)

    assert count == 17
    assert was_subsampled
    for source, got in zip(tensors, sampled, strict=True):
        assert torch.equal(got, source.index_select(0, idx))


def test_chunked_all_pairs_path_reports_chunked_without_full_pair_tensor() -> None:
    z = torch.randn(20, 5, 4, generator=torch.Generator().manual_seed(77))
    row, pair_obs, pairs = joint_structure_all_pairs_chunked_from_image_split(
        z[:10],
        z[10:],
        "local_local",
        2,
        gamma=0.5,
        covariance_eps=1e-5,
        projection_diagnostics=4,
        seed=0,
        fixed_rhos=[0.0],
        covariance_estimator="ridge",
        stats_chunk_size=3,
        gaussianity_max_samples=5,
    )

    assert row["all_pairs_computation"] == "chunked"
    assert row["statistics_accumulated_incrementally"]
    assert row["gaussianity_num_samples"] <= 5
    assert pair_obs == len(pairs) * 10
    assert row["shuffled_diagnostic_sample_aligned"]


def test_all_pairs_sufficient_stats_match_materialized_means_and_plus_minus_covariance() -> None:
    z = torch.randn(8, 5, 3, generator=torch.Generator().manual_seed(79))
    stats, _, _, _, _, _, _ = all_pairs_sample_and_stats(
        z,
        "local_local",
        2,
        shuffled=False,
        seed=0,
        stats_chunk_size=3,
        max_sample_rows=5,
    )
    z1, z2, _ = construct_pairs(z, "local_local", 2, estimator="all_pairs_clustered")
    ref = sufficient_statistics(z1, z2, chunk_size=z1.shape[0])

    for key in ["sum_x", "sum_y", "xtx", "yty", "xty"]:
        assert torch.allclose(stats[key], ref[key], atol=1e-6, rtol=1e-6)

    c11 = covariance_from_sufficient_stats(stats, which="x", denominator="unbiased")
    c22 = covariance_from_sufficient_stats(stats, which="y", denominator="unbiased")
    c12 = covariance_from_sufficient_stats(stats, which="xy", denominator="unbiased")
    analytic_plus = 0.5 * (c11 + c22 + c12 + c12.T)
    analytic_minus = 0.5 * (c11 + c22 - c12 - c12.T)
    mat_plus = covariance((z1 + z2) / math.sqrt(2.0))
    mat_minus = covariance((z1 - z2) / math.sqrt(2.0))

    assert torch.allclose(analytic_plus, mat_plus, atol=1e-6, rtol=1e-6)
    assert torch.allclose(analytic_minus, mat_minus, atol=1e-6, rtol=1e-6)


def test_all_pairs_diagnostic_sampling_is_deterministic_and_pair_balanced() -> None:
    z = torch.randn(50, 5, 3, generator=torch.Generator().manual_seed(80))
    first = all_pairs_sample_and_stats(
        z,
        "local_local",
        2,
        shuffled=False,
        seed=10,
        stats_chunk_size=7,
        max_sample_rows=6,
    )
    second = all_pairs_sample_and_stats(
        z,
        "local_local",
        2,
        shuffled=False,
        seed=10,
        stats_chunk_size=7,
        max_sample_rows=6,
    )
    identities = first[5]
    pair_counts = {}
    for pair_idx, _ in identities:
        pair_counts[pair_idx] = pair_counts.get(pair_idx, 0) + 1

    assert first[6] == second[6]
    assert identities == second[5]
    assert max(pair_counts.values()) - min(pair_counts.values()) <= 1


def test_materialized_all_pairs_guard_rejects_large_local_local_tensor() -> None:
    z = torch.randn(2000, 10, 2, generator=torch.Generator().manual_seed(78))

    with pytest.raises(ValueError, match="Refusing to materialize"):
        construct_pairs(z, "local_local", 2, estimator="all_pairs_clustered")
