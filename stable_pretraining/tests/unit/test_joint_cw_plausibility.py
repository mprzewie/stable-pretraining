"""Focused tests for the standalone joint-CW plausibility analysis."""

from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import analyze_joint_cw_plausibility as analysis

pytestmark = pytest.mark.unit


def correlated_gaussian(
    n: int,
    d: int,
    rho: float,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(n, d, generator=generator, dtype=torch.float64)
    noise = torch.randn(n, d, generator=generator, dtype=torch.float64)
    y = rho * x + math.sqrt(1.0 - rho * rho) * noise
    return x, y


def metric_kwargs(dimension: int) -> dict[str, object]:
    return {
        "ridge_eps": 1e-4,
        "gamma": 0.5,
        "gaussianity_max_samples": 256,
        "shape_directions": analysis.fixed_shape_directions(dimension, 16, 0),
        "diagnostic_seed": 0,
    }


def null_args(image_counts: list[int] | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        ridge_eps=1e-4,
        cw_gamma=0.5,
        num_structural_null_repetitions=8,
        num_shape_null_repetitions=8,
        null_seed=17,
        image_counts=image_counts or [128],
    )


def observed_row(
    *,
    checkpoint: str = "checkpoint",
    pair_type: str = "global_global",
    rho: float = 0.5,
    seed: int = 0,
    image_count: int = 128,
) -> dict[str, object]:
    calibration_count = image_count // 2
    x, y = correlated_gaussian(calibration_count, 4, rho, seed=seed)
    row, _ = analysis.calibration_metrics(
        x,
        y,
        ridge_eps=1e-4,
        gamma=0.5,
        gaussianity_max_samples=32,
        shape_directions=analysis.fixed_shape_directions(4, 4, 0),
        diagnostic_seed=seed,
    )
    return {
        "checkpoint": checkpoint,
        "training_progress": 50.0,
        "augmentation_id": seed // 3,
        "partition_id": seed % 3,
        "image_count": image_count,
        "calibration_count": calibration_count,
        "evaluation_count": calibration_count,
        "pair_type": pair_type,
        "projected_dimension": 4,
        **row,
    }


def test_correlated_gaussian_recovers_known_rho() -> None:
    x, y = correlated_gaussian(5000, 8, 0.65, seed=1)
    row, _ = analysis.calibration_metrics(x, y, **metric_kwargs(8))

    assert row["rho_hat"] == pytest.approx(0.65, abs=0.025)


def test_scalar_rho_sample_falls_inside_matched_null(tmp_path: Path) -> None:
    row = observed_row(rho=0.55, seed=19)
    args = null_args()
    args.num_structural_null_repetitions = 60
    analysis.apply_grouped_null_calibration([row], args, tmp_path)

    assert row["scalar_residual_null_p05"] <= row["scalar_residual"]
    assert row["scalar_residual"] <= row["scalar_residual_null_p95"]


def test_diagonal_correlations_increase_scalar_residual() -> None:
    generator = torch.Generator().manual_seed(2)
    x = torch.randn(6000, 8, generator=generator, dtype=torch.float64)
    noise = torch.randn(6000, 8, generator=generator, dtype=torch.float64)
    correlations = torch.linspace(0.1, 0.9, 8, dtype=torch.float64)
    anisotropic_y = x * correlations + noise * torch.sqrt(1.0 - correlations.square())
    scalar_x, scalar_y = correlated_gaussian(6000, 8, float(correlations.mean()), seed=2)

    anisotropic, _ = analysis.calibration_metrics(x, anisotropic_y, **metric_kwargs(8))
    scalar, _ = analysis.calibration_metrics(scalar_x, scalar_y, **metric_kwargs(8))

    assert anisotropic["scalar_residual"] > scalar["scalar_residual"] * 2
    assert anisotropic["explained_diagonal"] > anisotropic["explained_scalar"]


def test_low_rank_dependence_has_distinctive_spectrum() -> None:
    generator = torch.Generator().manual_seed(3)
    x = torch.randn(8000, 8, generator=generator, dtype=torch.float64)
    y = torch.randn(8000, 8, generator=generator, dtype=torch.float64)
    y[:, 0] = 0.9 * x[:, 0] + math.sqrt(1.0 - 0.9**2) * y[:, 0]
    row, _ = analysis.calibration_metrics(x, y, **metric_kwargs(8))
    singular = torch.tensor(__import__("json").loads(row["singular_values_R"]))

    assert singular[0] > 5 * singular[1]
    assert row["explained_rank_1"] > 0.9


def test_covariance_matched_exact_duplicates_pass(tmp_path: Path) -> None:
    generator = torch.Generator().manual_seed(4)
    scales = torch.linspace(0.2, 3.0, 8)
    base = torch.randn(64, 8, generator=generator) * scales
    embeddings = torch.randn(1, 64, 10, 8, generator=generator)
    embeddings[0, :, 0] = base
    record = analysis.CacheRecord("test", tmp_path / "cache.pt", embeddings, {}, 0.0)
    args = null_args([64])
    args.calibration_fraction = 0.5
    args.seed = 0
    args.num_structural_null_repetitions = 40
    results, nulls = analysis.duplicate_sanity_checks(
        record, torch.eye(8), args, tmp_path
    )

    assert len(nulls) == 1
    assert results[64]["duplicate_rho"] > 0.95
    assert results[64]["duplicate_sanity_pass"]


def test_nine_observations_share_one_structural_null(tmp_path: Path) -> None:
    rows = [observed_row(seed=seed) for seed in range(9)]
    output = analysis.apply_grouped_null_calibration(rows, null_args(), tmp_path)
    structural_keys = {row["structural_null_key"] for row in rows}
    structural_outputs = {
        key for key, values in output.items() if values[0]["null_null_type"] == "structural"
    }

    assert len(structural_keys) == 1
    assert structural_outputs == structural_keys


def test_shape_null_is_reused_across_rho_and_pair_types(tmp_path: Path) -> None:
    rows = [
        observed_row(checkpoint="a", pair_type="global_global", rho=0.2, seed=0),
        observed_row(checkpoint="b", pair_type="local_local", rho=0.8, seed=1),
    ]
    output = analysis.apply_grouped_null_calibration(rows, null_args(), tmp_path)
    shape_outputs = [
        key for key, values in output.items() if values[0]["null_null_type"] == "shape"
    ]

    assert rows[0]["shape_null_key"] == rows[1]["shape_null_key"]
    assert len(shape_outputs) == 1


def test_structural_null_loop_never_calls_cw_normality(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        analysis,
        "cw_normality",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("CW called")),
    )
    config = {
        "rho": 0.5,
        "calibration_count": 32,
        "dimension": 4,
        "ridge_eps": 1e-4,
    }

    rows = analysis.generate_structural_null_repetitions(
        config, repetitions=3, null_seed=0, cache_key="structural"
    )

    assert len(rows) == 3


def test_calibration_and_evaluation_ids_are_disjoint() -> None:
    calibration, evaluation = analysis.partition_and_split_ids(200, 100, 1, 0.5, 0)

    assert len(calibration) == 50
    assert len(evaluation) == 50
    assert set(calibration.tolist()).isdisjoint(evaluation.tolist())


@pytest.mark.parametrize("pair_type", analysis.PAIR_TYPES)
def test_each_image_contributes_one_balanced_deterministic_pair(pair_type: str) -> None:
    embeddings = torch.randn(224, 10, 4, generator=torch.Generator().manual_seed(5))
    image_ids = torch.arange(224)
    first = analysis.construct_balanced_pairs(embeddings, image_ids, pair_type, seed=17)
    second = analysis.construct_balanced_pairs(embeddings, image_ids, pair_type, seed=17)
    assignments = first[2]
    counts = {
        pair: assignments.count(pair)
        for pair in analysis.pair_pool(pair_type, embeddings.shape[1])
    }

    assert first[0].shape[0] == len(image_ids)
    assert torch.equal(first[0], second[0])
    assert assignments == second[2]
    assert max(counts.values()) - min(counts.values()) <= 1


def test_fixed_projection_is_identical_everywhere() -> None:
    first = analysis.deterministic_projection(32, 8, seed=11)
    second = analysis.deterministic_projection(32, 8, seed=11)

    assert torch.equal(first, second)
    assert torch.allclose(first.T @ first, torch.eye(8), atol=1e-6)


def test_cw_normality_never_receives_more_than_limit() -> None:
    analysis.MAX_CW_SAMPLES_SEEN = 0
    x, y = correlated_gaussian(2048, 8, 0.5, seed=6)
    kwargs = metric_kwargs(8)
    kwargs["gaussianity_max_samples"] = 128
    row, _ = analysis.calibration_metrics(x, y, **kwargs)

    assert row["gaussianity_num_samples"] == 128
    assert analysis.MAX_CW_SAMPLES_SEEN == 128


def test_group_summary_does_not_average_p_values(tmp_path: Path) -> None:
    rows = [observed_row(seed=seed) for seed in range(3)]
    analysis.apply_grouped_null_calibration(rows, null_args(), tmp_path)
    rows[0]["scalar_residual_null_upper_tail_p"] = 0.01
    rows[1]["scalar_residual_null_upper_tail_p"] = 0.50
    rows[2]["scalar_residual_null_upper_tail_p"] = 0.99
    aggregate = analysis.aggregate_rows(rows)[0]

    assert not any("upper_tail_p" in key for key in aggregate)
    assert "scalar_residual_fraction_above_null_p95" in aggregate
    assert "scalar_residual_median" in aggregate
    assert "scalar_residual_q25" in aggregate
    assert "scalar_residual_q75" in aggregate


def test_default_runtime_guard_has_at_most_eighteen_and_two_configs() -> None:
    rows = []
    for checkpoint in ("untrained", "middle", "final"):
        for image_count in (1000, 5000):
            for pair_type in analysis.PAIR_TYPES:
                row = observed_row(
                    checkpoint=checkpoint,
                    pair_type=pair_type,
                    image_count=image_count,
                )
                row["gaussianity_num_samples"] = min(image_count // 2, 1024)
                rows.append(row)
    args = null_args([1000, 5000])
    plan = analysis.null_runtime_plan(rows, args, checkpoint_count=3, emit=False)

    assert plan["structural_null_configurations"] == 18
    assert plan["shape_null_configurations"] == 2


def test_cpu_smoke_analysis(tmp_path: Path) -> None:
    analysis.MAX_CW_SAMPLES_SEEN = 0
    cache_path = tmp_path / "untrained.pt"
    embeddings = torch.randn(1, 32, 10, 12, generator=torch.Generator().manual_seed(7))
    torch.save(
        {
            "embeddings": embeddings,
            "metadata": {
                "checkpoint": "untrained",
                "checkpoint_path": "",
                "training_progress": 0.0,
                "view_order": ["global_1", "global_2"] + [f"local_{index}" for index in range(1, 9)],
            },
        },
        cache_path,
    )
    output_dir = tmp_path / "output"
    args = analysis.parse_args(
        [
            "--cache-file",
            str(cache_path),
            "--num-images",
            "32",
            "--image-counts",
            "32",
            "--projection-dim",
            "8",
            "--num-partitions",
            "1",
            "--num-augmentation-draws",
            "1",
            "--num-null-repetitions",
            "2",
            "--gaussianity-max-samples",
            "16",
            "--num-shape-directions",
            "4",
            "--output-dir",
            str(output_dir),
        ]
    )
    outputs = analysis.run(args)

    assert all(path.exists() for path in outputs.values())
    assert (output_dir / "run_config.json").exists()
    assert analysis.MAX_CW_SAMPLES_SEEN <= 16
