"""Tests for cached-embedding MSE/CW gradient-conflict analysis."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
import torch

import analyze_joint_cw_plausibility as analysis

pytestmark = pytest.mark.unit


def sample_embeddings(
    batch_size: int = 4,
    num_views: int = 4,
    dimension: int = 6,
) -> torch.Tensor:
    return torch.randn(
        batch_size,
        num_views,
        dimension,
        generator=torch.Generator().manual_seed(123),
    )


def test_decomposed_cw_loss_matches_production() -> None:
    result = analysis.validate_cw_decomposition(sample_embeddings())

    assert result["relative_loss_error"] < 1e-5


def test_decomposed_cw_gradient_matches_production() -> None:
    result = analysis.validate_cw_decomposition(sample_embeddings())

    assert result["gradient_cosine"] > 0.99999
    assert result["relative_gradient_error"] < 1e-4


def test_cw_component_gradients_reconstruct_full_gradient() -> None:
    result = analysis.validate_cw_decomposition(sample_embeddings())

    assert result["component_reconstruction_error"] < 1e-4


def test_negative_gradient_cosine_predicts_positive_other_loss_change() -> None:
    z = torch.ones(2, 2, 1)
    gradient_a = torch.ones_like(z)
    gradient_b = -torch.ones_like(z)
    delta_b = analysis.normalized_descent_step(z, gradient_b, 1e-3)

    assert analysis.tensor_cosine(gradient_a, gradient_b) == pytest.approx(-1.0)
    assert float((gradient_a * delta_b).sum()) > 0.0


def test_per_image_cosine_aggregation_has_expected_shape() -> None:
    left = sample_embeddings(batch_size=7)
    right = -left
    values = analysis.per_image_cosines(left, right)
    summary = analysis.per_image_cosine_summary(left, right)

    assert values.shape == (7,)
    assert summary["median_per_image_cosine_mse_cw"] == pytest.approx(-1.0)
    assert summary["fraction_images_negative_cosine"] == pytest.approx(1.0)


def test_global_and_local_gradient_view_selection() -> None:
    gradient = torch.arange(2 * 10 * 3).reshape(2, 10, 3)

    assert torch.equal(
        analysis.select_gradient_view_group(gradient, "global"),
        gradient[:, :2],
    )
    assert torch.equal(
        analysis.select_gradient_view_group(gradient, "local"),
        gradient[:, 2:],
    )


@pytest.mark.parametrize("num_views", [8, 10])
def test_dynamic_cache_view_layout_and_gradient_analysis(
    tmp_path: Path, num_views: int
) -> None:
    cache_path = tmp_path / f"embeddings-{num_views}.pt"
    view_order = ["global_1", "global_2"] + [
        f"local_{index}" for index in range(1, num_views - 1)
    ]
    embeddings = torch.randn(
        1,
        8,
        num_views,
        6,
        generator=torch.Generator().manual_seed(num_views),
    )
    torch.save(
        {
            "embeddings": embeddings,
            "metadata": {
                "checkpoint": f"views-{num_views}",
                "training_progress": 0.0,
                "num_views": num_views,
                "num_local_views": num_views - 2,
                "view_order": view_order,
            },
        },
        cache_path,
    )
    args = analysis.parse_args(
        [
            "--analysis-mode",
            "gradients",
            "--cache-file",
            str(cache_path),
            "--device",
            "cpu",
            "--output-dir",
            str(tmp_path / "output"),
        ]
    )
    records = analysis.discover_cache_records(args)

    assert analysis.validate_record_view_layouts(records) == (
        view_order,
        num_views - 2,
    )
    metrics = analysis.analyze_gradient_batch(
        embeddings[0, :4],
        args=args,
        augmentation_id=0,
        batch_id=0,
        device=torch.device("cpu"),
    )
    assert metrics["within_pair_type_gradient_reconstruction_error"] < 1e-4
    assert "cosine_mse_cw_exclusion_only" in metrics
    assert "cosine_mse_cw_within_gg" in metrics
    assert "measured_relative_mse_change_after_within_ll_cw_step" in metrics


def test_deterministic_gradient_batching_is_reproducible() -> None:
    first = analysis.deterministic_gradient_batches(100, 8, 5, seed=9)
    second = analysis.deterministic_gradient_batches(100, 8, 5, seed=9)

    assert len(first) == 5
    assert all(torch.equal(left, right) for left, right in zip(first, second, strict=True))
    assert torch.unique(torch.cat(first)).numel() == 40


def test_ep_control_directions_are_deterministic() -> None:
    z = sample_embeddings(batch_size=3, dimension=5)
    first_loss, first_gradient = analysis.production_ep_value_and_grad(
        z, num_slices=16, projection_seed=42
    )
    second_loss, second_gradient = analysis.production_ep_value_and_grad(
        z, num_slices=16, projection_seed=42
    )

    assert first_loss == pytest.approx(second_loss, abs=0.0)
    assert torch.equal(first_gradient, second_gradient)


@pytest.fixture(scope="module")
def gradient_smoke_outputs(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[dict[str, Path], Path]:
    root = tmp_path_factory.mktemp("joint-cw-gradient-smoke")
    cache_path = root / "embeddings.pt"
    embeddings = torch.randn(
        1,
        16,
        10,
        12,
        generator=torch.Generator().manual_seed(7),
    )
    torch.save(
        {
            "embeddings": embeddings,
            "metadata": {
                "checkpoint": "untrained",
                "checkpoint_path": "",
                "training_progress": 0.0,
                "view_order": ["global_1", "global_2"]
                + [f"local_{index}" for index in range(1, 9)],
            },
        },
        cache_path,
    )
    output_dir = root / "output"
    args = analysis.parse_args(
        [
            "--analysis-mode",
            "gradients",
            "--cache-file",
            str(cache_path),
            "--gradient-batch-size",
            "4",
            "--gradient-num-batches",
            "2",
            "--device",
            "cpu",
            "--output-dir",
            str(output_dir),
        ]
    )
    outputs = analysis.run(args)
    return outputs, output_dir


def test_gradient_mode_uses_original_feature_dimension(
    gradient_smoke_outputs: tuple[dict[str, Path], Path],
) -> None:
    outputs, _ = gradient_smoke_outputs
    with outputs["gradient_raw"].open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    assert {int(row["feature_dimension"]) for row in rows} == {12}
    assert {int(row["effective_sample_count"]) for row in rows} == {40}
    assert {
        "cosine_mse_cw_exclusion_only",
        "cosine_mse_cw_within_gg",
        "cosine_mse_cw_within_gl",
        "cosine_mse_cw_within_ll",
    }.issubset(rows[0])


def test_gradient_cpu_smoke_completes(
    gradient_smoke_outputs: tuple[dict[str, Path], Path],
) -> None:
    outputs, output_dir = gradient_smoke_outputs
    assert all(path.exists() for path in outputs.values())
    assert not (output_dir / "joint_cw_plausibility_raw.csv").exists()
    config = json.loads((output_dir / "gradient_run_config.json").read_text())
    assert config["view_order"] == ["global_1", "global_2"] + [
        f"local_{index}" for index in range(1, 9)
    ]
    assert config["number_of_local_views"] == 8
