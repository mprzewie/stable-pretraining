#!/usr/bin/env python3
"""Analyze finite-slice LeJEPA EP/SIGReg convergence to CWReg.

This script deliberately reuses the production LeJEPA regularizers:
``SlicedEppsPulley`` and ``CWReg`` from ``stable_pretraining.methods.lejepa``.
The input convention follows ``LeJEPA._compute_loss``: projector outputs are
stored per image/per view, then every regularizer batch is flattened in
view-major order as ``all_projected.reshape(-1, feature_dim)``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stable_pretraining.data import HFDataset  # noqa: E402
from stable_pretraining.data import transforms as spt_t  # noqa: E402
from stable_pretraining.data.gpu_transforms import (  # noqa: E402
    GPUColorJitter,
    GPUCompose,
    GPUGaussianBlur,
    GPUNormalize,
    GPURandomGrayscale,
    GPURandomHorizontalFlip,
    GPURandomResizedCrop,
    GPURandomSolarize,
    GroupedMultiView,
)
from stable_pretraining.methods.lejepa import CWReg, LeJEPA, SlicedEppsPulley  # noqa: E402


EPS = 1e-12
CHECKPOINT_SUFFIXES = {".ckpt", ".pt", ".pth"}


@dataclass(frozen=True)
class CheckpointSpec:
    label: str
    path: Path | None
    hparams: dict[str, Any] | None = None


class ImageFolderDict(Dataset):
    """Thin dict wrapper around torchvision ImageFolder."""

    def __init__(self, root: Path, transform):
        from torchvision.datasets import ImageFolder

        self.dataset = ImageFolder(str(root), transform=None)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        image, label = self.dataset[index]
        sample = {"image": image, "label": label}
        return self.transform(sample)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Empirical finite-slice EP/SIGReg convergence to CWReg."
    )
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--checkpoint", action="append", type=Path, default=[])
    parser.add_argument("--checkpoint-label", action="append", default=[])
    parser.add_argument(
        "--checkpoint-density",
        type=float,
        default=1.0,
        help=(
            "Approximate fraction of discovered non-last checkpoints to analyze. "
            "For example, 0.25 selects about 25%% spread across training time."
        ),
    )
    parser.add_argument(
        "--max-checkpoints",
        type=int,
        default=None,
        help="Maximum number of discovered non-last checkpoints to analyze.",
    )
    parser.add_argument(
        "--include-last-checkpoints",
        action="store_true",
        help="Include last*.ckpt files in addition to epoch checkpoints.",
    )
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--dataset-name", default="clane9/imagenet-100")
    parser.add_argument("--dataset-split", choices=["train", "validation", "val"], default="validation")
    parser.add_argument("--dataset-revision", default=None)
    parser.add_argument("--dataset-cache-dir", type=Path, default=None)
    parser.add_argument("--num-images", type=int, default=5000)
    parser.add_argument("--image-batch-size", type=int, default=128)
    parser.add_argument("--num-partitions", type=int, default=5)
    parser.add_argument("--num-projection-seeds", type=int, default=20)
    parser.add_argument(
        "--slice-counts",
        type=int,
        nargs="+",
        default=[1, 4, 16, 64, 256, 1024, 4096, 8192, 16384],
    )
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--include-untrained", dest="include_untrained", action="store_true", default=True)
    parser.add_argument("--no-include-untrained", dest="include_untrained", action="store_false")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument(
        "--encoder-name",
        default=os.environ.get("ENCODER_NAME"),
        help="Backbone architecture. Defaults to the checkpoint hparam when available.",
    )
    parser.add_argument("--drop-path-rate", type=float, default=None)
    parser.add_argument("--pretrained-backbone", action="store_true")
    parser.add_argument(
        "--model-sigreg",
        choices=["ep", "cw"],
        default=os.environ.get("SIGREG"),
        help=(
            "Regularizer type used only to instantiate the LeJEPA checkpoint "
            "container. The convergence analysis always evaluates both "
            "production SlicedEppsPulley and CWReg separately on cached Z."
        ),
    )
    parser.add_argument("--local-crop-size", type=int, default=int(os.environ.get("LOCAL_CROP_SIZE", "96")))
    parser.add_argument("--t-max", type=float, default=3.0)
    parser.add_argument("--n-points", type=int, default=17)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--resume-cache", dest="resume_cache", action="store_true", default=True)
    parser.add_argument("--no-resume-cache", dest="resume_cache", action="store_false")
    parser.add_argument("--model-mode", choices=["train", "eval"], default="train")
    parser.add_argument("--keep-last-batch", action="store_true")
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def default_stable_pretraining_data_dir() -> Path:
    storage_root = Path(os.environ.get("STORAGE_ROOT", Path.home() / "storage"))
    return Path(
        os.environ.get(
            "STABLE_PRETRAINING_DATA_DIR",
            storage_root / "datasets" / "stable-pretraining",
        )
    ).expanduser()


def infer_dataset_cache_dir(args: argparse.Namespace) -> Path | None:
    if args.dataset_cache_dir is not None:
        return args.dataset_cache_dir.expanduser()
    if args.dataset_root is not None:
        return None

    known_names = {
        "clane9/imagenet-100": "imagenet100",
        "frgfm/imagenette": "imagenet10",
    }
    cache_name = known_names.get(args.dataset_name)
    if cache_name is None:
        return None

    candidate = default_stable_pretraining_data_dir() / cache_name
    return candidate if candidate.exists() else None


def apply_debug_defaults(args: argparse.Namespace) -> None:
    if not args.debug:
        return
    args.num_images = min(args.num_images, 256)
    args.image_batch_size = min(args.image_batch_size, 32)
    args.num_partitions = min(args.num_partitions, 2)
    args.num_projection_seeds = min(args.num_projection_seeds, 3)
    args.slice_counts = [m for m in args.slice_counts if m <= 64] or [1, 4, 16]
    args.num_workers = min(args.num_workers, 2)
    args.bootstrap_samples = min(args.bootstrap_samples, 100)


def slugify(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return value.strip("._-") or "checkpoint"


def checkpoint_specs(args: argparse.Namespace) -> list[CheckpointSpec]:
    paths = [p.resolve() for p in args.checkpoint]
    if args.checkpoint_dir is not None:
        discovered = sorted(
            (
                p
                for p in args.checkpoint_dir.rglob("*")
                if p.is_file()
                and p.suffix in CHECKPOINT_SUFFIXES
                and (args.include_last_checkpoints or not p.name.startswith("last"))
            ),
            key=checkpoint_sort_key,
        )
        discovered = subsample_checkpoints(
            discovered,
            density=args.checkpoint_density,
            max_checkpoints=args.max_checkpoints,
        )
        paths.extend(p.resolve() for p in discovered)

    specs: list[CheckpointSpec] = []
    if args.include_untrained:
        specs.append(CheckpointSpec("untrained", None, None))

    labels = list(args.checkpoint_label)
    if labels and len(labels) != len(paths):
        raise ValueError(
            f"Got {len(labels)} checkpoint labels for {len(paths)} checkpoints."
        )

    for i, path in enumerate(paths):
        label = labels[i] if labels else path.stem
        resolved = path.resolve()
        specs.append(
            CheckpointSpec(
                label=slugify(label),
                path=resolved,
                hparams=read_checkpoint_hparams(resolved),
            )
        )
    return specs


def checkpoint_sort_key(path: Path) -> tuple[int, int | float, str]:
    match = re.search(r"epoch[=-](\d+)", path.name)
    if match:
        return (0, int(match.group(1)), path.name)
    if path.name.startswith("last"):
        return (2, math.inf, path.name)
    return (1, math.inf, path.name)


def subsample_checkpoints(
    paths: list[Path],
    density: float,
    max_checkpoints: int | None,
) -> list[Path]:
    if density <= 0:
        raise ValueError("--checkpoint-density must be positive.")
    if not paths:
        return []

    target = len(paths)
    if density < 1.0:
        target = max(1, int(round(len(paths) * density)))
    if max_checkpoints is not None:
        if max_checkpoints <= 0:
            raise ValueError("--max-checkpoints must be positive.")
        target = min(target, max_checkpoints)
    if target >= len(paths):
        return paths

    selected = sorted(
        {
            int(round(index))
            for index in np.linspace(0, len(paths) - 1, target)
        }
    )
    return [paths[index] for index in selected]


def read_checkpoint_hparams(path: Path) -> dict[str, Any]:
    if path.suffix not in CHECKPOINT_SUFFIXES:
        return {}
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] Could not inspect checkpoint hparams for {path}: {exc}", file=sys.stderr)
        return {}
    hparams = checkpoint.get("hyper_parameters", {}) if isinstance(checkpoint, dict) else {}
    return hparams if isinstance(hparams, dict) else {}


def effective_encoder_name(args: argparse.Namespace, spec: CheckpointSpec) -> str:
    if args.encoder_name:
        return args.encoder_name
    if spec.hparams and spec.hparams.get("module.model.encoder_name"):
        return str(spec.hparams["module.model.encoder_name"])
    return "vit_small_patch16_224"


def effective_drop_path_rate(args: argparse.Namespace, spec: CheckpointSpec) -> float:
    if args.drop_path_rate is not None:
        return args.drop_path_rate
    if spec.hparams and spec.hparams.get("module.model.drop_path_rate") is not None:
        return float(spec.hparams["module.model.drop_path_rate"])
    return float(os.environ.get("DROP_PATH_RATE", "0.1"))


def effective_model_sigreg(args: argparse.Namespace, spec: CheckpointSpec) -> str:
    if args.model_sigreg:
        return args.model_sigreg
    if spec.hparams and spec.hparams.get("module.model.sigreg"):
        value = str(spec.hparams["module.model.sigreg"])
        if value in {"ep", "cw"}:
            return value
    return "ep"


def checkpoint_epoch(spec: CheckpointSpec) -> int | None:
    if spec.path is None:
        return None
    match = re.search(r"epoch[=-](\d+)", spec.path.name)
    return int(match.group(1)) if match else None


def checkpoint_training_percent(spec: CheckpointSpec) -> float | None:
    if spec.path is None:
        return 0.0
    epoch = checkpoint_epoch(spec)
    if epoch is None or not spec.hparams:
        return None
    max_epochs = spec.hparams.get("trainer.max_epochs")
    if max_epochs is None:
        return None
    try:
        max_epochs_float = float(max_epochs)
    except (TypeError, ValueError):
        return None
    if max_epochs_float <= 0:
        return None
    return min(100.0, max(0.0, 100.0 * float(epoch + 1) / max_epochs_float))


def checkpoint_display_label(checkpoint: str, percent: float | None) -> str:
    if checkpoint == "untrained":
        return "untrained"
    if percent is None or not math.isfinite(percent):
        return checkpoint
    rounded = round(percent)
    if abs(percent - rounded) < 0.05:
        return f"{rounded}% trained"
    return f"{percent:.1f}% trained"


def checkpoint_plot_sort_key(row: dict[str, Any]) -> tuple[int, float, str]:
    checkpoint = str(row["checkpoint"])
    if checkpoint == "untrained":
        return (0, 0.0, checkpoint)
    percent = row.get("training_percent")
    if percent is None or percent == "":
        return (2, math.inf, checkpoint)
    try:
        percent_value = float(percent)
    except (TypeError, ValueError):
        return (2, math.inf, checkpoint)
    return (1, percent_value, checkpoint)


def checkpoint_plot_specs(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_checkpoint: dict[str, dict[str, Any]] = {}
    for row in rows:
        checkpoint = str(row["checkpoint"])
        if checkpoint not in by_checkpoint:
            by_checkpoint[checkpoint] = row
    return sorted(by_checkpoint.values(), key=checkpoint_plot_sort_key)


def checkpoint_color(row: dict[str, Any], cmap) -> Any:
    percent = row.get("training_percent")
    if percent is None or percent == "":
        value = 0.5
    else:
        try:
            value = min(1.0, max(0.0, float(percent) / 100.0))
        except (TypeError, ValueError):
            value = 0.5
    return cmap(value)


def format_slice_tick(slice_count: int) -> str:
    if slice_count > 0 and slice_count & (slice_count - 1) == 0:
        return rf"$2^{{{int(math.log2(slice_count))}}}$"
    return str(slice_count)


def metric_filename(name: str) -> str:
    return slugify(name.replace("_mean", ""))


def write_caption(output_dir: Path, filename: str, caption: str) -> None:
    (output_dir / f"{filename}.txt").write_text(caption.strip() + "\n")


def set_plot_theme() -> None:
    try:
        import seaborn as sns

        sns.set_theme()
    except ModuleNotFoundError:
        import matplotlib.pyplot as plt

        plt.style.use("seaborn-v0_8")


def plot_metric_panel(
    ax: Any,
    aggregate: list[dict[str, Any]],
    metric: str,
    ylabel: str,
    ref: float,
    *,
    ylog: bool = False,
    show_legend: bool = False,
) -> None:
    import matplotlib.pyplot as plt

    plot_specs = checkpoint_plot_specs(aggregate)
    cmap = plt.get_cmap("coolwarm")
    slice_ticks = sorted({row["slice_count"] for row in aggregate})
    for spec_row in plot_specs:
        checkpoint = spec_row["checkpoint"]
        label = spec_row.get("display_label", checkpoint)
        color = checkpoint_color(spec_row, cmap)
        rows = sorted(
            [row for row in aggregate if row["checkpoint"] == checkpoint],
            key=lambda row: row["slice_count"],
        )
        x = np.asarray([row["slice_count"] for row in rows], dtype=float)
        y = np.asarray([row[f"{metric}_mean"] for row in rows], dtype=float)
        std = np.asarray([row.get(f"{metric}_std", 0.0) for row in rows], dtype=float)
        ax.plot(x, y, marker="o", linewidth=1.8, color=color, label=label)
        ax.fill_between(x, y - std, y + std, color=color, alpha=0.16)
    ax.axhline(ref, color="black", linestyle="--", linewidth=0.8, alpha=0.55)
    ax.set_xscale("log", base=2)
    ax.set_xticks(slice_ticks)
    ax.set_xticklabels([format_slice_tick(tick) for tick in slice_ticks])
    if ylog:
        ax.set_yscale("log")
    ax.set_xlabel("EP slices")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    if show_legend:
        ax.legend(frameon=False, fontsize=8)


def save_single_metric_figure(
    aggregate: list[dict[str, Any]],
    output_dir: Path,
    metric: str,
    ylabel: str,
    ref: float,
    *,
    filename: str,
    caption: str,
    ylog: bool = False,
) -> None:
    import matplotlib.pyplot as plt

    set_plot_theme()
    fig, ax = plt.subplots(1, 1, figsize=(5.0, 3.6))
    plot_metric_panel(
        ax,
        aggregate,
        metric,
        ylabel,
        ref,
        ylog=ylog,
        show_legend=True,
    )
    fig.tight_layout()
    fig.savefig(output_dir / f"{filename}.pdf")
    write_caption(output_dir, filename, caption)
    plt.close(fig)


def cpu_transform():
    return spt_t.Compose(
        spt_t.RGB(),
        spt_t.Resize(size=[256, 256]),
        spt_t.ToImage(),
    )


def gpu_multiview_transform(local_crop_size: int) -> GroupedMultiView:
    global_chain = GPUCompose(
        [
            GPURandomResizedCrop(size=224, scale=[0.3, 1.0]),
            GPURandomHorizontalFlip(p=0.5),
            GPUColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, p=0.8),
            GPURandomGrayscale(p=0.2),
            GPUGaussianBlur(kernel_size=23, sigma=[0.1, 2.0], p=0.5),
            GPURandomSolarize(thresholds=0.5, additions=0.0, p=0.2),
            GPUNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ],
        compile=False,
    )
    local_chain = GPUCompose(
        [
            GPURandomResizedCrop(size=local_crop_size, scale=[0.05, 0.3]),
            GPURandomHorizontalFlip(p=0.5),
            GPUColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, p=0.8),
            GPURandomGrayscale(p=0.2),
            GPUGaussianBlur(kernel_size=23, sigma=[0.1, 2.0], p=0.5),
            GPURandomSolarize(thresholds=0.5, additions=0.0, p=0.2),
            GPUNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ],
        compile=False,
    )
    return GroupedMultiView(
        groups={
            "global": {"chain": global_chain, "names": ["global_1", "global_2"]},
            "local": {
                "chain": local_chain,
                "names": ["local_1", "local_2", "local_3", "local_4", "local_5", "local_6"],
            },
        }
    )


def build_dataset(args: argparse.Namespace) -> Dataset:
    transform = cpu_transform()
    split = "validation" if args.dataset_split == "val" else args.dataset_split
    if args.dataset_root is not None:
        root = args.dataset_root
        candidate = root / split
        if candidate.exists():
            root = candidate
        return ImageFolderDict(root, transform)

    kwargs: dict[str, Any] = {"split": split, "transform": transform}
    if args.dataset_revision is not None and args.dataset_revision != "null":
        kwargs["revision"] = args.dataset_revision
    dataset_cache_dir = infer_dataset_cache_dir(args)
    if dataset_cache_dir is not None:
        kwargs["cache_dir"] = str(dataset_cache_dir)
        print(f"[data] Using dataset cache: {dataset_cache_dir}", flush=True)
    return HFDataset(args.dataset_name, **kwargs)


def deterministic_subset(dataset: Dataset, num_images: int, seed: int) -> Subset:
    n = len(dataset)
    if num_images > n:
        raise ValueError(f"Requested {num_images} images, but dataset only has {n}.")
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(n, generator=generator)[:num_images].tolist()
    return Subset(dataset, indices)


def build_model(args: argparse.Namespace) -> LeJEPA:
    return build_model_for_spec(
        args,
        encoder_name=args.encoder_name or "vit_small_patch16_224",
        drop_path_rate=args.drop_path_rate
        if args.drop_path_rate is not None
        else float(os.environ.get("DROP_PATH_RATE", "0.1")),
        model_sigreg=args.model_sigreg or "ep",
    )


def build_model_for_spec(
    args: argparse.Namespace,
    encoder_name: str,
    drop_path_rate: float,
    model_sigreg: str,
) -> LeJEPA:
    model = LeJEPA(
        encoder_name=encoder_name,
        pretrained=args.pretrained_backbone,
        drop_path_rate=drop_path_rate,
        sigreg=model_sigreg,
        override_sr_gamma=args.gamma,
        n_slices=max(args.slice_counts),
        t_max=args.t_max,
        n_points=args.n_points,
    )
    return model


def load_checkpoint(model: LeJEPA, path: Path) -> None:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint.get("state_dict", checkpoint)
    if not isinstance(state, dict):
        raise ValueError(f"Checkpoint {path} does not contain a state dict.")

    candidates = [state]
    stripped_model = {
        key.removeprefix("model."): value
        for key, value in state.items()
        if key.startswith("model.")
    }
    if stripped_model:
        candidates.insert(0, stripped_model)

    last_error: Exception | None = None
    for candidate in candidates:
        try:
            missing, unexpected = model.load_state_dict(candidate, strict=False)
            bad_missing = [
                key
                for key in missing
                if key.startswith("backbone.") or key.startswith("projector.")
            ]
            if bad_missing:
                raise RuntimeError(f"Missing model keys: {bad_missing[:8]}")
            if unexpected:
                print(
                    f"[warn] Ignored {len(unexpected)} unexpected keys from {path.name}.",
                    file=sys.stderr,
                )
            return
        except Exception as exc:  # noqa: BLE001
            last_error = exc
    raise RuntimeError(f"Could not load {path}: {last_error}")


def set_batchnorm_no_running_updates(module: torch.nn.Module) -> None:
    for submodule in module.modules():
        if isinstance(submodule, torch.nn.modules.batchnorm._BatchNorm):
            submodule.momentum = 0.0


def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    out = {}
    for key, value in batch.items():
        out[key] = value.to(device, non_blocking=True) if torch.is_tensor(value) else value
    return out


@torch.no_grad()
def extract_projected_embeddings(
    args: argparse.Namespace,
    spec: CheckpointSpec,
    dataset: Dataset,
    cache_path: Path,
) -> tuple[torch.Tensor, dict[str, Any]]:
    if args.resume_cache and cache_path.exists():
        payload = torch.load(cache_path, map_location="cpu", weights_only=False)
        return payload["embeddings"], payload["metadata"]

    device = torch.device(args.device)
    encoder_name = effective_encoder_name(args, spec)
    drop_path_rate = effective_drop_path_rate(args, spec)
    model_sigreg = effective_model_sigreg(args, spec)
    print(
        f"[model] {spec.label}: encoder={encoder_name} "
        f"drop_path_rate={drop_path_rate} model_sigreg={model_sigreg}",
        flush=True,
    )
    model = build_model_for_spec(args, encoder_name, drop_path_rate, model_sigreg)
    if spec.path is not None:
        load_checkpoint(model, spec.path)
    model.to(device)
    if args.model_mode == "train":
        model.train()
        set_batchnorm_no_running_updates(model)
    else:
        model.eval()

    gpu_transform = gpu_multiview_transform(args.local_crop_size).to(device)
    loader = DataLoader(
        dataset,
        batch_size=args.image_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=args.num_workers > 0,
    )

    chunks: list[torch.Tensor] = []
    cache_batches: list[dict[str, Any]] = []
    num_images = 0
    num_views = None
    for batch_idx, batch in enumerate(loader):
        if args.max_batches is not None and batch_idx >= args.max_batches:
            break
        batch = move_batch_to_device(batch, device)
        torch.manual_seed(args.seed + batch_idx)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(args.seed + batch_idx)
        views = gpu_transform(batch)
        global_images = [views["global_1"]["image"], views["global_2"]["image"]]
        local_images = [
            views["local_1"]["image"],
            views["local_2"]["image"],
            views["local_3"]["image"],
            views["local_4"]["image"],
            views["local_5"]["image"],
            views["local_6"]["image"],
        ]
        g_features = model.backbone(torch.cat(global_images))
        l_features = model.backbone(torch.cat(local_images))
        projected = model.projector(torch.cat([g_features, l_features])).float()

        batch_size = global_images[0].shape[0]
        n_views = len(global_images) + len(local_images)
        per_image = projected.view(n_views, batch_size, -1).permute(1, 0, 2).cpu()
        chunks.append(per_image)
        num_images += batch_size
        num_views = n_views
        cache_batches.append(
            {
                "batch_index": batch_idx,
                "num_images": int(batch_size),
                "num_views": int(n_views),
                "effective_N": int(batch_size * n_views),
                "feature_dim": int(per_image.shape[-1]),
                "dtype": str(per_image.dtype),
            }
        )
        print(
            f"[cache] {spec.label}: batch={batch_idx} images={num_images}",
            flush=True,
        )

    embeddings = torch.cat(chunks, dim=0).contiguous()
    metadata = {
        "checkpoint": spec.label,
        "checkpoint_path": str(spec.path) if spec.path is not None else "",
        "num_images": int(embeddings.shape[0]),
        "num_views": int(num_views or 0),
        "effective_regularizer_sample_count_full": int(embeddings.shape[0] * (num_views or 0)),
        "feature_dim": int(embeddings.shape[-1]),
        "dtype": str(embeddings.dtype),
        "encoder_name": encoder_name,
        "model_sigreg": model_sigreg,
        "drop_path_rate": drop_path_rate,
        "model_mode": args.model_mode,
        "local_crop_size": args.local_crop_size,
        "batches": cache_batches,
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"embeddings": embeddings, "metadata": metadata}, cache_path)
    cache_path.with_suffix(".json").write_text(json.dumps(metadata, indent=2))
    return embeddings, metadata


def cache_key(args: argparse.Namespace, spec: CheckpointSpec) -> str:
    encoder_name = effective_encoder_name(args, spec)
    model_sigreg = effective_model_sigreg(args, spec)
    source = {
        "label": spec.label,
        "path": str(spec.path) if spec.path is not None else "untrained",
        "encoder_name": encoder_name,
        "model_sigreg": model_sigreg,
        "num_images": args.num_images,
        "image_batch_size": args.image_batch_size,
        "dataset_root": str(args.dataset_root) if args.dataset_root is not None else "",
        "dataset_name": args.dataset_name,
        "dataset_split": args.dataset_split,
        "dataset_revision": args.dataset_revision,
        "dataset_cache_dir": str(infer_dataset_cache_dir(args) or ""),
        "seed": args.seed,
        "local_crop_size": args.local_crop_size,
        "model_mode": args.model_mode,
    }
    digest = hashlib.sha1(json.dumps(source, sort_keys=True).encode()).hexdigest()[:10]
    return f"{slugify(spec.label)}-{digest}.pt"


def flattened_regularizer_batch(embeddings: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    batch = embeddings.index_select(0, indices)
    return batch.permute(1, 0, 2).reshape(-1, batch.shape[-1]).contiguous()


def synchronize_if_cuda(tensor: torch.Tensor) -> None:
    if tensor.is_cuda:
        torch.cuda.synchronize(tensor.device)


def regularizer_value_and_grad(
    module: torch.nn.Module,
    z: torch.Tensor,
) -> tuple[float, torch.Tensor, float]:
    x = z.detach().clone().float().requires_grad_(True)
    synchronize_if_cuda(x)
    start = time.perf_counter()
    loss = module(x)
    synchronize_if_cuda(x)
    elapsed_ms = 1000.0 * (time.perf_counter() - start)
    grad = torch.autograd.grad(loss, x)[0].detach()
    return float(loss.detach().cpu()), grad.cpu(), elapsed_ms


def cw_forward_flops_approx(effective_n: int, feature_dim: int) -> float:
    pairwise_terms = 2.0 * effective_n * effective_n * feature_dim
    marginal_terms = 2.0 * effective_n * feature_dim
    return pairwise_terms + marginal_terms


def ep_forward_flops_approx(
    effective_n: int,
    feature_dim: int,
    slice_count: int,
    n_points: int,
) -> float:
    projection_terms = 2.0 * effective_n * feature_dim * slice_count
    quadrature_terms = 6.0 * effective_n * slice_count * n_points
    target_terms = 6.0 * slice_count * n_points
    return projection_terms + quadrature_terms + target_terms


def pearson(x: Iterable[float], y: Iterable[float]) -> float:
    a = np.asarray(list(x), dtype=np.float64)
    b = np.asarray(list(y), dtype=np.float64)
    if a.size < 2 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def rankdata(values: Iterable[float]) -> np.ndarray:
    values = np.asarray(list(values), dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    i = 0
    while i < values.size:
        j = i + 1
        while j < values.size and values[order[j]] == values[order[i]]:
            j += 1
        ranks[order[i:j]] = (i + j - 1) / 2.0
        i = j
    return ranks


def spearman(x: Iterable[float], y: Iterable[float]) -> float:
    return pearson(rankdata(x), rankdata(y))


def bootstrap_ci(values: list[float], samples: int, seed: int) -> tuple[float, float]:
    clean = np.asarray([v for v in values if math.isfinite(v)], dtype=np.float64)
    if clean.size == 0:
        return float("nan"), float("nan")
    if clean.size == 1 or samples <= 0:
        return float(clean[0]), float(clean[0])
    rng = np.random.default_rng(seed)
    means = np.empty(samples, dtype=np.float64)
    for i in range(samples):
        draw = rng.choice(clean, size=clean.size, replace=True)
        means[i] = draw.mean()
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def run_regularizer_analysis(
    args: argparse.Namespace,
    spec: CheckpointSpec,
    embeddings: torch.Tensor,
    metadata: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    corr_rows: list[dict[str, Any]] = []
    num_images, num_views, feature_dim = embeddings.shape
    base_indices = torch.arange(num_images)
    device = torch.device(args.device)
    cw = CWReg(gamma=args.gamma).to(device)
    training_epoch = checkpoint_epoch(spec)
    training_percent = checkpoint_training_percent(spec)
    display_label = checkpoint_display_label(spec.label, training_percent)

    for partition_id in range(args.num_partitions):
        generator = torch.Generator().manual_seed(args.seed + 1009 * partition_id)
        perm = base_indices[torch.randperm(num_images, generator=generator)]
        max_full = (num_images // args.image_batch_size) * args.image_batch_size
        if args.keep_last_batch:
            batches = list(perm.split(args.image_batch_size))
        else:
            batches = list(perm[:max_full].split(args.image_batch_size))
        if args.max_batches is not None:
            batches = batches[: args.max_batches]

        cw_losses_by_batch: dict[int, float] = {}
        cw_grads_by_batch: dict[int, torch.Tensor] = {}
        cw_times_by_batch: dict[int, float] = {}
        for batch_id, batch_indices in enumerate(batches):
            z_cpu = flattened_regularizer_batch(embeddings, batch_indices)
            z_device = z_cpu.to(device)
            cw_loss, cw_grad, cw_time_ms = regularizer_value_and_grad(cw, z_device)
            cw_losses_by_batch[batch_id] = cw_loss
            cw_grads_by_batch[batch_id] = cw_grad
            cw_times_by_batch[batch_id] = cw_time_ms

        for slice_count in args.slice_counts:
            ep = SlicedEppsPulley(
                num_slices=slice_count,
                t_max=args.t_max,
                n_points=args.n_points,
                gamma=args.gamma,
            ).to(device)
            ep_losses_by_seed: dict[int, list[float]] = {
                seed: [] for seed in range(args.num_projection_seeds)
            }
            for projection_seed in range(args.num_projection_seeds):
                for batch_id, batch_indices in enumerate(batches):
                    z_cpu = flattened_regularizer_batch(embeddings, batch_indices)
                    z_device = z_cpu.to(device)
                    ep.global_step.fill_(projection_seed)
                    ep_loss, ep_grad, ep_time_ms = regularizer_value_and_grad(ep, z_device)
                    cw_loss = cw_losses_by_batch[batch_id]
                    cw_grad = cw_grads_by_batch[batch_id]
                    effective_n = int(batch_indices.numel() * num_views)
                    ep_losses_by_seed[projection_seed].append(ep_loss)

                    grad_cosine = F.cosine_similarity(
                        ep_grad.flatten(), cw_grad.flatten(), dim=0
                    ).item()
                    cw_grad_norm = cw_grad.norm().item()
                    ep_grad_norm = ep_grad.norm().item()
                    rows.append(
                        {
                            "checkpoint": spec.label,
                            "checkpoint_path": str(spec.path) if spec.path else "",
                            "display_label": display_label,
                            "training_epoch": training_epoch if training_epoch is not None else "",
                            "training_percent": (
                                training_percent if training_percent is not None else ""
                            ),
                            "partition_id": partition_id,
                            "batch_id": batch_id,
                            "slice_count": slice_count,
                            "projection_seed": projection_seed,
                            "num_images": int(batch_indices.numel()),
                            "num_views": num_views,
                            "effective_N": effective_n,
                            "feature_dim": feature_dim,
                            "cw_loss": cw_loss,
                            "ep_loss": ep_loss,
                            "cw_loss_time_ms": cw_times_by_batch[batch_id],
                            "ep_loss_time_ms": ep_time_ms,
                            "cw_loss_flops_approx": cw_forward_flops_approx(
                                effective_n, feature_dim
                            ),
                            "ep_loss_flops_approx": ep_forward_flops_approx(
                                effective_n,
                                feature_dim,
                                slice_count,
                                args.n_points,
                            ),
                            "relative_loss_error": abs(ep_loss - cw_loss) / (abs(cw_loss) + EPS),
                            "gradient_cosine": grad_cosine,
                            "gradient_norm_ratio": ep_grad_norm / (cw_grad_norm + EPS),
                            "relative_gradient_error": (ep_grad - cw_grad).norm().item()
                            / (cw_grad_norm + EPS),
                        }
                    )

            for projection_seed, ep_losses in ep_losses_by_seed.items():
                ordered_cw = [cw_losses_by_batch[i] for i in range(len(ep_losses))]
                corr_rows.append(
                    {
                        "checkpoint": spec.label,
                        "display_label": display_label,
                        "training_epoch": training_epoch if training_epoch is not None else "",
                        "training_percent": (
                            training_percent if training_percent is not None else ""
                        ),
                        "partition_id": partition_id,
                        "slice_count": slice_count,
                        "projection_seed": projection_seed,
                        "spearman": spearman(ep_losses, ordered_cw),
                        "pearson": pearson(ep_losses, ordered_cw),
                    }
                )

    print(
        f"[analysis] {spec.label}: rows={len(rows)} "
        f"feature_dim={metadata['feature_dim']} views={metadata['num_views']}",
        flush=True,
    )
    return rows, corr_rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def try_write_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    try:
        import pandas as pd

        pd.DataFrame(rows).to_parquet(path, index=False)
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] Could not write parquet {path}: {exc}", file=sys.stderr)


def aggregate_rows(
    rows: list[dict[str, Any]],
    corr_rows: list[dict[str, Any]],
    bootstrap_samples: int,
    seed: int,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((row["checkpoint"], row["slice_count"]), []).append(row)

    corr_grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in corr_rows:
        corr_grouped.setdefault((row["checkpoint"], row["slice_count"]), []).append(row)

    out: list[dict[str, Any]] = []
    for key in sorted(grouped, key=lambda x: (x[0], x[1])):
        checkpoint, slice_count = key
        values = grouped[key]
        corr_values = corr_grouped.get(key, [])
        ep_by_batch: dict[tuple[int, int], list[float]] = {}
        cos_by_batch: dict[tuple[int, int], list[float]] = {}
        for row in values:
            batch_key = (row["partition_id"], row["batch_id"])
            ep_by_batch.setdefault(batch_key, []).append(row["ep_loss"])
            cos_by_batch.setdefault(batch_key, []).append(row["gradient_cosine"])

        directional_variability = [
            float(np.std(v, ddof=1) / (abs(np.mean(v)) + EPS))
            for v in ep_by_batch.values()
            if len(v) > 1
        ]
        grad_cosine_seed_std = [
            float(np.std(v, ddof=1)) for v in cos_by_batch.values() if len(v) > 1
        ]

        first = values[0]
        result: dict[str, Any] = {
            "checkpoint": checkpoint,
            "checkpoint_path": first.get("checkpoint_path", ""),
            "display_label": first.get("display_label", checkpoint),
            "training_epoch": first.get("training_epoch", ""),
            "training_percent": first.get("training_percent", ""),
            "slice_count": slice_count,
        }
        metric_names = [
            "gradient_cosine",
            "relative_loss_error",
            "gradient_norm_ratio",
            "relative_gradient_error",
            "cw_loss_time_ms",
            "ep_loss_time_ms",
            "cw_loss_flops_approx",
            "ep_loss_flops_approx",
        ]
        for metric in metric_names:
            vals = [float(row[metric]) for row in values]
            result[f"{metric}_mean"] = float(np.mean(vals))
            result[f"{metric}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            lo, hi = bootstrap_ci(vals, bootstrap_samples, seed + slice_count)
            result[f"{metric}_ci95_low"] = lo
            result[f"{metric}_ci95_high"] = hi

        for metric in ["spearman", "pearson"]:
            vals = [float(row[metric]) for row in corr_values if math.isfinite(float(row[metric]))]
            result[f"{metric}_mean"] = float(np.mean(vals)) if vals else float("nan")
            result[f"{metric}_std"] = (
                float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            )
            lo, hi = bootstrap_ci(vals, bootstrap_samples, seed + slice_count + 17)
            result[f"{metric}_ci95_low"] = lo
            result[f"{metric}_ci95_high"] = hi

        result["directional_loss_variability_mean"] = (
            float(np.mean(directional_variability)) if directional_variability else float("nan")
        )
        result["directional_loss_variability_std"] = (
            float(np.std(directional_variability, ddof=1))
            if len(directional_variability) > 1
            else 0.0
        )
        result["gradient_cosine_seed_std_mean"] = (
            float(np.mean(grad_cosine_seed_std)) if grad_cosine_seed_std else float("nan")
        )
        out.append(result)
    return out


def plot_main(aggregate: list[dict[str, Any]], output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    set_plot_theme()
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.6), sharex=True)
    panels = [
        (
            "directional_loss_variability",
            "Relative EP loss variability",
            0.0,
            "ep_cw_convergence_loss_variability",
            "Directional variability of the finite-slice Epps-Pulley loss as a function of the number of projection slices. Curves show checkpoints ordered by training progress; shaded regions denote empirical variability across projection seeds and repartitioned batches.",
        ),
        (
            "relative_loss_error",
            "Relative loss discrepancy",
            0.0,
            "ep_cw_convergence_loss_discrepancy",
            "Relative discrepancy between the finite-slice Epps-Pulley regularizer and the closed-form Cramer-Wold regularizer evaluated on identical detached representation batches. Lower values indicate closer agreement in loss scale.",
        ),
        (
            "spearman",
            "Spearman batch-loss correlation",
            1.0,
            "ep_cw_convergence_spearman_loss_correlation",
            "Spearman correlation between finite-slice Epps-Pulley losses and closed-form Cramer-Wold losses across representation batches. Higher values indicate that EP preserves the CW ranking of batches.",
        ),
    ]
    for ax, (metric, ylabel, ref, filename, caption) in zip(axes, panels):
        plot_metric_panel(
            ax,
            aggregate,
            metric,
            ylabel,
            ref,
            show_legend=(ax is axes[-1]),
        )
        save_single_metric_figure(
            aggregate,
            output_dir,
            metric,
            ylabel,
            ref,
            filename=filename,
            caption=caption,
        )
    fig.tight_layout()
    fig.savefig(output_dir / "ep_cw_convergence_main.pdf")
    write_caption(
        output_dir,
        "ep_cw_convergence_main",
        "Loss-level convergence diagnostics for finite-slice Epps-Pulley regularization against the closed-form Cramer-Wold regularizer. Panels show directional loss variability, relative loss discrepancy, and Spearman batch-loss correlation as the number of projection slices increases.",
    )
    plt.close(fig)


def plot_appendix(aggregate: list[dict[str, Any]], output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    set_plot_theme()
    panels = [
        (
            "gradient_cosine",
            "Gradient cosine similarity",
            1.0,
            "ep_cw_convergence_gradient_cosine",
            "Cosine similarity between gradients of finite-slice Epps-Pulley and closed-form Cramer-Wold losses with respect to the same detached regularizer input tensor. Values approaching one indicate directional agreement.",
            False,
        ),
        (
            "relative_gradient_error",
            "Relative gradient error",
            0.0,
            "ep_cw_convergence_relative_gradient_error",
            "Relative norm of the difference between finite-slice Epps-Pulley and closed-form Cramer-Wold gradients. Lower values indicate closer agreement of the two gradient fields.",
            False,
        ),
        (
            "gradient_norm_ratio",
            "Gradient norm ratio",
            1.0,
            "ep_cw_convergence_gradient_norm_ratio",
            "Ratio of finite-slice Epps-Pulley gradient norm to closed-form Cramer-Wold gradient norm. Values approaching one indicate agreement in gradient scale.",
            False,
        ),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.6), sharex=True)
    for ax, (metric, ylabel, ref, filename, caption, ylog) in zip(axes, panels):
        plot_metric_panel(
            ax,
            aggregate,
            metric,
            ylabel,
            ref,
            ylog=ylog,
            show_legend=(ax is axes[-1]),
        )
        save_single_metric_figure(
            aggregate,
            output_dir,
            metric,
            ylabel,
            ref,
            filename=filename,
            caption=caption,
            ylog=ylog,
        )
    fig.tight_layout()
    fig.savefig(output_dir / "ep_cw_convergence_appendix.pdf")
    write_caption(
        output_dir,
        "ep_cw_convergence_appendix",
        "Gradient-level convergence diagnostics for finite-slice Epps-Pulley regularization against the closed-form Cramer-Wold regularizer. Panels show gradient cosine similarity, relative gradient error, and gradient norm ratio as the number of projection slices increases.",
    )
    plt.close(fig)


def plot_costs(aggregate: list[dict[str, Any]], output_dir: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    if not aggregate or "ep_loss_time_ms_mean" not in aggregate[0]:
        return

    set_plot_theme()
    cost_color = "#1f77b4"
    reference = checkpoint_plot_specs(aggregate)[0]
    reference_checkpoint = reference["checkpoint"]
    reference_label = reference.get("display_label", reference_checkpoint)
    reference_rows = [
        row for row in aggregate if row["checkpoint"] == reference_checkpoint
    ]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6), sharex=True)
    panels = [
        (
            "loss_time_ms",
            "Average forward loss time (ms)",
            "ep_cw_convergence_cost_time",
            "Measured forward loss evaluation time for finite-slice Epps-Pulley and closed-form Cramer-Wold regularizers on a single reference checkpoint. EP cost grows with the number of projection slices, while CW is independent of slice count.",
        ),
        (
            "loss_flops_approx",
            "Approx. forward loss FLOPs",
            "ep_cw_convergence_cost_flops",
            "Approximate analytic forward FLOP counts for finite-slice Epps-Pulley and closed-form Cramer-Wold regularizers. Counts are estimated from tensor shapes and dominant operations rather than profiler hardware counters.",
        ),
    ]
    slice_ticks = sorted({row["slice_count"] for row in reference_rows})
    def draw_cost_axis(ax: Any, suffix: str, ylabel: str) -> None:
        x = np.asarray(slice_ticks, dtype=float)
        ep_mean = []
        ep_std = []
        cw_mean = []
        cw_std = []
        for slice_count in slice_ticks:
            ep_vals = [
                float(row[f"ep_{suffix}_mean"])
                for row in reference_rows
                if row["slice_count"] == slice_count
                and math.isfinite(float(row[f"ep_{suffix}_mean"]))
            ]
            cw_vals = [
                float(row[f"cw_{suffix}_mean"])
                for row in reference_rows
                if row["slice_count"] == slice_count
                and math.isfinite(float(row[f"cw_{suffix}_mean"]))
            ]
            ep_mean.append(float(np.mean(ep_vals)) if ep_vals else float("nan"))
            ep_std.append(float(np.std(ep_vals, ddof=1)) if len(ep_vals) > 1 else 0.0)
            cw_mean.append(float(np.mean(cw_vals)) if cw_vals else float("nan"))
            cw_std.append(float(np.std(cw_vals, ddof=1)) if len(cw_vals) > 1 else 0.0)

        ep_mean_arr = np.asarray(ep_mean, dtype=float)
        ep_std_arr = np.asarray(ep_std, dtype=float)
        cw_mean_arr = np.asarray(cw_mean, dtype=float)
        cw_std_arr = np.asarray(cw_std, dtype=float)
        ax.plot(x, ep_mean_arr, marker="o", color=cost_color, linewidth=1.8, label="EP")
        ax.fill_between(
            x,
            ep_mean_arr - ep_std_arr,
            ep_mean_arr + ep_std_arr,
            color=cost_color,
            alpha=0.10,
        )
        ax.plot(x, cw_mean_arr, color=cost_color, linestyle="--", linewidth=1.5, label="CW")
        ax.fill_between(
            x,
            cw_mean_arr - cw_std_arr,
            cw_mean_arr + cw_std_arr,
            color=cost_color,
            alpha=0.06,
        )
        ax.set_xscale("log", base=2)
        ax.set_xticks(slice_ticks)
        ax.set_xticklabels([format_slice_tick(tick) for tick in slice_ticks])
        ax.set_yscale("log")
        ax.set_xlabel("EP slices")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)

    for ax, (suffix, ylabel, filename, caption) in zip(axes, panels):
        draw_cost_axis(ax, suffix, ylabel)
        single_fig, single_ax = plt.subplots(1, 1, figsize=(5.0, 3.6))
        draw_cost_axis(single_ax, suffix, ylabel)
        single_ax.legend(
            handles=[
                Line2D([0], [0], color=cost_color, linewidth=1.8, linestyle="-", label="EP"),
                Line2D([0], [0], color=cost_color, linewidth=1.5, linestyle="--", label="CW"),
            ],
            frameon=False,
            fontsize=8,
        )
        single_fig.tight_layout()
        single_fig.savefig(output_dir / f"{filename}.pdf")
        write_caption(output_dir, filename, caption)
        plt.close(single_fig)

    method_handles = [
        Line2D([0], [0], color=cost_color, linewidth=1.8, linestyle="-", label="EP"),
        Line2D([0], [0], color=cost_color, linewidth=1.5, linestyle="--", label="CW"),
    ]
    axes[-1].legend(handles=method_handles, frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "ep_cw_convergence_costs.pdf")
    write_caption(
        output_dir,
        "ep_cw_convergence_costs",
        "Computational cost comparison for finite-slice Epps-Pulley and closed-form Cramer-Wold regularizers on a single reference checkpoint. Panels report measured forward loss time and approximate analytic forward FLOPs as a function of the EP slice count.",
    )
    plt.close(fig)


def main() -> None:
    args = parse_args()
    apply_debug_defaults(args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.cache_dir or (args.output_dir / "embedding_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)

    specs = checkpoint_specs(args)
    if not specs:
        raise ValueError("No checkpoints selected and --no-include-untrained was used.")
    print("[checkpoints] Selected:", flush=True)
    for spec in specs:
        suffix = f" ({spec.path})" if spec.path is not None else ""
        print(f"  - {spec.label}{suffix}", flush=True)

    dataset = deterministic_subset(build_dataset(args), args.num_images, args.seed)
    all_rows: list[dict[str, Any]] = []
    all_corr_rows: list[dict[str, Any]] = []
    cache_manifest: list[dict[str, Any]] = []

    for spec in specs:
        cache_path = cache_dir / cache_key(args, spec)
        try:
            embeddings, metadata = extract_projected_embeddings(args, spec, dataset, cache_path)
        except Exception as exc:  # noqa: BLE001
            if spec.path is None:
                raise
            print(f"[warn] Skipping incompatible checkpoint {spec.path}: {exc}", file=sys.stderr)
            continue
        cache_manifest.append({"cache_path": str(cache_path), **metadata})
        rows, corr_rows = run_regularizer_analysis(args, spec, embeddings, metadata)
        all_rows.extend(rows)
        all_corr_rows.extend(corr_rows)

    raw_csv = args.output_dir / "ep_cw_convergence_raw.csv"
    corr_csv = args.output_dir / "ep_cw_convergence_correlations.csv"
    aggregate_csv = args.output_dir / "ep_cw_convergence_aggregate.csv"
    write_csv(raw_csv, all_rows)
    write_csv(corr_csv, all_corr_rows)
    aggregate = aggregate_rows(all_rows, all_corr_rows, args.bootstrap_samples, args.seed)
    write_csv(aggregate_csv, aggregate)
    try_write_parquet(args.output_dir / "ep_cw_convergence_raw.parquet", all_rows)
    try_write_parquet(args.output_dir / "ep_cw_convergence_aggregate.parquet", aggregate)
    (args.output_dir / "embedding_cache_manifest.json").write_text(
        json.dumps(cache_manifest, indent=2)
    )
    (args.output_dir / "run_config.json").write_text(
        json.dumps(vars(args), indent=2, default=str)
    )
    if aggregate:
        plot_main(aggregate, args.output_dir)
        plot_appendix(aggregate, args.output_dir)
        plot_costs(aggregate, args.output_dir)

    print(f"Wrote raw observations: {raw_csv}")
    print(f"Wrote aggregate results: {aggregate_csv}")


if __name__ == "__main__":
    main()
