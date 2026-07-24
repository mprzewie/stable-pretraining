#!/usr/bin/env python3
"""Analyze grouped multi-view CW gradients for LeJEPA projector outputs.

This is an analysis-only companion to ``analyze_ep_cw_convergence.py``.  It
keeps the production LeJEPA conventions:

* projector outputs are produced in view-major order ``[V, B, D]``;
* the alignment loss is exactly ``LeJEPA._compute_loss``'s invariance term,
  ``(mean(global_views) - all_views).square().mean()``;
* CWReg is the production closed-form Cramer-Wold normality regularizer,
  ``2*pi*N*cw_torch.metric.cw_normality``.

The script preserves image/view grouping while decomposing the CW
sample-sample V-statistic into diagonal, within-instance, and between-instance
terms.  Outputs are CSV only; no parquet files are written.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import sys
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
from stable_pretraining.methods.lejepa import CWReg, LeJEPA  # noqa: E402
from cw_torch.metric import cw_normality  # noqa: E402

EPS = 1e-12
CHECKPOINT_SUFFIXES = {".ckpt", ".pt", ".pth"}
CW_NORMALITY_SCALE = 0.28209479177


@dataclass(frozen=True)
class CheckpointSpec:
    label: str
    path: Path | None
    hparams: dict[str, Any] | None = None


@dataclass(frozen=True)
class ViewSelection:
    indices: tuple[int, ...]
    num_global: int

    @property
    def num_views(self) -> int:
        return len(self.indices)

    @property
    def num_local(self) -> int:
        return self.num_views - self.num_global


class ImageFolderDict(Dataset):
    def __init__(self, root: Path, transform: Any):
        from torchvision.datasets import ImageFolder

        self.dataset = ImageFolder(str(root), transform=None)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        image, label = self.dataset[index]
        return self.transform({"image": image, "label": label, "image_id": index})


def parse_args() -> argparse.Namespace:
    command = "gradient"
    argv = sys.argv[1:]
    if argv and argv[0] in {"gradient", "joint-structure"}:
        command = argv[0]
        argv = argv[1:]
    parser = argparse.ArgumentParser(
        description="Analyze within-instance CW gradients in grouped LeJEPA multi-view batches."
    )
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--checkpoint", action="append", type=Path, default=[])
    parser.add_argument("--checkpoint-label", action="append", default=[])
    parser.add_argument("--max-checkpoints", type=int, default=5)
    parser.add_argument("--include-last-checkpoints", action="store_true")
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--dataset-name", default="clane9/imagenet-100")
    parser.add_argument("--dataset-split", choices=["train", "validation", "val"], default="validation")
    parser.add_argument("--dataset-revision", default=None)
    parser.add_argument("--dataset-cache-dir", type=Path, default=None)
    parser.add_argument("--num-images", type=int, default=5000)
    parser.add_argument("--image-batch-size", type=int, default=128)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[32, 64, 128])
    parser.add_argument("--view-counts", type=int, nargs="+", default=[2, 4, 6, 10])
    parser.add_argument("--image-counts", type=int, nargs="+", default=[500, 1000, 2500, 5000])
    parser.add_argument("--stats-chunk-size", type=int, default=128)
    parser.add_argument("--analysis-projection-dim", type=int, default=256)
    parser.add_argument("--num-partitions", type=int, default=5)
    parser.add_argument("--num-augmentation-draws", type=int, default=3)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--covariance-eps", type=float, default=1e-4)
    parser.add_argument("--covariance-estimator", choices=["oas", "ridge"], default="oas")
    parser.add_argument("--calibration-fraction", type=float, default=0.5)
    parser.add_argument("--num-null-repetitions", type=int, default=20)
    parser.add_argument("--null-full-space", action="store_true")
    parser.add_argument("--include-full-space", action="store_true")
    parser.add_argument("--include-same-view-full-space", action="store_true")
    parser.add_argument("--include-all-pairs", action="store_true")
    parser.add_argument("--include-all-pairs-full-space", action="store_true")
    parser.add_argument("--null-scope", choices=["representative", "all"], default="representative")
    parser.add_argument("--null-seed", type=int, default=0)
    parser.add_argument("--gaussianity-max-samples", type=int, default=1024)
    parser.add_argument("--projection-diagnostics", type=int, default=64)
    parser.add_argument("--fixed-rhos", type=float, nargs="+", default=[0.0, 0.5, 0.8, 0.9, 0.95, 0.99])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--include-untrained", dest="include_untrained", action="store_true", default=True)
    parser.add_argument("--no-include-untrained", dest="include_untrained", action="store_false")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--encoder-name", default=os.environ.get("ENCODER_NAME"))
    parser.add_argument("--drop-path-rate", type=float, default=None)
    parser.add_argument("--pretrained-backbone", action="store_true")
    parser.add_argument("--model-sigreg", choices=["ep", "cw"], default=os.environ.get("SIGREG", "ep"))
    parser.add_argument("--local-crop-size", type=int, default=int(os.environ.get("LOCAL_CROP_SIZE", "96")))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--resume-cache", dest="resume_cache", action="store_true", default=True)
    parser.add_argument("--no-resume-cache", dest="resume_cache", action="store_false")
    parser.add_argument("--model-mode", choices=["train", "eval"], default="train")
    parser.add_argument("--keep-last-batch", action="store_true")
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--finite-step", type=float, default=0.0)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args(argv)
    args.command = command
    return args


def apply_debug_defaults(args: argparse.Namespace) -> None:
    if not args.debug:
        return
    args.num_images = min(args.num_images, 96)
    args.image_batch_size = min(args.image_batch_size, 32)
    args.batch_sizes = [b for b in args.batch_sizes if b <= 32] or [16, 32]
    args.image_counts = [n for n in args.image_counts if n <= args.num_images] or [args.num_images]
    args.view_counts = [v for v in args.view_counts if v <= 4] or [2, 4]
    args.num_partitions = min(args.num_partitions, 2)
    args.num_augmentation_draws = min(args.num_augmentation_draws, 1)
    args.num_workers = min(args.num_workers, 2)
    args.max_batches = min(args.max_batches or 2, 2)
    args.projection_diagnostics = min(args.projection_diagnostics, 16)
    args.analysis_projection_dim = min(args.analysis_projection_dim, 16)
    args.num_null_repetitions = min(args.num_null_repetitions, 5)


def slugify(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return value.strip("._-") or "checkpoint"


def checkpoint_sort_key(path: Path) -> tuple[int, int | float, str]:
    match = re.search(r"epoch[=-](\d+)", path.name)
    if match:
        return (0, int(match.group(1)), path.name)
    if path.name.startswith("last"):
        return (2, math.inf, path.name)
    return (1, math.inf, path.name)


def uniformly_subsample(paths: list[Path], max_count: int | None) -> list[Path]:
    if max_count is None or max_count >= len(paths):
        return paths
    if max_count <= 0:
        raise ValueError("--max-checkpoints must be positive.")
    indices = sorted({int(round(i)) for i in np.linspace(0, len(paths) - 1, max_count)})
    return [paths[i] for i in indices]


def read_checkpoint_hparams(path: Path) -> dict[str, Any]:
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] Could not inspect checkpoint hparams for {path}: {exc}", file=sys.stderr)
        return {}
    hparams = checkpoint.get("hyper_parameters", {}) if isinstance(checkpoint, dict) else {}
    return hparams if isinstance(hparams, dict) else {}


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
        paths.extend(p.resolve() for p in uniformly_subsample(discovered, args.max_checkpoints))

    labels = list(args.checkpoint_label)
    if labels and len(labels) != len(paths):
        raise ValueError(f"Got {len(labels)} labels for {len(paths)} checkpoints.")

    specs: list[CheckpointSpec] = []
    if args.include_untrained:
        specs.append(CheckpointSpec("untrained", None, None))
    for i, path in enumerate(paths):
        label = labels[i] if labels else path.stem
        specs.append(CheckpointSpec(slugify(label), path, read_checkpoint_hparams(path)))
    return specs


def checkpoint_epoch(spec: CheckpointSpec) -> int | None:
    if spec.path is None:
        return None
    match = re.search(r"epoch[=-](\d+)", spec.path.name)
    return int(match.group(1)) if match else None


def training_progress(spec: CheckpointSpec, all_specs: list[CheckpointSpec]) -> float:
    if spec.path is None:
        return 0.0
    if spec.hparams and spec.hparams.get("trainer.max_epochs") is not None:
        try:
            return min(100.0, 100.0 * float((checkpoint_epoch(spec) or 0) + 1) / float(spec.hparams["trainer.max_epochs"]))
        except (TypeError, ValueError, ZeroDivisionError):
            pass
    epochs = [checkpoint_epoch(s) for s in all_specs if s.path is not None]
    epochs = [e for e in epochs if e is not None]
    epoch = checkpoint_epoch(spec)
    if epoch is None or not epochs:
        return float("nan")
    return 100.0 * (epoch - min(epochs)) / max(1, max(epochs) - min(epochs))


def effective_encoder_name(args: argparse.Namespace, spec: CheckpointSpec) -> str:
    if args.encoder_name:
        return args.encoder_name
    if spec.hparams and spec.hparams.get("module.model.encoder_name"):
        return str(spec.hparams["module.model.encoder_name"])
    return "vit_tiny_patch16_224"


def effective_drop_path_rate(args: argparse.Namespace, spec: CheckpointSpec) -> float:
    if args.drop_path_rate is not None:
        return args.drop_path_rate
    if spec.hparams and spec.hparams.get("module.model.drop_path_rate") is not None:
        return float(spec.hparams["module.model.drop_path_rate"])
    return float(os.environ.get("DROP_PATH_RATE", "0.1"))


def cpu_transform() -> Any:
    return spt_t.Compose(spt_t.RGB(), spt_t.Resize(size=[256, 256]), spt_t.ToImage())


def default_stable_pretraining_data_dir() -> Path:
    storage_root = Path(os.environ.get("STORAGE_ROOT", Path.home() / "storage"))
    return Path(os.environ.get("STABLE_PRETRAINING_DATA_DIR", storage_root / "datasets" / "stable-pretraining"))


def infer_dataset_cache_dir(args: argparse.Namespace) -> Path | None:
    if args.dataset_cache_dir is not None:
        return args.dataset_cache_dir.expanduser()
    if args.dataset_root is not None:
        return None
    known_names = {"clane9/imagenet-100": "imagenet100", "frgfm/imagenette": "imagenet10"}
    cache_name = known_names.get(args.dataset_name)
    if cache_name is None:
        return None
    candidate = default_stable_pretraining_data_dir() / cache_name
    return candidate if candidate.exists() else None


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
    cache_dir = infer_dataset_cache_dir(args)
    if cache_dir is not None:
        kwargs["cache_dir"] = str(cache_dir)
        print(f"[data] Using dataset cache: {cache_dir}", flush=True)
    return HFDataset(args.dataset_name, **kwargs)


def deterministic_subset(dataset: Dataset, num_images: int, seed: int) -> Subset:
    n = len(dataset)
    if num_images > n:
        raise ValueError(f"Requested {num_images} images, but dataset only has {n}.")
    indices = torch.randperm(n, generator=torch.Generator().manual_seed(seed))[:num_images].tolist()
    return Subset(dataset, indices)


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
            "local": {"chain": local_chain, "names": [f"local_{i}" for i in range(1, 9)]},
        }
    )


def build_model(args: argparse.Namespace, spec: CheckpointSpec) -> LeJEPA:
    return LeJEPA(
        encoder_name=effective_encoder_name(args, spec),
        pretrained=args.pretrained_backbone,
        drop_path_rate=effective_drop_path_rate(args, spec),
        sigreg=args.model_sigreg,
        override_sr_gamma=args.gamma,
    )


def load_checkpoint(model: LeJEPA, path: Path) -> None:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint.get("state_dict", checkpoint)
    candidates = [state]
    stripped = {k.removeprefix("model."): v for k, v in state.items() if k.startswith("model.")}
    if stripped:
        candidates.insert(0, stripped)
    last_error: Exception | None = None
    for candidate in candidates:
        try:
            missing, unexpected = model.load_state_dict(candidate, strict=False)
            bad_missing = [k for k in missing if k.startswith("backbone.") or k.startswith("projector.")]
            if bad_missing:
                raise RuntimeError(f"Missing model keys: {bad_missing[:8]}")
            if unexpected:
                print(f"[warn] Ignored {len(unexpected)} unexpected keys from {path.name}.", file=sys.stderr)
            return
        except Exception as exc:  # noqa: BLE001
            last_error = exc
    raise RuntimeError(f"Could not load {path}: {last_error}")


def set_batchnorm_no_running_updates(module: torch.nn.Module) -> None:
    for submodule in module.modules():
        if isinstance(submodule, torch.nn.modules.batchnorm._BatchNorm):
            submodule.momentum = 0.0


def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}


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
    model = build_model(args, spec)
    if spec.path is not None:
        load_checkpoint(model, spec.path)
    model.to(device)
    if args.model_mode == "train":
        model.train()
        set_batchnorm_no_running_updates(model)
    else:
        model.eval()

    transform = gpu_multiview_transform(args.local_crop_size).to(device)
    loader = DataLoader(
        dataset,
        batch_size=args.image_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=args.num_workers > 0,
    )

    chunks: list[torch.Tensor] = []
    batches: list[dict[str, Any]] = []
    num_images = 0
    for augmentation_id in range(args.num_augmentation_draws):
        aug_chunks: list[torch.Tensor] = []
        aug_images = 0
        for batch_idx, batch in enumerate(loader):
            if args.max_batches is not None and batch_idx >= args.max_batches:
                break
            batch = move_batch_to_device(batch, device)
            seed = args.seed + 100_003 * augmentation_id + batch_idx
            torch.manual_seed(seed)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(seed)
            views = transform(batch)
            global_images = [views["global_1"]["image"], views["global_2"]["image"]]
            local_images = [views[f"local_{i}"]["image"] for i in range(1, 9)]
            features = torch.cat(
                [
                    model.backbone(torch.cat(global_images)),
                    model.backbone(torch.cat(local_images)),
                ]
            )
            projected = model.projector(features).float()
            batch_size = global_images[0].shape[0]
            n_views = len(global_images) + len(local_images)
            per_image = projected.view(n_views, batch_size, -1).permute(1, 0, 2).cpu()
            aug_chunks.append(per_image)
            aug_images += batch_size
            batches.append(
                {
                    "augmentation_id": augmentation_id,
                    "batch_index": batch_idx,
                    "num_images": batch_size,
                    "num_views": n_views,
                    "feature_dim": per_image.shape[-1],
                    "augmentation_seed": seed,
                }
            )
            print(f"[cache] {spec.label}: aug={augmentation_id} batch={batch_idx} images={aug_images}", flush=True)
        chunks.append(torch.cat(aug_chunks, dim=0))
        num_images = max(num_images, aug_images)

    embeddings = torch.stack(chunks, dim=0).contiguous()
    metadata = {
        "checkpoint": spec.label,
        "checkpoint_path": str(spec.path) if spec.path is not None else "",
        "num_augmentation_draws": int(embeddings.shape[0]),
        "num_images": int(embeddings.shape[1]),
        "num_views": int(embeddings.shape[2]),
        "num_global_views": 2,
        "num_local_views": int(embeddings.shape[2] - 2),
        "feature_dim": int(embeddings.shape[3]),
        "dtype": str(embeddings.dtype),
        "view_order": ["global_1", "global_2"] + [f"local_{i}" for i in range(1, 9)],
        "flattening": "production LeJEPA view-major: z_bvd.permute(1,0,2).reshape(-1,D)",
        "batches": batches,
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"embeddings": embeddings, "metadata": metadata}, cache_path)
    cache_path.with_suffix(".json").write_text(json.dumps(metadata, indent=2))
    return embeddings, metadata


def cache_key(args: argparse.Namespace, spec: CheckpointSpec) -> str:
    source = {
        "label": spec.label,
        "path": str(spec.path) if spec.path is not None else "untrained",
        "encoder": effective_encoder_name(args, spec),
        "num_images": args.num_images,
        "image_batch_size": args.image_batch_size,
        "num_augmentation_draws": args.num_augmentation_draws,
        "dataset_root": str(args.dataset_root) if args.dataset_root else "",
        "dataset_name": args.dataset_name,
        "dataset_split": args.dataset_split,
        "seed": args.seed,
        "local_crop_size": args.local_crop_size,
        "model_mode": args.model_mode,
    }
    digest = hashlib.sha1(json.dumps(source, sort_keys=True).encode()).hexdigest()[:10]
    return f"{slugify(spec.label)}-{digest}.pt"


def view_selection(num_views: int, requested: int) -> ViewSelection:
    if requested > num_views:
        raise ValueError(f"Requested {requested} views, but cache has {num_views}.")
    if requested < 2:
        raise ValueError("At least two views are required.")
    indices = tuple(range(requested))
    num_global = min(2, requested)
    return ViewSelection(indices=indices, num_global=num_global)


def flatten_view_major(z_bvd: torch.Tensor) -> torch.Tensor:
    return z_bvd.permute(1, 0, 2).reshape(-1, z_bvd.shape[-1]).contiguous()


def production_alignment_loss(z_bvd: torch.Tensor, num_global: int) -> torch.Tensor:
    z_vbd = z_bvd.permute(1, 0, 2)
    centers = z_vbd[:num_global].mean(0)
    return (centers.unsqueeze(0) - z_vbd).square().mean()


def pair_masks(batch_size: int, num_views: int, num_global: int, device: torch.device | None = None) -> dict[str, torch.Tensor]:
    image_ids = torch.arange(batch_size, device=device).repeat(num_views)
    view_ids = torch.arange(num_views, device=device).repeat_interleave(batch_size)
    same_image = image_ids[:, None] == image_ids[None, :]
    same_view = view_ids[:, None] == view_ids[None, :]
    diagonal = same_image & same_view
    within = same_image & ~same_view
    between = ~same_image
    global_view = view_ids < num_global
    local_view = ~global_view
    gg = within & global_view[:, None] & global_view[None, :]
    gl = within & (
        (global_view[:, None] & local_view[None, :])
        | (local_view[:, None] & global_view[None, :])
    )
    ll = within & local_view[:, None] & local_view[None, :]
    return {
        "diagonal": diagonal,
        "within": within,
        "between": between,
        "within_gg": gg,
        "within_gl": gl,
        "within_ll": ll,
    }


def cw_kernel_matrix(z_flat: torch.Tensor, gamma: float | torch.Tensor) -> torch.Tensor:
    n, d = z_flat.shape
    gamma_t = torch.as_tensor(gamma, dtype=z_flat.dtype, device=z_flat.device)
    k_const = 1.0 / (2.0 * d - 3.0)
    dist2 = torch.cdist(z_flat, z_flat).square()
    return torch.rsqrt(gamma_t + k_const * dist2)


def cw_target_term(z_flat: torch.Tensor, gamma: float | torch.Tensor) -> torch.Tensor:
    n, d = z_flat.shape
    gamma_t = torch.as_tensor(gamma, dtype=z_flat.dtype, device=z_flat.device)
    k_const = 1.0 / (2.0 * d - 3.0)
    sample_prior = torch.rsqrt(gamma_t + 0.5 + k_const * z_flat.square().sum(dim=1)).mean()
    return torch.rsqrt(1.0 + gamma_t) - 2.0 * sample_prior


def cw_outer_scale(num_samples: int) -> float:
    return 2.0 * math.pi * num_samples * CW_NORMALITY_SCALE


def cw_component_losses(z_bvd: torch.Tensor, gamma: float, num_global: int) -> dict[str, torch.Tensor]:
    batch_size, num_views, _ = z_bvd.shape
    z_flat = flatten_view_major(z_bvd)
    n = z_flat.shape[0]
    kernel = cw_kernel_matrix(z_flat, gamma)
    masks = pair_masks(batch_size, num_views, num_global, z_bvd.device)
    scale = cw_outer_scale(n)
    losses = {
        name: scale * kernel[mask].sum() / (n * n)
        for name, mask in masks.items()
    }
    losses["pair_full"] = losses["diagonal"] + losses["within"] + losses["between"]
    losses["target"] = scale * cw_target_term(z_flat, gamma)
    losses["full"] = losses["pair_full"] + losses["target"]
    denom = batch_size * (batch_size - 1) * num_views * num_views
    if denom <= 0:
        losses["group_pair"] = torch.full((), float("nan"), device=z_bvd.device, dtype=z_bvd.dtype)
        losses["group_full"] = torch.full((), float("nan"), device=z_bvd.device, dtype=z_bvd.dtype)
    else:
        losses["group_pair"] = scale * kernel[masks["between"]].sum() / denom
        losses["group_full"] = losses["group_pair"] + losses["target"]
    losses["exclusion_only_full"] = losses["full"] - losses["within"]
    return losses


def grad(loss: torch.Tensor, z: torch.Tensor, retain_graph: bool = True) -> torch.Tensor:
    return torch.autograd.grad(loss, z, retain_graph=retain_graph, allow_unused=False)[0]


def norm(t: torch.Tensor) -> float:
    return float(t.detach().norm().cpu())


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(F.cosine_similarity(a.detach().flatten(), b.detach().flatten(), dim=0).cpu())


def signed_effect(component: torch.Tensor, align: torch.Tensor) -> float:
    return float((component.detach().flatten() @ align.detach().flatten()).cpu() / (align.detach().square().sum().cpu() + EPS))


def delta_align(component: torch.Tensor, align: torch.Tensor) -> float:
    return float((-(align.detach().flatten() @ component.detach().flatten())).cpu())


def geometric_diagnostics(z_bvd: torch.Tensor, gamma: float, num_global: int) -> dict[str, float]:
    batch_size, num_views, _ = z_bvd.shape
    z_flat = flatten_view_major(z_bvd)
    dist2 = torch.cdist(z_flat, z_flat).square()
    kernel = cw_kernel_matrix(z_flat, gamma)
    masks = pair_masks(batch_size, num_views, num_global, z_bvd.device)
    return {
        "mean_within_distance": float(dist2[masks["within"]].mean().detach().cpu()),
        "mean_between_distance": float(dist2[masks["between"]].mean().detach().cpu()),
        "mean_within_kernel": float(kernel[masks["within"]].mean().detach().cpu()),
        "mean_between_kernel": float(kernel[masks["between"]].mean().detach().cpu()),
        "within_between_kernel_ratio": float((kernel[masks["within"]].mean() / (kernel[masks["between"]].mean() + EPS)).detach().cpu()),
    }


def finite_step_check(z: torch.Tensor, align_grad: torch.Tensor, component_grad: torch.Tensor, num_global: int, step: float) -> tuple[float, float]:
    before = production_alignment_loss(z, num_global)
    after = production_alignment_loss((z - step * component_grad).detach(), num_global)
    measured = float((after - before).detach().cpu())
    predicted = float((-step * (align_grad.detach().flatten() @ component_grad.detach().flatten())).cpu())
    return measured, predicted


def analyze_batch(z_cpu: torch.Tensor, gamma: float, num_global: int, finite_step: float = 0.0) -> dict[str, float]:
    z = z_cpu.detach().clone().float().requires_grad_(True)
    losses = cw_component_losses(z, gamma, num_global)
    production_cw = CWReg(gamma=gamma)(flatten_view_major(z))
    alignment_loss = production_alignment_loss(z, num_global)

    g_full = grad(losses["full"], z)
    g_production = grad(production_cw, z)
    g_within = grad(losses["within"], z)
    g_between = grad(losses["between"], z)
    g_diagonal = grad(losses["diagonal"], z)
    g_target = grad(losses["target"], z)
    g_group = grad(losses["group_full"], z)
    g_exclusion = grad(losses["exclusion_only_full"], z)
    g_align = grad(alignment_loss, z)
    g_gg = grad(losses["within_gg"], z)
    g_gl = grad(losses["within_gl"], z)
    g_ll = grad(losses["within_ll"], z, retain_graph=False)

    recon = g_within + g_between + g_diagonal + g_target
    within_type_recon = g_gg + g_gl + g_ll
    full_norm = norm(g_full)
    between_norm = norm(g_between)
    align_norm = norm(g_align)
    group_norm = norm(g_group)
    geom = geometric_diagnostics(z.detach(), gamma, num_global)

    row = {
        "cw_loss_full": float(losses["full"].detach().cpu()),
        "cw_loss_production": float(production_cw.detach().cpu()),
        "cw_pair_within_loss": float(losses["within"].detach().cpu()),
        "cw_pair_between_loss": float(losses["between"].detach().cpu()),
        "cw_pair_diagonal_loss": float(losses["diagonal"].detach().cpu()),
        "cw_target_loss": float(losses["target"].detach().cpu()),
        "group_cw_loss": float(losses["group_full"].detach().cpu()),
        "exclusion_only_cw_loss": float(losses["exclusion_only_full"].detach().cpu()),
        "alignment_loss": float(alignment_loss.detach().cpu()),
        "norm_cw_full": full_norm,
        "norm_cw_production": norm(g_production),
        "norm_pair_within": norm(g_within),
        "norm_pair_between": between_norm,
        "norm_pair_diagonal": norm(g_diagonal),
        "norm_target": norm(g_target),
        "norm_group_cw": group_norm,
        "norm_exclusion_only_cw": norm(g_exclusion),
        "norm_alignment": align_norm,
        "norm_within_gg": norm(g_gg),
        "norm_within_gl": norm(g_gl),
        "norm_within_ll": norm(g_ll),
        "within_fraction_of_cw_gradient": norm(g_within) / (full_norm + EPS),
        "within_to_between_gradient_ratio": norm(g_within) / (between_norm + EPS),
        "within_to_alignment_gradient_ratio": norm(g_within) / (align_norm + EPS),
        "relative_full_group_gradient_difference": norm(g_full - g_group) / (full_norm + EPS),
        "cosine_within_alignment": cosine(g_within, g_align),
        "cosine_between_alignment": cosine(g_between, g_align),
        "cosine_target_alignment": cosine(g_target, g_align),
        "cosine_full_cw_alignment": cosine(g_full, g_align),
        "cosine_group_cw_alignment": cosine(g_group, g_align),
        "cosine_full_group_cw": cosine(g_full, g_group),
        "cosine_gg_alignment": cosine(g_gg, g_align),
        "cosine_gl_alignment": cosine(g_gl, g_align),
        "cosine_ll_alignment": cosine(g_ll, g_align),
        "signed_within_effect_on_alignment": signed_effect(g_within, g_align),
        "signed_between_effect_on_alignment": signed_effect(g_between, g_align),
        "signed_target_effect_on_alignment": signed_effect(g_target, g_align),
        "delta_align_within": delta_align(g_within, g_align),
        "delta_align_between": delta_align(g_between, g_align),
        "delta_align_target": delta_align(g_target, g_align),
        "gradient_reconstruction_error": norm(g_full - recon) / (full_norm + EPS),
        "production_cw_gradient_error": norm(g_full - g_production) / (norm(g_production) + EPS),
        "within_type_gradient_reconstruction_error": norm(g_within - within_type_recon) / (norm(g_within) + EPS),
        **geom,
    }
    if finite_step > 0:
        measured, predicted = finite_step_check(z.detach(), g_align, g_within, num_global, finite_step)
        row["finite_step_within_measured_delta_alignment"] = measured
        row["finite_step_within_predicted_delta_alignment"] = predicted
    return row


def deterministic_batches(num_images: int, batch_size: int, partition_id: int, seed: int, keep_last: bool) -> list[torch.Tensor]:
    generator = torch.Generator().manual_seed(seed + 1009 * partition_id)
    perm = torch.randperm(num_images, generator=generator)
    if keep_last:
        return list(perm.split(batch_size))
    usable = (num_images // batch_size) * batch_size
    return list(perm[:usable].split(batch_size))


def run_analysis_for_checkpoint(
    args: argparse.Namespace,
    spec: CheckpointSpec,
    all_specs: list[CheckpointSpec],
    embeddings: torch.Tensor,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    progress = training_progress(spec, all_specs)
    num_augs, num_images, cached_views, feature_dim = embeddings.shape
    for augmentation_id in range(num_augs):
        for partition_id in range(args.num_partitions):
            for batch_size in args.batch_sizes:
                batches = deterministic_batches(num_images, batch_size, partition_id, args.seed, args.keep_last_batch)
                if args.max_batches is not None:
                    batches = batches[: args.max_batches]
                for batch_id, indices in enumerate(batches):
                    for requested_views in args.view_counts:
                        selection = view_selection(cached_views, requested_views)
                        z = embeddings[augmentation_id].index_select(0, indices).index_select(1, torch.tensor(selection.indices))
                        metrics = analyze_batch(z, args.gamma, selection.num_global, args.finite_step)
                        rows.append(
                            {
                                "checkpoint": spec.label,
                                "checkpoint_path": str(spec.path) if spec.path is not None else "",
                                "training_progress": progress,
                                "partition_id": partition_id,
                                "augmentation_id": augmentation_id,
                                "batch_id": batch_id,
                                "batch_size": int(indices.numel()),
                                "num_views": selection.num_views,
                                "num_global_views": selection.num_global,
                                "num_local_views": selection.num_local,
                                "feature_dim": feature_dim,
                                "effective_N": int(indices.numel() * selection.num_views),
                                "within_pair_fraction": (selection.num_views - 1)
                                / (indices.numel() * selection.num_views - 1),
                                **metrics,
                            }
                        )
                        print(
                            f"[analysis] {spec.label} aug={augmentation_id} part={partition_id} "
                            f"B={indices.numel()} V={selection.num_views} batch={batch_id}",
                            flush=True,
                        )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def quantile(values: list[float], q: float) -> float:
    clean = np.asarray([v for v in values if math.isfinite(v)], dtype=np.float64)
    return float(np.quantile(clean, q)) if clean.size else float("nan")


def aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metrics = [
        "within_fraction_of_cw_gradient",
        "within_to_between_gradient_ratio",
        "cosine_within_alignment",
        "cosine_between_alignment",
        "cosine_target_alignment",
        "cosine_full_cw_alignment",
        "cosine_group_cw_alignment",
        "cosine_full_group_cw",
        "relative_full_group_gradient_difference",
        "norm_pair_within",
        "norm_pair_between",
        "norm_target",
        "norm_cw_full",
        "norm_group_cw",
        "signed_within_effect_on_alignment",
        "mean_within_distance",
        "mean_between_distance",
        "mean_within_kernel",
        "mean_between_kernel",
        "norm_within_gg",
        "norm_within_gl",
        "norm_within_ll",
    ]
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        for pair_type in ["within", "between", "target", "full", "group", "within_gg", "within_gl", "within_ll"]:
            key = (
                row["checkpoint"],
                row["training_progress"],
                row["batch_size"],
                row["num_views"],
                pair_type,
            )
            groups.setdefault(key, []).append(row)
    out: list[dict[str, Any]] = []
    for (checkpoint, progress, batch_size, num_views, pair_type), group_rows in groups.items():
        row: dict[str, Any] = {
            "checkpoint": checkpoint,
            "training_progress": progress,
            "batch_size": batch_size,
            "num_views": num_views,
            "pair_type": pair_type,
            "count": len(group_rows),
        }
        for metric in metrics:
            vals = [float(r[metric]) for r in group_rows if r.get(metric) not in ("", None)]
            clean = [v for v in vals if math.isfinite(v)]
            row[f"{metric}_mean"] = float(np.mean(clean)) if clean else float("nan")
            row[f"{metric}_std"] = float(np.std(clean, ddof=1)) if len(clean) > 1 else 0.0
            row[f"{metric}_median"] = quantile(clean, 0.5)
            row[f"{metric}_iqr"] = quantile(clean, 0.75) - quantile(clean, 0.25)
            row[f"{metric}_q05"] = quantile(clean, 0.05)
            row[f"{metric}_q95"] = quantile(clean, 0.95)
        out.append(row)
    return out


def set_plot_theme() -> None:
    try:
        import seaborn as sns

        sns.set_theme()
    except ModuleNotFoundError:
        import matplotlib.pyplot as plt

        plt.style.use("seaborn-v0_8")


def plot_metric(rows: list[dict[str, Any]], output_dir: Path, metric: str, ylabel: str, filename: str, hue_key: str = "checkpoint") -> None:
    import matplotlib.pyplot as plt

    set_plot_theme()
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("pair_type") != "within":
            continue
        if int(row.get("batch_size", 0)) != 128 or int(row.get("num_views", 0)) != 10:
            continue
        grouped.setdefault(str(row[hue_key]), []).append(row)
    for label, group in sorted(grouped.items()):
        group = sorted(group, key=lambda r: float(r["training_progress"]))
        x = np.asarray([float(r["training_progress"]) for r in group])
        y = np.asarray([float(r[f"{metric}_mean"]) for r in group])
        err = np.asarray([float(r[f"{metric}_std"]) for r in group])
        ax.plot(x, y, marker="o", label=label)
        ax.fill_between(x, y - err, y + err, alpha=0.16)
    ax.set_xlabel("training progress (%)")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / f"{filename}.pdf")
    plt.close(fig)


def plot_sensitivity(rows: list[dict[str, Any]], output_dir: Path, x_key: str, filename: str) -> None:
    import matplotlib.pyplot as plt

    set_plot_theme()
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8))
    latest_progress = max(float(r["training_progress"]) for r in rows if math.isfinite(float(r["training_progress"])))
    subset = [
        r
        for r in rows
        if r.get("pair_type") == "within"
        and abs(float(r["training_progress"]) - latest_progress) < 1e-8
    ]
    for metric, ax, ylabel in [
        ("within_fraction_of_cw_gradient", axes[0], "||g_within|| / ||g_cw||"),
        ("cosine_within_alignment", axes[1], "cosine(within, alignment)"),
    ]:
        grouped: dict[int, list[float]] = {}
        for row in subset:
            grouped.setdefault(int(row[x_key]), []).append(float(row[f"{metric}_mean"]))
        xs = sorted(grouped)
        ys = [float(np.mean(grouped[x])) for x in xs]
        ax.plot(xs, ys, marker="o")
        ax.set_xlabel(x_key)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / f"{filename}.pdf")
    plt.close(fig)


def write_summary_report(output_dir: Path, aggregate: list[dict[str, Any]]) -> None:
    target = [
        r for r in aggregate
        if r.get("pair_type") == "within"
        and int(r.get("batch_size", 0)) == 128
        and int(r.get("num_views", 0)) == 10
    ]
    lines = [
        "# Multi-view CW gradient interaction summary",
        "",
        "This report is generated from detached LeJEPA projector outputs and does not modify training.",
        "",
    ]
    if target:
        latest = max(target, key=lambda r: float(r["training_progress"]))
        share = float(latest["within_fraction_of_cw_gradient_median"])
        cosine_align = float(latest["cosine_within_alignment_median"])
        full_group_cos = float(latest["cosine_full_group_cw_median"])
        rel_group = float(latest["relative_full_group_gradient_difference_median"])
        lines.extend(
            [
                f"At the latest analyzed checkpoint, median within-gradient share is {share:.4g}.",
                f"Median cosine(within, alignment) is {cosine_align:.4g}.",
                f"Median cosine(full CW, group-aware CW) is {full_group_cos:.4g}.",
                f"Median relative full/group gradient difference is {rel_group:.4g}.",
                "",
            ]
        )
        if share < 0.01 and full_group_cos > 0.99:
            conclusion = "likely negligible by the heuristic thresholds"
        elif share >= 0.03 or full_group_cos < 0.98 or cosine_align < -0.1:
            conclusion = "potentially meaningful by the heuristic thresholds"
        else:
            conclusion = "borderline or mixed by the heuristic thresholds"
        lines.append(f"Overall heuristic conclusion: {conclusion}.")
    else:
        lines.append("No B=128, V=10 aggregate rows were available for the default decision summary.")
    (output_dir / "multiview_cw_gradient_summary.md").write_text("\n".join(lines) + "\n")


def center(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mean = x.mean(dim=0)
    return x - mean, mean


def covariance(x: torch.Tensor) -> torch.Tensor:
    if x.shape[0] < 2:
        return torch.zeros(x.shape[1], x.shape[1], dtype=x.dtype, device=x.device)
    xc, _ = center(x)
    return xc.T @ xc / (x.shape[0] - 1)


def cross_covariance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    if x.shape[0] < 2:
        return torch.zeros(x.shape[1], y.shape[1], dtype=x.dtype, device=x.device)
    xc, _ = center(x)
    yc, _ = center(y)
    return xc.T @ yc / (x.shape[0] - 1)


def stable_inv_sqrt(cov: torch.Tensor, eps: float) -> torch.Tensor:
    cov = 0.5 * (cov + cov.T)
    scale = torch.trace(cov).clamp_min(EPS) / cov.shape[0]
    regularized = cov + (eps * scale) * torch.eye(cov.shape[0], dtype=cov.dtype, device=cov.device)
    evals, evecs = torch.linalg.eigh(regularized)
    return (evecs * torch.rsqrt(evals.clamp_min(EPS))) @ evecs.T


def stable_sqrt(cov: torch.Tensor, eps: float) -> torch.Tensor:
    cov = 0.5 * (cov + cov.T)
    scale = torch.trace(cov).clamp_min(EPS) / cov.shape[0]
    regularized = cov + (eps * scale) * torch.eye(cov.shape[0], dtype=cov.dtype, device=cov.device)
    evals, evecs = torch.linalg.eigh(regularized)
    return (evecs * torch.sqrt(evals.clamp_min(EPS))) @ evecs.T


def covariance_spectrum_hash(cov: torch.Tensor, *, decimals: int = 6) -> str:
    evals = torch.linalg.eigvalsh(0.5 * (cov + cov.T)).detach().cpu().numpy()
    rounded = np.round(evals, decimals=decimals)
    return hashlib.sha1(rounded.tobytes()).hexdigest()


def effective_rank_from_spectrum(values: torch.Tensor) -> float:
    vals = values.detach().abs().float()
    total = vals.sum()
    if total <= 0:
        return 0.0
    p = vals / total
    entropy = -(p * torch.log(p.clamp_min(EPS))).sum()
    return float(torch.exp(entropy).cpu())


def isotropy_error(cov: torch.Tensor) -> float:
    alpha = torch.trace(cov) / cov.shape[0]
    return float((cov - alpha * torch.eye(cov.shape[0], device=cov.device, dtype=cov.dtype)).norm().cpu() / (cov.norm().cpu() + EPS))


def condition_number(evals: torch.Tensor) -> float:
    vals = evals.detach().abs().float()
    positive = vals[vals > 1e-12]
    if positive.numel() == 0:
        return float("nan")
    return float((positive.max() / positive.min()).cpu())


def marginal_stats(x: torch.Tensor, prefix: str) -> dict[str, float]:
    cov = covariance(x)
    evals = torch.linalg.eigvalsh(0.5 * (cov + cov.T)).clamp_min(0)
    norms = x.norm(dim=1)
    return {
        f"{prefix}_mean_norm": float(x.mean(dim=0).norm().cpu()),
        f"{prefix}_variance_mean": float(torch.diagonal(cov).mean().cpu()),
        f"{prefix}_variance_std": float(torch.diagonal(cov).std(unbiased=False).cpu()),
        f"{prefix}_cov_top_eigenvalue": float(evals.max().cpu()),
        f"{prefix}_cov_min_eigenvalue": float(evals.min().cpu()),
        f"{prefix}_effective_rank": effective_rank_from_spectrum(evals),
        f"{prefix}_condition_number": condition_number(evals),
        f"{prefix}_isotropy_error": isotropy_error(cov),
        f"{prefix}_mean_squared_norm": float(x.square().sum(dim=1).mean().cpu()),
        f"{prefix}_norm_median": quantile([float(v) for v in norms.cpu()], 0.5),
        f"{prefix}_norm_q05": quantile([float(v) for v in norms.cpu()], 0.05),
        f"{prefix}_norm_q95": quantile([float(v) for v in norms.cpu()], 0.95),
    }


def standardize_total_variance(x: torch.Tensor) -> torch.Tensor:
    xc, _ = center(x)
    cov = covariance(xc)
    scale = torch.sqrt((torch.trace(cov) / cov.shape[0]).clamp_min(EPS))
    return xc / scale


def skewness_and_kurtosis(x: torch.Tensor) -> tuple[float, float]:
    xc = x - x.mean(dim=0)
    std = x.std(dim=0, unbiased=False).clamp_min(1e-8)
    z = xc / std
    skew = z.pow(3).mean(dim=0).abs().mean()
    kurt = (z.pow(4).mean(dim=0) - 3.0).abs().mean()
    return float(skew.cpu()), float(kurt.cpu())


def projection_shape_stats(x: torch.Tensor, num_projections: int, seed: int) -> tuple[float, float]:
    g = torch.Generator(device=x.device).manual_seed(seed)
    directions = torch.randn(x.shape[1], num_projections, generator=g, device=x.device)
    directions = directions / directions.norm(dim=0).clamp_min(EPS)
    projected = x @ directions
    return skewness_and_kurtosis(projected)


def roc_auc(scores_positive: torch.Tensor, scores_negative: torch.Tensor) -> float:
    scores = torch.cat([scores_positive, scores_negative]).detach().cpu().numpy()
    labels = np.concatenate([np.ones(scores_positive.numel()), np.zeros(scores_negative.numel())])
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    i = 0
    while i < scores.size:
        j = i + 1
        while j < scores.size and scores[order[j]] == scores[order[i]]:
            j += 1
        ranks[order[i:j]] = (i + j + 1) / 2.0
        i = j
    pos_ranks = ranks[labels == 1]
    n_pos = scores_positive.numel()
    n_neg = scores_negative.numel()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((pos_ranks.sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def correlated_gaussian_energy(z1: torch.Tensor, z2: torch.Tensor, rho: float) -> torch.Tensor:
    rho = float(max(min(rho, 0.999), -0.999))
    numerator = z1.square().sum(dim=1) + z2.square().sum(dim=1) - 2.0 * rho * (z1 * z2).sum(dim=1)
    return numerator / (2.0 * (1.0 - rho * rho))


def ordered_pair_pool(num_views: int, num_global: int, pair_type: str) -> list[tuple[int, int]]:
    globals_ = list(range(num_global))
    locals_ = list(range(num_global, num_views))
    if pair_type == "global_global":
        return [(i, j) for i in globals_ for j in globals_ if i != j]
    if pair_type == "global_local":
        return [(i, j) for i in globals_ for j in locals_] + [
            (i, j) for i in locals_ for j in globals_
        ]
    if pair_type == "local_local":
        return [(i, j) for i in locals_ for j in locals_ if i != j]
    if pair_type == "same_view_duplicate":
        return [(i, i) for i in range(num_views)]
    if pair_type == "positive_pooled_frequency_weighted":
        return (
            ordered_pair_pool(num_views, num_global, "global_global")
            + ordered_pair_pool(num_views, num_global, "global_local")
            + ordered_pair_pool(num_views, num_global, "local_local")
        )
    if pair_type == "positive_pooled_equal_weighted":
        # The caller evaluates the three components separately and averages rows;
        # this sentinel is intentionally not expanded here.
        raise ValueError("positive_pooled_equal_weighted is an aggregate label.")
    raise ValueError(f"Unknown pair_type={pair_type!r}")


def positive_pair_indices(num_views: int, num_global: int, pair_type: str) -> list[tuple[int, int]]:
    return ordered_pair_pool(num_views, num_global, pair_type)


def balanced_pair_assignment(
    num_images: int,
    num_views: int,
    num_global: int,
    pair_type: str,
    *,
    seed: int = 0,
) -> list[tuple[int, int]]:
    pool = ordered_pair_pool(num_views, num_global, pair_type)
    if not pool:
        raise ValueError(f"Pair type {pair_type!r} has no valid pairs.")
    offset = seed % len(pool)
    return [pool[(offset + i) % len(pool)] for i in range(num_images)]


def construct_pairs(
    z_bvd: torch.Tensor,
    pair_type: str,
    num_global: int,
    *,
    shuffled: bool = False,
    seed: int = 0,
    estimator: str = "all_pairs_clustered",
    max_materialized_rows: int = 8192,
) -> tuple[torch.Tensor, torch.Tensor, list[tuple[int, int]]]:
    if estimator == "balanced_one_pair_per_image":
        assignments = balanced_pair_assignment(
            z_bvd.shape[0], z_bvd.shape[1], num_global, pair_type, seed=seed
        )
        left = torch.stack([z_bvd[i, v] for i, (v, _) in enumerate(assignments)])
        right = torch.stack([z_bvd[i, w] for i, (_, w) in enumerate(assignments)])
        if shuffled:
            perm = torch.randperm(z_bvd.shape[0], generator=torch.Generator(device=z_bvd.device).manual_seed(seed), device=z_bvd.device)
            if z_bvd.shape[0] > 1 and torch.equal(perm, torch.arange(z_bvd.shape[0], device=z_bvd.device)):
                perm = perm.roll(1)
            right = right.index_select(0, perm)
        return left, right, assignments

    if estimator != "all_pairs_clustered":
        raise ValueError(f"Unknown estimator={estimator!r}")
    pairs = ordered_pair_pool(z_bvd.shape[1], num_global, pair_type)
    total_rows = len(pairs) * z_bvd.shape[0]
    if total_rows > max_materialized_rows:
        raise ValueError(
            "Refusing to materialize a large all_pairs_clustered tensor; "
            "use the chunked joint-structure all-pairs path instead."
        )
    left: list[torch.Tensor] = []
    right: list[torch.Tensor] = []
    generator = torch.Generator(device=z_bvd.device).manual_seed(seed)
    for i, j in pairs:
        z1 = z_bvd[:, i]
        z2 = z_bvd[:, j]
        if shuffled:
            perm = torch.randperm(z_bvd.shape[0], generator=generator, device=z_bvd.device)
            if z_bvd.shape[0] > 1 and torch.equal(perm, torch.arange(z_bvd.shape[0], device=z_bvd.device)):
                perm = perm.roll(1)
            z2 = z2.index_select(0, perm)
        left.append(z1)
        right.append(z2)
    return torch.cat(left, dim=0), torch.cat(right, dim=0), pairs


def split_calibration_evaluation(
    num_images: int, fraction: float, seed: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    if not 0 < fraction < 1:
        raise ValueError("--calibration-fraction must be in (0, 1).")
    perm = torch.randperm(num_images, generator=torch.Generator(device=device).manual_seed(seed), device=device)
    n_cal = max(1, min(num_images - 1, int(round(num_images * fraction))))
    return perm[:n_cal], perm[n_cal:]


def sufficient_statistics(
    x: torch.Tensor,
    y: torch.Tensor | None = None,
    *,
    chunk_size: int | None = None,
) -> dict[str, torch.Tensor | int]:
    if y is None:
        y = x
    if x.shape[0] != y.shape[0]:
        raise ValueError("Sufficient statistics require matching sample counts.")
    n = x.shape[0]
    chunk = max(1, int(chunk_size or n))
    sum_x = torch.zeros(x.shape[1], device=x.device, dtype=torch.float64)
    sum_y = torch.zeros(y.shape[1], device=y.device, dtype=torch.float64)
    xtx = torch.zeros(x.shape[1], x.shape[1], device=x.device, dtype=torch.float64)
    yty = torch.zeros(y.shape[1], y.shape[1], device=y.device, dtype=torch.float64)
    xty = torch.zeros(x.shape[1], y.shape[1], device=x.device, dtype=torch.float64)
    for start in range(0, n, chunk):
        xs = x[start : start + chunk].double()
        ys = y[start : start + chunk].double()
        sum_x += xs.sum(dim=0)
        sum_y += ys.sum(dim=0)
        xtx += xs.T @ xs
        yty += ys.T @ ys
        xty += xs.T @ ys
    return {"n": n, "sum_x": sum_x, "sum_y": sum_y, "xtx": xtx, "yty": yty, "xty": xty}


def covariance_from_sufficient_stats(
    stats: dict[str, torch.Tensor | int],
    *,
    which: str,
    denominator: str = "unbiased",
) -> torch.Tensor:
    n = int(stats["n"])
    if denominator == "unbiased" and n < 2:
        raise ValueError("Need at least two samples for unbiased covariance.")
    denom = n - 1 if denominator == "unbiased" else n
    if which == "x":
        total = stats["xtx"]
        mean_outer = torch.outer(stats["sum_x"], stats["sum_x"]) / n
    elif which == "y":
        total = stats["yty"]
        mean_outer = torch.outer(stats["sum_y"], stats["sum_y"]) / n
    elif which == "xy":
        total = stats["xty"]
        mean_outer = torch.outer(stats["sum_x"], stats["sum_y"]) / n
    else:
        raise ValueError(f"Unknown covariance block {which!r}.")
    return ((total - mean_outer) / denom).float()


def oas_covariance(
    x: torch.Tensor,
    *,
    eps: float,
    chunk_size: int | None = None,
) -> tuple[torch.Tensor, float, int]:
    stats = sufficient_statistics(x, chunk_size=chunk_size)
    n = int(stats["n"])
    sum_x = stats["sum_x"]
    xtx = stats["xtx"]
    emp_mle64 = (xtx - torch.outer(sum_x, sum_x) / n) / n
    emp_unbiased = covariance_from_sufficient_stats(stats, which="x", denominator="unbiased")
    rank = int(torch.linalg.matrix_rank(emp_unbiased).detach().cpu())
    _, p = x.shape
    mu = torch.trace(emp_mle64) / p
    alpha = (emp_mle64 * emp_mle64).mean()
    denom = (n + 1.0) * (alpha - mu.square() / p)
    if float(denom.abs().detach().cpu()) < EPS:
        shrinkage = torch.tensor(1.0, device=x.device, dtype=emp_mle64.dtype)
    else:
        shrinkage = ((alpha + mu.square()) / denom).clamp(0.0, 1.0)
    eye = torch.eye(p, device=x.device, dtype=emp_mle64.dtype)
    shrunk = (1.0 - shrinkage) * emp_mle64 + shrinkage * mu * eye
    shrunk = shrunk + eps * mu.clamp_min(EPS) * eye
    return shrunk.to(dtype=x.dtype), float(shrinkage.detach().cpu()), rank


def covariance_estimate(
    x: torch.Tensor,
    *,
    method: str,
    eps: float,
    chunk_size: int | None = None,
) -> tuple[torch.Tensor, float, int]:
    stats = sufficient_statistics(x, chunk_size=chunk_size)
    emp = covariance_from_sufficient_stats(stats, which="x", denominator="unbiased")
    rank = int(torch.linalg.matrix_rank(emp).detach().cpu())
    if method == "ridge":
        alpha = torch.trace(emp).clamp_min(EPS) / emp.shape[0]
        return emp + eps * alpha * torch.eye(emp.shape[0], device=emp.device, dtype=emp.dtype), 0.0, rank
    if method != "oas":
        raise ValueError(f"Unknown covariance estimator {method!r}")
    return oas_covariance(x, eps=eps, chunk_size=chunk_size)


def covariance_estimate_from_stats(
    stats: dict[str, torch.Tensor | int],
    *,
    which: str,
    method: str,
    eps: float,
) -> tuple[torch.Tensor, float, int]:
    emp = covariance_from_sufficient_stats(stats, which=which, denominator="unbiased")
    rank = int(torch.linalg.matrix_rank(emp).detach().cpu())
    if method == "ridge":
        alpha = torch.trace(emp).clamp_min(EPS) / emp.shape[0]
        return emp + eps * alpha * torch.eye(emp.shape[0], device=emp.device, dtype=emp.dtype), 0.0, rank
    if method != "oas":
        raise ValueError(f"Unknown covariance estimator {method!r}")
    n = int(stats["n"])
    if which == "x":
        total = stats["xtx"]
        summed = stats["sum_x"]
    elif which == "y":
        total = stats["yty"]
        summed = stats["sum_y"]
    else:
        raise ValueError("OAS covariance can only be estimated for x or y blocks.")
    emp_mle64 = (total - torch.outer(summed, summed) / n) / n
    p = emp_mle64.shape[0]
    mu = torch.trace(emp_mle64) / p
    alpha = (emp_mle64 * emp_mle64).mean()
    denom = (n + 1.0) * (alpha - mu.square() / p)
    if float(denom.abs().detach().cpu()) < EPS:
        shrinkage = torch.tensor(1.0, device=emp_mle64.device, dtype=emp_mle64.dtype)
    else:
        shrinkage = ((alpha + mu.square()) / denom).clamp(0.0, 1.0)
    eye = torch.eye(p, device=emp_mle64.device, dtype=emp_mle64.dtype)
    shrunk = (1.0 - shrinkage) * emp_mle64 + shrinkage * mu * eye
    shrunk = shrunk + eps * mu.clamp_min(EPS) * eye
    return shrunk.float(), float(shrinkage.detach().cpu()), rank


def deterministic_projection(original_dim: int, projected_dim: int, seed: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    q = min(projected_dim, original_dim)
    generator = torch.Generator(device=device).manual_seed(seed + 7919 * original_dim + q)
    mat = torch.randn(original_dim, q, generator=generator, device=device, dtype=dtype)
    qmat, _ = torch.linalg.qr(mat, mode="reduced")
    return qmat[:, :q]


def resolve_fixed_projection_dim(
    requested_dim: int,
    *,
    image_counts: list[int],
    available_images: int,
    calibration_fraction: float,
    original_dim: int,
) -> int:
    counts = [min(int(count), available_images) for count in image_counts]
    if not counts:
        raise ValueError("At least one image count is required.")
    min_cal = min(max(1, min(count - 1, int(round(count * calibration_fraction)))) for count in counts)
    q = min(int(requested_dim), min_cal - 1, int(original_dim))
    if q < 1:
        raise ValueError("Projected analysis needs at least two calibration images.")
    return q


def apply_analysis_space(
    z1: torch.Tensor,
    z2: torch.Tensor,
    shuffled_z2: torch.Tensor,
    *,
    analysis_space: str,
    projection_dim: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    if analysis_space == "full":
        return z1, z2, shuffled_z2, z1.shape[1]
    if analysis_space != "projected":
        raise ValueError(f"Unknown analysis_space={analysis_space!r}")
    q = min(projection_dim, z1.shape[0] - 1, z1.shape[1])
    if q < 1:
        raise ValueError("Projected analysis requires at least two samples.")
    proj = deterministic_projection(z1.shape[1], q, seed, z1.device, z1.dtype)
    return z1 @ proj, z2 @ proj, shuffled_z2 @ proj, q


def apply_analysis_space_bvd(
    z_bvd: torch.Tensor,
    *,
    analysis_space: str,
    projection_dim: int,
    seed: int,
) -> tuple[torch.Tensor, int]:
    if analysis_space == "full":
        return z_bvd, z_bvd.shape[-1]
    if analysis_space != "projected":
        raise ValueError(f"Unknown analysis_space={analysis_space!r}")
    q = int(projection_dim)
    if q < 1 or q > z_bvd.shape[-1] or q > z_bvd.shape[0] - 1:
        raise ValueError("Projected analysis dimension must be pre-resolved and valid for every image count.")
    proj = deterministic_projection(z_bvd.shape[-1], q, seed, z_bvd.device, z_bvd.dtype)
    return torch.einsum("bvd,dq->bvq", z_bvd, proj), q


MAX_CW_NORMALITY_SAMPLES_OBSERVED = 0


def deterministic_row_subsample(
    x: torch.Tensor,
    max_samples: int,
    *,
    seed: int,
) -> tuple[torch.Tensor, bool]:
    if x.shape[0] <= max_samples:
        return x, False
    generator = torch.Generator(device=x.device).manual_seed(seed)
    idx = torch.randperm(x.shape[0], generator=generator, device=x.device)[:max_samples]
    idx, _ = torch.sort(idx)
    return x.index_select(0, idx), True


def shared_row_subsample(
    tensors: list[torch.Tensor],
    max_samples: int,
    *,
    seed: int,
) -> tuple[list[torch.Tensor], int, bool, torch.Tensor]:
    if not tensors:
        raise ValueError("At least one tensor is required.")
    n = min(t.shape[0] for t in tensors)
    count = min(n, max_samples)
    if count < n:
        generator = torch.Generator(device=tensors[0].device).manual_seed(seed)
        idx = torch.randperm(n, generator=generator, device=tensors[0].device)[:count]
        idx, _ = torch.sort(idx)
        return [t[:n].index_select(0, idx) for t in tensors], count, True, idx
    idx = torch.arange(n, device=tensors[0].device)
    return [t[:n] for t in tensors], count, False, idx


def unscaled_cw_normality_score(
    x: torch.Tensor,
    gamma: float,
    *,
    max_samples: int | None = None,
    seed: int = 0,
) -> float:
    global MAX_CW_NORMALITY_SAMPLES_OBSERVED
    if max_samples is not None:
        x, _ = deterministic_row_subsample(x, max_samples, seed=seed)
    MAX_CW_NORMALITY_SAMPLES_OBSERVED = max(MAX_CW_NORMALITY_SAMPLES_OBSERVED, int(x.shape[0]))
    standardized = standardize_total_variance(x)
    gamma_t = torch.as_tensor(gamma, device=x.device, dtype=x.dtype)
    return float(cw_normality(standardized, gamma_t).detach().cpu())


def gaussianity_score_with_metadata(
    x: torch.Tensor,
    gamma: float,
    *,
    max_samples: int,
    seed: int,
) -> tuple[float, int, bool]:
    xs, was_subsampled = deterministic_row_subsample(x, max_samples, seed=seed)
    return unscaled_cw_normality_score(xs, gamma), int(xs.shape[0]), was_subsampled


def unscaled_cw_normality_score_presampled(x: torch.Tensor, gamma: float) -> float:
    return unscaled_cw_normality_score(x, gamma)


def shape_only_cw_normality(
    x: torch.Tensor,
    gamma: float,
    *,
    covariance_eps: float,
    covariance_estimator: str,
    stats_chunk_size: int | None,
) -> float:
    cov, _, _ = covariance_estimate(
        x,
        method=covariance_estimator,
        eps=covariance_eps,
        chunk_size=stats_chunk_size,
    )
    inv = stable_inv_sqrt(cov, covariance_eps)
    whitened = whiten_with_calibration(x, x.mean(dim=0), inv)
    return unscaled_cw_normality_score_presampled(whitened, gamma)


def whiten_with_calibration(x: torch.Tensor, mean: torch.Tensor, inv_sqrt: torch.Tensor) -> torch.Tensor:
    return (x - mean) @ inv_sqrt.T


def structural_metrics_from_calibration(
    z1_cal: torch.Tensor,
    z2_cal: torch.Tensor,
    z1_eval: torch.Tensor,
    z2_eval: torch.Tensor,
    shuffled_eval: torch.Tensor,
    *,
    gamma: float,
    covariance_eps: float,
    covariance_estimator: str,
    projection_diagnostics: int,
    seed: int,
    fixed_rhos: list[float],
    stats_chunk_size: int | None = None,
    pair_type: str = "",
    gaussianity_max_samples: int = 1024,
    calibration_stats: dict[str, torch.Tensor | int] | None = None,
) -> dict[str, float]:
    z1_cal = z1_cal.float()
    z2_cal = z2_cal.float()
    z1_eval = z1_eval.float()
    z2_eval = z2_eval.float()
    shuffled_eval = shuffled_eval.float()
    if calibration_stats is None:
        mean1 = z1_cal.mean(dim=0)
        mean2 = z2_cal.mean(dim=0)
        c11, shrink1, rank1 = covariance_estimate(
            z1_cal, method=covariance_estimator, eps=covariance_eps, chunk_size=stats_chunk_size
        )
        c22, shrink2, rank2 = covariance_estimate(
            z2_cal, method=covariance_estimator, eps=covariance_eps, chunk_size=stats_chunk_size
        )
        joint_stats = sufficient_statistics(z1_cal, z2_cal, chunk_size=stats_chunk_size)
    else:
        n_stats = int(calibration_stats["n"])
        mean1 = (calibration_stats["sum_x"] / n_stats).to(device=z1_cal.device, dtype=z1_cal.dtype)
        mean2 = (calibration_stats["sum_y"] / n_stats).to(device=z2_cal.device, dtype=z2_cal.dtype)
        c11, shrink1, rank1 = covariance_estimate_from_stats(
            calibration_stats, which="x", method=covariance_estimator, eps=covariance_eps
        )
        c22, shrink2, rank2 = covariance_estimate_from_stats(
            calibration_stats, which="y", method=covariance_estimator, eps=covariance_eps
        )
        joint_stats = calibration_stats
    c12 = covariance_from_sufficient_stats(
        joint_stats,
        which="xy",
        denominator="mle" if covariance_estimator == "oas" else "unbiased",
    )
    inv1 = stable_inv_sqrt(c11, covariance_eps)
    inv2 = stable_inv_sqrt(c22, covariance_eps)
    r = inv1 @ c12 @ inv2
    diag = torch.diagonal(r)
    rho_hat = torch.trace(r) / r.shape[0]
    eye = torch.eye(r.shape[0], device=r.device, dtype=r.dtype)
    offdiag = r - torch.diag(diag)
    scalar_residual = (r - rho_hat * eye).norm() / (r.norm() + EPS)
    explained_rho_i = 1.0 - (r - rho_hat * eye).square().sum() / (r.square().sum() + EPS)
    explained_diagonal = 1.0 - offdiag.square().sum() / (r.square().sum() + EPS)
    sym = 0.5 * (r + r.T)
    antisym = 0.5 * (r - r.T)
    singular = torch.linalg.svdvals(r)
    energy = singular.square().sum().clamp_min(EPS)

    rho_hat_unconstrained = float(rho_hat.detach().cpu())
    rho_used_for_energy = max(min(rho_hat_unconstrained, 0.999), -0.999)
    rho_was_clipped = rho_used_for_energy != rho_hat_unconstrained

    w1_cal = whiten_with_calibration(z1_cal, mean1, inv1)
    w2_cal = whiten_with_calibration(z2_cal, mean2, inv2)
    w1_eval = whiten_with_calibration(z1_eval, mean1, inv1)
    w2_eval = whiten_with_calibration(z2_eval, mean2, inv2)
    wshuf_eval = whiten_with_calibration(shuffled_eval, mean2, inv2)
    same_view = pair_type == "same_view_duplicate"
    if same_view:
        e_true = torch.full((w1_eval.shape[0],), float("nan"), device=w1_eval.device)
        e_shuf = torch.full((w1_eval.shape[0],), float("nan"), device=w1_eval.device)
    else:
        e_true = correlated_gaussian_energy(w1_eval, w2_eval, rho_used_for_energy)
        e_shuf = correlated_gaussian_energy(w1_eval, wshuf_eval, rho_used_for_energy)
    cos_true = F.cosine_similarity(z1_eval, z2_eval, dim=1)
    cos_shuf = F.cosine_similarity(z1_eval, shuffled_eval, dim=1)
    wcos_true = F.cosine_similarity(w1_eval, w2_eval, dim=1)
    wcos_shuf = F.cosine_similarity(w1_eval, wshuf_eval, dim=1)
    dist_true = (z1_eval - z2_eval).square().sum(dim=1)
    dist_shuf = (z1_eval - shuffled_eval).square().sum(dim=1)

    w_plus = (w1_cal + w2_cal) / math.sqrt(2.0)
    w_minus = (w1_cal - w2_cal) / math.sqrt(2.0)
    if calibration_stats is None:
        c_plus = covariance_from_sufficient_stats(sufficient_statistics(w_plus, chunk_size=stats_chunk_size), which="x", denominator="unbiased")
        c_minus = covariance_from_sufficient_stats(sufficient_statistics(w_minus, chunk_size=stats_chunk_size), which="x", denominator="unbiased")
        c_pm = covariance_from_sufficient_stats(sufficient_statistics(w_plus, w_minus, chunk_size=stats_chunk_size), which="xy", denominator="unbiased")
    else:
        c11w = inv1 @ c11 @ inv1.T
        c22w = inv2 @ c22 @ inv2.T
        c12w = r
        c_plus = 0.5 * (c11w + c22w + c12w + c12w.T)
        c_minus = 0.5 * (c11w + c22w - c12w - c12w.T)
        c_pm = 0.5 * (c11w - c22w - c12w + c12w.T)
    variance_ratio = torch.trace(c_plus) / torch.trace(c_minus).clamp_min(EPS)
    z_plus = (z1_cal + z2_cal) / math.sqrt(2.0)
    z_minus = (z1_cal - z2_cal) / math.sqrt(2.0)
    if calibration_stats is None:
        raw_c_plus = covariance_from_sufficient_stats(sufficient_statistics(z_plus, chunk_size=stats_chunk_size), which="x", denominator="unbiased")
        raw_c_minus = covariance_from_sufficient_stats(sufficient_statistics(z_minus, chunk_size=stats_chunk_size), which="x", denominator="unbiased")
    else:
        raw_c_plus = 0.5 * (c11 + c22 + c12 + c12.T)
        raw_c_minus = 0.5 * (c11 + c22 - c12 - c12.T)
    (
        z1_g,
        z2_g,
        w_plus_g,
        w_minus_g,
        z_plus_g,
        z_minus_g,
    ), gaussianity_n, gaussianity_subsampled, gaussianity_idx = shared_row_subsample(
        [z1_cal, z2_cal, w_plus, w_minus, z_plus, z_minus],
        gaussianity_max_samples,
        seed=seed + 1,
    )
    g_z1 = unscaled_cw_normality_score_presampled(z1_g, gamma)
    g_z2 = unscaled_cw_normality_score_presampled(z2_g, gamma)
    g_plus = unscaled_cw_normality_score_presampled(w_plus_g, gamma)
    g_minus = unscaled_cw_normality_score_presampled(w_minus_g, gamma)
    raw_g_plus = unscaled_cw_normality_score_presampled(z_plus_g, gamma)
    raw_g_minus = unscaled_cw_normality_score_presampled(z_minus_g, gamma)
    shape_g_plus = shape_only_cw_normality(
        w_plus_g,
        gamma,
        covariance_eps=covariance_eps,
        covariance_estimator=covariance_estimator,
        stats_chunk_size=stats_chunk_size,
    )
    shape_g_minus = shape_only_cw_normality(
        w_minus_g,
        gamma,
        covariance_eps=covariance_eps,
        covariance_estimator=covariance_estimator,
        stats_chunk_size=stats_chunk_size,
    )

    row: dict[str, float] = {
        "rho_hat": rho_hat_unconstrained,
        "rho_hat_unconstrained": rho_hat_unconstrained,
        "rho_used_for_energy": float("nan") if same_view else rho_used_for_energy,
        "rho_was_clipped": rho_was_clipped,
        "scalar_model_relative_residual": float(scalar_residual.detach().cpu()),
        "explained_rho_i": float(explained_rho_i.detach().cpu()),
        "explained_diagonal": float(explained_diagonal.detach().cpu()),
        "relative_offdiagonal_energy": float((offdiag.norm() / (r.norm() + EPS)).detach().cpu()),
        "antisymmetric_energy": float((antisym.norm() / (r.norm() + EPS)).detach().cpu()),
        "symmetric_energy": float((sym.norm() / (r.norm() + EPS)).detach().cpu()),
        "cross_correlation_effective_rank": effective_rank_from_spectrum(singular),
        "singular_value_dispersion": float((singular.std(unbiased=False) / (singular.mean() + EPS)).detach().cpu()),
        "top_singular_value": float(singular.max().detach().cpu()),
        "min_singular_value": float(singular.min().detach().cpu()),
        "max_singular_value": float(singular.max().detach().cpu()),
        "diagonal_mean": float(diag.mean().detach().cpu()),
        "diagonal_std": float(diag.std(unbiased=False).detach().cpu()),
        "negative_diagonal_fraction": float((diag < 0).float().mean().detach().cpu()),
        "shrinkage_coefficient_C11": shrink1,
        "shrinkage_coefficient_C22": shrink2,
        "covariance_rank_C11": rank1,
        "covariance_rank_C22": rank2,
        "z1_isotropy_error": isotropy_error(c11),
        "z2_isotropy_error": isotropy_error(c22),
        "z1_effective_rank": effective_rank_from_spectrum(torch.linalg.eigvalsh(0.5 * (c11 + c11.T)).clamp_min(0)),
        "z2_effective_rank": effective_rank_from_spectrum(torch.linalg.eigvalsh(0.5 * (c22 + c22.T)).clamp_min(0)),
        "plus_isotropy_error": isotropy_error(c_plus),
        "minus_isotropy_error": isotropy_error(c_minus),
        "raw_plus_isotropy_error": isotropy_error(raw_c_plus),
        "raw_minus_isotropy_error": isotropy_error(raw_c_minus),
        "plus_effective_rank": effective_rank_from_spectrum(torch.linalg.eigvalsh(0.5 * (c_plus + c_plus.T)).clamp_min(0)),
        "minus_effective_rank": effective_rank_from_spectrum(torch.linalg.eigvalsh(0.5 * (c_minus + c_minus.T)).clamp_min(0)),
        "plus_condition_number": condition_number(torch.linalg.eigvalsh(0.5 * (c_plus + c_plus.T)).clamp_min(0)),
        "minus_condition_number": condition_number(torch.linalg.eigvalsh(0.5 * (c_minus + c_minus.T)).clamp_min(0)),
        "plus_minus_crosscov_norm": float((c_pm.norm() / (c_plus.norm() + c_minus.norm() + EPS)).detach().cpu()),
        "plus_minus_variance_ratio": float(variance_ratio.detach().cpu()),
        "unscaled_cw_normality_z1": g_z1,
        "unscaled_cw_normality_z2": g_z2,
        "unscaled_cw_normality_plus": g_plus,
        "unscaled_cw_normality_minus": g_minus,
        "target_fit_cw_plus": g_plus,
        "target_fit_cw_minus": g_minus,
        "shape_cw_plus": shape_g_plus,
        "shape_cw_minus": shape_g_minus,
        "raw_unscaled_cw_normality_plus": raw_g_plus,
        "raw_unscaled_cw_normality_minus": raw_g_minus,
        "gaussianity_num_samples": gaussianity_n,
        "gaussianity_was_subsampled": gaussianity_subsampled,
        "gaussianity_sample_indices": json.dumps([int(i) for i in gaussianity_idx.detach().cpu().tolist()]),
        "gaussianity_score_type": "unscaled_cw_normality",
        "diagnostic_sample_based": True,
        "diagnostic_sample_size": gaussianity_n,
        "mean_pair_cosine": float(cos_true.mean().detach().cpu()),
        "median_pair_cosine": quantile([float(v) for v in cos_true.detach().cpu()], 0.5),
        "shuffled_mean_pair_cosine": float(cos_shuf.mean().detach().cpu()),
        "mean_pair_distance": float(dist_true.mean().detach().cpu()),
        "median_pair_distance": quantile([float(v) for v in dist_true.detach().cpu()], 0.5),
        "shuffled_mean_pair_distance": float(dist_shuf.mean().detach().cpu()),
        "heldout_true_pair_energy": float(e_true.mean().detach().cpu()) if not same_view else float("nan"),
        "heldout_shuffled_pair_energy": float(e_shuf.mean().detach().cpu()) if not same_view else float("nan"),
        "energy_auc": roc_auc(-e_true, -e_shuf) if not same_view else float("nan"),
        "cosine_auc": roc_auc(cos_true, cos_shuf),
        "distance_auc": roc_auc(-dist_true, -dist_shuf),
        "whitened_cosine_auc": roc_auc(wcos_true, wcos_shuf),
        "_c11_matrix": c11.detach().cpu(),
        "_c22_matrix": c22.detach().cpu(),
        "c11_spectrum_hash": covariance_spectrum_hash(c11),
        "c22_spectrum_hash": covariance_spectrum_hash(c22),
    }
    for k in [1, 4, 16, 32]:
        kk = min(k, singular.numel())
        row[f"explained_rank_{k}"] = float((singular[:kk].square().sum() / energy).detach().cpu())
    for i in range(min(8, singular.numel())):
        row[f"singular_value_{i+1}"] = float(singular[i].detach().cpu())
    plus_skew, plus_kurt = projection_shape_stats(standardize_total_variance(w_plus), projection_diagnostics, seed)
    minus_skew, minus_kurt = projection_shape_stats(standardize_total_variance(w_minus), projection_diagnostics, seed + 17)
    row["projection_skewness_plus"] = plus_skew
    row["projection_kurtosis_plus"] = plus_kurt
    row["projection_skewness_minus"] = minus_skew
    row["projection_kurtosis_minus"] = minus_kurt
    for fixed_rho in fixed_rhos:
        suffix = str(fixed_rho).replace(".", "p").replace("-", "m")
        if same_view:
            row[f"energy_auc_rho_{suffix}"] = float("nan")
        else:
            fixed_true = correlated_gaussian_energy(w1_eval, w2_eval, fixed_rho)
            fixed_shuf = correlated_gaussian_energy(w1_eval, wshuf_eval, fixed_rho)
            row[f"energy_auc_rho_{suffix}"] = roc_auc(-fixed_true, -fixed_shuf)
    return row


NULL_CALIBRATION_CACHE: dict[tuple[Any, ...], tuple[dict[str, float], list[dict[str, float]]]] = {}


def strip_internal_metric_tensors(row: dict[str, Any]) -> None:
    row.pop("_c11_matrix", None)
    row.pop("_c22_matrix", None)


def matched_null_rows(
    observed: dict[str, float],
    *,
    n_cal: int,
    n_eval: int,
    dim: int,
    repetitions: int,
    gamma: float,
    covariance_eps: float,
    covariance_estimator: str,
    projection_diagnostics: int,
    fixed_rhos: list[float],
    seed: int,
    stats_chunk_size: int | None = None,
    estimator: str = "balanced_one_pair_per_image",
    same_view_duplicate: bool = False,
    gaussianity_max_samples: int = 1024,
) -> tuple[dict[str, float], list[dict[str, float]], bool]:
    rho_unconstrained = float(observed["rho_hat"])
    rho = max(min(rho_unconstrained, 0.999), -0.999)
    c11_observed = observed.get("_c11_matrix")
    c22_observed = observed.get("_c22_matrix")
    if not isinstance(c11_observed, torch.Tensor):
        c11_observed = torch.eye(dim)
    if not isinstance(c22_observed, torch.Tensor):
        c22_observed = torch.eye(dim)
    c11_observed = c11_observed.float().cpu()
    c22_observed = c22_observed.float().cpu()
    c11_hash = covariance_spectrum_hash(c11_observed)
    c22_hash = covariance_spectrum_hash(c22_observed)
    cache_key_ = (
        n_cal,
        n_eval,
        dim,
        covariance_estimator,
        float(covariance_eps),
        float(gamma),
        int(repetitions),
        int(gaussianity_max_samples),
        int(stats_chunk_size or 0),
        bool(same_view_duplicate),
        c11_hash,
        c22_hash,
        round(rho, 5),
        int(seed),
    )
    if cache_key_ in NULL_CALIBRATION_CACHE:
        summary, rows = NULL_CALIBRATION_CACHE[cache_key_]
        return dict(summary), rows, True
    rows: list[dict[str, float]] = []
    for rep in range(repetitions):
        generator = torch.Generator().manual_seed(seed + 104729 * rep)
        total = n_cal + n_eval
        e1 = torch.randn(total, dim, generator=generator)
        sqrt_c11 = stable_sqrt(c11_observed, covariance_eps)
        x = e1 @ sqrt_c11.T
        if same_view_duplicate:
            y = x.clone()
        else:
            e2 = torch.randn(total, dim, generator=generator)
            sqrt_c22 = stable_sqrt(c22_observed, covariance_eps)
            y = (rho * e1 + math.sqrt(max(1.0 - rho * rho, 1e-8)) * e2) @ sqrt_c22.T
        shuffled = y.roll(1, dims=0)
        null_row = structural_metrics_from_calibration(
            x[:n_cal],
            y[:n_cal],
            x[n_cal:],
            y[n_cal:],
            shuffled[n_cal:],
            gamma=gamma,
            covariance_eps=covariance_eps,
            covariance_estimator=covariance_estimator,
            projection_diagnostics=projection_diagnostics,
            seed=seed + rep,
            fixed_rhos=fixed_rhos,
            stats_chunk_size=stats_chunk_size,
            pair_type="same_view_duplicate" if same_view_duplicate else "",
            gaussianity_max_samples=gaussianity_max_samples,
        )
        strip_internal_metric_tensors(null_row)
        rows.append(null_row)
    summary: dict[str, float] = {}
    calibrated = [
        "scalar_model_relative_residual",
        "relative_offdiagonal_energy",
        "antisymmetric_energy",
        "singular_value_dispersion",
        "plus_isotropy_error",
        "minus_isotropy_error",
        "plus_minus_crosscov_norm",
        "unscaled_cw_normality_plus",
        "unscaled_cw_normality_minus",
        "target_fit_cw_plus",
        "target_fit_cw_minus",
        "shape_cw_plus",
        "shape_cw_minus",
    ]
    for metric in calibrated:
        if metric not in observed:
            continue
        vals = [float(row[metric]) for row in rows if math.isfinite(float(row[metric]))]
        if not vals:
            continue
        obs = float(observed[metric])
        median = quantile(vals, 0.5)
        std = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        summary[f"{metric}_null_mean"] = float(np.mean(vals))
        summary[f"{metric}_null_median"] = median
        summary[f"{metric}_null_std"] = std
        summary[f"{metric}_null_q05"] = quantile(vals, 0.05)
        summary[f"{metric}_null_q95"] = quantile(vals, 0.95)
        summary[f"{metric}_observed_to_null_median"] = obs / (median + EPS)
        summary[f"{metric}_calibrated_z"] = (obs - float(np.mean(vals))) / (std + EPS)
        summary[f"{metric}_null_tail_probability"] = float(np.mean(np.asarray(vals) >= obs))
    NULL_CALIBRATION_CACHE[cache_key_] = (dict(summary), rows)
    return summary, rows, False


def joint_structure_metrics(
    z1: torch.Tensor,
    z2: torch.Tensor,
    shuffled_z2: torch.Tensor,
    *,
    gamma: float,
    covariance_eps: float,
    projection_diagnostics: int,
    seed: int,
    fixed_rhos: list[float],
    covariance_estimator: str = "oas",
    calibration_fraction: float = 0.5,
    num_null_repetitions: int = 0,
    stats_chunk_size: int | None = None,
    pair_type: str = "",
    estimator: str = "balanced_one_pair_per_image",
    gaussianity_max_samples: int = 1024,
    null_seed: int = 0,
) -> dict[str, float]:
    cal, eva = split_calibration_evaluation(z1.shape[0], calibration_fraction, seed, z1.device)
    observed = structural_metrics_from_calibration(
        z1.index_select(0, cal),
        z2.index_select(0, cal),
        z1.index_select(0, eva),
        z2.index_select(0, eva),
        shuffled_z2.index_select(0, eva),
        gamma=gamma,
        covariance_eps=covariance_eps,
        covariance_estimator=covariance_estimator,
        projection_diagnostics=projection_diagnostics,
        seed=seed,
        fixed_rhos=fixed_rhos,
        stats_chunk_size=stats_chunk_size,
        pair_type=pair_type,
        gaussianity_max_samples=gaussianity_max_samples,
    )
    observed["calibration_image_count"] = int(cal.numel())
    observed["evaluation_image_count"] = int(eva.numel())
    if num_null_repetitions > 0:
        null_summary, _, null_cached = matched_null_rows(
            observed,
            n_cal=int(cal.numel()),
            n_eval=int(eva.numel()),
            dim=z1.shape[1],
            repetitions=num_null_repetitions,
            gamma=gamma,
            covariance_eps=covariance_eps,
            covariance_estimator=covariance_estimator,
            projection_diagnostics=projection_diagnostics,
            fixed_rhos=fixed_rhos,
            seed=null_seed,
            stats_chunk_size=stats_chunk_size,
            estimator=estimator,
            same_view_duplicate=pair_type == "same_view_duplicate",
            gaussianity_max_samples=gaussianity_max_samples,
        )
        observed.update(null_summary)
        observed["null_results_cached"] = null_cached
    strip_internal_metric_tensors(observed)
    return observed


def joint_structure_metrics_from_image_split(
    z_cal_images: torch.Tensor,
    z_eval_images: torch.Tensor,
    pair_type: str,
    num_global: int,
    *,
    gamma: float,
    covariance_eps: float,
    projection_diagnostics: int,
    seed: int,
    fixed_rhos: list[float],
    covariance_estimator: str,
    estimator: str,
    num_null_repetitions: int,
    stats_chunk_size: int,
    gaussianity_max_samples: int = 1024,
    null_seed: int = 0,
) -> tuple[dict[str, float], int, list[tuple[int, int]]]:
    z1_cal, z2_cal, cal_pairs = construct_pairs(
        z_cal_images,
        pair_type,
        num_global,
        shuffled=False,
        seed=seed,
        estimator=estimator,
    )
    z1_eval, z2_eval, eval_pairs = construct_pairs(
        z_eval_images,
        pair_type,
        num_global,
        shuffled=False,
        seed=seed,
        estimator=estimator,
    )
    _, shuffled_eval, _ = construct_pairs(
        z_eval_images,
        pair_type,
        num_global,
        shuffled=True,
        seed=seed + 53,
        estimator=estimator,
    )
    observed = structural_metrics_from_calibration(
        z1_cal,
        z2_cal,
        z1_eval,
        z2_eval,
        shuffled_eval,
        gamma=gamma,
        covariance_eps=covariance_eps,
        covariance_estimator=covariance_estimator,
        projection_diagnostics=projection_diagnostics,
        seed=seed,
        fixed_rhos=fixed_rhos,
        stats_chunk_size=stats_chunk_size,
        pair_type=pair_type,
        gaussianity_max_samples=gaussianity_max_samples,
    )
    observed["calibration_image_count"] = int(z_cal_images.shape[0])
    observed["evaluation_image_count"] = int(z_eval_images.shape[0])
    observed["calibration_source_image_count"] = int(z_cal_images.shape[0])
    observed["evaluation_source_image_count"] = int(z_eval_images.shape[0])
    observed["calibration_pair_observations"] = int(z1_cal.shape[0])
    observed["evaluation_pair_observations"] = int(z1_eval.shape[0])
    observed["actual_stats_chunk_size"] = int(stats_chunk_size)
    observed["calibration_evaluation_image_ids_disjoint"] = True
    if num_null_repetitions > 0:
        null_summary, _, null_cached = matched_null_rows(
            observed,
            n_cal=int(z_cal_images.shape[0]),
            n_eval=int(z_eval_images.shape[0]),
            dim=z1_cal.shape[1],
            repetitions=num_null_repetitions,
            gamma=gamma,
            covariance_eps=covariance_eps,
            covariance_estimator=covariance_estimator,
            projection_diagnostics=projection_diagnostics,
            fixed_rhos=fixed_rhos,
            seed=null_seed,
            stats_chunk_size=stats_chunk_size,
            estimator=estimator,
            same_view_duplicate=pair_type == "same_view_duplicate",
            gaussianity_max_samples=gaussianity_max_samples,
        )
        observed.update(null_summary)
        observed["null_results_cached"] = null_cached
    else:
        observed["null_results_cached"] = False
    strip_internal_metric_tensors(observed)
    return observed, int(z1_eval.shape[0]), eval_pairs or cal_pairs


def all_pairs_sample_and_stats(
    z_bvd: torch.Tensor,
    pair_type: str,
    num_global: int,
    *,
    shuffled: bool,
    seed: int,
    stats_chunk_size: int,
    max_sample_rows: int,
    sample_identities: list[tuple[int, int]] | None = None,
) -> tuple[dict[str, torch.Tensor | int], torch.Tensor, torch.Tensor, int, list[tuple[int, int]], list[tuple[int, int]], str]:
    pairs = ordered_pair_pool(z_bvd.shape[1], num_global, pair_type)
    n_images = z_bvd.shape[0]
    if sample_identities is None:
        identities: list[tuple[int, int]] = []
        per_pair = max(1, math.ceil(max_sample_rows / max(1, len(pairs))))
        generator = torch.Generator(device=z_bvd.device).manual_seed(seed + 31_337)
        for pair_idx in range(len(pairs)):
            if n_images <= per_pair:
                image_ids = torch.arange(n_images, device=z_bvd.device)
            else:
                image_ids = torch.randperm(n_images, generator=generator, device=z_bvd.device)[:per_pair]
                image_ids, _ = torch.sort(image_ids)
            identities.extend((pair_idx, int(image_id)) for image_id in image_ids.detach().cpu().tolist())
        identities = identities[:max_sample_rows]
    else:
        identities = [(int(pair_idx), int(image_idx)) for pair_idx, image_idx in sample_identities]
    identity_set = set(identities)
    identities_json = json.dumps(identities, separators=(",", ":"))
    identity_hash = hashlib.sha1(identities_json.encode()).hexdigest()
    sum_x = torch.zeros(z_bvd.shape[2], device=z_bvd.device, dtype=torch.float64)
    sum_y = torch.zeros(z_bvd.shape[2], device=z_bvd.device, dtype=torch.float64)
    xtx = torch.zeros(z_bvd.shape[2], z_bvd.shape[2], device=z_bvd.device, dtype=torch.float64)
    yty = torch.zeros(z_bvd.shape[2], z_bvd.shape[2], device=z_bvd.device, dtype=torch.float64)
    xty = torch.zeros(z_bvd.shape[2], z_bvd.shape[2], device=z_bvd.device, dtype=torch.float64)
    sample_left: list[torch.Tensor] = []
    sample_right: list[torch.Tensor] = []
    total = 0
    generator = torch.Generator(device=z_bvd.device).manual_seed(seed)
    permutations = {
        pair: torch.randperm(n_images, generator=generator, device=z_bvd.device)
        for pair in pairs
    } if shuffled else {}
    for pair_idx, pair in enumerate(pairs):
        i, j = pair
        perm = permutations.get(pair)
        if perm is not None and n_images > 1 and torch.equal(perm, torch.arange(n_images, device=z_bvd.device)):
            perm = perm.roll(1)
        for start in range(0, n_images, max(1, stats_chunk_size)):
            end = min(start + max(1, stats_chunk_size), n_images)
            left = z_bvd[start:end, i]
            if perm is None:
                right = z_bvd[start:end, j]
            else:
                right = z_bvd.index_select(0, perm[start:end])[:, j]
            chunk_stats = sufficient_statistics(left, right, chunk_size=left.shape[0])
            sum_x += chunk_stats["sum_x"]
            sum_y += chunk_stats["sum_y"]
            xtx += chunk_stats["xtx"]
            yty += chunk_stats["yty"]
            xty += chunk_stats["xty"]
            local_indices = [
                image_idx - start
                for selected_pair_idx, image_idx in identity_set
                if selected_pair_idx == pair_idx and start <= image_idx < end
            ]
            if local_indices:
                idx = torch.tensor(local_indices, device=z_bvd.device, dtype=torch.long)
                sample_left.append(left.index_select(0, idx))
                sample_right.append(right.index_select(0, idx))
            total += left.shape[0]
    if sample_left:
        left_sample = torch.cat(sample_left, dim=0)
        right_sample = torch.cat(sample_right, dim=0)
    else:
        left_sample = z_bvd.new_empty((0, z_bvd.shape[2]))
        right_sample = z_bvd.new_empty((0, z_bvd.shape[2]))
    return (
        {"n": total, "sum_x": sum_x, "sum_y": sum_y, "xtx": xtx, "yty": yty, "xty": xty},
        left_sample,
        right_sample,
        total,
        pairs,
        identities,
        identity_hash,
    )


def joint_structure_all_pairs_chunked_from_image_split(
    z_cal_images: torch.Tensor,
    z_eval_images: torch.Tensor,
    pair_type: str,
    num_global: int,
    *,
    gamma: float,
    covariance_eps: float,
    projection_diagnostics: int,
    seed: int,
    fixed_rhos: list[float],
    covariance_estimator: str,
    stats_chunk_size: int,
    gaussianity_max_samples: int,
) -> tuple[dict[str, float], int, list[tuple[int, int]]]:
    max_sample_rows = max(gaussianity_max_samples, 2)
    cal_stats, z1_cal, z2_cal, cal_total, pairs, cal_sample_ids, cal_sample_hash = all_pairs_sample_and_stats(
        z_cal_images,
        pair_type,
        num_global,
        shuffled=False,
        seed=seed,
        stats_chunk_size=stats_chunk_size,
        max_sample_rows=max_sample_rows,
    )
    _, z1_eval, z2_eval, eval_total, _, eval_sample_ids, eval_sample_hash = all_pairs_sample_and_stats(
        z_eval_images,
        pair_type,
        num_global,
        shuffled=False,
        seed=seed,
        stats_chunk_size=stats_chunk_size,
        max_sample_rows=max_sample_rows,
    )
    _, _, shuffled_eval, _, _, shuffled_sample_ids, shuffled_sample_hash = all_pairs_sample_and_stats(
        z_eval_images,
        pair_type,
        num_global,
        shuffled=True,
        seed=seed + 53,
        stats_chunk_size=stats_chunk_size,
        max_sample_rows=max_sample_rows,
        sample_identities=eval_sample_ids,
    )
    observed = structural_metrics_from_calibration(
        z1_cal,
        z2_cal,
        z1_eval,
        z2_eval,
        shuffled_eval,
        gamma=gamma,
        covariance_eps=covariance_eps,
        covariance_estimator=covariance_estimator,
        projection_diagnostics=projection_diagnostics,
        seed=seed,
        fixed_rhos=fixed_rhos,
        stats_chunk_size=stats_chunk_size,
        pair_type=pair_type,
        gaussianity_max_samples=gaussianity_max_samples,
        calibration_stats=cal_stats,
    )
    observed["calibration_image_count"] = int(z_cal_images.shape[0])
    observed["evaluation_image_count"] = int(z_eval_images.shape[0])
    observed["calibration_source_image_count"] = int(z_cal_images.shape[0])
    observed["evaluation_source_image_count"] = int(z_eval_images.shape[0])
    observed["calibration_pair_observations"] = int(cal_total)
    observed["evaluation_pair_observations"] = int(eval_total)
    observed["actual_stats_chunk_size"] = int(stats_chunk_size)
    observed["calibration_evaluation_image_ids_disjoint"] = True
    observed["null_results_cached"] = False
    observed["statistics_accumulated_incrementally"] = True
    observed["all_pairs_computation"] = "chunked"
    observed["diagnostic_sample_identity_hash"] = eval_sample_hash
    observed["calibration_diagnostic_sample_identity_hash"] = cal_sample_hash
    observed["shuffled_diagnostic_sample_identity_hash"] = shuffled_sample_hash
    observed["diagnostic_sample_identities"] = json.dumps(eval_sample_ids)
    observed["shuffled_diagnostic_sample_aligned"] = eval_sample_ids == shuffled_sample_ids
    strip_internal_metric_tensors(observed)
    return observed, int(eval_total), pairs


def joint_structure_estimators(include_all_pairs: bool) -> list[str]:
    return ["balanced_one_pair_per_image"] + (["all_pairs_clustered"] if include_all_pairs else [])


def should_run_joint_null(
    *,
    estimator: str,
    analysis_space: str,
    null_full_space: bool,
    null_scope: str,
    partition_id: int,
    augmentation_id: int,
    pair_type: str,
) -> bool:
    if estimator != "balanced_one_pair_per_image":
        return False
    if analysis_space != "projected" and not null_full_space:
        return False
    if null_scope == "all":
        return True
    if null_scope != "representative":
        raise ValueError(f"Unknown null_scope={null_scope!r}")
    return (
        analysis_space == "projected"
        and partition_id == 0
        and augmentation_id == 0
        and pair_type in {"global_global", "global_local", "local_local", "same_view_duplicate"}
    )


def run_joint_structure_for_checkpoint(
    args: argparse.Namespace,
    spec: CheckpointSpec,
    all_specs: list[CheckpointSpec],
    embeddings: torch.Tensor,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    progress = training_progress(spec, all_specs)
    num_augs, num_images, cached_views, original_dim = embeddings.shape
    if not hasattr(args, "resolved_analysis_projection_dim"):
        args.resolved_analysis_projection_dim = resolve_fixed_projection_dim(
            args.analysis_projection_dim,
            image_counts=args.image_counts,
            available_images=num_images,
            calibration_fraction=args.calibration_fraction,
            original_dim=original_dim,
        )
        print(f"[joint] fixed projected analysis dimension: {args.resolved_analysis_projection_dim}", flush=True)
    pair_types = ["global_global", "global_local", "local_local", "same_view_duplicate"]
    analysis_spaces = ["projected"] + (["full"] if args.include_full_space else [])
    largest_count = min(max(int(c) for c in args.image_counts), num_images)
    for augmentation_id in range(num_augs):
        for partition_id in range(args.num_partitions):
            perm = torch.randperm(num_images, generator=torch.Generator().manual_seed(args.seed + 1009 * partition_id))
            for image_count in args.image_counts:
                count = min(int(image_count), num_images)
                selected = perm[:count]
                z_all = embeddings[augmentation_id].index_select(0, selected)
                selection = view_selection(cached_views, min(max(args.view_counts), cached_views))
                z_all = z_all.index_select(1, torch.tensor(selection.indices))
                cal_rel, eval_rel = split_calibration_evaluation(
                    count,
                    args.calibration_fraction,
                    args.seed + 20_011 * partition_id + 101 * augmentation_id,
                    z_all.device,
                )
                cal_source_ids = selected.index_select(0, cal_rel.cpu())
                eval_source_ids = selected.index_select(0, eval_rel.cpu())
                z_cal_base = z_all.index_select(0, cal_rel)
                z_eval_base = z_all.index_select(0, eval_rel)
                for pair_type in pair_types:
                    if pair_type == "local_local" and selection.num_local < 2:
                        continue
                    if pair_type == "global_local" and selection.num_local < 1:
                        continue
                    for estimator in joint_structure_estimators(args.include_all_pairs):
                        for analysis_space in analysis_spaces:
                            if analysis_space == "full" and (
                                count != largest_count
                                or estimator != "balanced_one_pair_per_image"
                                or pair_type == "same_view_duplicate" and not args.include_same_view_full_space
                                or partition_id != 0
                                or augmentation_id != 0
                            ):
                                continue
                            if (
                                estimator == "all_pairs_clustered"
                                and analysis_space == "full"
                                and not args.include_all_pairs_full_space
                            ):
                                continue
                            projection_dim = args.resolved_analysis_projection_dim
                            z_cal, analysis_dim = apply_analysis_space_bvd(
                                z_cal_base,
                                analysis_space=analysis_space,
                                projection_dim=projection_dim,
                                seed=args.seed,
                            )
                            z_eval, _ = apply_analysis_space_bvd(
                                z_eval_base,
                                analysis_space=analysis_space,
                                projection_dim=projection_dim,
                                seed=args.seed,
                            )
                            run_null = should_run_joint_null(
                                estimator=estimator,
                                analysis_space=analysis_space,
                                null_full_space=args.null_full_space,
                                null_scope=args.null_scope,
                                partition_id=partition_id,
                                augmentation_id=augmentation_id,
                                pair_type=pair_type,
                            )
                            if estimator == "all_pairs_clustered":
                                metrics, pair_observations, pairs = joint_structure_all_pairs_chunked_from_image_split(
                                    z_cal,
                                    z_eval,
                                    pair_type,
                                    selection.num_global,
                                    gamma=args.gamma,
                                    covariance_eps=args.covariance_eps,
                                    projection_diagnostics=args.projection_diagnostics,
                                    seed=args.seed + partition_id + 10_003 * augmentation_id,
                                    fixed_rhos=args.fixed_rhos,
                                    covariance_estimator=args.covariance_estimator,
                                    stats_chunk_size=args.stats_chunk_size,
                                    gaussianity_max_samples=args.gaussianity_max_samples,
                                )
                            else:
                                metrics, pair_observations, pairs = joint_structure_metrics_from_image_split(
                                    z_cal,
                                    z_eval,
                                    pair_type,
                                    selection.num_global,
                                    gamma=args.gamma,
                                    covariance_eps=args.covariance_eps,
                                    projection_diagnostics=args.projection_diagnostics,
                                    seed=args.seed + partition_id + 10_003 * augmentation_id,
                                    fixed_rhos=args.fixed_rhos,
                                    covariance_estimator=args.covariance_estimator,
                                    estimator=estimator,
                                    num_null_repetitions=args.num_null_repetitions if run_null else 0,
                                    stats_chunk_size=args.stats_chunk_size,
                                    gaussianity_max_samples=args.gaussianity_max_samples,
                                    null_seed=args.null_seed,
                                )
                            sanity = ""
                            if pair_type == "same_view_duplicate":
                                residual_ratio = metrics.get(
                                    "scalar_model_relative_residual_observed_to_null_median",
                                    float("nan"),
                                )
                                if not math.isfinite(residual_ratio):
                                    sanity = "not_calibrated"
                                else:
                                    sanity = "pass" if residual_ratio < 2.5 else "warn_covariance_estimator"
                            rows.append(
                                {
                                    "checkpoint": spec.label,
                                    "checkpoint_path": str(spec.path) if spec.path is not None else "",
                                    "training_progress": progress,
                                    "augmentation_id": augmentation_id,
                                    "partition_id": partition_id,
                                    "image_count": count,
                                    "num_images_used": count,
                                    "num_independent_image_groups": count,
                                    "num_pair_observations": pair_observations,
                                    "calibration_source_image_count": int(cal_source_ids.numel()),
                                    "evaluation_source_image_count": int(eval_source_ids.numel()),
                                    "calibration_source_image_ids": json.dumps([int(x) for x in cal_source_ids.tolist()]),
                                    "evaluation_source_image_ids": json.dumps([int(x) for x in eval_source_ids.tolist()]),
                                    "pair_type": pair_type,
                                    "pair_direction": (
                                        "all_pairs_symmetric_ordered"
                                        if estimator == "all_pairs_clustered"
                                        else "balanced_symmetric_ordered"
                                    ),
                                    "estimator": estimator,
                                    "analysis_space": analysis_space,
                                    "original_dimension": original_dim,
                                    "analysis_dimension": analysis_dim,
                                    "resolved_analysis_projection_dim": args.resolved_analysis_projection_dim,
                                    "stats_chunk_size": args.stats_chunk_size,
                                    "statistics_accumulated_incrementally": bool(
                                        metrics.get("statistics_accumulated_incrementally", False)
                                    ),
                                    "all_pairs_computation": metrics.get("all_pairs_computation", ""),
                                    "null_scope": args.null_scope,
                                    "null_seed": args.null_seed,
                                    "shrinkage_method": args.covariance_estimator,
                                    "effective_dimension": analysis_dim,
                                    "num_views": selection.num_views,
                                    "num_global_views": selection.num_global,
                                    "num_local_views": selection.num_local,
                                    "selected_view_pairs": json.dumps(pairs),
                                    "same_view_sanity_status": sanity,
                                    **metrics,
                                }
                            )
                            print(
                                f"[joint] {spec.label} aug={augmentation_id} part={partition_id} "
                                f"N={count} pair={pair_type} estimator={estimator} space={analysis_space}",
                                flush=True,
                            )
    return rows


def aggregate_joint_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metrics = [
        "rho_hat",
        "rho_hat_unconstrained",
        "rho_used_for_energy",
        "scalar_model_relative_residual",
        "explained_rho_i",
        "explained_diagonal",
        "explained_rank_1",
        "explained_rank_4",
        "explained_rank_16",
        "explained_rank_32",
        "relative_offdiagonal_energy",
        "antisymmetric_energy",
        "singular_value_dispersion",
        "cross_correlation_effective_rank",
        "top_singular_value",
        "plus_isotropy_error",
        "minus_isotropy_error",
        "raw_plus_isotropy_error",
        "raw_minus_isotropy_error",
        "plus_effective_rank",
        "minus_effective_rank",
        "plus_minus_crosscov_norm",
        "plus_minus_variance_ratio",
        "unscaled_cw_normality_plus",
        "unscaled_cw_normality_minus",
        "target_fit_cw_plus",
        "target_fit_cw_minus",
        "shape_cw_plus",
        "shape_cw_minus",
        "raw_unscaled_cw_normality_plus",
        "raw_unscaled_cw_normality_minus",
        "gaussianity_num_samples",
        "projection_skewness_plus",
        "projection_kurtosis_plus",
        "projection_skewness_minus",
        "projection_kurtosis_minus",
        "mean_pair_cosine",
        "median_pair_cosine",
        "mean_pair_distance",
        "median_pair_distance",
        "heldout_true_pair_energy",
        "heldout_shuffled_pair_energy",
        "energy_auc",
        "cosine_auc",
        "distance_auc",
        "whitened_cosine_auc",
        "z1_isotropy_error",
        "z2_isotropy_error",
        "shrinkage_coefficient_C11",
        "shrinkage_coefficient_C22",
        "scalar_model_relative_residual_observed_to_null_median",
        "relative_offdiagonal_energy_observed_to_null_median",
        "antisymmetric_energy_observed_to_null_median",
        "plus_isotropy_error_observed_to_null_median",
        "minus_isotropy_error_observed_to_null_median",
    ]
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = (
            row["checkpoint"],
            row["training_progress"],
            row["pair_type"],
            row["estimator"],
            row["analysis_space"],
            row["analysis_dimension"],
            row["image_count"],
        )
        groups.setdefault(key, []).append(row)
    out: list[dict[str, Any]] = []
    for (
        checkpoint,
        progress,
        pair_type,
        estimator,
        analysis_space,
        analysis_dimension,
        image_count,
    ), group_rows in groups.items():
        row: dict[str, Any] = {
            "checkpoint": checkpoint,
            "training_progress": progress,
            "pair_type": pair_type,
            "estimator": estimator,
            "analysis_space": analysis_space,
            "analysis_dimension": analysis_dimension,
            "image_count": image_count,
            "count": len(group_rows),
        }
        for metric in metrics:
            vals = [float(r[metric]) for r in group_rows if r.get(metric) not in ("", None)]
            clean = [v for v in vals if math.isfinite(v)]
            row[f"{metric}_mean"] = float(np.mean(clean)) if clean else float("nan")
            row[f"{metric}_std"] = float(np.std(clean, ddof=1)) if len(clean) > 1 else 0.0
            row[f"{metric}_median"] = quantile(clean, 0.5)
            row[f"{metric}_iqr"] = quantile(clean, 0.75) - quantile(clean, 0.25)
            row[f"{metric}_q05"] = quantile(clean, 0.05)
            row[f"{metric}_q95"] = quantile(clean, 0.95)
        out.append(row)
    return out


def plot_joint_metric(rows: list[dict[str, Any]], output_dir: Path, metric: str, ylabel: str, filename: str) -> None:
    import matplotlib.pyplot as plt

    set_plot_theme()
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    default_count = max(int(r["image_count"]) for r in rows)
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if int(row["image_count"]) != default_count:
            continue
        if row.get("estimator") != "balanced_one_pair_per_image" or row.get("analysis_space") != "projected":
            continue
        if row["pair_type"] == "same_view_duplicate":
            continue
        grouped.setdefault(str(row["pair_type"]), []).append(row)
    for label, group in sorted(grouped.items()):
        group = sorted(group, key=lambda r: float(r["training_progress"]))
        x = np.asarray([float(r["training_progress"]) for r in group])
        y = np.asarray([float(r[f"{metric}_mean"]) for r in group])
        err = np.asarray([float(r[f"{metric}_std"]) for r in group])
        ax.plot(x, y, marker="o", label=label)
        ax.fill_between(x, y - err, y + err, alpha=0.15)
    ax.set_xlabel("training progress (%)")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / f"{filename}.pdf")
    plt.close(fig)


def joint_stability_rows(rows: list[dict[str, Any]], analysis_space: str = "projected") -> list[dict[str, Any]]:
    return [
        r for r in rows
        if r.get("estimator") == "balanced_one_pair_per_image"
        and r.get("analysis_space") == analysis_space
        and r.get("pair_type") != "same_view_duplicate"
    ]


def plot_joint_stability(
    rows: list[dict[str, Any]],
    output_dir: Path,
    *,
    analysis_space: str = "projected",
    filename: str = "joint_structure_stability",
) -> None:
    import matplotlib.pyplot as plt

    set_plot_theme()
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8))
    space_rows = joint_stability_rows(rows, analysis_space)
    if not space_rows:
        plt.close(fig)
        return
    latest_progress = max(float(r["training_progress"]) for r in space_rows if math.isfinite(float(r["training_progress"])))
    for x_key, ax in [("image_count", axes[0]), ("analysis_dimension", axes[1])]:
        for pair_type in ["global_global", "global_local", "local_local"]:
            subset = [
                r for r in space_rows
                if r["pair_type"] == pair_type
                and abs(float(r["training_progress"]) - latest_progress) < 1e-8
            ]
            grouped: dict[int, list[float]] = {}
            for row in subset:
                grouped.setdefault(int(row[x_key]), []).append(float(row["scalar_model_relative_residual_mean"]))
            xs = sorted(grouped)
            if not xs:
                continue
            ys = [float(np.mean(grouped[x])) for x in xs]
            ax.plot(xs, ys, marker="o", label=pair_type)
        ax.set_xlabel(x_key)
        ax.set_ylabel(f"{analysis_space} rho-I residual")
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / f"{filename}.pdf")
    plt.close(fig)


def write_joint_summary(output_dir: Path, aggregate: list[dict[str, Any]], gradient_aggregate_path: Path | None = None) -> None:
    lines = [
        "# Joint-structure observational summary",
        "",
        "This analysis characterizes cached detached LeJEPA projector outputs. It does not implement Joint-CW, change training, optimize rho, or evaluate downstream training behavior.",
        "",
    ]
    default_rows = [
        r for r in aggregate
        if r["pair_type"] in {"global_global", "global_local", "local_local"}
        and r.get("estimator") == "balanced_one_pair_per_image"
        and r.get("analysis_space") == "projected"
        and int(r["image_count"]) == max(int(x["image_count"]) for x in aggregate)
    ]
    if default_rows:
        latest_progress = max(float(r["training_progress"]) for r in default_rows)
        lines.append("Latest projected primary estimates by pair type:")
        for row in sorted([r for r in default_rows if float(r["training_progress"]) == latest_progress], key=lambda r: r["pair_type"]):
            residual = float(row["scalar_model_relative_residual_median"])
            null_ratio = float(row.get("scalar_model_relative_residual_observed_to_null_median_median", float("nan")))
            rho = float(row["rho_hat_median"])
            diag_gain = float(row["explained_diagonal_median"]) - float(row["explained_rho_i_median"])
            auc = float(row["energy_auc_median"])
            cosine_auc = float(row["cosine_auc_median"])
            lines.append(
                f"- {row['pair_type']}: rho={rho:.4g}, residual={residual:.4g}, "
                f"residual/null={null_ratio:.4g}, diagonal-minus-scalar explained={diag_gain:.4g}, "
                f"energy AUC={auc:.4g}, cosine AUC={cosine_auc:.4g}."
            )
        lines.extend(
            [
                "",
                "Interpretation is calibrated against matched finite-sample nulls where available. "
                "Use pair-type-specific rows rather than pooled rows for model assessment.",
            ]
        )
    else:
        lines.append("No default pooled positive-pair rows were available for an automatic summary.")
    if gradient_aggregate_path is not None and gradient_aggregate_path.exists():
        lines.append("")
        lines.append(f"Existing group-aware gradient diagnostics can be joined from: {gradient_aggregate_path}")
    (output_dir / "joint_structure_summary.md").write_text("\n".join(lines) + "\n")


def run_joint_structure(args: argparse.Namespace, specs: list[CheckpointSpec], dataset: Dataset, cache_dir: Path) -> None:
    print("[inspection] joint-structure reuses cached embeddings [augmentation,image,view,feature]", flush=True)
    print("[inspection] view order is global_1, global_2, then local_1..local_8", flush=True)
    print("[inspection] projector output is the same tensor used by LeJEPA MSE and flattened CWReg", flush=True)
    all_rows: list[dict[str, Any]] = []
    manifest: list[dict[str, Any]] = []
    for spec in specs:
        cache_path = cache_dir / cache_key(args, spec)
        embeddings, metadata = extract_projected_embeddings(args, spec, dataset, cache_path)
        manifest.append(metadata)
        all_rows.extend(run_joint_structure_for_checkpoint(args, spec, specs, embeddings))

    aggregate = aggregate_joint_rows(all_rows)
    primary_rows = [
        row for row in all_rows if row.get("estimator") == "balanced_one_pair_per_image"
    ]
    all_pair_rows = [row for row in all_rows if row.get("estimator") == "all_pairs_clustered"]
    null_rows = []
    for row in primary_rows:
        null_row = {
            key: value
            for key, value in row.items()
            if "_null_" in key
            or key.endswith("_observed_to_null_median")
            or key.endswith("_calibrated_z")
            or key.endswith("_null_tail_probability")
        }
        null_row.update(
            {
                "checkpoint": row["checkpoint"],
                "training_progress": row["training_progress"],
                "augmentation_id": row["augmentation_id"],
                "partition_id": row["partition_id"],
                "image_count": row["image_count"],
                "pair_type": row["pair_type"],
                "analysis_space": row["analysis_space"],
                "estimator": row["estimator"],
                "rho_hat": row["rho_hat"],
                "rho_hat_unconstrained": row.get("rho_hat_unconstrained", row["rho_hat"]),
                "rho_used_for_energy": row.get("rho_used_for_energy", ""),
                "rho_was_clipped": row.get("rho_was_clipped", ""),
                "null_results_cached": row.get("null_results_cached", False),
            }
        )
        null_rows.append(null_row)
    write_csv(args.output_dir / "joint_structure_primary_raw.csv", primary_rows)
    write_csv(args.output_dir / "joint_structure_primary_aggregate.csv", aggregate_joint_rows(primary_rows))
    write_csv(args.output_dir / "joint_structure_all_pairs_raw.csv", all_pair_rows)
    write_csv(args.output_dir / "joint_structure_null_calibration.csv", null_rows)
    (args.output_dir / "embedding_cache_manifest.json").write_text(json.dumps(manifest, indent=2))
    (args.output_dir / "run_config.json").write_text(json.dumps(vars(args), indent=2, default=str))
    if aggregate:
        plot_joint_metric(aggregate, args.output_dir, "rho_hat", "rho_hat", "joint_rho_vs_training")
        plot_joint_metric(aggregate, args.output_dir, "scalar_model_relative_residual", "||R-rho I||/||R||", "joint_scalar_residual_vs_training")
        plot_joint_metric(aggregate, args.output_dir, "relative_offdiagonal_energy", "off-diagonal energy", "joint_offdiagonal_energy_vs_training")
        plot_joint_metric(aggregate, args.output_dir, "cross_correlation_effective_rank", "effective rank(R)", "joint_effective_rank_vs_training")
        plot_joint_metric(aggregate, args.output_dir, "unscaled_cw_normality_minus", "unscaled CW normality(z_minus)", "joint_minus_gaussianity_vs_training")
        plot_joint_metric(aggregate, args.output_dir, "energy_auc", "true-vs-shuffled energy AUC", "joint_true_shuffled_auc_vs_training")
        plot_joint_metric(aggregate, args.output_dir, "median_pair_cosine", "median pair cosine", "joint_pair_cosine_vs_training")
        plot_joint_stability(aggregate, args.output_dir)
        if any(row.get("analysis_space") == "full" for row in aggregate):
            plot_joint_stability(
                aggregate,
                args.output_dir,
                analysis_space="full",
                filename="joint_structure_stability_full_space",
            )
        write_joint_summary(args.output_dir, aggregate, args.output_dir / "multiview_cw_gradients_aggregate.csv")


def main() -> None:
    args = parse_args()
    apply_debug_defaults(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.cache_dir or (args.output_dir / "embedding_cache")
    specs = checkpoint_specs(args)
    dataset = deterministic_subset(build_dataset(args), args.num_images, args.seed)

    print("[inspection] production CWReg: stable_pretraining.methods.lejepa.CWReg", flush=True)
    print("[inspection] production EP/SIGReg: stable_pretraining.methods.lejepa.SlicedEppsPulley", flush=True)
    print("[inspection] LeJEPA alignment: centers = mean(first global views); MSE to all views", flush=True)
    print("[inspection] regularizer tensor: projector outputs [V,B,D] flattened view-major to [B*V,D]", flush=True)

    if args.command == "joint-structure":
        run_joint_structure(args, specs, dataset, cache_dir)
        return

    all_rows: list[dict[str, Any]] = []
    manifest: list[dict[str, Any]] = []
    for spec in specs:
        cache_path = cache_dir / cache_key(args, spec)
        embeddings, metadata = extract_projected_embeddings(args, spec, dataset, cache_path)
        manifest.append(metadata)
        all_rows.extend(run_analysis_for_checkpoint(args, spec, specs, embeddings))

    aggregate = aggregate_rows(all_rows)
    write_csv(args.output_dir / "multiview_cw_gradients_raw.csv", all_rows)
    write_csv(args.output_dir / "multiview_cw_gradients_aggregate.csv", aggregate)
    (args.output_dir / "embedding_cache_manifest.json").write_text(json.dumps(manifest, indent=2))
    (args.output_dir / "run_config.json").write_text(json.dumps(vars(args), indent=2, default=str))

    if aggregate:
        plot_metric(
            aggregate,
            args.output_dir,
            "within_fraction_of_cw_gradient",
            "||g_within|| / ||g_cw||",
            "within_gradient_share_vs_training",
        )
        plot_metric(
            aggregate,
            args.output_dir,
            "cosine_within_alignment",
            "cosine(within, alignment)",
            "within_alignment_cosine_vs_training",
        )
        plot_metric(
            aggregate,
            args.output_dir,
            "relative_full_group_gradient_difference",
            "||g_cw - g_group|| / ||g_cw||",
            "ordinary_vs_group_aware_cw",
        )
        plot_sensitivity(aggregate, args.output_dir, "num_views", "view_count_sensitivity")
        plot_sensitivity(aggregate, args.output_dir, "batch_size", "batch_size_sensitivity")
        write_summary_report(args.output_dir, aggregate)


if __name__ == "__main__":
    main()
