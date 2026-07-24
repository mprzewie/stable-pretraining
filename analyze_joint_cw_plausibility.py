#!/usr/bin/env python3
"""Observational plausibility analysis for a scalar-rho positive-view model.

This script analyzes cached LeJEPA projector embeddings without defining a
Joint-CW objective, changing training, or computing training gradients.
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
from typing import Any, Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from cw_torch.metric import cw_normality

EPS = 1e-12
PAIR_TYPES = ("global_global", "global_local", "local_local")
STRUCTURAL_NULL_METRICS = (
    "scalar_residual",
    "offdiagonal_ratio",
    "antisymmetric_ratio",
    "plus_isotropy_error",
    "minus_isotropy_error",
    "plus_minus_crosscov",
)
SHAPE_NULL_METRICS = (
    "shape_cw_plus",
    "shape_cw_minus",
)
NULL_METRICS = STRUCTURAL_NULL_METRICS + SHAPE_NULL_METRICS
MAX_CW_SAMPLES_SEEN = 0


@dataclass
class CacheRecord:
    label: str
    path: Path
    embeddings: torch.Tensor
    metadata: dict[str, Any]
    training_progress: float


@dataclass(frozen=True)
class CheckpointSpec:
    label: str
    path: Path | None
    hparams: dict[str, Any]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test whether positive-view embeddings are structurally compatible with R ~= rho I."
    )
    source = parser.add_argument_group("embedding source")
    source.add_argument("--cache-dir", type=Path)
    source.add_argument("--cache-file", action="append", type=Path, default=[])
    source.add_argument("--checkpoint-dir", type=Path)
    source.add_argument("--checkpoint", action="append", type=Path, default=[])
    source.add_argument("--checkpoint-label", action="append", default=[])
    source.add_argument("--max-checkpoints", type=int, default=3)
    source.add_argument("--include-untrained", action=argparse.BooleanOptionalAction, default=True)

    analysis = parser.add_argument_group("analysis")
    analysis.add_argument(
        "--analysis-mode",
        choices=["structure", "gradients", "both"],
        default="structure",
    )
    analysis.add_argument("--num-images", type=int, default=5000)
    analysis.add_argument("--image-counts", type=int, nargs="+", default=[1000, 5000])
    analysis.add_argument("--projection-dim", type=int, default=128)
    analysis.add_argument("--num-partitions", type=int, default=3)
    analysis.add_argument("--calibration-fraction", type=float, default=0.5)
    analysis.add_argument("--ridge-eps", type=float, default=1e-4)
    analysis.add_argument("--cw-gamma", type=float, default=0.5)
    analysis.add_argument("--gaussianity-max-samples", type=int, default=1024)
    analysis.add_argument("--num-shape-directions", type=int, default=64)
    analysis.add_argument("--num-structural-null-repetitions", type=int, default=None)
    analysis.add_argument("--num-shape-null-repetitions", type=int, default=None)
    analysis.add_argument(
        "--num-null-repetitions",
        type=int,
        default=None,
        help="deprecated alias setting both structural and shape null repetitions",
    )
    analysis.add_argument("--seed", type=int, default=0)
    analysis.add_argument("--null-seed", type=int, default=1729)
    analysis.add_argument("--output-dir", type=Path, required=True)
    analysis.add_argument("--debug", action="store_true")

    gradients = parser.add_argument_group("embedding-gradient analysis")
    gradients.add_argument("--gradient-batch-size", type=int, default=128)
    gradients.add_argument("--gradient-num-batches", type=int, default=10)
    gradients.add_argument("--gradient-cw-lambda", type=float, default=0.02)
    gradients.add_argument("--gradient-finite-step", type=float, default=0.001)
    gradients.add_argument("--gradient-include-ep-control", action="store_true")
    gradients.add_argument("--gradient-ep-lambda", type=float, default=0.05)
    gradients.add_argument("--gradient-ep-slices", type=int, default=1024)

    extraction = parser.add_argument_group("optional checkpoint extraction")
    extraction.add_argument("--dataset-root", type=Path)
    extraction.add_argument("--dataset-name", default="clane9/imagenet-100")
    extraction.add_argument("--dataset-split", choices=["train", "validation", "val"], default="validation")
    extraction.add_argument("--dataset-revision")
    extraction.add_argument("--dataset-cache-dir", type=Path)
    extraction.add_argument("--image-batch-size", type=int, default=128)
    extraction.add_argument("--num-augmentation-draws", type=int, default=3)
    extraction.add_argument("--num-workers", type=int, default=8)
    extraction.add_argument("--device", choices=["cpu", "cuda"], default="cuda" if torch.cuda.is_available() else "cpu")
    extraction.add_argument("--encoder-name")
    extraction.add_argument("--drop-path-rate", type=float)
    extraction.add_argument("--local-crop-size", type=int, default=96)
    extraction.add_argument("--num-local-views", type=int)
    extraction.add_argument("--model-sigreg", choices=["ep", "cw"], default="ep")
    extraction.add_argument("--model-mode", choices=["train", "eval"], default="train")

    args = parser.parse_args(argv)
    alias_repetitions = args.num_null_repetitions
    args.num_structural_null_repetitions = (
        alias_repetitions
        if args.num_structural_null_repetitions is None and alias_repetitions is not None
        else args.num_structural_null_repetitions
    )
    args.num_shape_null_repetitions = (
        alias_repetitions
        if args.num_shape_null_repetitions is None and alias_repetitions is not None
        else args.num_shape_null_repetitions
    )
    args.num_structural_null_repetitions = args.num_structural_null_repetitions or 100
    args.num_shape_null_repetitions = args.num_shape_null_repetitions or 100
    if args.debug:
        args.num_images = min(args.num_images, 64)
        args.image_counts = [n for n in args.image_counts if n <= args.num_images] or [args.num_images]
        args.projection_dim = min(args.projection_dim, 16)
        args.num_partitions = 1
        args.num_augmentation_draws = 1
        args.num_structural_null_repetitions = min(args.num_structural_null_repetitions, 3)
        args.num_shape_null_repetitions = min(args.num_shape_null_repetitions, 3)
        args.num_shape_directions = min(args.num_shape_directions, 8)
        args.gaussianity_max_samples = min(args.gaussianity_max_samples, 64)
        args.num_workers = 0
        args.gradient_batch_size = min(args.gradient_batch_size, 8)
        args.gradient_num_batches = min(args.gradient_num_batches, 2)
        args.gradient_ep_slices = min(args.gradient_ep_slices, 16)
    if not 0.0 < args.calibration_fraction < 1.0:
        parser.error("--calibration-fraction must be in (0, 1)")
    if args.gaussianity_max_samples < 2:
        parser.error("--gaussianity-max-samples must be at least 2")
    if args.num_local_views is not None and args.num_local_views < 0:
        parser.error("--num-local-views must be non-negative")
    if not (args.cache_dir or args.cache_file or args.checkpoint_dir or args.checkpoint):
        parser.error("provide embedding caches or checkpoints")
    return args


def stable_seed(*parts: Any) -> int:
    digest = hashlib.sha1("|".join(map(str, parts)).encode()).digest()
    return int.from_bytes(digest[:8], "little") % (2**31)


def slugify(value: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip()).strip("._-")
    return clean or "checkpoint"


def json_compatible(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(key): json_compatible(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_compatible(child) for child in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def epoch_from_text(value: str) -> int | None:
    match = re.search(r"epoch[=_-]?(\d+)", value)
    return int(match.group(1)) if match else None


def cache_payload(path: Path) -> tuple[torch.Tensor, dict[str, Any]] | None:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return None
    if not isinstance(payload, dict) or "embeddings" not in payload:
        return None
    embeddings = payload["embeddings"]
    if not torch.is_tensor(embeddings) or embeddings.ndim != 4:
        return None
    metadata = payload.get("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}
    return embeddings.float().contiguous(), metadata


def discover_cache_records(args: argparse.Namespace) -> list[CacheRecord]:
    explicit = [path.expanduser().resolve() for path in args.cache_file]
    discovered: list[Path] = []
    if args.cache_dir:
        discovered = sorted(args.cache_dir.expanduser().resolve().rglob("*.pt"))
    paths = list(dict.fromkeys(explicit + discovered))
    records: list[CacheRecord] = []
    for path in paths:
        loaded = cache_payload(path)
        if loaded is None:
            continue
        embeddings, metadata = loaded
        label = slugify(str(metadata.get("checkpoint", path.stem)))
        progress = metadata.get("training_progress", float("nan"))
        try:
            progress = float(progress)
        except (TypeError, ValueError):
            progress = float("nan")
        records.append(CacheRecord(label, path, embeddings, metadata, progress))
    infer_training_progress(records)
    if explicit:
        return records
    return select_initial_records(records, args.max_checkpoints)


def infer_training_progress(records: list[CacheRecord]) -> None:
    epochs: dict[int, int] = {}
    for index, record in enumerate(records):
        if math.isfinite(record.training_progress):
            continue
        if record.label.lower() == "untrained" or not record.metadata.get("checkpoint_path"):
            record.training_progress = 0.0
            continue
        epoch = epoch_from_text(str(record.metadata.get("checkpoint_path", record.path.name)))
        if epoch is not None:
            epochs[index] = epoch
    if epochs:
        final_epoch = max(epochs.values()) + 1
        for index, epoch in epochs.items():
            records[index].training_progress = 100.0 * (epoch + 1) / final_epoch


def record_sort_key(record: CacheRecord) -> tuple[float, str]:
    progress = record.training_progress
    if not math.isfinite(progress):
        epoch = epoch_from_text(str(record.metadata.get("checkpoint_path", record.path.name)))
        progress = float(epoch if epoch is not None else math.inf)
    return progress, record.label


def select_initial_records(records: list[CacheRecord], max_count: int) -> list[CacheRecord]:
    if max_count <= 0 or len(records) <= max_count:
        return sorted(records, key=record_sort_key)
    untrained = [record for record in records if record.label.lower() == "untrained" or record.training_progress == 0.0]
    trained = sorted([record for record in records if record not in untrained], key=record_sort_key)
    selected: list[CacheRecord] = []
    if untrained:
        selected.append(sorted(untrained, key=lambda record: record.label)[0])
    remaining = max_count - len(selected)
    if remaining > 0 and trained:
        if remaining == 2 and len(trained) > 1:
            indices = [len(trained) // 2, len(trained) - 1]
        else:
            indices = sorted({int(round(i)) for i in np.linspace(0, len(trained) - 1, remaining)})
        selected.extend(trained[index] for index in indices)
    return selected[:max_count]


def read_checkpoint_hparams(path: Path) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    hparams = checkpoint.get("hyper_parameters", {}) if isinstance(checkpoint, dict) else {}
    return hparams if isinstance(hparams, dict) else {}


def checkpoint_specs(args: argparse.Namespace) -> list[CheckpointSpec]:
    paths = [path.expanduser().resolve() for path in args.checkpoint]
    if args.checkpoint_dir:
        all_paths = sorted(
            (
                path.resolve()
                for path in args.checkpoint_dir.expanduser().rglob("*")
                if path.suffix in {".ckpt", ".pt", ".pth"} and not path.name.startswith("last")
            ),
            key=lambda path: (epoch_from_text(path.name) if epoch_from_text(path.name) is not None else math.inf, path.name),
        )
        if len(all_paths) > 2:
            all_paths = [all_paths[len(all_paths) // 2], all_paths[-1]]
        paths.extend(all_paths)
    labels = list(args.checkpoint_label)
    if labels and len(labels) != len(paths):
        raise ValueError("the number of checkpoint labels must match explicit and discovered checkpoints")
    specs = [CheckpointSpec("untrained", None, {})] if args.include_untrained else []
    for index, path in enumerate(paths):
        label = labels[index] if labels else path.stem
        specs.append(CheckpointSpec(slugify(label), path, read_checkpoint_hparams(path)))
    return specs


def _find_local_view_counts(value: Any, key_path: str = "") -> set[int]:
    counts: set[int] = set()
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{key_path}.{key}" if key_path else str(key)
            lowered = child_path.lower()
            if str(key).lower() == "num_local_views":
                try:
                    counts.add(int(child))
                except (TypeError, ValueError):
                    pass
            if (
                "local" in lowered
                and str(key).lower() == "names"
                and isinstance(child, (list, tuple))
                and all(re.fullmatch(r"local_\d+", str(name)) for name in child)
            ):
                counts.add(len(child))
            counts.update(_find_local_view_counts(child, child_path))
    elif isinstance(value, (list, tuple)):
        if value and all(re.fullmatch(r"local_\d+", str(item)) for item in value):
            counts.add(len(value))
        for index, child in enumerate(value):
            counts.update(_find_local_view_counts(child, f"{key_path}.{index}"))
    return counts


def resolve_extraction_num_local_views(
    args: argparse.Namespace, specs: Sequence[CheckpointSpec]
) -> int:
    if args.num_local_views is not None:
        return int(args.num_local_views)
    inferred: set[int] = set()
    for spec in specs:
        inferred.update(_find_local_view_counts(spec.hparams))
    if len(inferred) == 1:
        return inferred.pop()
    if not inferred:
        raise ValueError(
            "could not infer the local-view count from checkpoint hyperparameters; "
            "pass --num-local-views explicitly"
        )
    raise ValueError(
        f"checkpoint hyperparameters contain conflicting local-view counts: {sorted(inferred)}; "
        "pass --num-local-views explicitly"
    )


def expected_view_order(num_local_views: int) -> list[str]:
    return ["global_1", "global_2"] + [
        f"local_{index}" for index in range(1, num_local_views + 1)
    ]


def cache_view_layout(record: CacheRecord) -> tuple[list[str], int]:
    num_views = int(record.embeddings.shape[2])
    if num_views < 2:
        raise ValueError(f"{record.path} has {num_views} views; at least two are required")
    raw_order = record.metadata.get("view_order")
    if not isinstance(raw_order, (list, tuple)):
        raise ValueError(f"{record.path} does not contain cache metadata.view_order")
    view_order = [str(name) for name in raw_order]
    expected = expected_view_order(num_views - 2)
    if view_order != expected:
        raise ValueError(
            f"{record.path} has unsupported view order {view_order}; expected {expected}"
        )
    metadata_num_views = record.metadata.get("num_views")
    if metadata_num_views is not None and int(metadata_num_views) != num_views:
        raise ValueError(f"{record.path} metadata.num_views disagrees with its tensor shape")
    metadata_num_local = record.metadata.get("num_local_views")
    if metadata_num_local is not None and int(metadata_num_local) != num_views - 2:
        raise ValueError(f"{record.path} metadata.num_local_views disagrees with view_order")
    return view_order, num_views - 2


def validate_record_view_layouts(records: Sequence[CacheRecord]) -> tuple[list[str], int]:
    layouts = [cache_view_layout(record) for record in records]
    distinct = {(tuple(order), num_local) for order, num_local in layouts}
    if len(distinct) != 1:
        raise ValueError("all analyzed caches must use the same view order")
    order, num_local = layouts[0]
    return order, num_local


def _extraction_dependencies() -> dict[str, Any]:
    from stable_pretraining.data import HFDataset
    from stable_pretraining.data import transforms as spt_t
    from stable_pretraining.data.gpu_transforms import (
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
    from stable_pretraining.methods.lejepa import LeJEPA

    return locals()


def build_extraction_dataset(args: argparse.Namespace) -> Any:
    deps = _extraction_dependencies()
    spt_t = deps["spt_t"]
    transform = spt_t.Compose(spt_t.RGB(), spt_t.Resize(size=[256, 256]), spt_t.ToImage())
    split = "validation" if args.dataset_split == "val" else args.dataset_split
    if args.dataset_root:
        from torchvision.datasets import ImageFolder

        root = args.dataset_root / split if (args.dataset_root / split).exists() else args.dataset_root

        class ImageFolderDict(torch.utils.data.Dataset):
            def __init__(self) -> None:
                self.dataset = ImageFolder(str(root), transform=None)

            def __len__(self) -> int:
                return len(self.dataset)

            def __getitem__(self, index: int) -> dict[str, Any]:
                image, label = self.dataset[index]
                return transform({"image": image, "label": label, "image_id": index})

        dataset: Any = ImageFolderDict()
    else:
        kwargs: dict[str, Any] = {"split": split, "transform": transform}
        if args.dataset_revision:
            kwargs["revision"] = args.dataset_revision
        dataset_cache_dir = args.dataset_cache_dir
        if dataset_cache_dir is None:
            storage_root = Path(os.environ.get("STORAGE_ROOT", Path.home() / "storage"))
            cache_name = {"clane9/imagenet-100": "imagenet100", "frgfm/imagenette": "imagenet10"}.get(
                args.dataset_name
            )
            candidate = storage_root / "datasets" / "stable-pretraining" / str(cache_name)
            if cache_name and candidate.exists():
                dataset_cache_dir = candidate
        if dataset_cache_dir:
            kwargs["cache_dir"] = str(dataset_cache_dir)
        dataset = deps["HFDataset"](args.dataset_name, **kwargs)
    if args.num_images > len(dataset):
        raise ValueError(f"requested {args.num_images} images from a dataset containing {len(dataset)}")
    indices = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(args.seed))[: args.num_images]
    return torch.utils.data.Subset(dataset, indices.tolist())


def build_multiview_transform(
    args: argparse.Namespace, device: torch.device, num_local_views: int
) -> Any:
    deps = _extraction_dependencies()

    def common_ops() -> list[torch.nn.Module]:
        return [
            deps["GPURandomHorizontalFlip"](p=0.5),
            deps["GPUColorJitter"](brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, p=0.8),
            deps["GPURandomGrayscale"](p=0.2),
            deps["GPUGaussianBlur"](kernel_size=23, sigma=[0.1, 2.0], p=0.5),
            deps["GPURandomSolarize"](thresholds=0.5, additions=0.0, p=0.2),
            deps["GPUNormalize"](mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]

    global_chain = deps["GPUCompose"](
        [deps["GPURandomResizedCrop"](size=224, scale=[0.3, 1.0]), *common_ops()], compile=False
    )
    local_chain = deps["GPUCompose"](
        [deps["GPURandomResizedCrop"](size=args.local_crop_size, scale=[0.05, 0.3]), *common_ops()],
        compile=False,
    )
    return deps["GroupedMultiView"](
        groups={
            "global": {"chain": global_chain, "names": ["global_1", "global_2"]},
            "local": {
                "chain": local_chain,
                "names": [
                    f"local_{index}" for index in range(1, num_local_views + 1)
                ],
            },
        }
    ).to(device)


def effective_hparam(spec: CheckpointSpec, explicit: Any, key: str, default: Any) -> Any:
    if explicit is not None:
        return explicit
    return spec.hparams.get(key, default)


def build_extraction_model(args: argparse.Namespace, spec: CheckpointSpec) -> torch.nn.Module:
    LeJEPA = _extraction_dependencies()["LeJEPA"]
    if spec.path is None:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
    model = LeJEPA(
        encoder_name=effective_hparam(spec, args.encoder_name, "module.model.encoder_name", "vit_tiny_patch16_224"),
        pretrained=False,
        drop_path_rate=float(effective_hparam(spec, args.drop_path_rate, "module.model.drop_path_rate", 0.1)),
        sigreg=str(spec.hparams.get("module.model.sigreg", args.model_sigreg)),
        override_sr_gamma=float(spec.hparams.get("module.model.override_sr_gamma", 0.5)),
    )
    if spec.path:
        checkpoint = torch.load(spec.path, map_location="cpu", weights_only=False)
        state = checkpoint.get("state_dict", checkpoint)
        candidates = [
            {key.removeprefix("model."): value for key, value in state.items() if key.startswith("model.")},
            state,
        ]
        for candidate in candidates:
            if not candidate:
                continue
            missing, _ = model.load_state_dict(candidate, strict=False)
            if not any(key.startswith(("backbone.", "projector.")) for key in missing):
                break
        else:
            raise RuntimeError(f"could not load backbone/projector from {spec.path}")
    return model


@torch.no_grad()
def extract_checkpoint_records(args: argparse.Namespace) -> list[CacheRecord]:
    specs = checkpoint_specs(args)
    num_local_views = resolve_extraction_num_local_views(args, specs)
    view_order = expected_view_order(num_local_views)
    dataset = build_extraction_dataset(args)
    output_cache = args.cache_dir or (args.output_dir / "embedding_cache")
    output_cache.mkdir(parents=True, exist_ok=True)
    records: list[CacheRecord] = []
    epochs = [epoch_from_text(spec.path.name) for spec in specs if spec.path]
    final_epoch = max((epoch for epoch in epochs if epoch is not None), default=0) + 1
    for spec in specs:
        cache_identity = {
            "checkpoint": str(spec.path or "untrained"),
            "encoder_name": effective_hparam(
                spec, args.encoder_name, "module.model.encoder_name", "vit_tiny_patch16_224"
            ),
            "num_images": args.num_images,
            "num_augmentation_draws": args.num_augmentation_draws,
            "dataset_root": str(args.dataset_root or ""),
            "dataset_name": args.dataset_name,
            "dataset_split": args.dataset_split,
            "seed": args.seed,
            "local_crop_size": args.local_crop_size,
            "num_local_views": num_local_views,
            "model_mode": args.model_mode,
        }
        digest = hashlib.sha1(json.dumps(cache_identity, sort_keys=True).encode()).hexdigest()[:10]
        cache_path = output_cache / f"{spec.label}-{digest}.pt"
        loaded = cache_payload(cache_path) if cache_path.exists() else None
        if loaded is None:
            device = torch.device(args.device)
            model = build_extraction_model(args, spec).to(device)
            if args.model_mode == "train":
                model.train()
                for module in model.modules():
                    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                        module.momentum = 0.0
            else:
                model.eval()
            transform = build_multiview_transform(args, device, num_local_views)
            loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=args.image_batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=device.type == "cuda",
                persistent_workers=args.num_workers > 0,
            )
            draws: list[torch.Tensor] = []
            for augmentation_id in range(args.num_augmentation_draws):
                chunks: list[torch.Tensor] = []
                for batch_id, batch in enumerate(loader):
                    batch = {
                        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
                        for key, value in batch.items()
                    }
                    seed = args.seed + 100_003 * augmentation_id + batch_id
                    torch.manual_seed(seed)
                    if device.type == "cuda":
                        torch.cuda.manual_seed_all(seed)
                    views = transform(batch)
                    images = [views["global_1"]["image"], views["global_2"]["image"]]
                    images.extend(
                        views[f"local_{index}"]["image"]
                        for index in range(1, num_local_views + 1)
                    )
                    feature_groups = [model.backbone(torch.cat(images[:2]))]
                    if num_local_views:
                        feature_groups.append(model.backbone(torch.cat(images[2:])))
                    features = torch.cat(feature_groups)
                    projected = model.projector(features).float()
                    batch_size = images[0].shape[0]
                    chunks.append(
                        projected.reshape(len(view_order), batch_size, -1)
                        .permute(1, 0, 2)
                        .cpu()
                    )
                draws.append(torch.cat(chunks))
            embeddings = torch.stack(draws).contiguous()
            epoch = epoch_from_text(spec.path.name) if spec.path else None
            progress = 0.0 if spec.path is None else 100.0 * ((epoch or 0) + 1) / final_epoch
            metadata = {
                "checkpoint": spec.label,
                "checkpoint_path": str(spec.path or ""),
                "training_progress": progress,
                "num_augmentation_draws": embeddings.shape[0],
                "num_images": embeddings.shape[1],
                "num_views": embeddings.shape[2],
                "num_global_views": 2,
                "num_local_views": num_local_views,
                "feature_dim": embeddings.shape[3],
                "view_order": view_order,
                "untrained_initialization": (
                    "reconstructed_deterministically_from_analysis_seed"
                    if spec.path is None
                    else "checkpoint"
                ),
                "untrained_matches_training_initialization": False if spec.path is None else None,
                "analysis_seed": args.seed,
            }
            torch.save({"embeddings": embeddings, "metadata": metadata}, cache_path)
            cache_path.with_suffix(".json").write_text(json.dumps(metadata, indent=2))
            loaded = embeddings, metadata
        embeddings, metadata = loaded
        records.append(
            CacheRecord(
                spec.label,
                cache_path,
                embeddings,
                metadata,
                float(metadata.get("training_progress", float("nan"))),
            )
        )
    return records


def deterministic_projection(original_dim: int, projected_dim: int, seed: int) -> torch.Tensor:
    if projected_dim > original_dim:
        raise ValueError(f"projection dimension {projected_dim} exceeds feature dimension {original_dim}")
    generator = torch.Generator().manual_seed(stable_seed("analysis_projection", seed, original_dim, projected_dim))
    matrix = torch.randn(original_dim, projected_dim, generator=generator, dtype=torch.float64)
    q, _ = torch.linalg.qr(matrix, mode="reduced")
    return q[:, :projected_dim].float()


def pair_pool(pair_type: str, num_views: int) -> list[tuple[int, int]]:
    if num_views < 2:
        raise ValueError("at least two views are required")
    globals_ = [0, 1]
    locals_ = list(range(2, num_views))
    if pair_type == "global_global":
        return [(0, 1), (1, 0)]
    if pair_type == "global_local":
        return [(g, local) for g in globals_ for local in locals_] + [
            (local, g) for local in locals_ for g in globals_
        ]
    if pair_type == "local_local":
        return [(left, right) for left in locals_ for right in locals_ if left != right]
    raise ValueError(f"unknown pair type: {pair_type}")


def balanced_pair_assignments(
    num_images: int, pair_type: str, seed: int, num_views: int
) -> list[tuple[int, int]]:
    pool = pair_pool(pair_type, num_views)
    if not pool:
        raise ValueError(f"pair type {pair_type} requires local views")
    offset = seed % len(pool)
    return [pool[(offset + index) % len(pool)] for index in range(num_images)]


def construct_balanced_pairs(
    embeddings: torch.Tensor,
    image_ids: torch.Tensor,
    pair_type: str,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, list[tuple[int, int]], torch.Tensor]:
    if embeddings.ndim != 3 or embeddings.shape[1] < 2:
        raise ValueError("expected projected embeddings with shape [image, view>=2, feature]")
    assignments = balanced_pair_assignments(
        len(image_ids), pair_type, seed, embeddings.shape[1]
    )
    selected = embeddings.index_select(0, image_ids)
    rows = torch.arange(len(image_ids))
    left = selected[rows, torch.tensor([pair[0] for pair in assignments])]
    right = selected[rows, torch.tensor([pair[1] for pair in assignments])]
    return left, right, assignments, image_ids.clone()


def partition_and_split_ids(
    available_images: int,
    image_count: int,
    partition_id: int,
    calibration_fraction: float,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if image_count > available_images:
        raise ValueError(f"requested {image_count} images from a cache containing {available_images}")
    generator = torch.Generator().manual_seed(stable_seed("partition", seed, partition_id))
    selected = torch.randperm(available_images, generator=generator)[:image_count]
    split_generator = torch.Generator().manual_seed(stable_seed("split", seed, partition_id, image_count))
    selected = selected[torch.randperm(image_count, generator=split_generator)]
    calibration_count = max(2, min(image_count - 2, int(round(image_count * calibration_fraction))))
    return selected[:calibration_count], selected[calibration_count:]


def sample_covariance(x: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
    if x.shape[0] < 2:
        raise ValueError("sample covariance requires at least two rows")
    y = x if y is None else y
    xc = x - x.mean(dim=0)
    yc = y - y.mean(dim=0)
    return xc.T @ yc / (x.shape[0] - 1)


def ridge_covariance(x: torch.Tensor, ridge_eps: float) -> torch.Tensor:
    cov = sample_covariance(x)
    scale = torch.trace(cov).clamp_min(EPS) / cov.shape[0]
    return cov + ridge_eps * scale * torch.eye(cov.shape[0], dtype=cov.dtype, device=cov.device)


def inverse_sqrt(cov: torch.Tensor) -> torch.Tensor:
    cov = 0.5 * (cov + cov.T)
    eigenvalues, eigenvectors = torch.linalg.eigh(cov)
    return (eigenvectors * torch.rsqrt(eigenvalues.clamp_min(EPS))) @ eigenvectors.T


def isotropy_error(cov: torch.Tensor) -> float:
    alpha = torch.trace(cov) / cov.shape[0]
    eye = torch.eye(cov.shape[0], dtype=cov.dtype, device=cov.device)
    return float(((cov - alpha * eye).norm() / cov.norm().clamp_min(EPS)).cpu())


def effective_rank(values: torch.Tensor) -> float:
    values = values.detach().abs()
    probabilities = values / values.sum().clamp_min(EPS)
    return float(torch.exp(-(probabilities * probabilities.clamp_min(EPS).log()).sum()).cpu())


def deterministic_subsample(x: torch.Tensor, max_samples: int, seed: int) -> tuple[torch.Tensor, int]:
    count = min(x.shape[0], max_samples)
    if count == x.shape[0]:
        return x, count
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(x.shape[0], generator=generator)[:count].sort().values
    return x.index_select(0, indices), count


def shape_cw_score(
    x: torch.Tensor,
    ridge_eps: float,
    gamma: float,
    max_samples: int,
    seed: int,
) -> tuple[float, int, torch.Tensor]:
    global MAX_CW_SAMPLES_SEEN
    sampled, count = deterministic_subsample(x, max_samples, seed)
    centered = sampled - sampled.mean(dim=0)
    whitened = centered @ inverse_sqrt(ridge_covariance(centered, ridge_eps)).T
    MAX_CW_SAMPLES_SEEN = max(MAX_CW_SAMPLES_SEEN, count)
    score = cw_normality(
        whitened.float(), torch.tensor(gamma, dtype=torch.float32, device=whitened.device)
    )
    return float(score.detach().cpu()), count, whitened


def fixed_shape_directions(dimension: int, count: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(stable_seed("shape_directions", seed, dimension, count))
    directions = torch.randn(dimension, count, generator=generator, dtype=torch.float64)
    return directions / directions.norm(dim=0).clamp_min(EPS)


def projection_shape_stats(x: torch.Tensor, directions: torch.Tensor) -> tuple[float, float]:
    projected = (x - x.mean(dim=0)) @ directions
    standardized = projected / projected.std(dim=0, unbiased=False).clamp_min(1e-8)
    skewness = standardized.pow(3).mean(dim=0).abs().mean()
    excess_kurtosis = (standardized.pow(4).mean(dim=0) - 3.0).abs().mean()
    return float(skewness.cpu()), float(excess_kurtosis.cpu())


def roc_auc(positive_scores: torch.Tensor, negative_scores: torch.Tensor) -> float:
    scores = torch.cat([positive_scores, negative_scores]).detach().cpu().numpy()
    labels = np.concatenate([np.ones(positive_scores.numel()), np.zeros(negative_scores.numel())])
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    index = 0
    while index < scores.size:
        end = index + 1
        while end < scores.size and scores[order[end]] == scores[order[index]]:
            end += 1
        ranks[order[index:end]] = (index + end + 1) / 2.0
        index = end
    n_positive = positive_scores.numel()
    n_negative = negative_scores.numel()
    return float(
        (ranks[labels == 1].sum() - n_positive * (n_positive + 1) / 2.0)
        / (n_positive * n_negative)
    )


def calibration_metrics(
    z1: torch.Tensor,
    z2: torch.Tensor,
    *,
    ridge_eps: float,
    gamma: float,
    gaussianity_max_samples: int,
    shape_directions: torch.Tensor,
    diagnostic_seed: int,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    z1 = z1.double()
    z2 = z2.double()
    mean1 = z1.mean(dim=0)
    mean2 = z2.mean(dim=0)
    c11 = ridge_covariance(z1, ridge_eps)
    c22 = ridge_covariance(z2, ridge_eps)
    c12 = sample_covariance(z1, z2)
    w1_transform = inverse_sqrt(c11)
    w2_transform = inverse_sqrt(c22)
    r = w1_transform @ c12 @ w2_transform
    diagonal = torch.diagonal(r)
    rho_hat = torch.trace(r) / r.shape[0]
    eye = torch.eye(r.shape[0], dtype=r.dtype)
    norm_sq = r.square().sum().clamp_min(EPS)
    offdiagonal = r - torch.diag(diagonal)
    singular_values = torch.linalg.svdvals(r)
    singular_energy = singular_values.square()

    w1 = (z1 - mean1) @ w1_transform.T
    w2 = (z2 - mean2) @ w2_transform.T
    w_plus = (w1 + w2) / math.sqrt(2.0)
    w_minus = (w1 - w2) / math.sqrt(2.0)
    c_plus = sample_covariance(w_plus)
    c_minus = sample_covariance(w_minus)
    c_plus_minus = sample_covariance(w_plus, w_minus)
    plus_eigenvalues = torch.linalg.eigvalsh(0.5 * (c_plus + c_plus.T)).clamp_min(0)
    minus_eigenvalues = torch.linalg.eigvalsh(0.5 * (c_minus + c_minus.T)).clamp_min(0)
    cross_normalizer = torch.sqrt(c_plus.square().sum() * c_minus.square().sum()).clamp_min(EPS)

    shape_plus, shape_count, shape_plus_rows = shape_cw_score(
        w_plus, ridge_eps, gamma, gaussianity_max_samples, diagnostic_seed
    )
    shape_minus, minus_count, shape_minus_rows = shape_cw_score(
        w_minus, ridge_eps, gamma, gaussianity_max_samples, diagnostic_seed
    )
    if shape_count != minus_count:
        raise RuntimeError("plus and minus Gaussianity diagnostics used different sample counts")
    plus_skew, plus_kurtosis = projection_shape_stats(shape_plus_rows, shape_directions)
    minus_skew, minus_kurtosis = projection_shape_stats(shape_minus_rows, shape_directions)

    metrics: dict[str, Any] = {
        "rho_hat": float(rho_hat.cpu()),
        "scalar_residual": float(((r - rho_hat * eye).norm() / r.norm().clamp_min(EPS)).cpu()),
        "offdiagonal_ratio": float((offdiagonal.norm() / r.norm().clamp_min(EPS)).cpu()),
        "antisymmetric_ratio": float(((r - r.T).norm() / (2.0 * r.norm().clamp_min(EPS))).cpu()),
        "diagonal_mean": float(diagonal.mean().cpu()),
        "diagonal_std": float(diagonal.std(unbiased=False).cpu()),
        "fraction_negative_diagonal": float((diagonal < 0).double().mean().cpu()),
        "effective_rank_R": effective_rank(singular_values),
        "explained_scalar": float((1.0 - (r - rho_hat * eye).square().sum() / norm_sq).cpu()),
        "explained_diagonal": float((1.0 - offdiagonal.square().sum() / norm_sq).cpu()),
        "plus_isotropy_error": isotropy_error(c_plus),
        "minus_isotropy_error": isotropy_error(c_minus),
        "plus_effective_rank": effective_rank(plus_eigenvalues),
        "minus_effective_rank": effective_rank(minus_eigenvalues),
        "plus_minus_crosscov": float((c_plus_minus.norm() / cross_normalizer).cpu()),
        "plus_minus_variance_ratio": float(
            (torch.trace(c_plus) / torch.trace(c_minus).clamp_min(EPS)).cpu()
        ),
        "shape_cw_plus": shape_plus,
        "shape_cw_minus": shape_minus,
        "gaussianity_num_samples": shape_count,
        "projection_skewness_plus": plus_skew,
        "projection_skewness_minus": minus_skew,
        "projection_kurtosis_plus": plus_kurtosis,
        "projection_kurtosis_minus": minus_kurtosis,
        "singular_values_R": json.dumps([float(value) for value in singular_values.cpu()]),
        "plus_covariance_eigenvalues": json.dumps([float(value) for value in plus_eigenvalues.flip(0).cpu()]),
        "minus_covariance_eigenvalues": json.dumps([float(value) for value in minus_eigenvalues.flip(0).cpu()]),
    }
    for rank in (1, 4, 16):
        metrics[f"explained_rank_{rank}"] = float(
            (singular_energy[: min(rank, len(singular_energy))].sum() / singular_energy.sum().clamp_min(EPS)).cpu()
        )
    for rank, value in enumerate(singular_values[:16], start=1):
        metrics[f"singular_value_{rank}"] = float(value.cpu())
    transforms = {
        "mean1": mean1,
        "mean2": mean2,
        "whitener1": w1_transform,
        "whitener2": w2_transform,
        "R": r,
    }
    return metrics, transforms


def shuffled_evaluation_metrics(
    z1: torch.Tensor,
    z2: torch.Tensor,
    transforms: dict[str, torch.Tensor],
    seed: int,
) -> dict[str, float]:
    z1 = z1.double()
    z2 = z2.double()
    w1 = (z1 - transforms["mean1"]) @ transforms["whitener1"].T
    w2 = (z2 - transforms["mean2"]) @ transforms["whitener2"].T
    if len(w2) < 2:
        raise ValueError("shuffled evaluation requires at least two source images")
    shift = 1 + seed % (len(w2) - 1)
    permutation = torch.arange(len(w2)).roll(shift)
    shuffled = w2.index_select(0, permutation)
    true_cosine = F.cosine_similarity(w1, w2, dim=1)
    shuffled_cosine = F.cosine_similarity(w1, shuffled, dim=1)
    true_distance = (w1 - w2).square().sum(dim=1)
    shuffled_distance = (w1 - shuffled).square().sum(dim=1)
    return {
        "true_cosine_mean": float(true_cosine.mean().cpu()),
        "true_cosine_median": float(true_cosine.median().cpu()),
        "shuffled_cosine_mean": float(shuffled_cosine.mean().cpu()),
        "shuffled_cosine_median": float(shuffled_cosine.median().cpu()),
        "true_distance_mean": float(true_distance.mean().cpu()),
        "true_distance_median": float(true_distance.median().cpu()),
        "shuffled_distance_mean": float(shuffled_distance.mean().cpu()),
        "shuffled_distance_median": float(shuffled_distance.median().cpu()),
        "cosine_auc": roc_auc(true_cosine, shuffled_cosine),
        "distance_auc": roc_auc(-true_distance, -shuffled_distance),
    }


def structural_null_metrics(
    z1: torch.Tensor,
    z2: torch.Tensor,
    *,
    ridge_eps: float,
) -> dict[str, float]:
    """Calculate structural metrics without invoking CW normality."""
    z1 = z1.double()
    z2 = z2.double()
    c11 = ridge_covariance(z1, ridge_eps)
    c22 = ridge_covariance(z2, ridge_eps)
    c12 = sample_covariance(z1, z2)
    whitener1 = inverse_sqrt(c11)
    whitener2 = inverse_sqrt(c22)
    r = whitener1 @ c12 @ whitener2
    diagonal = torch.diagonal(r)
    rho = torch.trace(r) / r.shape[0]
    eye = torch.eye(r.shape[0], dtype=r.dtype)
    w1 = (z1 - z1.mean(dim=0)) @ whitener1.T
    w2 = (z2 - z2.mean(dim=0)) @ whitener2.T
    w_plus = (w1 + w2) / math.sqrt(2.0)
    w_minus = (w1 - w2) / math.sqrt(2.0)
    c_plus = sample_covariance(w_plus)
    c_minus = sample_covariance(w_minus)
    c_plus_minus = sample_covariance(w_plus, w_minus)
    cross_normalizer = torch.sqrt(c_plus.square().sum() * c_minus.square().sum()).clamp_min(EPS)
    return {
        "rho_hat": float(rho.cpu()),
        "scalar_residual": float(((r - rho * eye).norm() / r.norm().clamp_min(EPS)).cpu()),
        "offdiagonal_ratio": float(
            ((r - torch.diag(diagonal)).norm() / r.norm().clamp_min(EPS)).cpu()
        ),
        "antisymmetric_ratio": float(
            ((r - r.T).norm() / (2.0 * r.norm().clamp_min(EPS))).cpu()
        ),
        "plus_isotropy_error": isotropy_error(c_plus),
        "minus_isotropy_error": isotropy_error(c_minus),
        "plus_minus_crosscov": float((c_plus_minus.norm() / cross_normalizer).cpu()),
    }


def summarize_null(
    observed: dict[str, Any],
    repetitions: list[dict[str, Any]],
    metrics: Sequence[str],
) -> dict[str, float]:
    calibrated: dict[str, float] = {}
    for metric in metrics:
        values = np.asarray([float(row[metric]) for row in repetitions], dtype=np.float64)
        median = float(np.median(values))
        calibrated[f"{metric}_null_median"] = median
        calibrated[f"{metric}_null_p05"] = float(np.quantile(values, 0.05))
        calibrated[f"{metric}_null_p95"] = float(np.quantile(values, 0.95))
        calibrated[f"{metric}_observed_to_null_median"] = float(observed[metric]) / (abs(median) + EPS)
        calibrated[f"{metric}_null_upper_tail_p"] = float(
            (1 + np.count_nonzero(values >= float(observed[metric]))) / (len(values) + 1)
        )
    return calibrated


def cached_null_repetitions(
    config: dict[str, Any],
    generate: Any,
    *,
    cache_dir: Path,
) -> tuple[list[dict[str, Any]], str, bool]:
    key = hashlib.sha1(json.dumps(config, sort_keys=True).encode()).hexdigest()
    path = cache_dir / f"{key}.json"
    cached = path.exists()
    if cached:
        payload = json.loads(path.read_text())
        repetitions_data = payload["repetitions"]
    else:
        repetitions_data = generate(key)
        cache_dir.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"config": config, "repetitions": repetitions_data}, indent=2))
    repetitions_data = [{f"null_{name}": value for name, value in config.items()} | row for row in repetitions_data]
    return repetitions_data, key, cached


def structural_group_key(row: dict[str, Any]) -> tuple[str, int, str]:
    return str(row["checkpoint"]), int(row["image_count"]), str(row["pair_type"])


def shape_group_key(
    row: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[int, int, float, float, int]:
    return (
        int(row["calibration_count"]),
        int(row["projected_dimension"]),
        float(args.ridge_eps),
        float(args.cw_gamma),
        int(row["gaussianity_num_samples"]),
    )


def grouped_observations(
    rows: list[dict[str, Any]],
) -> dict[tuple[str, int, str], list[dict[str, Any]]]:
    groups: dict[tuple[str, int, str], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(structural_group_key(row), []).append(row)
    return groups


def null_runtime_plan(
    rows: list[dict[str, Any]],
    args: argparse.Namespace,
    *,
    checkpoint_count: int,
    emit: bool = True,
) -> dict[str, int]:
    structural_count = len(grouped_observations(rows))
    shape_count = len({shape_group_key(row, args) for row in rows})
    plan = {
        "observed_rows": len(rows),
        "structural_null_configurations": structural_count,
        "shape_null_configurations": shape_count,
        "total_structural_null_repetitions": structural_count
        * args.num_structural_null_repetitions,
        "total_shape_null_repetitions": shape_count * args.num_shape_null_repetitions,
    }
    if emit:
        print(f"[null-plan] observed rows: {plan['observed_rows']}", flush=True)
        print(
            f"[null-plan] structural-null configurations: {structural_count}; "
            f"total repetitions: {plan['total_structural_null_repetitions']}",
            flush=True,
        )
        print(
            f"[null-plan] shape-null configurations: {shape_count}; "
            f"total repetitions: {plan['total_shape_null_repetitions']}",
            flush=True,
        )
    if checkpoint_count == 3 and len(set(args.image_counts)) <= 2:
        assert structural_count <= 18, "default analysis must use at most 18 structural nulls"
        assert shape_count <= 2, "default analysis must use at most two shape nulls"
    return plan


def generate_structural_null_repetitions(
    config: dict[str, Any],
    *,
    repetitions: int,
    null_seed: int,
    cache_key: str,
) -> list[dict[str, Any]]:
    generated: list[dict[str, Any]] = []
    rho = float(np.clip(config["rho"], -0.999999, 0.999999))
    for repetition in range(repetitions):
        generator = torch.Generator().manual_seed(
            stable_seed("structural_null", null_seed, cache_key, repetition)
        )
        x = torch.randn(
            config["calibration_count"],
            config["dimension"],
            generator=generator,
            dtype=torch.float64,
        )
        noise = torch.randn(
            config["calibration_count"],
            config["dimension"],
            generator=generator,
            dtype=torch.float64,
        )
        y = rho * x + math.sqrt(1.0 - rho**2) * noise
        generated.append(structural_null_metrics(x, y, ridge_eps=config["ridge_eps"]))
    return generated


def generate_shape_null_repetitions(
    config: dict[str, Any],
    *,
    repetitions: int,
    null_seed: int,
    cache_key: str,
) -> list[dict[str, Any]]:
    generated: list[dict[str, Any]] = []
    for repetition in range(repetitions):
        generator = torch.Generator().manual_seed(
            stable_seed("shape_null", null_seed, cache_key, repetition)
        )
        plus = torch.randn(
            config["gaussianity_num_samples"],
            config["dimension"],
            generator=generator,
            dtype=torch.float64,
        )
        minus = torch.randn(
            config["gaussianity_num_samples"],
            config["dimension"],
            generator=generator,
            dtype=torch.float64,
        )
        plus_score, _, _ = shape_cw_score(
            plus,
            config["ridge_eps"],
            config["gamma"],
            config["gaussianity_num_samples"],
            stable_seed(cache_key, repetition, "plus"),
        )
        minus_score, _, _ = shape_cw_score(
            minus,
            config["ridge_eps"],
            config["gamma"],
            config["gaussianity_num_samples"],
            stable_seed(cache_key, repetition, "minus"),
        )
        generated.append({"shape_cw_plus": plus_score, "shape_cw_minus": minus_score})
    return generated


def apply_grouped_null_calibration(
    rows: list[dict[str, Any]],
    args: argparse.Namespace,
    cache_dir: Path,
) -> dict[str, list[dict[str, Any]]]:
    null_output: dict[str, list[dict[str, Any]]] = {}
    for (checkpoint, image_count, pair_type), members in grouped_observations(rows).items():
        rho = float(np.median([float(row["rho_hat"]) for row in members]))
        calibration_count = int(members[0]["calibration_count"])
        dimension = int(members[0]["projected_dimension"])
        config = {
            "null_type": "structural",
            "checkpoint": checkpoint,
            "image_count": image_count,
            "pair_type": pair_type,
            "calibration_count": calibration_count,
            "dimension": dimension,
            "rho": round(rho, 4),
            "ridge_eps": args.ridge_eps,
            "repetitions": args.num_structural_null_repetitions,
            "null_seed": args.null_seed,
        }

        def generate_structural(key: str) -> list[dict[str, Any]]:
            return generate_structural_null_repetitions(
                config,
                repetitions=args.num_structural_null_repetitions,
                null_seed=args.null_seed,
                cache_key=key,
            )

        repetitions, key, cached = cached_null_repetitions(
            config, generate_structural, cache_dir=cache_dir
        )
        null_output[key] = repetitions
        for row in members:
            row.update(summarize_null(row, repetitions, STRUCTURAL_NULL_METRICS))
            row["structural_null_key"] = key
            row["structural_null_results_cached"] = cached
            row["structural_null_group_median_rho"] = rho

    shape_groups: dict[tuple[int, int, float, float, int], list[dict[str, Any]]] = {}
    for row in rows:
        shape_groups.setdefault(shape_group_key(row, args), []).append(row)
    for (calibration_count, dimension, ridge_eps, gamma, sample_count), members in shape_groups.items():
        config = {
            "null_type": "shape",
            "calibration_count": calibration_count,
            "dimension": dimension,
            "ridge_eps": ridge_eps,
            "gamma": gamma,
            "gaussianity_num_samples": sample_count,
            "repetitions": args.num_shape_null_repetitions,
            "null_seed": args.null_seed,
        }

        def generate_shape(key: str) -> list[dict[str, Any]]:
            return generate_shape_null_repetitions(
                config,
                repetitions=args.num_shape_null_repetitions,
                null_seed=args.null_seed,
                cache_key=key,
            )

        repetitions, key, cached = cached_null_repetitions(
            config, generate_shape, cache_dir=cache_dir
        )
        null_output[key] = repetitions
        for row in members:
            row.update(summarize_null(row, repetitions, SHAPE_NULL_METRICS))
            row["shape_null_key"] = key
            row["shape_null_results_cached"] = cached
    return null_output


def analyze_observation(
    projected: torch.Tensor,
    *,
    calibration_ids: torch.Tensor,
    evaluation_ids: torch.Tensor,
    pair_type: str,
    assignment_seed: int,
    args: argparse.Namespace,
    shape_directions: torch.Tensor,
) -> dict[str, Any]:
    z1_cal, z2_cal, _, cal_sources = construct_balanced_pairs(
        projected, calibration_ids, pair_type, assignment_seed
    )
    z1_eval, z2_eval, _, eval_sources = construct_balanced_pairs(
        projected, evaluation_ids, pair_type, assignment_seed + 1
    )
    if set(cal_sources.tolist()).intersection(eval_sources.tolist()):
        raise RuntimeError("calibration/evaluation source-image leakage")
    metrics, transforms = calibration_metrics(
        z1_cal,
        z2_cal,
        ridge_eps=args.ridge_eps,
        gamma=args.cw_gamma,
        gaussianity_max_samples=args.gaussianity_max_samples,
        shape_directions=shape_directions,
        diagnostic_seed=stable_seed("diagnostic", assignment_seed),
    )
    metrics.update(
        shuffled_evaluation_metrics(
            z1_eval,
            z2_eval,
            transforms,
            stable_seed("shuffle", assignment_seed),
        )
    )
    return metrics


def covariance_sqrt(cov: torch.Tensor) -> torch.Tensor:
    cov = 0.5 * (cov + cov.T)
    eigenvalues, eigenvectors = torch.linalg.eigh(cov)
    return (eigenvectors * torch.sqrt(eigenvalues.clamp_min(0.0))) @ eigenvectors.T


def covariance_hash(cov: torch.Tensor) -> str:
    rounded = np.round(cov.detach().cpu().numpy(), decimals=7)
    return hashlib.sha1(rounded.tobytes()).hexdigest()


def duplicate_sanity_checks(
    record: CacheRecord,
    projection: torch.Tensor,
    args: argparse.Namespace,
    null_cache_dir: Path,
) -> tuple[dict[int, dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    projected = torch.einsum("ivd,dq->ivq", record.embeddings[0], projection)
    results: dict[int, dict[str, Any]] = {}
    nulls: dict[str, list[dict[str, Any]]] = {}
    for image_count in args.image_counts:
        calibration_ids, evaluation_ids = partition_and_split_ids(
            projected.shape[0], image_count, 0, args.calibration_fraction, args.seed
        )
        ids = torch.cat([calibration_ids, evaluation_ids])
        x = projected.index_select(0, ids)[:, 0]
        observed = structural_null_metrics(x, x.clone(), ridge_eps=args.ridge_eps)
        observed_covariance = sample_covariance(x.double())
        covariance_root = covariance_sqrt(observed_covariance)
        config = {
            "null_type": "duplicate",
            "image_count": image_count,
            "dimension": projection.shape[1],
            "ridge_eps": args.ridge_eps,
            "covariance_hash": covariance_hash(observed_covariance),
            "repetitions": args.num_structural_null_repetitions,
            "null_seed": args.null_seed,
        }

        def generate_duplicate(key: str) -> list[dict[str, Any]]:
            generated: list[dict[str, Any]] = []
            for repetition in range(args.num_structural_null_repetitions):
                generator = torch.Generator().manual_seed(
                    stable_seed("duplicate_null", args.null_seed, key, repetition)
                )
                noise = torch.randn(
                    image_count, projection.shape[1], generator=generator, dtype=torch.float64
                )
                x_null = noise @ covariance_root.T
                generated.append(
                    structural_null_metrics(x_null, x_null.clone(), ridge_eps=args.ridge_eps)
                )
            return generated

        null_rows, key, cached = cached_null_repetitions(
            config, generate_duplicate, cache_dir=null_cache_dir
        )
        calibrated = summarize_null(observed, null_rows, STRUCTURAL_NULL_METRICS)
        lower = calibrated["scalar_residual_null_p05"]
        upper = calibrated["scalar_residual_null_p95"]
        pass_status = observed["scalar_residual"] <= upper and observed["rho_hat"] >= 0.95
        results[image_count] = {
            "duplicate_rho": observed["rho_hat"],
            "duplicate_scalar_residual": observed["scalar_residual"],
            "duplicate_null_interval": json.dumps([lower, upper]),
            "duplicate_sanity_pass": pass_status,
            "duplicate_null_results_cached": cached,
        }
        nulls[key] = null_rows
    return results, nulls


def numeric_value(value: Any) -> float | None:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float, np.number)) and math.isfinite(float(value)):
        return float(value)
    return None


def aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    keys = ("checkpoint", "training_progress", "image_count", "pair_type", "projected_dimension")
    for row in rows:
        groups.setdefault(tuple(row[key] for key in keys), []).append(row)
    output: list[dict[str, Any]] = []
    for group_key, members in groups.items():
        aggregate = dict(zip(keys, group_key, strict=True))
        aggregate["num_observations"] = len(members)
        columns = sorted(set().union(*(member.keys() for member in members)))
        for column in columns:
            if "_null_" in column or column.endswith("_null_upper_tail_p"):
                continue
            values = [numeric_value(member.get(column)) for member in members]
            finite = [value for value in values if value is not None]
            if finite and len(finite) == len(members):
                aggregate[f"{column}_median"] = float(np.median(finite))
                aggregate[f"{column}_q25"] = float(np.quantile(finite, 0.25))
                aggregate[f"{column}_q75"] = float(np.quantile(finite, 0.75))
        for metric in NULL_METRICS:
            observed = np.asarray([float(member[metric]) for member in members])
            null_median = float(members[0][f"{metric}_null_median"])
            null_p05 = float(members[0][f"{metric}_null_p05"])
            null_p95 = float(members[0][f"{metric}_null_p95"])
            aggregate[f"{metric}_null_median"] = null_median
            aggregate[f"{metric}_null_p05"] = null_p05
            aggregate[f"{metric}_null_p95"] = null_p95
            aggregate[f"{metric}_median_observed_to_null_median"] = float(
                np.median(observed) / (abs(null_median) + EPS)
            )
            aggregate[f"{metric}_fraction_above_null_p95"] = float(
                np.mean(observed > null_p95)
            )
        for array_column in ("singular_values_R", "plus_covariance_eigenvalues", "minus_covariance_eigenvalues"):
            arrays = [np.asarray(json.loads(member[array_column]), dtype=float) for member in members]
            if arrays:
                aggregate[f"{array_column}_median"] = json.dumps(
                    np.median(np.stack(arrays), axis=0).tolist()
                )
        output.append(aggregate)
    return output


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = sorted(set().union(*(row.keys() for row in rows))) if rows else []
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def plot_results(rows: list[dict[str, Any]], output_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        print(f"[warn] Plotting dependencies unavailable: {exc}", file=sys.stderr)
        return
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        pass
    if not rows:
        return
    largest_count = max(int(row["image_count"]) for row in rows)
    selected = [row for row in rows if int(row["image_count"]) == largest_count]

    def grouped_values(pair_type: str, metric: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        groups: dict[float, list[float]] = {}
        for row in selected:
            if row["pair_type"] != pair_type:
                continue
            groups.setdefault(float(row["training_progress"]), []).append(float(row[metric]))
        x = np.asarray(sorted(groups))
        means = np.asarray([np.mean(groups[value]) for value in x])
        stds = np.asarray([np.std(groups[value], ddof=1) if len(groups[value]) > 1 else 0.0 for value in x])
        return x, means, stds

    def plot_metric(metric: str, ylabel: str, *, null_band: bool = False) -> None:
        figure, axes = plt.subplots(1, 3, figsize=(12, 3.6), sharex=True)
        for axis, pair_type in zip(axes, PAIR_TYPES, strict=True):
            x, means, stds = grouped_values(pair_type, metric)
            axis.plot(x, means, marker="o")
            axis.fill_between(x, means - stds, means + stds, alpha=0.18)
            if null_band:
                null_x, lower, _ = grouped_values(pair_type, f"{metric}_null_p05")
                _, upper, _ = grouped_values(pair_type, f"{metric}_null_p95")
                axis.fill_between(null_x, lower, upper, color="0.5", alpha=0.2, label="matched null 5-95%")
                axis.legend(fontsize=8)
            axis.set_title(pair_type.replace("_", " "))
            axis.set_xlabel("Training progress (%)")
            axis.set_ylabel(ylabel if axis is axes[0] else "")
        figure.tight_layout()
        figure.savefig(output_dir / f"{metric}.pdf", bbox_inches="tight")
        plt.close(figure)

    plot_metric("rho_hat", r"$\hat{\rho}$")
    plot_metric("scalar_residual", "Scalar-model residual", null_band=True)
    plot_metric("offdiagonal_ratio", "Off-diagonal ratio")
    plot_metric("plus_isotropy_error", "Common-coordinate isotropy error")
    plot_metric("minus_isotropy_error", "Difference-coordinate isotropy error")
    plot_metric("plus_minus_crosscov", "Normalized common/difference cross-covariance")
    plot_metric("shape_cw_plus", "Whitened common-coordinate CW normality")
    plot_metric("shape_cw_minus", "Whitened difference-coordinate CW normality")
    plot_metric("cosine_auc", "True-vs-shuffled cosine AUC")

    figure, axes = plt.subplots(1, 3, figsize=(12, 3.6), sharex=True)
    for axis, pair_type in zip(axes, PAIR_TYPES, strict=True):
        for rank in range(1, 5):
            x, means, _ = grouped_values(pair_type, f"singular_value_{rank}")
            axis.plot(x, means, marker="o", label=f"s{rank}")
        axis.set_title(pair_type.replace("_", " "))
        axis.set_xlabel("Training progress (%)")
        axis.set_ylabel("Singular value" if axis is axes[0] else "")
        axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output_dir / "singular_value_spectrum.pdf", bbox_inches="tight")
    plt.close(figure)


def summary_report(aggregate_rows_: list[dict[str, Any]], duplicate: dict[int, dict[str, Any]]) -> str:
    largest_count = max(int(row["image_count"]) for row in aggregate_rows_)
    lines = [
        "# Joint positive-view structural plausibility",
        "",
        "This is an observational analysis of cached representations. It does not evaluate optimization behavior or whether a Joint-CW training objective would work.",
        "",
    ]
    pair_summaries: dict[str, dict[str, float]] = {}
    for pair_type in PAIR_TYPES:
        selected = [
            row
            for row in aggregate_rows_
            if row["pair_type"] == pair_type and int(row["image_count"]) == largest_count
        ]
        finite_progress = [
            float(row["training_progress"])
            for row in selected
            if math.isfinite(float(row["training_progress"]))
        ]
        latest_progress = max(finite_progress) if finite_progress else float(selected[-1]["training_progress"])
        latest = min(
            selected,
            key=lambda row: abs(float(row["training_progress"]) - latest_progress)
            if math.isfinite(float(row["training_progress"]))
            else math.inf,
        )
        rho_values = [float(row["rho_hat_median"]) for row in selected]
        scalar_fractions = [
            float(row["scalar_residual_fraction_above_null_p95"]) for row in selected
        ]
        diagonal_advantage = float(latest["explained_diagonal_median"]) - float(
            latest["explained_scalar_median"]
        )
        pair_summaries[pair_type] = {
            "rho": float(latest["rho_hat_median"]),
            "scalar_fraction": float(latest["scalar_residual_fraction_above_null_p95"]),
            "diagonal_advantage": diagonal_advantage,
            "rho_span": max(rho_values) - min(rho_values),
            "fraction_span": max(scalar_fractions) - min(scalar_fractions),
        }
        lines.extend(
            [
                f"## {pair_type.replace('_', ' ').title()}",
                "",
                f"- Fitted rho: median {float(latest['rho_hat_median']):.3f}, IQR "
                f"[{float(latest['rho_hat_q25']):.3f}, {float(latest['rho_hat_q75']):.3f}]. "
                f"Checkpoint medians span {min(rho_values):.3f} to {max(rho_values):.3f}.",
                f"- Scalar residual: median {float(latest['scalar_residual_median']):.3f}, IQR "
                f"[{float(latest['scalar_residual_q25']):.3f}, {float(latest['scalar_residual_q75']):.3f}]; "
                f"null median {float(latest['scalar_residual_null_median']):.3f}, null 5th-95th "
                f"[{float(latest['scalar_residual_null_p05']):.3f}, {float(latest['scalar_residual_null_p95']):.3f}]. "
                f"Median/null={float(latest['scalar_residual_median_observed_to_null_median']):.2f}; "
                f"fraction above null 95th={float(latest['scalar_residual_fraction_above_null_p95']):.2f}.",
                f"- Explained energy: scalar {float(latest['explained_scalar_median']):.3f}, "
                f"diagonal {float(latest['explained_diagonal_median']):.3f} "
                f"(advantage {diagonal_advantage:.3f}), rank-4 {float(latest['explained_rank_4_median']):.3f}, "
                f"rank-16 {float(latest['explained_rank_16_median']):.3f}.",
                f"- Common/difference isotropy errors: {float(latest['plus_isotropy_error_median']):.3f} / "
                f"{float(latest['minus_isotropy_error_median']):.3f}; fractions above matched-null 95th "
                f"{float(latest['plus_isotropy_error_fraction_above_null_p95']):.2f} / "
                f"{float(latest['minus_isotropy_error_fraction_above_null_p95']):.2f}.",
                f"- Shape-only CW median/null: "
                f"{float(latest['shape_cw_plus_median_observed_to_null_median']):.2f} / "
                f"{float(latest['shape_cw_minus_median_observed_to_null_median']):.2f}; fractions above "
                f"Gaussian-null 95th {float(latest['shape_cw_plus_fraction_above_null_p95']):.2f} / "
                f"{float(latest['shape_cw_minus_fraction_above_null_p95']):.2f}.",
                f"- Stability across checkpoints: rho-span {max(rho_values) - min(rho_values):.3f}; "
                f"fraction-above-null span {max(scalar_fractions) - min(scalar_fractions):.2f}. "
                "Augmentation/partition stability is represented by the IQR and fraction-above-null statistics.",
                "",
            ]
        )
    lines.append("## Estimator sanity")
    lines.append("")
    for image_count, result in sorted(duplicate.items()):
        lines.append(
            f"- {image_count} images: rho={result['duplicate_rho']:.4f}, residual={result['duplicate_scalar_residual']:.4f}, "
            f"null interval={result['duplicate_null_interval']}, pass={result['duplicate_sanity_pass']}."
        )
    rho_values = [values["rho"] for values in pair_summaries.values()]
    unstable = any(
        values["rho_span"] >= 0.2 or values["fraction_span"] >= 0.75
        for values in pair_summaries.values()
    )
    anisotropic = sum(
        values["scalar_fraction"] > 0.5 and values["diagonal_advantage"] > 0.05
        for values in pair_summaries.values()
    )
    if unstable:
        verdict = "results are unstable or inconclusive"
    elif anisotropic >= 2:
        verdict = "substantial anisotropic structure remains"
    elif max(rho_values) - min(rho_values) >= 0.1:
        verdict = "dependence is strongly pair-type-specific"
    elif all(values["scalar_fraction"] <= 0.25 for values in pair_summaries.values()):
        verdict = "scalar-rho is a reasonable first approximation"
    else:
        verdict = "results are unstable or inconclusive"
    lines.extend(
        [
            "",
            "## Structural conclusion",
            "",
            verdict,
            "",
            "This conclusion concerns structural plausibility only. It does not show that Joint-CW would train successfully or replace the current MSE alignment term.",
        ]
    )
    return "\n".join(lines) + "\n"


def gradient_dependencies() -> dict[str, Any]:
    from assets.cli.analyze_multiview_cw_gradients import (
        cw_component_losses,
        flatten_view_major,
        production_alignment_loss,
    )
    from stable_pretraining.methods.lejepa import CWReg, SlicedEppsPulley

    return {
        "CWReg": CWReg,
        "SlicedEppsPulley": SlicedEppsPulley,
        "cw_component_losses": cw_component_losses,
        "flatten_view_major": flatten_view_major,
        "production_alignment_loss": production_alignment_loss,
    }


def deterministic_gradient_batches(
    num_images: int,
    batch_size: int,
    num_batches: int,
    seed: int,
) -> list[torch.Tensor]:
    required = batch_size * num_batches
    if required > num_images:
        raise ValueError(
            f"gradient analysis requests {required} source images, but the cache has {num_images}"
        )
    generator = torch.Generator().manual_seed(stable_seed("gradient_batches", seed))
    permutation = torch.randperm(num_images, generator=generator)[:required]
    return list(permutation.split(batch_size))


def tensor_cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    left_flat = left.detach().flatten()
    right_flat = right.detach().flatten()
    denominator = left_flat.norm() * right_flat.norm()
    if float(denominator.cpu()) <= EPS:
        return float("nan")
    return float(((left_flat @ right_flat) / denominator).cpu())


def tensor_norm(value: torch.Tensor) -> float:
    return float(value.detach().norm().cpu())


def per_image_cosines(
    left: torch.Tensor,
    right: torch.Tensor,
) -> torch.Tensor:
    left_rows = left.detach().reshape(left.shape[0], -1)
    right_rows = right.detach().reshape(right.shape[0], -1)
    return F.cosine_similarity(left_rows, right_rows, dim=1)


def select_gradient_view_group(
    gradient: torch.Tensor,
    group: str,
) -> torch.Tensor:
    if group == "global":
        return gradient[:, :2]
    if group == "local":
        return gradient[:, 2:]
    raise ValueError(f"unknown view group: {group}")


def per_image_cosine_summary(
    left: torch.Tensor,
    right: torch.Tensor,
    *,
    suffix: str = "",
) -> dict[str, float]:
    values = per_image_cosines(left, right).cpu().numpy()
    ending = f"_{suffix}" if suffix else ""
    return {
        f"median_per_image_cosine_mse_cw{ending}": float(np.median(values)),
        f"q25_per_image_cosine_mse_cw{ending}": float(np.quantile(values, 0.25)),
        f"q75_per_image_cosine_mse_cw{ending}": float(np.quantile(values, 0.75)),
        f"fraction_images_negative_cosine{ending}": float(np.mean(values < 0.0)),
    }


def normalized_descent_step(
    z: torch.Tensor,
    gradient: torch.Tensor,
    step_fraction: float,
) -> torch.Tensor:
    z_rms = z.detach().square().mean().sqrt()
    gradient_rms = gradient.detach().square().mean().sqrt().clamp_min(EPS)
    return -step_fraction * z_rms * gradient.detach() / gradient_rms


def mse_change_after_component_step(
    z: torch.Tensor,
    mse_loss: torch.Tensor,
    g_mse: torch.Tensor,
    component_gradient: torch.Tensor,
    step_fraction: float,
    production_alignment_loss: Any,
) -> tuple[float, float]:
    if float(component_gradient.detach().norm().cpu()) <= EPS:
        return float("nan"), float("nan")
    delta = normalized_descent_step(z, component_gradient, step_fraction)
    after = production_alignment_loss((z.detach() + delta).detach(), num_global=2)
    denominator = mse_loss.detach().abs().clamp_min(EPS)
    measured = (after - mse_loss.detach()) / denominator
    predicted = (g_mse.detach() * delta).sum() / denominator
    return float(measured.cpu()), float(predicted.cpu())


def production_ep_value_and_grad(
    z_bvd: torch.Tensor,
    *,
    num_slices: int,
    projection_seed: int,
) -> tuple[float, torch.Tensor]:
    deps = gradient_dependencies()
    z = z_bvd.detach().clone().float().requires_grad_(True)
    ep = deps["SlicedEppsPulley"](
        num_slices=num_slices,
        n_points=17,
        t_max=3.0,
        gamma=0.5,
    ).to(z.device)
    ep.global_step.fill_(projection_seed)
    loss = ep(deps["flatten_view_major"](z))
    gradient = torch.autograd.grad(loss, z, retain_graph=False)[0]
    return float(loss.detach().cpu()), gradient.detach()


def validate_cw_decomposition(
    z_bvd: torch.Tensor,
    *,
    gamma: float = 0.5,
) -> dict[str, float]:
    deps = gradient_dependencies()
    z = z_bvd.detach().clone().float().requires_grad_(True)
    losses = deps["cw_component_losses"](z, gamma=gamma, num_global=2)
    production = deps["CWReg"](gamma=gamma).to(z.device)(
        deps["flatten_view_major"](z)
    )
    relative_loss_error = float(
        ((losses["full"] - production).abs() / production.detach().abs().clamp_min(EPS))
        .detach()
        .cpu()
    )
    decomposed_gradient = torch.autograd.grad(
        losses["full"], z, retain_graph=True
    )[0]
    production_gradient = torch.autograd.grad(
        production, z, retain_graph=True
    )[0]
    component_gradient = sum(
        (
            torch.autograd.grad(losses[name], z, retain_graph=True)[0]
            for name in ("diagonal", "within", "between", "target")
        ),
        torch.zeros_like(z),
    )
    gradient_cosine = tensor_cosine(decomposed_gradient, production_gradient)
    relative_gradient_error = float(
        (
            (decomposed_gradient - production_gradient).norm()
            / production_gradient.norm().clamp_min(EPS)
        )
        .detach()
        .cpu()
    )
    component_reconstruction_error = float(
        (
            (component_gradient - decomposed_gradient).norm()
            / decomposed_gradient.norm().clamp_min(EPS)
        )
        .detach()
        .cpu()
    )
    if relative_loss_error >= 1e-5:
        raise RuntimeError(
            f"CW decomposition loss validation failed: relative error={relative_loss_error:.3e}"
        )
    if gradient_cosine <= 0.99999 or relative_gradient_error >= 1e-4:
        raise RuntimeError(
            "CW decomposition gradient validation failed: "
            f"cosine={gradient_cosine:.8f}, relative error={relative_gradient_error:.3e}"
        )
    if component_reconstruction_error >= 1e-4:
        raise RuntimeError(
            "CW component gradient reconstruction failed: "
            f"relative error={component_reconstruction_error:.3e}"
        )
    return {
        "relative_loss_error": relative_loss_error,
        "gradient_cosine": gradient_cosine,
        "relative_gradient_error": relative_gradient_error,
        "component_reconstruction_error": component_reconstruction_error,
    }


def analyze_gradient_batch(
    z_cpu: torch.Tensor,
    *,
    args: argparse.Namespace,
    augmentation_id: int,
    batch_id: int,
    device: torch.device,
) -> dict[str, Any]:
    deps = gradient_dependencies()
    z = z_cpu.to(device=device, dtype=torch.float32).detach().clone().requires_grad_(True)
    production_cw = deps["CWReg"](gamma=args.cw_gamma).to(device)(
        deps["flatten_view_major"](z)
    )
    g_cw_full = torch.autograd.grad(production_cw, z, retain_graph=False)[0]
    losses = deps["cw_component_losses"](z, gamma=args.cw_gamma, num_global=2)
    mse_loss = deps["production_alignment_loss"](z, num_global=2)

    g_cw_within = torch.autograd.grad(losses["within"], z, retain_graph=True)[0]
    g_cw_between = torch.autograd.grad(losses["between"], z, retain_graph=True)[0]
    g_cw_target = torch.autograd.grad(losses["target"], z, retain_graph=True)[0]
    g_cw_exclusion = torch.autograd.grad(
        losses["exclusion_only_full"], z, retain_graph=True
    )[0]
    g_cw_group = torch.autograd.grad(losses["group_full"], z, retain_graph=True)[0]
    g_cw_within_gg = torch.autograd.grad(
        losses["within_gg"], z, retain_graph=True
    )[0]
    g_cw_within_gl = torch.autograd.grad(
        losses["within_gl"], z, retain_graph=True
    )[0]
    g_cw_within_ll = torch.autograd.grad(
        losses["within_ll"], z, retain_graph=False
    )[0]
    g_mse = torch.autograd.grad(mse_loss, z, retain_graph=False)[0]

    within_type_reconstruction_error = float(
        (
            (
                g_cw_within
                - g_cw_within_gg
                - g_cw_within_gl
                - g_cw_within_ll
            ).norm()
            / g_cw_within.norm().clamp_min(EPS)
        )
        .detach()
        .cpu()
    )
    if within_type_reconstruction_error >= 1e-4:
        raise RuntimeError(
            "CW within-pair gradient reconstruction failed: "
            f"relative error={within_type_reconstruction_error:.3e}"
        )

    mse_norm = g_mse.norm().clamp_min(EPS)
    weighted_cw = args.gradient_cw_lambda * g_cw_full
    weighted_within = args.gradient_cw_lambda * g_cw_within
    weighted_exclusion = args.gradient_cw_lambda * g_cw_exclusion
    weighted_within_gg = args.gradient_cw_lambda * g_cw_within_gg
    weighted_within_gl = args.gradient_cw_lambda * g_cw_within_gl
    weighted_within_ll = args.gradient_cw_lambda * g_cw_within_ll
    cancellation = (g_mse + weighted_cw).norm() / (
        g_mse.norm() + weighted_cw.norm()
    ).clamp_min(EPS)
    exclusion_cancellation = (g_mse + weighted_exclusion).norm() / (
        g_mse.norm() + weighted_exclusion.norm()
    ).clamp_min(EPS)

    delta_cw = normalized_descent_step(
        z, g_cw_full, args.gradient_finite_step
    )
    mse_after_cw = deps["production_alignment_loss"](
        (z.detach() + delta_cw).detach(), num_global=2
    )
    measured_mse_change = (mse_after_cw - mse_loss.detach()) / mse_loss.detach().abs().clamp_min(EPS)
    predicted_mse_change = (g_mse.detach() * delta_cw).sum() / mse_loss.detach().abs().clamp_min(EPS)

    delta_mse = normalized_descent_step(
        z, g_mse, args.gradient_finite_step
    )
    cw_after_mse = deps["CWReg"](gamma=args.cw_gamma).to(device)(
        deps["flatten_view_major"]((z.detach() + delta_mse).detach())
    )
    measured_cw_change = (cw_after_mse - production_cw.detach()) / production_cw.detach().abs().clamp_min(EPS)
    predicted_cw_change = (g_cw_full.detach() * delta_mse).sum() / production_cw.detach().abs().clamp_min(EPS)
    component_steps = {
        name: mse_change_after_component_step(
            z,
            mse_loss,
            g_mse,
            gradient,
            args.gradient_finite_step,
            deps["production_alignment_loss"],
        )
        for name, gradient in (
            ("within_cw", g_cw_within),
            ("within_gg_cw", g_cw_within_gg),
            ("within_gl_cw", g_cw_within_gl),
            ("within_ll_cw", g_cw_within_ll),
        )
    }

    row: dict[str, Any] = {
        "gradient_space": "cached_projector_embeddings",
        "parameter_gradients_computed": False,
        "mse_loss": float(mse_loss.detach().cpu()),
        "cw_loss_full": float(production_cw.detach().cpu()),
        "cw_loss_diagonal": float(losses["diagonal"].detach().cpu()),
        "cw_loss_within": float(losses["within"].detach().cpu()),
        "cw_loss_between": float(losses["between"].detach().cpu()),
        "cw_loss_target": float(losses["target"].detach().cpu()),
        "cw_loss_exclusion_only": float(
            losses["exclusion_only_full"].detach().cpu()
        ),
        "cw_loss_group_aware": float(losses["group_full"].detach().cpu()),
        "cosine_mse_cw_full": tensor_cosine(g_mse, g_cw_full),
        "cosine_mse_cw_within": tensor_cosine(g_mse, g_cw_within),
        "cosine_mse_cw_between": tensor_cosine(g_mse, g_cw_between),
        "cosine_mse_cw_target": tensor_cosine(g_mse, g_cw_target),
        "cosine_mse_cw_exclusion_only": tensor_cosine(g_mse, g_cw_exclusion),
        "cosine_mse_cw_group_aware": tensor_cosine(g_mse, g_cw_group),
        "cosine_mse_cw_within_gg": tensor_cosine(g_mse, g_cw_within_gg),
        "cosine_mse_cw_within_gl": tensor_cosine(g_mse, g_cw_within_gl),
        "cosine_mse_cw_within_ll": tensor_cosine(g_mse, g_cw_within_ll),
        "norm_g_mse": tensor_norm(g_mse),
        "norm_g_cw_full": tensor_norm(g_cw_full),
        "norm_g_cw_within": tensor_norm(g_cw_within),
        "norm_g_cw_between": tensor_norm(g_cw_between),
        "norm_g_cw_target": tensor_norm(g_cw_target),
        "norm_g_cw_exclusion_only": tensor_norm(g_cw_exclusion),
        "norm_g_cw_group_aware": tensor_norm(g_cw_group),
        "norm_g_cw_within_gg": tensor_norm(g_cw_within_gg),
        "norm_g_cw_within_gl": tensor_norm(g_cw_within_gl),
        "norm_g_cw_within_ll": tensor_norm(g_cw_within_ll),
        "weighted_cw_to_mse_norm": float(
            (weighted_cw.norm() / mse_norm).detach().cpu()
        ),
        "weighted_within_to_mse_norm": float(
            (weighted_within.norm() / mse_norm).detach().cpu()
        ),
        "weighted_exclusion_only_to_mse_norm": float(
            (weighted_exclusion.norm() / mse_norm).detach().cpu()
        ),
        "weighted_within_gg_to_mse_norm": float(
            (weighted_within_gg.norm() / mse_norm).detach().cpu()
        ),
        "weighted_within_gl_to_mse_norm": float(
            (weighted_within_gl.norm() / mse_norm).detach().cpu()
        ),
        "weighted_within_ll_to_mse_norm": float(
            (weighted_within_ll.norm() / mse_norm).detach().cpu()
        ),
        "cancellation_ratio": float(cancellation.detach().cpu()),
        "exclusion_only_cancellation_ratio": float(
            exclusion_cancellation.detach().cpu()
        ),
        "cw_projection_on_mse": float(
            ((weighted_cw.detach() * g_mse.detach()).sum() / mse_norm.square())
            .detach()
            .cpu()
        ),
        "within_projection_on_mse": float(
            ((weighted_within.detach() * g_mse.detach()).sum() / mse_norm.square())
            .detach()
            .cpu()
        ),
        "within_gg_projection_on_mse": float(
            ((weighted_within_gg.detach() * g_mse.detach()).sum() / mse_norm.square())
            .detach()
            .cpu()
        ),
        "within_gl_projection_on_mse": float(
            ((weighted_within_gl.detach() * g_mse.detach()).sum() / mse_norm.square())
            .detach()
            .cpu()
        ),
        "within_ll_projection_on_mse": float(
            ((weighted_within_ll.detach() * g_mse.detach()).sum() / mse_norm.square())
            .detach()
            .cpu()
        ),
        "within_pair_type_gradient_reconstruction_error": within_type_reconstruction_error,
        "cosine_mse_cw_global_views": tensor_cosine(
            select_gradient_view_group(g_mse, "global"),
            select_gradient_view_group(g_cw_full, "global"),
        ),
        "cosine_mse_cw_local_views": tensor_cosine(
            select_gradient_view_group(g_mse, "local"),
            select_gradient_view_group(g_cw_full, "local"),
        ),
        "cosine_mse_within_global_views": tensor_cosine(
            select_gradient_view_group(g_mse, "global"),
            select_gradient_view_group(g_cw_within, "global"),
        ),
        "cosine_mse_within_local_views": tensor_cosine(
            select_gradient_view_group(g_mse, "local"),
            select_gradient_view_group(g_cw_within, "local"),
        ),
        "measured_relative_mse_change_after_cw_step": float(
            measured_mse_change.detach().cpu()
        ),
        "predicted_relative_mse_change_after_cw_step": float(
            predicted_mse_change.detach().cpu()
        ),
        "measured_relative_cw_change_after_mse_step": float(
            measured_cw_change.detach().cpu()
        ),
        "predicted_relative_cw_change_after_mse_step": float(
            predicted_cw_change.detach().cpu()
        ),
    }
    for component_name, (measured, predicted) in component_steps.items():
        row[f"measured_relative_mse_change_after_{component_name}_step"] = measured
        row[f"predicted_relative_mse_change_after_{component_name}_step"] = predicted
    row.update(per_image_cosine_summary(g_mse, g_cw_full))

    if args.gradient_include_ep_control:
        ep_seed = stable_seed(
            "ep_control", args.seed, augmentation_id, batch_id
        )
        ep_loss, g_ep = production_ep_value_and_grad(
            z.detach(),
            num_slices=args.gradient_ep_slices,
            projection_seed=ep_seed,
        )
        weighted_ep = args.gradient_ep_lambda * g_ep.to(device)
        ep_cancellation = (g_mse + weighted_ep).norm() / (
            g_mse.norm() + weighted_ep.norm()
        ).clamp_min(EPS)
        ep_image_cosines = per_image_cosines(g_mse, g_ep.to(device))
        row.update(
            {
                "cosine_mse_ep": tensor_cosine(g_mse, g_ep.to(device)),
                "ep_loss": ep_loss,
                "norm_g_ep": tensor_norm(g_ep),
                "weighted_ep_to_mse_norm": float(
                    (weighted_ep.norm() / mse_norm).detach().cpu()
                ),
                "ep_cancellation_ratio": float(ep_cancellation.detach().cpu()),
                "fraction_images_negative_cosine_ep": float(
                    (ep_image_cosines < 0).float().mean().cpu()
                ),
            }
        )
    return row


def aggregate_gradient_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, float], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(
            (str(row["checkpoint"]), float(row["training_progress"])), []
        ).append(row)
    output: list[dict[str, Any]] = []
    for (checkpoint, progress), members in groups.items():
        aggregate: dict[str, Any] = {
            "checkpoint": checkpoint,
            "training_progress": progress,
            "num_observations": len(members),
            "gradient_space": "cached_projector_embeddings",
            "parameter_gradients_computed": False,
        }
        columns = sorted(set().union(*(member.keys() for member in members)))
        for column in columns:
            values = [numeric_value(member.get(column)) for member in members]
            finite = [value for value in values if value is not None]
            if finite and len(finite) == len(members):
                aggregate[f"{column}_median"] = float(np.median(finite))
                aggregate[f"{column}_q25"] = float(np.quantile(finite, 0.25))
                aggregate[f"{column}_q75"] = float(np.quantile(finite, 0.75))
                aggregate[f"{column}_p05"] = float(np.quantile(finite, 0.05))
                aggregate[f"{column}_p95"] = float(np.quantile(finite, 0.95))
        output.append(aggregate)
    return sorted(output, key=lambda row: float(row["training_progress"]))


def plot_gradient_results(
    aggregate: list[dict[str, Any]],
    output_dir: Path,
    *,
    include_ep: bool,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        print(f"[warn] Gradient plotting dependencies unavailable: {exc}", file=sys.stderr)
        return
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        pass

    def single_metric(metric: str, ylabel: str, *, zero_line: bool = False) -> None:
        x = np.asarray([float(row["training_progress"]) for row in aggregate])
        median = np.asarray([float(row[f"{metric}_median"]) for row in aggregate])
        q25 = np.asarray([float(row[f"{metric}_q25"]) for row in aggregate])
        q75 = np.asarray([float(row[f"{metric}_q75"]) for row in aggregate])
        figure, axis = plt.subplots(figsize=(5.5, 3.8))
        axis.plot(x, median, marker="o")
        axis.fill_between(x, q25, q75, alpha=0.2, label="batch IQR")
        if zero_line:
            axis.axhline(0.0, color="black", linewidth=1, linestyle="--")
        axis.set_xlabel("Training progress (%)")
        axis.set_ylabel(ylabel)
        axis.legend()
        figure.tight_layout()
        figure.savefig(output_dir / f"gradient_{metric}.pdf", bbox_inches="tight")
        plt.close(figure)

    single_metric("cosine_mse_cw_full", "cosine(MSE, full CW)", zero_line=True)
    single_metric("cosine_mse_cw_within", "cosine(MSE, within-image CW)", zero_line=True)
    single_metric(
        "cosine_mse_cw_exclusion_only",
        "cosine(MSE, exclusion-only CW)",
        zero_line=True,
    )
    single_metric(
        "cosine_mse_cw_group_aware",
        "cosine(MSE, group-aware CW)",
        zero_line=True,
    )
    single_metric("weighted_cw_to_mse_norm", "weighted CW / MSE gradient norm")
    single_metric(
        "weighted_within_to_mse_norm",
        "weighted within-CW / MSE gradient norm",
    )
    single_metric("cancellation_ratio", "gradient cancellation ratio")
    single_metric(
        "fraction_images_negative_cosine",
        "fraction of images with negative cosine",
    )
    single_metric("cw_projection_on_mse", "CW projection on MSE", zero_line=True)

    x = np.asarray([float(row["training_progress"]) for row in aggregate])
    figure, axis = plt.subplots(figsize=(5.5, 3.8))
    for metric, label in (
        ("cosine_mse_cw_global_views", "global views"),
        ("cosine_mse_cw_local_views", "local views"),
    ):
        axis.plot(
            x,
            [float(row[f"{metric}_median"]) for row in aggregate],
            marker="o",
            label=label,
        )
    axis.axhline(0.0, color="black", linewidth=1, linestyle="--")
    axis.set_xlabel("Training progress (%)")
    axis.set_ylabel("cosine(MSE, full CW)")
    axis.legend()
    figure.tight_layout()
    figure.savefig(
        output_dir / "gradient_global_vs_local_conflict.pdf", bbox_inches="tight"
    )
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(5.5, 3.8))
    for metric, label in (
        ("cosine_mse_cw_within_gg", "global-global"),
        ("cosine_mse_cw_within_gl", "global-local"),
        ("cosine_mse_cw_within_ll", "local-local"),
    ):
        axis.plot(
            x,
            [float(row[f"{metric}_median"]) for row in aggregate],
            marker="o",
            label=label,
        )
    axis.axhline(0.0, color="black", linewidth=1, linestyle="--")
    axis.set_xlabel("Training progress (%)")
    axis.set_ylabel("cosine(MSE, within-pair CW)")
    axis.legend()
    figure.tight_layout()
    figure.savefig(
        output_dir / "gradient_within_pair_type_conflict.pdf",
        bbox_inches="tight",
    )
    plt.close(figure)
    if include_ep:
        single_metric("cosine_mse_ep", "cosine(MSE, EP)", zero_line=True)


def gradient_summary(aggregate: list[dict[str, Any]], *, include_ep: bool) -> str:
    latest = aggregate[-1]
    full_cosines = [float(row["cosine_mse_cw_full_median"]) for row in aggregate]
    exclusion_cosines = [
        float(row["cosine_mse_cw_exclusion_only_median"]) for row in aggregate
    ]
    weighted = [float(row["weighted_cw_to_mse_norm_median"]) for row in aggregate]
    signs_mixed = min(full_cosines) < 0.0 <= max(full_cosines)
    if signs_mixed:
        conclusion = "evidence is mixed across checkpoints"
    elif min(full_cosines) >= -0.05 or max(weighted) < 0.01:
        conclusion = "no meaningful embedding-gradient conflict detected"
    elif (
        np.median(exclusion_cosines) > np.median(full_cosines)
        and float(latest["weighted_within_to_mse_norm_median"]) >= 0.05
    ):
        conclusion = "conflict is concentrated in same-image CW interactions"
    elif np.median(full_cosines) <= -0.1 and np.median(weighted) >= 0.05:
        conclusion = "CW and MSE measurably conflict"
    else:
        conclusion = "weak or intermittent conflict"

    lines = [
        "# Embedding-gradient conflict analysis",
        "",
        "All gradients in this report are with respect to cached projector outputs. No backbone or model-parameter gradients were computed.",
        "",
        f"- Full MSE/CW cosine at the latest checkpoint: {float(latest['cosine_mse_cw_full_median']):.3f}, "
        f"batch/draw IQR [{float(latest['cosine_mse_cw_full_q25']):.3f}, {float(latest['cosine_mse_cw_full_q75']):.3f}]; "
        f"checkpoint range {min(full_cosines):.3f} to {max(full_cosines):.3f}.",
        f"- Removing only the within-image term changes the latest median cosine from "
        f"{float(latest['cosine_mse_cw_full_median']):.3f} to "
        f"{float(latest['cosine_mse_cw_exclusion_only_median']):.3f}; its cancellation ratio is "
        f"{float(latest['exclusion_only_cancellation_ratio_median']):.3f}.",
        f"- The separately renormalized group-aware estimator has median cosine "
        f"{float(latest['cosine_mse_cw_group_aware_median']):.3f}. It changes both pair inclusion "
        "and the between-image normalization.",
        f"- GG/GL/LL within-term cosines: "
        f"{float(latest['cosine_mse_cw_within_gg_median']):.3f} / "
        f"{float(latest['cosine_mse_cw_within_gl_median']):.3f} / "
        f"{float(latest['cosine_mse_cw_within_ll_median']):.3f}.",
        f"- Weighted full/within/GG/GL/LL CW-to-MSE norm ratios: "
        f"{float(latest['weighted_cw_to_mse_norm_median']):.3f} / "
        f"{float(latest['weighted_within_to_mse_norm_median']):.3f} / "
        f"{float(latest['weighted_within_gg_to_mse_norm_median']):.3f} / "
        f"{float(latest['weighted_within_gl_to_mse_norm_median']):.3f} / "
        f"{float(latest['weighted_within_ll_to_mse_norm_median']):.3f}.",
        f"- Combined-gradient cancellation ratio: {float(latest['cancellation_ratio_median']):.3f}.",
        f"- Global/local view cosines: {float(latest['cosine_mse_cw_global_views_median']):.3f} / {float(latest['cosine_mse_cw_local_views_median']):.3f}.",
        f"- Fraction of images with negative full-CW cosine: {float(latest['fraction_images_negative_cosine_median']):.3f}.",
        f"- Finite CW step, measured/predicted relative MSE change: {float(latest['measured_relative_mse_change_after_cw_step_median']):.3e} / {float(latest['predicted_relative_mse_change_after_cw_step_median']):.3e}.",
        f"- Finite within-CW step, measured/predicted relative MSE change: "
        f"{float(latest['measured_relative_mse_change_after_within_cw_step_median']):.3e} / "
        f"{float(latest['predicted_relative_mse_change_after_within_cw_step_median']):.3e}.",
        f"- Finite GG/GL/LL steps, measured relative MSE changes: "
        f"{float(latest['measured_relative_mse_change_after_within_gg_cw_step_median']):.3e} / "
        f"{float(latest['measured_relative_mse_change_after_within_gl_cw_step_median']):.3e} / "
        f"{float(latest['measured_relative_mse_change_after_within_ll_cw_step_median']):.3e}.",
        f"- Finite MSE step, measured/predicted relative CW change: {float(latest['measured_relative_cw_change_after_mse_step_median']):.3e} / {float(latest['predicted_relative_cw_change_after_mse_step_median']):.3e}.",
    ]
    if include_ep:
        lines.append(
            f"- EP control cosine and negative-image fraction: {float(latest['cosine_mse_ep_median']):.3f} / "
            f"{float(latest['fraction_images_negative_cosine_ep_median']):.3f}."
        )
    lines.extend(
        [
            "",
            "## Conclusion",
            "",
            conclusion,
            "",
            "Embedding-space conflict does not necessarily imply parameter-space or optimization-level conflict.",
        ]
    )
    return "\n".join(lines) + "\n"


def load_records_for_analysis(args: argparse.Namespace) -> list[CacheRecord]:
    checkpoint_mode = bool(args.checkpoint_dir or args.checkpoint)
    records = discover_cache_records(args) if not checkpoint_mode or args.cache_file else []
    if not records and checkpoint_mode:
        records = extract_checkpoint_records(args)
    if not records:
        raise ValueError("no compatible embedding caches or checkpoints were found")
    return records


def run_gradient_analysis(args: argparse.Namespace) -> dict[str, Path]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = load_records_for_analysis(args)
    view_order, num_local_views = validate_record_view_layouts(records)
    feature_dimensions = {int(record.embeddings.shape[-1]) for record in records}
    if len(feature_dimensions) != 1:
        raise ValueError("all gradient-analysis caches must have the same original feature dimension")
    feature_dimension = feature_dimensions.pop()
    draws = min(args.num_augmentation_draws, min(record.embeddings.shape[0] for record in records))
    num_views = int(records[0].embeddings.shape[2])
    flattened_count = args.gradient_batch_size * num_views
    print(f"[gradient-plan] checkpoints: {len(records)}", flush=True)
    print(f"[gradient-plan] augmentation draws: {draws}", flush=True)
    print(f"[gradient-plan] gradient batches per draw: {args.gradient_num_batches}", flush=True)
    print(f"[gradient-plan] batch size: {args.gradient_batch_size}", flush=True)
    print(f"[gradient-plan] views: {num_views}", flush=True)
    print(f"[gradient-plan] original feature dimension: {feature_dimension}", flush=True)
    print(f"[gradient-plan] flattened CW sample count: {flattened_count}", flush=True)
    print(f"[gradient-plan] EP control enabled: {args.gradient_include_ep_control}", flush=True)

    device = torch.device(args.device)
    synthetic = torch.randn(
        4, 3, min(feature_dimension, 8), generator=torch.Generator().manual_seed(args.seed)
    ).to(device)
    synthetic_validation = validate_cw_decomposition(synthetic)
    first_batches = deterministic_gradient_batches(
        min(record.embeddings.shape[1] for record in records),
        args.gradient_batch_size,
        args.gradient_num_batches,
        args.seed,
    )
    real_validation = validate_cw_decomposition(
        records[0].embeddings[0].index_select(0, first_batches[0]).to(device)
    )
    print(
        "[gradient-validation] synthetic and real CW decomposition checks passed",
        flush=True,
    )

    rows: list[dict[str, Any]] = []
    shared_batches = deterministic_gradient_batches(
        min(record.embeddings.shape[1] for record in records),
        args.gradient_batch_size,
        args.gradient_num_batches,
        args.seed,
    )
    for record in records:
        for augmentation_id in range(draws):
            embeddings = record.embeddings[augmentation_id]
            for batch_id, image_ids in enumerate(shared_batches):
                z_batch = embeddings.index_select(0, image_ids)
                metrics = analyze_gradient_batch(
                    z_batch,
                    args=args,
                    augmentation_id=augmentation_id,
                    batch_id=batch_id,
                    device=device,
                )
                rows.append(
                    {
                        "checkpoint": record.label,
                        "checkpoint_path": str(record.metadata.get("checkpoint_path", "")),
                        "training_progress": record.training_progress,
                        "augmentation_id": augmentation_id,
                        "batch_id": batch_id,
                        "batch_size": z_batch.shape[0],
                        "num_views": z_batch.shape[1],
                        "feature_dimension": z_batch.shape[2],
                        "effective_sample_count": z_batch.shape[0] * z_batch.shape[1],
                        **metrics,
                    }
                )
                print(
                    f"[gradient] {record.label} aug={augmentation_id} batch={batch_id}",
                    flush=True,
                )

    aggregate = aggregate_gradient_rows(rows)
    raw_path = args.output_dir / "gradient_conflict_raw.csv"
    aggregate_path = args.output_dir / "gradient_conflict_aggregate.csv"
    summary_path = args.output_dir / "gradient_conflict_summary.md"
    write_csv(raw_path, rows)
    write_csv(aggregate_path, aggregate)
    summary_path.write_text(
        gradient_summary(aggregate, include_ep=args.gradient_include_ep_control)
    )
    plot_gradient_results(
        aggregate,
        args.output_dir,
        include_ep=args.gradient_include_ep_control,
    )
    gradient_config = {
        "analyzed_checkpoints": [
            {
                "label": record.label,
                "checkpoint_path": str(record.metadata.get("checkpoint_path", "")),
                "cache_path": str(record.path),
                "cache_metadata": json_compatible(record.metadata),
            }
            for record in records
        ],
        "number_of_checkpoints": len(records),
        "number_of_augmentation_draws": draws,
        "number_of_gradient_batches": args.gradient_num_batches,
        "batch_size": args.gradient_batch_size,
        "number_of_views": num_views,
        "view_order": view_order,
        "number_of_global_views": 2,
        "number_of_local_views": num_local_views,
        "original_feature_dimension": feature_dimension,
        "flattened_cw_sample_count": flattened_count,
        "cw_gamma": args.cw_gamma,
        "cw_lambda": args.gradient_cw_lambda,
        "ep_lambda": args.gradient_ep_lambda,
        "ep_slices": args.gradient_ep_slices,
        "finite_step_fraction": args.gradient_finite_step,
        "model_mode": args.model_mode,
        "seed": args.seed,
        "device": str(device),
        "ep_control_enabled": args.gradient_include_ep_control,
        "untrained_model_note": (
            "An untrained cache extracted by this script is reconstructed deterministically "
            "from the analysis seed and is not necessarily the training run's exact "
            "initialization unless an epoch-zero checkpoint was supplied."
        ),
        "synthetic_decomposition_validation": synthetic_validation,
        "real_decomposition_validation": real_validation,
    }
    (args.output_dir / "gradient_run_config.json").write_text(
        json.dumps(gradient_config, indent=2)
    )
    return {
        "gradient_raw": raw_path,
        "gradient_aggregate": aggregate_path,
        "gradient_summary": summary_path,
    }


def run_structure_analysis(args: argparse.Namespace) -> dict[str, Path]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    null_cache_dir = args.output_dir / "null_cache"
    checkpoint_mode = bool(args.checkpoint_dir or args.checkpoint)
    records = discover_cache_records(args) if not checkpoint_mode or args.cache_file else []
    if not records and checkpoint_mode:
        records = extract_checkpoint_records(args)
    if not records:
        raise ValueError("no compatible embedding caches or checkpoints were found")
    view_order, num_local_views = validate_record_view_layouts(records)
    dimensions = {record.embeddings.shape[-1] for record in records}
    if len(dimensions) != 1:
        raise ValueError(f"all caches must have the same feature dimension, got {sorted(dimensions)}")
    available = min(record.embeddings.shape[1] for record in records)
    if max(args.image_counts) > available:
        raise ValueError(f"largest requested image count is {max(args.image_counts)}, cache has {available}")
    if max(args.image_counts) > args.num_images:
        raise ValueError("--image-counts cannot exceed --num-images")
    original_dimension = dimensions.pop()
    projection = deterministic_projection(original_dimension, args.projection_dim, args.seed)
    shape_directions = fixed_shape_directions(args.projection_dim, args.num_shape_directions, args.seed)

    raw_rows: list[dict[str, Any]] = []
    for record in records:
        draws = min(args.num_augmentation_draws, record.embeddings.shape[0])
        for augmentation_id in range(draws):
            projected = torch.einsum("ivd,dq->ivq", record.embeddings[augmentation_id], projection)
            for partition_id in range(args.num_partitions):
                for image_count in args.image_counts:
                    calibration_ids, evaluation_ids = partition_and_split_ids(
                        projected.shape[0],
                        image_count,
                        partition_id,
                        args.calibration_fraction,
                        args.seed,
                    )
                    for pair_type in PAIR_TYPES:
                        assignment_seed = stable_seed(
                            "pairs", args.seed, augmentation_id, partition_id, image_count, pair_type
                        )
                        metrics = analyze_observation(
                            projected,
                            calibration_ids=calibration_ids,
                            evaluation_ids=evaluation_ids,
                            pair_type=pair_type,
                            assignment_seed=assignment_seed,
                            args=args,
                            shape_directions=shape_directions,
                        )
                        raw_rows.append(
                            {
                                "checkpoint": record.label,
                                "checkpoint_path": str(record.metadata.get("checkpoint_path", "")),
                                "training_progress": record.training_progress,
                                "augmentation_id": augmentation_id,
                                "partition_id": partition_id,
                                "image_count": image_count,
                                "calibration_count": len(calibration_ids),
                                "evaluation_count": len(evaluation_ids),
                                "pair_type": pair_type,
                                "projected_dimension": args.projection_dim,
                                **metrics,
                            }
                        )
                        print(
                            f"[analysis] {record.label} aug={augmentation_id} partition={partition_id} "
                            f"n={image_count} pair={pair_type}",
                            flush=True,
                        )

    runtime_plan = null_runtime_plan(
        raw_rows, args, checkpoint_count=len(records), emit=True
    )
    null_output = apply_grouped_null_calibration(raw_rows, args, null_cache_dir)
    duplicate, duplicate_nulls = duplicate_sanity_checks(
        records[0], projection, args, null_cache_dir
    )
    null_output.update(duplicate_nulls)
    for row in raw_rows:
        row.update(duplicate[int(row["image_count"])])

    aggregate = aggregate_rows(raw_rows)
    null_rows_flat: list[dict[str, Any]] = []
    for key, repetitions in null_output.items():
        for repetition, row in enumerate(repetitions):
            null_rows_flat.append({"null_key": key, "repetition": repetition, **row})

    raw_path = args.output_dir / "joint_cw_plausibility_raw.csv"
    aggregate_path = args.output_dir / "joint_cw_plausibility_aggregate.csv"
    null_path = args.output_dir / "joint_cw_plausibility_nulls.csv"
    summary_path = args.output_dir / "joint_cw_plausibility_summary.md"
    write_csv(raw_path, raw_rows)
    write_csv(aggregate_path, aggregate)
    write_csv(null_path, null_rows_flat)
    summary_path.write_text(summary_report(aggregate, duplicate))
    run_config = {
        **{
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
            if key not in {"cache_file", "checkpoint"}
        },
        "cache_file": [str(path) for path in args.cache_file],
        "checkpoint": [str(path) for path in args.checkpoint],
        "fixed_projection_shape": list(projection.shape),
        "fixed_projection_sha1": hashlib.sha1(projection.numpy().tobytes()).hexdigest(),
        "maximum_cw_normality_samples_observed": MAX_CW_SAMPLES_SEEN,
        "analyzed_checkpoints": [record.label for record in records],
        "analyzed_cache_paths": [str(record.path) for record in records],
        "view_order": view_order,
        "number_of_global_views": 2,
        "number_of_local_views": num_local_views,
        "null_runtime_plan": runtime_plan,
    }
    (args.output_dir / "run_config.json").write_text(json.dumps(run_config, indent=2))
    plot_results(raw_rows, args.output_dir)
    return {
        "raw": raw_path,
        "aggregate": aggregate_path,
        "nulls": null_path,
        "summary": summary_path,
    }


def run(args: argparse.Namespace) -> dict[str, Path]:
    outputs: dict[str, Path] = {}
    if args.analysis_mode in {"structure", "both"}:
        outputs.update(run_structure_analysis(args))
    if args.analysis_mode in {"gradients", "both"}:
        outputs.update(run_gradient_analysis(args))
    return outputs


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    outputs = run(args)
    print("[done] " + ", ".join(f"{name}={path}" for name, path in outputs.items()), flush=True)


if __name__ == "__main__":
    main()
