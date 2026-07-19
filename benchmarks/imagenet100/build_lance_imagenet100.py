#!/usr/bin/env python
"""Build Lance-backed ImageNet100 splits from the HuggingFace dataset cache."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import datasets

from stable_pretraining.data.images import build_lance_image_dataset


def _none_if_null(value: str | None) -> str | None:
    if value is None or value.lower() in {"", "none", "null"}:
        return None
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    storage_root = Path(os.environ.get("STORAGE_ROOT", Path.home() / "storage"))
    default_cache = storage_root / "datasets" / "stable-pretraining" / "imagenet100"
    default_out = storage_root / "datasets" / "stable-pretraining" / "imagenet100_lance"

    parser.add_argument(
        "--dataset", default=os.environ.get("DATASET_PATH", "clane9/imagenet-100")
    )
    parser.add_argument(
        "--revision", default=_none_if_null(os.environ.get("DATASET_REVISION"))
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(os.environ.get("HF_CACHE_DIR", default_cache)),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(os.environ.get("LANCE_IMAGENET100_DIR", default_out)),
    )
    parser.add_argument("--splits", nargs="+", default=["train", "validation"])
    parser.add_argument("--image-format", choices=["jpeg", "webp"], default="jpeg")
    parser.add_argument("--quality", type=int, default=90)
    parser.add_argument("--max-size", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument(
        "--workers",
        type=int,
        default=int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1)),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for split in args.splits:
        out = args.output_dir / f"{split}.lance"
        print(f"Loading {args.dataset} split={split} cache_dir={args.cache_dir}")
        ds = datasets.load_dataset(
            args.dataset,
            split=split,
            revision=args.revision,
            cache_dir=str(args.cache_dir),
        )
        print(f"Building {out}")
        build_lance_image_dataset(
            ds,
            out,
            image_format=args.image_format,
            quality=args.quality,
            max_size=args.max_size,
            batch_size=args.batch_size,
            workers=args.workers,
            overwrite=args.overwrite,
        )


if __name__ == "__main__":
    main()
