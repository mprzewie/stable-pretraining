#!/usr/bin/env python3
"""Dump LeJEPA W&B sweep results to CSV.

Environment defaults match the local submit scripts:

  WANDB_API_KEY       required
  WANDB_ENTITY        default: gmum
  WANDB_PROJECT       default: spt_cw_jepa
  WANDB_GROUPS        optional comma/space-separated exact group names
  WANDB_GROUP_PREFIX  optional group prefix used to narrow groups
  WANDB_CSV           default: figures/wandb_results.csv

Only finished W&B runs are considered. If no group filter is supplied, all
non-empty groups with finished runs in the project are queried. Rows are
group-level experiments; matching ``-pretrain``, ``-probe``, and ``-probe_dino``
runs are merged by seed/base name inside each group, then probe metrics are
reported as group means and standard deviations.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import statistics
from collections import defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

try:
    from tqdm import tqdm
except ImportError:

    def tqdm(iterable, **_: Any):
        return iterable


STAGES = ("pretrain", "probe_dino", "probe")
FIELDNAMES = [
    "run_name",
    "dataset",
    "backbone",
    "sigreg",
    "gamma",
    "lambda",
    "lp_classic",
    "lp_classic_std",
    "lp_dino",
    "lp_dino_std",
    "pretraining_epochs",
    "batch_size",
    "n_global_views",
    "n_local_views",
    "tags",
]


def _split_env_list(value: str | None) -> list[str]:
    if not value:
        return []
    return [item for item in re.split(r"[\s,]+", value.strip()) if item]


def _flatten(obj: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(obj, Mapping):
        out = {}
        for key, value in obj.items():
            full_key = f"{prefix}.{key}" if prefix else str(key)
            out.update(_flatten(value, full_key))
        return out
    return {prefix: obj}


def _summary_dict(run: Any) -> dict[str, Any]:
    summary = getattr(run, "summary", {})
    if hasattr(summary, "_json_dict"):
        return dict(summary._json_dict)
    return dict(summary)


def _config_dict(run: Any) -> dict[str, Any]:
    config = dict(getattr(run, "config", {}) or {})
    return _flatten(config)


def _first(mapping: Mapping[str, Any], keys: Iterable[str]) -> Any:
    for key in keys:
        value = mapping.get(key)
        if value not in (None, ""):
            return value
    return ""


def _metric(summary: Mapping[str, Any], candidates: Iterable[str]) -> Any:
    value = _first(summary, candidates)
    if isinstance(value, Mapping) and "value" in value:
        return value["value"]
    return value


def _is_empty(value: Any) -> bool:
    if value in (None, ""):
        return True
    if isinstance(value, str) and value.lower() in {"null", "none", "nan"}:
        return True
    return False


def _as_float(value: Any) -> float | None:
    if _is_empty(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalise_tag(value: str) -> str:
    value = value.removeprefix("g").removeprefix("l")
    if value in {"silverman", "silverm", "null", "none", "None"}:
        return "silverman"
    return value.replace("p", ".")


def _parse_group(group: str) -> dict[str, str]:
    match = re.search(r"-(ep|cw)-g([^-]+)-l([^-]+)$", group)
    if not match:
        return {"sigreg": "", "gamma": "", "lambda": ""}
    sigreg, gamma, lamb = match.groups()
    return {
        "sigreg": sigreg,
        "gamma": _normalise_tag(gamma),
        "lambda": _normalise_tag(lamb),
    }


def _stage_from_run_name(name: str) -> str:
    for stage in STAGES:
        if name.endswith(f"-{stage}"):
            return stage
    return ""


def _base_run_name(name: str) -> str:
    stage = _stage_from_run_name(name)
    if not stage:
        return name
    return name[: -(len(stage) + 1)]


def _infer_dataset(config: Mapping[str, Any], group: str) -> str:
    dataset = _first(
        config,
        (
            "dataset",
            "DATASET",
            "data.train.dataset._args_.0",
            "data.train.dataset._args_",
            "data.train.dataset.path",
        ),
    )
    if dataset:
        return str(dataset)
    if "inet100" in group or "imagenet100" in group:
        return "imagenet100"
    if "inet10" in group or "imagenet10" in group:
        return "imagenet10"
    return ""


def _infer_backbone(config: Mapping[str, Any]) -> str:
    return str(
        _first(
            config,
            (
                "encoder_name",
                "ENCODER_NAME",
                "module.model.encoder_name",
                "module.backbone.model_name",
            ),
        )
    )


def _count_view_names(config: Mapping[str, Any], prefix: str) -> int | str:
    names = set()
    pattern = re.compile(rf"^{re.escape(prefix)}_\d+$")
    for value in config.values():
        values = value if isinstance(value, list) else [value]
        for item in values:
            if isinstance(item, str) and pattern.match(item):
                names.add(item)
    return len(names) if names else ""


def _row_for(base_name: str, runs: Mapping[str, Any]) -> dict[str, Any]:
    pretrain = runs.get("pretrain")
    classic = runs.get("probe")
    dino = runs.get("probe_dino")
    reference = pretrain or classic or dino
    group = getattr(reference, "group", "") or ""
    parsed = _parse_group(group)
    tags = sorted(
        {
            str(tag)
            for run in runs.values()
            for tag in (getattr(run, "tags", None) or [])
        }
    )

    configs = {
        stage: _config_dict(run)
        for stage, run in runs.items()
        if run is not None
    }
    pretrain_config = configs.get("pretrain", {})
    reference_config = (
        configs.get("probe") or configs.get("probe_dino") or pretrain_config
    )

    classic_summary = _summary_dict(classic) if classic is not None else {}
    dino_summary = _summary_dict(dino) if dino is not None else {}
    dataset = _infer_dataset(reference_config, group)
    dataset_metric_suffixes = ["in-domain", dataset, "imagenet100", "imagenet10", "val"]

    classic_keys = [
        f"eval/linear_probe_simple_top1_epoch/{suffix}"
        for suffix in dataset_metric_suffixes
        if suffix
    ]
    classic_keys += [
        f"eval/linear_probe_top1_epoch/{suffix}"
        for suffix in dataset_metric_suffixes
        if suffix
    ]
    classic_keys.append("eval/linear_probe_top1_epoch")
    dino_keys = [
        f"eval/linear_probe_dinov3_style_max_top1_epoch/{suffix}"
        for suffix in dataset_metric_suffixes
        if suffix
    ]

    n_global = _count_view_names(pretrain_config, "global")
    n_local = _count_view_names(pretrain_config, "local")
    if n_global == "":
        n_global = os.environ.get("DEFAULT_GLOBAL_VIEWS", "2")
    if n_local == "":
        n_local = os.environ.get("DEFAULT_LOCAL_VIEWS", "6")

    sigreg = _first(pretrain_config, ("sigreg", "module.model.sigreg", "SIGREG"))
    gamma = _first(
        pretrain_config,
        (
            "override_sr_gamma",
            "module.model.override_sr_gamma",
            "OVERRIDE_SR_GAMMA",
        ),
    )
    lamb = _first(pretrain_config, ("lamb", "module.model.lamb", "LAMB"))

    return {
        "run_name": base_name,
        "_group": group,
        "dataset": dataset,
        "backbone": _infer_backbone(reference_config),
        "sigreg": parsed["sigreg"] if _is_empty(sigreg) else sigreg,
        "gamma": parsed["gamma"] if _is_empty(gamma) else gamma,
        "lambda": parsed["lambda"] if _is_empty(lamb) else lamb,
        "lp_classic": _metric(classic_summary, classic_keys),
        "lp_classic_std": "",
        "lp_dino": _metric(dino_summary, dino_keys),
        "lp_dino_std": "",
        "pretraining_epochs": _first(
            pretrain_config,
            ("trainer.max_epochs", "max_epochs", "epochs", "EPOCHS"),
        ),
        "batch_size": _first(
            pretrain_config,
            ("batch_size", "BATCH_SIZE", "data.train.batch_size"),
        ),
        "n_global_views": n_global,
        "n_local_views": n_local,
        "tags": ",".join(tags),
    }


def _mean(values: list[float]) -> float | str:
    return statistics.mean(values) if values else ""


def _std(values: list[float]) -> float | str:
    return statistics.stdev(values) if len(values) > 1 else ""


def _most_complete_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return max(
        rows,
        key=lambda row: sum(
            1
            for key, value in row.items()
            if key != "_group" and not _is_empty(value)
        ),
    )


def _aggregate_group_row(group: str, seed_rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not seed_rows:
        return None

    classic_values = [
        value
        for value in (_as_float(row.get("lp_classic")) for row in seed_rows)
        if value is not None
    ]
    dino_values = [
        value
        for value in (_as_float(row.get("lp_dino")) for row in seed_rows)
        if value is not None
    ]
    if not classic_values and not dino_values:
        return None

    reference = dict(_most_complete_row(seed_rows))
    reference["run_name"] = group
    reference["lp_classic"] = _mean(classic_values)
    reference["lp_classic_std"] = _std(classic_values)
    reference["lp_dino"] = _mean(dino_values)
    reference["lp_dino_std"] = _std(dino_values)
    reference["tags"] = ",".join(
        sorted(
            {
                tag
                for row in seed_rows
                for tag in str(row.get("tags", "")).split(",")
                if tag
            }
        )
    )
    reference.pop("_group", None)
    return reference


def _runs_for_group(api: Any, path: str, group: str) -> list[Any]:
    return list(api.runs(path, filters={"state": "finished", "group": group}))


def _discover_groups(api: Any, path: str, prefix: str = "") -> list[str]:
    groups = set()
    desc = (
        f"Discovering groups in {path}"
        f"{f' prefix={prefix!r}' if prefix else ''}"
    )
    for run in tqdm(api.runs(path, filters={"state": "finished"}), desc=desc):
        group = getattr(run, "group", "") or ""
        if group and group.startswith(prefix):
            groups.add(group)
    return sorted(groups)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entity", default=os.environ.get("WANDB_ENTITY", "gmum"))
    parser.add_argument(
        "--project",
        default=os.environ.get("WANDB_PROJECT", "spt_cw_jepa"),
    )
    parser.add_argument(
        "--output",
        default=os.environ.get("WANDB_CSV", "figures/wandb_results.csv"),
    )
    parser.add_argument(
        "--groups",
        nargs="*",
        default=_split_env_list(os.environ.get("WANDB_GROUPS")),
    )
    parser.add_argument("--group-prefix", default=os.environ.get("WANDB_GROUP_PREFIX", ""))
    args = parser.parse_args()

    api_key = os.environ.get("WANDB_API_KEY")
    if not api_key:
        raise SystemExit("WANDB_API_KEY must be set.")

    import wandb

    wandb.login(key=api_key, relogin=True)
    api = wandb.Api()
    path = f"{args.entity}/{args.project}"

    groups = args.groups or _discover_groups(api, path, args.group_prefix)

    rows: list[dict[str, Any]] = []
    for group in tqdm(groups, desc=f"Querying {len(groups)} group(s)"):
        runs = _runs_for_group(api, path, group)
        by_base: dict[str, dict[str, Any]] = defaultdict(dict)
        for run in tqdm(runs, desc=group, leave=False):
            stage = _stage_from_run_name(run.name)
            if not stage:
                continue
            by_base[_base_run_name(run.name)][stage] = run
        seed_rows = [
            _row_for(base_name, grouped_runs)
            for base_name, grouped_runs in sorted(by_base.items())
            if "probe" in grouped_runs or "probe_dino" in grouped_runs
        ]
        group_row = _aggregate_group_row(group, seed_rows)
        if group_row is not None:
            rows.append(group_row)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
