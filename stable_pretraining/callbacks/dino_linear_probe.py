from __future__ import annotations

import math
import re
import types
from collections.abc import Iterable
from typing import Any

import torch
from lightning.pytorch import LightningModule
from loguru import logger as logging

from .utils import TrainableCallback, detach_tensors, log_header


def _as_list(value: Any) -> list[Any]:
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    if isinstance(value, Iterable):
        return list(value)
    return [value]


def _tag(value: Any) -> str:
    text = f"{float(value):.0e}" if isinstance(value, float) else str(value)
    text = text.replace("-", "m").replace("+", "")
    return re.sub(r"[^0-9A-Za-z]+", "p", text).strip("p")


class DinoStyleLinearProbe(TrainableCallback):
    """DINO-style linear probing sweep over feature variants and learning rates.

    The callback trains all linear classifiers in one pass over the same frozen
    backbone features and augmented images.
    """

    def __init__(
        self,
        module: LightningModule,
        name: str,
        target: str,
        num_classes: int,
        embed_dim: int,
        learning_rates: Iterable[float],
        n_last_blocks: Iterable[int] = (1, 2, 4),
        pools: Iterable[str] = ("cls", "avgpool"),
        backbone_attr: str = "backbone",
        is_vit: bool | None = None,
        batch_size: int = 256,
        lr_scale_base: int = 256,
        optimizers: Iterable[str] = ("sgd",),
        momentum: float = 0.9,
        weight_decay: float = 0.0,
        adamw_weight_decay: float = 0.0,
        eval_dataset_name: str | None = None,
        in_domain_dataset_name: str | None = None,
        log_train_metrics: bool = False,
    ) -> None:
        self.target = target
        self.num_classes = int(num_classes)
        self.embed_dim = int(embed_dim)
        self.learning_rates = [float(lr) for lr in _as_list(learning_rates)]
        self.n_last_blocks = [int(n) for n in _as_list(n_last_blocks)]
        self.pools = [str(pool) for pool in _as_list(pools)]
        self.backbone_attr = backbone_attr
        self.is_vit = is_vit
        self.batch_size = int(batch_size)
        self.lr_scale_base = int(lr_scale_base)
        self.optimizer_names = [str(opt).lower() for opt in _as_list(optimizers)]
        self.momentum = float(momentum)
        self.weight_decay = float(weight_decay)
        self.adamw_weight_decay = float(adamw_weight_decay)
        self.eval_dataset_name = eval_dataset_name
        self.in_domain_dataset_name = in_domain_dataset_name
        self.log_train_metrics = bool(log_train_metrics)
        self._head_specs = self._build_head_specs()
        self._val_stats: dict[str, dict[str, torch.Tensor]] = {}

        super().__init__(
            module=module,
            name=name,
            optimizer=None,
            scheduler={"type": "CosineAnnealingLR", "T_max": 1, "eta_min": 0.0},
        )

        log_header("DinoStyleLinearProbe")
        logging.info(f"  name: {self.name}")
        logging.info(f"  requested heads: {len(self._head_specs)}")
        logging.info(f"  target: {self.target}")
        logging.info(f"  eval_dataset_name: {self.eval_dataset_name}")
        logging.info(f"  in_domain_dataset_name: {self.in_domain_dataset_name}")
        self.wrap_forward(module)

    def _build_head_specs(self) -> list[dict[str, Any]]:
        specs = []
        for n_blocks in self.n_last_blocks:
            for pool in self.pools:
                if pool not in {"cls", "avgpool", "cls_avgpool"}:
                    raise ValueError(
                        "DinoStyleLinearProbe pools must be one of "
                        "'cls', 'avgpool', or 'cls_avgpool'"
                    )
                dim_multiplier = n_blocks
                if pool == "cls_avgpool":
                    dim_multiplier += 1
                dim = self.embed_dim * dim_multiplier
                for optimizer_name in self.optimizer_names:
                    if optimizer_name not in {"sgd", "adamw"}:
                        raise ValueError(
                            "DinoStyleLinearProbe optimizers must be 'sgd' or 'adamw'"
                        )
                    for lr in self.learning_rates:
                        specs.append(
                            {
                                "name": (
                                    f"{optimizer_name}_{n_blocks}blk_{pool}_lr{_tag(lr)}"
                                ),
                                "optimizer": optimizer_name,
                                "n_last_blocks": n_blocks,
                                "pool": pool,
                                "lr": lr,
                                "dim": dim,
                            }
                        )
        return specs

    def configure_model(self, pl_module: LightningModule) -> torch.nn.Module:
        backbone = getattr(pl_module, self.backbone_attr)
        if not self._backbone_is_vit(backbone):
            self.n_last_blocks = [1]
            self.pools = ["avgpool"]
            self._head_specs = self._build_head_specs()
        logging.info(f"  configured heads: {len(self._head_specs)}")
        heads = torch.nn.ModuleDict()
        for spec in self._head_specs:
            head = torch.nn.Linear(spec["dim"], self.num_classes)
            torch.nn.init.normal_(head.weight, mean=0.0, std=0.01)
            torch.nn.init.zeros_(head.bias)
            heads[spec["name"]] = head
        return heads

    def setup_optimizer(self, pl_module: LightningModule):
        world_size = getattr(getattr(pl_module, "trainer", None), "world_size", 1) or 1
        scale = (self.batch_size * int(world_size)) / self.lr_scale_base
        groups_by_optimizer = {"sgd": [], "adamw": []}
        for spec in self._head_specs:
            groups_by_optimizer[spec["optimizer"]].append(
                {
                    "params": self.module[spec["name"]].parameters(),
                    "lr": spec["lr"] * scale,
                    "initial_lr": spec["lr"] * scale,
                }
            )
        optimizers = []
        if groups_by_optimizer["sgd"]:
            optimizers.append(
                torch.optim.SGD(
                    groups_by_optimizer["sgd"],
                    momentum=self.momentum,
                    weight_decay=self.weight_decay,
                )
            )
        if groups_by_optimizer["adamw"]:
            optimizers.append(
                torch.optim.AdamW(
                    groups_by_optimizer["adamw"],
                    weight_decay=self.adamw_weight_decay,
                )
            )
        return optimizers[0] if len(optimizers) == 1 else optimizers

    def setup_scheduler(self, optimizer, pl_module: LightningModule):
        trainer = getattr(pl_module, "trainer", None)
        t_max = getattr(trainer, "estimated_stepping_batches", None)
        t_max = max(int(t_max or 1), 1)
        if isinstance(optimizer, list):
            return [
                torch.optim.lr_scheduler.CosineAnnealingLR(
                    opt,
                    T_max=t_max,
                    eta_min=0.0,
                )
                for opt in optimizer
            ]
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=t_max,
            eta_min=0.0,
        )

    def _backbone_is_vit(self, backbone: torch.nn.Module) -> bool:
        if self.is_vit is not None:
            return self.is_vit
        model_name = getattr(backbone, "default_cfg", {}).get("architecture", "")
        return hasattr(backbone, "get_intermediate_layers") or "vit" in str(
            model_name
        ).lower()

    def _extract_features(
        self,
        backbone: torch.nn.Module,
        images: torch.Tensor,
    ) -> dict[tuple[int, str], torch.Tensor]:
        if not self._backbone_is_vit(backbone):
            embedding = backbone(images)
            return {(1, "avgpool"): embedding}

        max_blocks = max(self.n_last_blocks)
        if not hasattr(backbone, "get_intermediate_layers"):
            raise ValueError(
                "DINO-style ViT probing requires a backbone with "
                "get_intermediate_layers()."
            )
        intermediate = backbone.get_intermediate_layers(images, n=max_blocks)
        features = {}
        for n_blocks in self.n_last_blocks:
            selected = intermediate[-n_blocks:]
            cls = torch.cat([tokens[:, 0] for tokens in selected], dim=-1)
            avgpool = torch.cat(
                [tokens[:, 1:].mean(dim=1) for tokens in selected],
                dim=-1,
            )
            features[(n_blocks, "cls")] = cls
            features[(n_blocks, "avgpool")] = avgpool
            features[(n_blocks, "cls_avgpool")] = torch.cat(
                [cls, selected[-1][:, 1:].mean(dim=1)],
                dim=-1,
            )
        return features

    def _update_val_stats(
        self,
        head_name: str,
        logits: torch.Tensor,
        target: torch.Tensor,
    ) -> None:
        with torch.no_grad():
            total = torch.tensor(target.numel(), device=logits.device)
            top1 = logits.argmax(dim=1).eq(target).sum()
            topk = min(5, self.num_classes)
            top5 = (
                logits.topk(topk, dim=1).indices.eq(target[:, None]).any(dim=1).sum()
            )
        stats = self._val_stats.setdefault(
            head_name,
            {
                "top1": torch.tensor(0, device=logits.device),
                "top5": torch.tensor(0, device=logits.device),
                "total": torch.tensor(0, device=logits.device),
            },
        )
        stats["top1"] = stats["top1"] + top1
        stats["top5"] = stats["top5"] + top5
        stats["total"] = stats["total"] + total

    def wrap_forward(self, pl_module: LightningModule) -> None:
        fn = pl_module.forward

        def new_forward(self, batch, stage, callback=self, fn=fn):
            outputs = fn(batch, stage)
            backbone = getattr(self, callback.backbone_attr)
            backbone.eval()
            target = detach_tensors(batch[callback.target].long())
            with torch.no_grad():
                features = callback._extract_features(backbone, batch["image"])

            losses = []
            for spec in callback._head_specs:
                feature = detach_tensors(
                    features[(spec["n_last_blocks"], spec["pool"])]
                )
                logits = callback.module[spec["name"]](feature)
                if stage == "fit":
                    losses.append(torch.nn.functional.cross_entropy(logits, target))
                    if callback.log_train_metrics:
                        acc = logits.argmax(dim=1).eq(target).float().mean()
                        self.log(
                            f"train/{callback.name}_{spec['name']}_top1",
                            acc,
                            on_step=False,
                            on_epoch=True,
                            sync_dist=True,
                        )
                elif stage == "validate":
                    callback._update_val_stats(spec["name"], logits, target)

            if stage == "fit" and losses:
                loss = torch.stack(losses).sum()
                outputs["loss"] = outputs.get(
                    "loss", torch.tensor(0.0, device=loss.device)
                ) + loss
                self.log(
                    f"train/{callback.name}_loss",
                    loss,
                    on_step=True,
                    on_epoch=True,
                    sync_dist=True,
                )
            return outputs

        pl_module.forward = types.MethodType(new_forward, pl_module)

    def on_validation_epoch_start(self, trainer, pl_module) -> None:
        self._val_stats = {}

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        dataset = self.eval_dataset_name or "val"
        logs = {}
        top1_values = []
        best_top1 = -math.inf
        best_name = None
        for head_name, stats in self._val_stats.items():
            total = stats["total"].clamp_min(1)
            top1 = stats["top1"].float() / total
            top5 = stats["top5"].float() / total
            top1_values.append(top1)
            logs[f"eval/{self.name}_{head_name}_top1_epoch/{dataset}"] = top1
            logs[f"eval/{self.name}_{head_name}_top5_epoch/{dataset}"] = top5
            if dataset == self.in_domain_dataset_name:
                logs[f"eval/{self.name}_{head_name}_top1_epoch/in-domain"] = top1
                logs[f"eval/{self.name}_{head_name}_top5_epoch/in-domain"] = top5
            top1_value = float(top1.detach().cpu())
            if top1_value > best_top1:
                best_top1 = top1_value
                best_name = head_name
        if best_name is not None:
            top1_stack = torch.stack(top1_values)
            max_top1 = top1_stack.max()
            mean_top1 = top1_stack.mean()
            logs[f"eval/{self.name}_max_top1_epoch/{dataset}"] = max_top1
            logs[f"eval/{self.name}_mean_top1_epoch/{dataset}"] = mean_top1
            if dataset == self.in_domain_dataset_name:
                logs[f"eval/{self.name}_max_top1_epoch/in-domain"] = max_top1
                logs[f"eval/{self.name}_mean_top1_epoch/in-domain"] = mean_top1
            logging.info(
                f"  {self.name}: best {dataset} top1={best_top1:.4f} ({best_name})"
            )
        if logs:
            pl_module.log_dict(logs, on_step=False, on_epoch=True, sync_dist=False)
