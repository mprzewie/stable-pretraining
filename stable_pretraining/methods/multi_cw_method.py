from dataclasses import dataclass
from numbers import Real
from typing import Optional

import timm
import torch
import torch.nn as nn
from transformers.utils import ModelOutput

from stable_pretraining import Module
from stable_pretraining.backbone import MLP
from stable_pretraining.methods.lejepa import _grouped_pair_diagnostics
from stable_pretraining.methods.multi_cw import (
    MultiViewBlockCWLoss,
    MultiViewCWVariant,
)


@dataclass
class MultiCWOutput(ModelOutput):
    """Multiview CW loss, embeddings, and detached diagnostics."""

    loss: torch.Tensor = None
    embedding: torch.Tensor = None
    cw_loss: torch.Tensor = None
    diagnostics: Optional[dict[str, torch.Tensor]] = None


class MultiCW(Module):
    """Train an encoder with image-grouped multiview Cramér–Wold loss."""

    def __init__(
        self,
        encoder_name: str = "vit_base_patch16_224",
        projector: Optional[nn.Module] = None,
        pretrained: bool = False,
        drop_path_rate: float = 0.1,
        override_sr_gamma: float | str | None = None,
        n_global: int = 2,
        n_local: int = 6,
        rho_gg: float = 0.88,
        rho_gl: float = 0.72,
        rho_ll: float = 0.61,
        cw_variant: MultiViewCWVariant = "canonical",
        w_joint: float = 0.60,
        w_collective_major: float = 0.25,
        w_collective_minor: float = 0.05,
        w_global_residual: float = 0.03,
        w_local_residual: float = 0.07,
    ):
        super().__init__()

        self.backbone = timm.create_model(
            encoder_name,
            pretrained=pretrained,
            num_classes=0,
            **({"dynamic_img_size": True} if "vit" in encoder_name else {}),
            drop_path_rate=drop_path_rate,
        )

        embed_dim = getattr(self.backbone, "num_features", None)
        if embed_dim is None:
            embed_dim = getattr(self.backbone, "embed_dim", None)
        if embed_dim is None:
            raise AttributeError(
                f"Backbone {encoder_name!r} exposes neither embed_dim nor num_features."
            )

        if projector is None:
            projector = nn.Sequential(
                nn.Linear(embed_dim, 512, bias=True),
                MLP(
                    in_channels=512,
                    hidden_channels=[2048, 2048, 512],
                    norm_layer="batch_norm",
                    activation_layer=nn.ReLU,
                    inplace=True,
                    dropout=0.0,
                ),
            )
        self.projector = projector

        if override_sr_gamma is None:
            sr_gamma: float | None = 0.5
        elif override_sr_gamma == "silverman":
            sr_gamma = None
        elif isinstance(override_sr_gamma, Real):
            sr_gamma = float(override_sr_gamma)
            if sr_gamma <= 0:
                raise ValueError("override_sr_gamma must be positive.")
        else:
            raise ValueError(
                "override_sr_gamma must be None, 'silverman', or a positive number."
            )

        self.multiview_cw = MultiViewBlockCWLoss(
            gamma=sr_gamma,
            variant=cw_variant,
            n_global=n_global,
            n_local=n_local,
            rho_gg=rho_gg,
            rho_gl=rho_gl,
            rho_ll=rho_ll,
            w_joint=w_joint,
            w_collective_major=w_collective_major,
            w_collective_minor=w_collective_minor,
            w_global_residual=w_global_residual,
            w_local_residual=w_local_residual,
        )
        self.n_global = n_global
        self.n_local = n_local
        self.diagnostic_rhos = {
            "gg": rho_gg,
            "gl": rho_gl,
            "ll": rho_ll,
        }
        self.embed_dim = embed_dim

    def _compute_loss(
        self,
        all_projected: torch.Tensor,
        n_global: int,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute multiview CW without changing the image sampling unit."""
        if n_global != self.n_global:
            raise ValueError(f"Expected {self.n_global} global views, got {n_global}")
        expected_views = self.n_global + self.n_local
        if all_projected.ndim != 3:
            raise ValueError(
                "Expected projected views with shape [V, B, D], "
                f"got {tuple(all_projected.shape)}"
            )
        if all_projected.shape[0] != expected_views:
            raise ValueError(
                f"Expected {expected_views} projected views, "
                f"got {all_projected.shape[0]}"
            )

        image_major = all_projected.permute(1, 0, 2)
        return self.multiview_cw(image_major)

    def _validate_views(
        self,
        global_views: list[torch.Tensor],
        local_views: list[torch.Tensor],
    ) -> None:
        """Validate view counts and aligned per-view batch dimensions."""
        if len(global_views) != self.n_global:
            raise ValueError(
                f"Expected {self.n_global} global views, got {len(global_views)}"
            )
        if len(local_views) != self.n_local:
            raise ValueError(
                f"Expected {self.n_local} local views, got {len(local_views)}"
            )

        shapes = [tuple(view.shape) for view in global_views + local_views]
        if any(len(shape) < 1 for shape in shapes):
            raise ValueError(f"Every view must have a batch dimension: {shapes}")
        batch_sizes = {shape[0] for shape in shapes}
        if len(batch_sizes) != 1:
            raise ValueError(
                "All views must preserve the same batch size and image order; "
                f"got shapes {shapes}"
            )

    def forward(
        self,
        global_views: Optional[list[torch.Tensor]] = None,
        local_views: Optional[list[torch.Tensor]] = None,
        images: Optional[torch.Tensor] = None,
    ) -> MultiCWOutput:
        if self.training:
            if global_views is None or local_views is None:
                raise ValueError(
                    "global_views and local_views must be provided in training mode"
                )
            self._validate_views(global_views, local_views)

            g_features = self.backbone(torch.cat(global_views))
            l_features = self.backbone(torch.cat(local_views))
            all_features = torch.cat([g_features, l_features])
            all_projected = self.projector(all_features)

            batch_size = global_views[0].shape[0]
            view_count = len(global_views) + len(local_views)
            all_projected = all_projected.reshape(view_count, batch_size, -1)

            loss, loss_diagnostics = self._compute_loss(
                all_projected,
                len(global_views),
            )
            diagnostics = _grouped_pair_diagnostics(
                all_projected,
                len(global_views),
                self.diagnostic_rhos,
            )
            diagnostics.update(loss_diagnostics)

            return MultiCWOutput(
                loss=loss,
                cw_loss=loss,
                embedding=g_features.detach(),
                diagnostics=diagnostics,
            )

        if images is None:
            raise ValueError("images must be provided in evaluation mode")
        embedding = self.backbone(images)
        zero = embedding.new_zeros(())
        return MultiCWOutput(
            loss=zero,
            cw_loss=zero,
            embedding=embedding,
        )
