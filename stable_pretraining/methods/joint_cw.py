from collections import defaultdict
from dataclasses import dataclass
import math
from numbers import Real
from typing import Optional

import timm
import torch
import torch.nn as nn
from transformers.utils import ModelOutput

from stable_pretraining import Module
from stable_pretraining.backbone import MLP
from stable_pretraining.methods.lejepa import CWReg


class JointCWLoss(Module):
    """Pairwise Joint-CW objective in whitened sum and difference coordinates."""

    def __init__(
        self,
        gamma: float | None = 0.5,
        rho: float = 0.7,
        beta: float = 0.5,
        w_plus: Optional[float] = None,
    ):
        super().__init__()
        self.cwreg = CWReg(gamma=gamma)
        if not -1 < rho < 1:
            raise ValueError(f"rho must be between -1 and 1 but got {rho=}")
        if not 0 <= beta <= 1:
            raise ValueError(f"beta must be between 0 and 1 but got {beta=}")
        w_plus = w_plus if w_plus is not None else beta / 2
        if not 0 <= w_plus <= beta:
            raise ValueError(f"w_plus must be between 0 and {beta=} but got {w_plus=}")
        self.rho = rho
        self.beta = beta
        self.w_plus = w_plus

    def forward(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if z1.ndim != 2 or z2.ndim != 2:
            raise ValueError("Input tensors must have two dimensions.")
        if z1.shape != z2.shape:
            raise ValueError("Input tensors must have the same shape.")

        z_plus = (z1 + z2) / math.sqrt(2 * (1 + self.rho))
        z_minus = (z1 - z2) / math.sqrt(2 * (1 - self.rho))
        joint = torch.cat([z_plus, z_minus], dim=-1)
        cw_joint = self.cwreg(joint)

        cw_plus = self.cwreg(z_plus)
        cw_minus = self.cwreg(z_minus)

        cw_blocks = self.w_plus * cw_plus + (self.beta - self.w_plus) * cw_minus
        weighted_joint = (1.0 - self.beta) * cw_joint
        weighted_blocks = cw_blocks
        loss = weighted_joint + weighted_blocks

        diagnostics = {
            "cw_joint": cw_joint.detach(),
            "cw_plus": cw_plus.detach(),
            "cw_minus": cw_minus.detach(),
            "cw_blocks": cw_blocks.detach(),
            "weighted_joint": weighted_joint.detach(),
            "weighted_blocks": weighted_blocks.detach(),
        }
        return loss, diagnostics


@dataclass
class JointCWOutput(ModelOutput):
    """Joint-CW losses, embeddings, and detached pair diagnostics."""

    loss: torch.Tensor = None
    embedding: torch.Tensor = None
    gg_loss: torch.Tensor = None
    gl_loss: torch.Tensor = None
    ll_loss: torch.Tensor = None
    diagnostics: Optional[dict[str, torch.Tensor]] = None


class JointCW(Module):
    """Train an encoder with averaged pairwise GG, GL, and LL Joint-CW."""

    def __init__(
        self,
        encoder_name: str = "vit_base_patch16_224",
        projector: Optional[nn.Module] = None,
        pretrained: bool = False,
        drop_path_rate: float = 0.1,
        override_sr_gamma: float | str | None = None,
        rho_gg: float = 0.88,
        rho_gl: float = 0.72,
        rho_ll: float = 0.61,
        beta: float = 1.0,
        w_plus: Optional[float] = None,
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

        self.jcw_gg = JointCWLoss(
            gamma=sr_gamma,
            rho=rho_gg,
            beta=beta,
            w_plus=w_plus,
        )
        self.jcw_gl = JointCWLoss(
            gamma=sr_gamma,
            rho=rho_gl,
            beta=beta,
            w_plus=w_plus,
        )
        self.jcw_ll = JointCWLoss(
            gamma=sr_gamma,
            rho=rho_ll,
            beta=beta,
            w_plus=w_plus,
        )
        self.beta = beta
        self.w_plus = self.jcw_gg.w_plus
        self.embed_dim = embed_dim

    def _compute_loss(
        self,
        all_projected: torch.Tensor,
        n_global: int,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        dict[str, torch.Tensor],
    ]:
        """Average pairwise losses within the GG, GL, and LL groups."""
        gg_losses = []
        gl_losses = []
        ll_losses = []
        diagnostics: dict[str, list[torch.Tensor]] = defaultdict(list)

        for i in range(len(all_projected)):
            for j in range(i + 1, len(all_projected)):
                if i < n_global and j < n_global:
                    group = "gg"
                    objective = self.jcw_gg
                    losses = gg_losses
                elif i < n_global or j < n_global:
                    group = "gl"
                    objective = self.jcw_gl
                    losses = gl_losses
                else:
                    group = "ll"
                    objective = self.jcw_ll
                    losses = ll_losses

                pair_loss, pair_diagnostics = objective(
                    all_projected[i],
                    all_projected[j],
                )
                losses.append(pair_loss)
                for name, value in pair_diagnostics.items():
                    diagnostics[f"{group}/{name}"].append(value)

        grouped_diagnostics = {
            name: torch.stack(values).mean() for name, values in diagnostics.items()
        }
        return (
            torch.stack(gg_losses).mean(),
            torch.stack(gl_losses).mean(),
            torch.stack(ll_losses).mean(),
            grouped_diagnostics,
        )

    def forward(
        self,
        global_views: Optional[list[torch.Tensor]] = None,
        local_views: Optional[list[torch.Tensor]] = None,
        images: Optional[torch.Tensor] = None,
    ) -> JointCWOutput:
        if self.training:
            if global_views is None or local_views is None:
                raise ValueError(
                    "global_views and local_views must be provided in training mode"
                )

            g_features = self.backbone(torch.cat(global_views))
            l_features = self.backbone(torch.cat(local_views))
            all_features = torch.cat([g_features, l_features])
            all_projected = self.projector(all_features)

            batch_size = global_views[0].shape[0]
            view_count = len(global_views) + len(local_views)
            all_projected = all_projected.reshape(view_count, batch_size, -1)

            gg_loss, gl_loss, ll_loss, loss_diagnostics = self._compute_loss(
                all_projected,
                len(global_views),
            )
            loss = (gg_loss + gl_loss + ll_loss) / 3
            return JointCWOutput(
                loss=loss,
                gg_loss=gg_loss,
                gl_loss=gl_loss,
                ll_loss=ll_loss,
                embedding=g_features.detach(),
                diagnostics=loss_diagnostics,
            )

        if images is None:
            raise ValueError("images must be provided in evaluation mode")
        embedding = self.backbone(images)
        zero = embedding.new_zeros(())
        return JointCWOutput(
            loss=zero,
            gg_loss=zero,
            gl_loss=zero,
            ll_loss=zero,
            embedding=embedding,
        )
