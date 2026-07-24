from dataclasses import dataclass
from transformers.utils import ModelOutput
from typing import Optional
from numbers import Real
import math

import timm
import torch
import torch.nn as nn
from torch.distributed.nn import all_reduce

from stable_pretraining import Module
from stable_pretraining.backbone import MLP
from stable_pretraining.methods.lejepa import CWReg, _grouped_pair_diagnostics

from cw_torch.gamma import silverman_rule_of_thumb
from cw_torch.metric import cw_normality




class JointCWLoss(Module):

    def __init__(self, gamma: float = 0.5, rho: float = 0.7):
        super().__init__()
        self.cwreg = CWReg(gamma=gamma)
        assert -1 < rho < 1, f"rho must be between -1 and 1 but got {rho=}"
        self.rho = rho

    def forward(self, z1: torch.Tensor, z2: torch.Tensor):
        assert len(z1.shape) == len(z2.shape) == 2, "Input tensors must have 2 dimensions."
        assert z1.shape == z2.shape, "Input tensors must have the same shape."
        
        z_plus  = (z1 + z2) / math.sqrt(2 * (1 + self.rho))
        z_minus = (z1 - z2) / math.sqrt(2 * (1 - self.rho))
        joint = torch.cat([z_plus, z_minus], dim=-1)
        return self.cwreg(joint)
    
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
    def __init__(
        self,
        encoder_name: str = "vit_base_patch16_224",
        projector: Optional[nn.Module] = None,
        # n_slices: int = 1024,
        # t_max: float = 3.0,
        # n_points: int = 17,
        # lamb: float = 0.02,
        pretrained: bool = False,
        drop_path_rate: float = 0.1,
        # sigreg: str = "ep",
        override_sr_gamma: float | str | None = None,
        rho_gg = 0.88,
        rho_gl = 0.72,
        rho_ll = 0.61,
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
            # Method-specific defaults:
            # original EP uses gamma=0.5
            # CW uses Silverman
            sr_gamma = 0.5
        
        elif override_sr_gamma == "silverman":
            sr_gamma = None
        
        elif isinstance(override_sr_gamma, Real):
            sr_gamma = float(override_sr_gamma)
            if sr_gamma <= 0:
                raise ValueError("override_sr_gamma must be positive.")
        

        self.jcw_gg = JointCWLoss(gamma=sr_gamma, rho=rho_gg)
        self.jcw_gl = JointCWLoss(gamma=sr_gamma, rho=rho_gl)
        self.jcw_ll = JointCWLoss(gamma=sr_gamma, rho=rho_ll)
        self.diagnostic_rhos = {
            "gg": rho_gg,
            "gl": rho_gl,
            "ll": rho_ll,
        }

        # self.lamb = lamb
        self.embed_dim = embed_dim

    def _compute_loss(
        self,
        all_projected: torch.Tensor,
        n_global: int,
    ):
        """Compute the LeJEPA loss.

        :param all_projected: All view projections [V, N, K].
        :param n_global: Number of global views.
        :param sigreg: SlicedEppsPulley module.
        :param lamb: SIGReg weight λ.
        :return: Tuple of (total_loss, inv_loss, sigreg_loss).
        """
        gg_losses = []
        gl_losses = []
        ll_losses = []

        for i in range(len(all_projected)):
            for j in range(i + 1, len(all_projected)):
                if i < n_global and j < n_global:
                    gg_losses.append(self.jcw_gg(all_projected[i], all_projected[j]))
                elif i < n_global or j < n_global:
                    gl_losses.append(self.jcw_gl(all_projected[i], all_projected[j]))
                else:
                    ll_losses.append(self.jcw_ll(all_projected[i], all_projected[j]))
                    
        return torch.stack(gg_losses).mean(), torch.stack(gl_losses).mean(), torch.stack(ll_losses).mean()


    def forward(
        self,
        global_views: Optional[list[torch.Tensor]] = None,
        local_views: Optional[list[torch.Tensor]] = None,
        images: Optional[torch.Tensor] = None,
    ) -> JointCWOutput:
        if self.training:
            assert global_views is not None and local_views is not None, (
                "global_views and local_views must be provided in training mode"
            )

            g_features = self.backbone(torch.cat(global_views))
            l_features = self.backbone(torch.cat(local_views))

            all_features = torch.cat([g_features, l_features])
            all_projected = self.projector(all_features)

            bs = global_views[0].shape[0]
            n_views = len(global_views) + len(local_views)
            all_projected = all_projected.view(n_views, bs, -1)

            gg_loss, gl_loss, ll_loss = self._compute_loss(
                all_projected, len(global_views)
            )
            loss = (gg_loss + gl_loss + ll_loss) / 3
            diagnostics = _grouped_pair_diagnostics(
                all_projected,
                len(global_views),
                self.diagnostic_rhos,
            )

            embedding = g_features.detach()
            return JointCWOutput(
                loss=loss,
                gg_loss=gg_loss,
                gl_loss=gl_loss,
                ll_loss=ll_loss,
                embedding=embedding,
                diagnostics=diagnostics,
            )
        else:
            assert images is not None, "images must be provided in eval mode"
            embedding = self.backbone(images)
            zero = torch.tensor(0.0, device=images.device)
            return JointCWOutput(
                loss=zero,
                gg_loss=zero,
                gl_loss=zero,
                ll_loss=zero,
                embedding=embedding,
            )   
