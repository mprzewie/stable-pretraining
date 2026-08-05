"""LeJEPA: Latent Embedding Joint-Embedding Predictive Architecture.

Self-supervised learning via multi-view invariance combined with a
sliced goodness-of-fit test (SIGReg) that pushes embeddings toward
an isotropic Gaussian.

References:
    Balestriero & LeCun. "LeJEPA: Provable and Scalable
    Self-Supervised Learning Without the Heuristics." 2025.
    https://arxiv.org/abs/2511.08544

Example::

    from stable_pretraining.methods import LeJEPA

    model = LeJEPA("vit_small_patch16_224")

    global_images = [torch.randn(4, 3, 224, 224)] * 2
    all_images = [torch.randn(4, 3, 224, 224)] * 6
    model.train()
    output = model(global_images, all_images)
    output.loss.backward()

    model.eval()
    output = model(images=torch.randn(4, 3, 224, 224))
    features = output.embedding  # [N, D]
"""

from dataclasses import dataclass
from transformers.utils import ModelOutput
from typing import Optional
from numbers import Real
import math

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.nn import all_reduce

from stable_pretraining import Module
from stable_pretraining.backbone import MLP

from cw_torch.gamma import silverman_rule_of_thumb
from cw_torch.metric import cw_normality


@torch.no_grad()
def anchor_diagnostics(
    all_projected: torch.Tensor,
    n_global: int,
    *,
    compute_spectrum: bool = False,
) -> dict[str, torch.Tensor]:
    """Diagnostics aligned with the Anchor-CW hypothesis.

    The routine compares global-view anchors with all-view anchors and measures
    between-image anchor variance versus within-image augmentation variance. It
    intentionally avoids the expensive per-view ``B x B`` similarity matrices
    used by the previous joint-CW diagnostics.

    Args:
        all_projected: Projected views with shape ``[views, batch, features]``.
        n_global: Number of leading global views.
        compute_spectrum: Also compute effective/stable ranks of the anchor
            matrices. This is useful occasionally, but more expensive than the
            remaining scalar diagnostics.

    Returns:
        Detached scalar diagnostics. All calculations are performed in float32.
    """
    if all_projected.ndim != 3:
        raise ValueError(
            "all_projected must have shape [views, batch, features]."
        )

    n_views, batch_size, _ = all_projected.shape
    if not 0 < n_global <= n_views:
        raise ValueError(
            f"n_global must be in [1, {n_views}], got {n_global}."
        )
    if batch_size < 1:
        raise ValueError("all_projected must contain at least one sample.")

    eps = 1e-8
    z = all_projected.detach().float()
    global_views = z[:n_global]
    local_views = z[n_global:]

    global_centers = global_views.mean(dim=0)
    all_centers = z.mean(dim=0)

    global_centered = global_centers - global_centers.mean(dim=0, keepdim=True)
    all_centered = all_centers - all_centers.mean(dim=0, keepdim=True)

    global_residuals = global_views - global_centers.unsqueeze(0)
    all_residuals = z - all_centers.unsqueeze(0)

    total_centered = z - z.mean(dim=(0, 1), keepdim=True)
    total_variance = total_centered.square().mean()
    all_between_variance = all_centered.square().mean()
    all_within_variance = all_residuals.square().mean()

    diagnostics: dict[str, torch.Tensor] = {
        # Geometry of the two possible Anchor-CW inputs.
        "anchor/global/feature_variance": global_centered.square().mean(),
        "anchor/all/feature_variance": all_between_variance,
        "anchor/global/mean_norm": global_centers.norm(dim=-1).mean(),
        "anchor/all/mean_norm": all_centers.norm(dim=-1).mean(),
        "anchor/global_all/mse": (global_centers - all_centers).square().mean(),
        "anchor/global_all/cosine": F.cosine_similarity(
            global_centers, all_centers, dim=-1, eps=eps
        ).mean(),

        # Within-image augmentation spread under the two center definitions.
        "residual/global_views_to_global/mse": global_residuals.square().mean(),
        "residual/all_views_to_all/mse": all_within_variance,

        # Exact ANOVA-style decomposition for the all-view mean.
        "variance/all/total": total_variance,
        "variance/all/between": all_between_variance,
        "variance/all/within": all_within_variance,
        "variance/all/between_fraction": all_between_variance
        / total_variance.clamp_min(eps),
        "variance/all/decomposition_error": (
            total_variance - all_between_variance - all_within_variance
        ).abs()
        / total_variance.clamp_min(eps),
    }

    if len(local_views) > 0:
        local_centers = local_views.mean(dim=0)
        local_to_global = local_views - global_centers.unsqueeze(0)
        diagnostics.update(
            {
                "residual/local_views_to_global/mse": local_to_global.square().mean(),
                "anchor/global_local/mse": (
                    global_centers - local_centers
                ).square().mean(),
                "anchor/global_local/cosine": F.cosine_similarity(
                    global_centers, local_centers, dim=-1, eps=eps
                ).mean(),
            }
        )

    if compute_spectrum:
        diagnostics.update(_anchor_spectrum_diagnostics(global_centers, "global"))
        diagnostics.update(_anchor_spectrum_diagnostics(all_centers, "all"))

    return diagnostics


@torch.no_grad()
def _anchor_spectrum_diagnostics(
    anchors: torch.Tensor, prefix: str
) -> dict[str, torch.Tensor]:
    """Return rank diagnostics without forming a feature covariance matrix."""
    eps = 1e-12
    centered = anchors.float() - anchors.float().mean(dim=0, keepdim=True)
    if centered.shape[0] < 2:
        zero = centered.new_zeros(())
        return {
            f"anchor/{prefix}/effective_rank": zero,
            f"anchor/{prefix}/stable_rank": zero,
        }

    singular_values = torch.linalg.svdvals(centered)
    energy = singular_values.square()
    total = energy.sum().clamp_min(eps)
    probabilities = energy / total
    effective_rank = torch.exp(
        -(probabilities * probabilities.clamp_min(eps).log()).sum()
    )
    stable_rank = total / energy.max().clamp_min(eps)
    return {
        f"anchor/{prefix}/effective_rank": effective_rank,
        f"anchor/{prefix}/stable_rank": stable_rank,
    }


class EppsPulley(nn.Module):
    """Epps-Pulley goodness-of-fit test for univariate normality.

    Projects data onto a grid of points and computes the Epps-Pulley statistic.

    :param t_max: Integration upper bound.
    :param n_points: Number of integration points.
    """

    def __init__(self, t_max: float = 3.0, n_points: int = 17, gamma: float | None = 0.5):
        super().__init__()
        assert n_points % 2 == 1

        self._is_ddp = (
            torch.distributed.is_available() and torch.distributed.is_initialized()
        )
        self.world_size = torch.distributed.get_world_size() if self._is_ddp else 1

        t = torch.linspace(0, t_max, n_points)
        dt = t_max / (n_points - 1)
        self.register_buffer("t", t)

        phi = (-0.5 * t**2).exp()
        self.register_buffer("phi", phi)

        weights = torch.full((n_points,), 2 * dt)
        weights[[0, -1]] = dt
        self.register_buffer("weights", weights)
        self.gamma = gamma

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """:param x: Samples [N, S] (N samples, S slices).

        :return: Per-slice statistic [S].
        """
        N = x.size(0)
        x_t = x.unsqueeze(-1) * self.t
        cos_mean = x_t.cos().mean(0)
        sin_mean = x_t.sin().mean(0)

        if self._is_ddp:
            all_reduce(cos_mean, op=torch.distributed.ReduceOp.AVG)
            all_reduce(sin_mean, op=torch.distributed.ReduceOp.AVG)

        err = (cos_mean - self.phi).square() + sin_mean.square()
        if self.gamma is not None:
            gamma = torch.as_tensor(self.gamma, device=x.device, dtype=x.dtype)
        else:
            gamma = silverman_rule_of_thumb(
                    sample_stddev=1.0,
                    sample_count=N * self.world_size,
                ).to(device=x.device, dtype=x.dtype)
            
        window = (-gamma * self.t.square()).exp()
        weights = self.weights * window
            
        return (err @ weights) * N * self.world_size


class SlicedEppsPulley(nn.Module):
    """Sliced Epps-Pulley goodness-of-fit test for multivariate normality.

    Projects data onto random 1-D directions and averages the univariate
    Epps-Pulley statistics.  A synchronised step counter seeds the random
    projections so all DDP ranks sample identical directions.

    :param num_slices: Number of random 1-D projections.
    :param t_max: EP integration upper bound.
    :param n_points: EP quadrature nodes.
    """

    def __init__(self, num_slices: int = 1024, t_max: float = 3.0, n_points: int = 17, gamma: float | None = 0.5):
        super().__init__()
        self._is_ddp = (
            torch.distributed.is_available() and torch.distributed.is_initialized()
        )
        self.num_slices = num_slices
        self.ep = EppsPulley(t_max=t_max, n_points=n_points, gamma=gamma)
        self.register_buffer("global_step", torch.zeros((), dtype=torch.long))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """:param x: Embeddings [N, D].

        :return: Scalar mean EP statistic.
        """
        with torch.no_grad():
            step = self.global_step.clone()

            if self._is_ddp:
                # All ranks increment global_step in lockstep, so this
                # broadcast is redundant under normal synchronous training.
                # It is kept as a safety net against step drift from
                # uneven batches (e.g. drop_last=False).
                torch.distributed.broadcast(step, src=0)

            g = torch.Generator(device=x.device).manual_seed(step.item())
            A = torch.randn(x.size(-1), self.num_slices, device=x.device, generator=g)
            A = A / A.norm(p=2, dim=0)
            self.global_step.add_(1)

        proj = x @ A
        return self.ep(proj).mean()


class CWReg(nn.Module):
    """Closed-form Cramer-Wold regularizer toward a standard Gaussian."""

    def __init__(self, gamma: float | None = None):
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.gamma is None:
            gamma = silverman_rule_of_thumb(
                sample_stddev=1.0,
                sample_count=x.shape[0],
            ).to(device=x.device, dtype=x.dtype)
        else:
            gamma = torch.as_tensor(self.gamma, device=x.device, dtype=x.dtype)

        return 2.0 * math.pi * x.shape[0] * cw_normality(x, gamma)
    
class ClusterUCWReg(nn.Module):
    """Cluster-U Cramér–Wold regularizer toward N(0, I).

    Expects representations grouped as [V, B, D]:
        V: views per image
        B: independent images
        D: feature dimension
    """

    def __init__(self, gamma: float = 0.5):
        super().__init__()
        if gamma <= 0:
            raise ValueError(f"gamma must be positive, got {gamma}.")
        self.gamma = gamma

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(
                f"Expected x with shape [V, B, D], got {tuple(x.shape)}."
            )

        num_views, num_groups, feature_dim = x.shape

        if num_groups < 2:
            raise ValueError(
                "Cluster-U requires at least two independent image groups."
            )

        gamma = torch.as_tensor(
            self.gamma,
            device=x.device,
            dtype=x.dtype,
        )

        # [V, B, D] -> [B, V, D]
        grouped = x.permute(1, 0, 2).contiguous()

        # Ordinary CW over all BV views.
        flat = grouped.reshape(num_groups * num_views, feature_dim)
        pooled_cw = cw_normality(flat, gamma)

        # Average CW computed separately inside each image group.
        within_cw = torch.stack(
            [
                cw_normality(group_views, gamma)
                for group_views in grouped
            ]
        ).mean()

        # Replaces the pooled data-data V-term with a U-statistic
        # over independent image groups, while preserving all-view
        # sample-to-Gaussian terms.
        cluster_u_cw = (
            num_groups * pooled_cw - within_cw
        ) / (num_groups - 1)

        # Preserve the exact external scaling used by CWReg.
        sample_count = num_groups * num_views
        return 2.0 * math.pi * sample_count * cluster_u_cw


@dataclass
class LeJEPAOutput(ModelOutput):
    """Output from LeJEPA forward pass.

    :ivar loss: Combined invariance + SIGReg loss (0 in eval mode).
    :ivar embedding: Backbone embeddings [V*N, D] (train) or [N, D] (eval).
    :ivar inv_loss: Invariance component.
    :ivar sigreg_loss: Epps-Pulley goodness-of-fit component.
    :ivar diagnostics: Optional detached Anchor-CW geometry metrics. Diagnostics
        are disabled by default.
    """

    loss: torch.Tensor = None
    embedding: torch.Tensor = None
    inv_loss: torch.Tensor = None
    sigreg_loss: torch.Tensor = None
    diagnostics: Optional[dict[str, torch.Tensor]] = None


class LeJEPA(Module):
    """LeJEPA: multi-view invariance + sliced Epps-Pulley SIGReg.

    Architecture:
        - **Backbone**: timm ViT (CLS-pooled, ``num_classes=0``)
        - **Projector**: MLP projection head
        - **Loss**: ``invariance + (λ * SIGReg)``

    Centers are computed from global-view projections only.  The invariance
    term penalises the MSE between each view's projection and the center.
    The SIGReg term is a sliced goodness-of-fit test that pushes
    projected embeddings toward an isotropic Gaussian, averaged over views.

    :param encoder_name: timm model name (e.g., ``"vit_base_patch16_224"``)
    :param projector: Optional projection head.  When ``None``, a 3-layer
        BN+ReLU MLP (``embed_dim → 2048 → 2048 → 512``) is created.
    :param n_slices: Random projection directions for the goodness-of-fit test (default: 1024)
    :param t_max: EP integration upper bound (default: 3.0)
    :param n_points: EP quadrature nodes (default: 17)
    :param lamb: SIGReg weight λ (default: 0.02)
    :param pretrained: Load pretrained timm weights
    :param apply_sigreg_on: Inputs regularized by SIGReg/CW. ``"all"`` uses
        every projected view, ``"centers_global"`` uses one global-view mean
        per image, ``"centers_all"`` uses one all-view mean per image,
        ``"all_global"`` uses every global view, and ``"one_global"`` uses
        the first global view.
        The invariance MSE always retains LeJEPA's global-view center.
    :param diagnostics_every_n_steps: Compute lightweight Anchor-CW diagnostics
        every given number of training steps. ``None`` disables diagnostics.
    :param diagnostics_compute_spectrum: Include effective/stable anchor ranks
        whenever diagnostics are computed.

    Example::

        model = LeJEPA("vit_base_patch16_224")
        images = torch.randn(4, 3, 224, 224)

        model.train()
        output = model(
            global_views=[images, images],
            all_views=[images, images, images, images],
        )
        output.loss.backward()

        model.eval()
        output = model(images=images)
        features = output.embedding  # [4, 768]

    Example with Lightning::

        import lightning as pl
        from stable_pretraining.methods import LeJEPA


        class LeJEPALightning(pl.LightningModule):
            def __init__(self):
                super().__init__()
                self.model = LeJEPA("vit_base_patch16_224")

            def training_step(self, batch, batch_idx):
                views = [v["image"] for v in batch["views"]]
                output = self.model(global_views=views, all_views=views)
                self.log("loss", output.loss)
                return output.loss

            def configure_optimizers(self):
                return torch.optim.AdamW(self.parameters(), lr=1e-3)
    """

    def __init__(
        self,
        encoder_name: str = "vit_base_patch16_224",
        projector: Optional[nn.Module] = None,
        n_slices: int = 1024,
        t_max: float = 3.0,
        n_points: int = 17,
        lamb: float = 0.02,
        pretrained: bool = False,
        drop_path_rate: float = 0.1,
        sigreg: str = "ep",
        apply_sigreg_on: str = "all",
        override_sr_gamma: float | str | None = None,
        diagnostics_every_n_steps: int | None = None,
        diagnostics_compute_spectrum: bool = False,
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
            # Fixed default used by the matched EP/CW experiments.
            # Silverman remains available explicitly via ``"silverman"``.
            sr_gamma = 0.5
        
        elif override_sr_gamma == "silverman":
            sr_gamma = None
        
        elif isinstance(override_sr_gamma, Real):
            sr_gamma = float(override_sr_gamma)
            if sr_gamma <= 0:
                raise ValueError("override_sr_gamma must be positive.")
        else:
            raise ValueError(
                "override_sr_gamma must be None, 'silverman', "
                f"or a positive number, got {override_sr_gamma!r}"
            )
    
        if sigreg == "ep":
            self.sigreg = SlicedEppsPulley(
                num_slices=n_slices, t_max=t_max, n_points=n_points, gamma=sr_gamma
            )
        elif sigreg == "cw":
            self.sigreg = CWReg(gamma=sr_gamma)
        elif sigreg == "cluster_ucw":
            self.sigreg = ClusterUCWReg(gamma=sr_gamma)
        else:
            raise ValueError(
                f"Unknown LeJEPA sigreg={sigreg!r}; expected 'ep', 'cw', or 'cluster_ucw'"
            )

        valid_sigreg_inputs = {
            "all",
            "all_global",
            "centers_all",
            "centers_global",
            "one_global",
        }
        if apply_sigreg_on not in valid_sigreg_inputs:
            raise ValueError(
                f"Unknown apply_sigreg_on={apply_sigreg_on!r}; expected one of "
                f"{sorted(valid_sigreg_inputs)}."
            )
        if diagnostics_every_n_steps is not None and diagnostics_every_n_steps <= 0:
            raise ValueError("diagnostics_every_n_steps must be positive or None.")

        self.lamb = lamb
        self.embed_dim = embed_dim
        self.apply_sigreg_on = apply_sigreg_on
        self.diagnostics_every_n_steps = diagnostics_every_n_steps
        self.diagnostics_compute_spectrum = diagnostics_compute_spectrum
        self.register_buffer(
            "_diagnostic_step", torch.zeros((), dtype=torch.long), persistent=False
        )

    def _compute_loss(
        self,
        all_projected: torch.Tensor,
        n_global: int,
        sigreg: nn.Module,
        lamb: float,
    ):
        """Compute the LeJEPA loss.

        :param all_projected: All view projections [V, N, K].
        :param n_global: Number of global views.
        :param sigreg: SlicedEppsPulley module.
        :param lamb: SIGReg weight λ.
        :return: Tuple of (total_loss, inv_loss, sigreg_loss).
        """
        global_centers = all_projected[:n_global].mean(dim=0)  # [N, K]
        all_centers = all_projected.mean(dim=0)  # [N, K]


        inv_loss = (global_centers.unsqueeze(0) - all_projected).square().mean()

        if self.apply_sigreg_on == "all":
            sigreg_inputs = all_projected.flatten(0, 1)
        elif self.apply_sigreg_on == "centers_global":
            sigreg_inputs = global_centers
        elif self.apply_sigreg_on == "centers_all":
            sigreg_inputs = all_centers
        elif self.apply_sigreg_on == "all_global":
            sigreg_inputs = all_projected[:n_global].flatten(0, 1)
        elif self.apply_sigreg_on == "one_global":
            sigreg_inputs = all_projected[0]
        else:  # Guarded in __init__; retained for defensive programming.
            raise RuntimeError(
                f"Unexpected apply_sigreg_on={self.apply_sigreg_on!r}."
            )

        sigreg_loss = sigreg(sigreg_inputs)
        loss = inv_loss + lamb * sigreg_loss
        return loss, inv_loss, sigreg_loss

    def forward(
        self,
        global_views: Optional[list[torch.Tensor]] = None,
        local_views: Optional[list[torch.Tensor]] = None,
        images: Optional[torch.Tensor] = None,
    ) -> LeJEPAOutput:
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

            loss, inv_loss, sigreg_loss = self._compute_loss(
                all_projected, len(global_views), self.sigreg, self.lamb
            )
            diagnostics = None
            if self.diagnostics_every_n_steps is not None:
                step = int(self._diagnostic_step.item())
                if step % self.diagnostics_every_n_steps == 0:
                    diagnostics = anchor_diagnostics(
                        all_projected,
                        len(global_views),
                        compute_spectrum=self.diagnostics_compute_spectrum,
                    )
                self._diagnostic_step.add_(1)

            embedding = g_features.detach()
            return LeJEPAOutput(
                loss=loss,
                embedding=embedding,
                inv_loss=inv_loss,
                sigreg_loss=sigreg_loss,
                diagnostics=diagnostics,
            )
        else:
            assert images is not None, "images must be provided in eval mode"
            embedding = self.backbone(images)
            zero = torch.tensor(0.0, device=images.device)
            return LeJEPAOutput(
                loss=zero,
                embedding=embedding,
                inv_loss=zero,
                sigreg_loss=zero,
            )
