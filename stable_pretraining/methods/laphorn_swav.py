"""SwAV with one-pass exact-marginal LaPHorn assignments.

This variant keeps SwAV's encoder, projector, prototypes, multi-crop objective,
and optional composable queue path, but replaces iterative Sinkhorn-Knopp
normalisation with the directional LaPHorn coupling.

References:
    Caron et al. "Unsupervised Learning of Visual Features by Contrasting
    Cluster Assignments." NeurIPS 2020. https://arxiv.org/abs/2006.09882
    Struski et al. "Laphorn: Exact-Marginal Neural Layers Without Iterative
    Normalisation." 2026.
"""

from typing import Optional, Sequence, Union

import torch.nn as nn

from stable_pretraining.losses import LaphornSwAVLoss

from .swav import SwAV


class LaphornSwAV(SwAV):
    """SwAV using LaPHorn instead of Sinkhorn-Knopp for assignments.

    :param encoder_name: timm model name or pre-built ``nn.Module``.
    :param projector_dims: ``(hidden, output)`` projector dimensions.
    :param n_prototypes: Number of prototypes.
    :param temperature: Temperature of the swapped-prediction softmax.
    :param assignment_temperature: Scale applied to prototype scores before
        LaPHorn. This is separate from the LaPHorn strip width.
    :param laphorn_h: Optional explicit positive strip width.
    :param laphorn_log_h: Optional strip width in log space. Mutually exclusive
        with ``laphorn_h``.
    :param h_from_logits: Automatic width parameterisation.
    :param low_resolution: Adapt the first convolution for low-res input.
    :param pretrained: Load pretrained timm weights.
    :param dynamic_img_size: Allow timm ViTs to accept varying crop sizes.

    Note:
        Prototype indices are the ordered axis used by LaPHorn's strip. This
        makes the experimental variant sensitive to prototype ordering at
        finite strip width.
    """

    def __init__(
        self,
        encoder_name: Union[str, nn.Module] = "vit_small_patch16_224",
        projector_dims: Sequence[int] = (2048, 128),
        n_prototypes: int = 3000,
        temperature: float = 0.1,
        assignment_temperature: float = 0.05,
        laphorn_h: Optional[float] = None,
        laphorn_log_h: Optional[float] = None,
        h_from_logits: str = "neg_mean",
        low_resolution: bool = False,
        pretrained: bool = False,
        dynamic_img_size: bool = True,
    ) -> None:
        super().__init__(
            encoder_name=encoder_name,
            projector_dims=projector_dims,
            n_prototypes=n_prototypes,
            temperature=temperature,
            low_resolution=low_resolution,
            pretrained=pretrained,
            dynamic_img_size=dynamic_img_size,
        )
        self.swav_loss = LaphornSwAVLoss(
            temperature=temperature,
            assignment_temperature=assignment_temperature,
            laphorn_h=laphorn_h,
            laphorn_log_h=laphorn_log_h,
            h_from_logits=h_from_logits,
        )


__all__ = ["LaphornSwAV"]
