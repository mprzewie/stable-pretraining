"""SwAV swapped-prediction loss with LaPHorn assignments."""

from typing import Optional

import torch

from stable_pretraining.utils.laphorn_directional import coupling

from .joint_embedding import SwAVLoss


class LaphornSwAVLoss(SwAVLoss):
    """Compute the SwAV loss using one-pass LaPHorn assignments.

    Args:
        temperature: Temperature of the swapped-prediction softmax.
        assignment_temperature: Scale applied to prototype scores before the
            directional LaPHorn coupling. This plays the role of SwAV's score
            temperature, but is not LaPHorn's strip width.
        laphorn_h: Optional positive strip width. If omitted, the LaPHorn
            implementation derives it from the mean scaled score.
        laphorn_log_h: Optional strip width in log space. Mutually exclusive
            with ``laphorn_h``.
        h_from_logits: Automatic width parameterisation used when neither
            explicit width is supplied.

    Note:
        LaPHorn produces a unit-mass coupling with row marginal ``1 / B``.
        SwAV needs a categorical target per sample, so assignments are scaled
        by ``B``. Their rows then sum to one and each prototype receives total
        mass ``B / K``. Prototype indices define LaPHorn's ordered strip axis.
    """

    def __init__(
        self,
        temperature: float = 0.1,
        assignment_temperature: float = 0.05,
        laphorn_h: Optional[float] = None,
        laphorn_log_h: Optional[float] = None,
        h_from_logits: str = "neg_mean",
    ) -> None:
        if assignment_temperature <= 0:
            raise ValueError("assignment_temperature must be positive")
        if laphorn_h is not None and laphorn_log_h is not None:
            raise ValueError("pass at most one of laphorn_h and laphorn_log_h")
        super().__init__(temperature=temperature)
        self.assignment_temperature = assignment_temperature
        self.laphorn_h = laphorn_h
        self.laphorn_log_h = laphorn_log_h
        self.h_from_logits = h_from_logits

    @torch.no_grad()
    def assignments(self, scores: torch.Tensor) -> torch.Tensor:
        """Return balanced categorical targets from prototype scores.

        Args:
            scores: Prototype scores with shape ``(batch, n_prototypes)``.

        Returns:
            A nonnegative tensor of the same shape whose rows sum to one and
            whose columns sum to ``batch / n_prototypes`` (up to dtype error).

        Note:
            Coupling construction runs in float32 for mixed-precision stability,
            matching the existing Sinkhorn assignment path.
        """
        if scores.ndim != 2:
            raise ValueError(
                f"scores must have shape (batch, n_prototypes), got {scores.shape}"
            )
        scaled_scores = scores.float() / self.assignment_temperature
        plan = coupling(
            scaled_scores,
            h=self.laphorn_h,
            log_h=self.laphorn_log_h,
            h_from_logits=self.h_from_logits,
        )
        return plan * scores.shape[0]


__all__ = ["LaphornSwAVLoss"]
