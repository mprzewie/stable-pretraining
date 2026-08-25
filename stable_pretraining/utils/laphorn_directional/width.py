"""Shared kernel-width parameterisation for PyTorch and fused CUDA paths."""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def _exp_finite(log_h, logits):
    """Exponentiate a log-width without ever returning zero or infinity.

    Clipping is only a finite-precision guard. In the normal numerical range
    this is exactly ``exp(log_h)`` and has the usual gradient.
    """
    if torch.is_tensor(log_h):
        log_h = log_h.to(device=logits.device, dtype=logits.dtype)
    else:
        log_h = torch.as_tensor(log_h, device=logits.device, dtype=logits.dtype)
    finfo = torch.finfo(logits.dtype)
    # On the unit-spaced strip grid, widths below machine epsilon and above
    # its reciprocal are already numerically saturated. Keeping that range
    # also makes divisions by h safe in the PyTorch and float32 CUDA kernels.
    lo = math.log(finfo.eps)
    hi = -lo
    return torch.exp(log_h.clamp(min=lo, max=hi))


def resolve_width(logits, h=None, log_h=None, h_from_logits="neg_mean"):
    """Resolve the positive strip width.

    Default automatic parameterisation::

        log_h = -mean(logits)
        h = exp(log_h)

    A global logit shift therefore leaves the directional softmax unchanged
    while controlling its strip hardness multiplicatively. ``h`` and
    ``log_h`` are mutually exclusive. The former ``softplus`` and ``exp``
    modes remain available only for explicit backward compatibility.
    """
    if h is not None and log_h is not None:
        raise ValueError("pass at most one of h and log_h")
    if h is not None:
        if torch.is_tensor(h):
            return h.to(device=logits.device, dtype=logits.dtype)
        return torch.as_tensor(float(h), device=logits.device, dtype=logits.dtype)
    if log_h is not None:
        return _exp_finite(log_h, logits)

    mu = logits.mean()
    if h_from_logits in (None, "neg_mean", "log"):
        return _exp_finite(-mu, logits)
    if h_from_logits == "softplus":
        return F.softplus(mu)
    if h_from_logits == "exp":
        return torch.exp(mu)
    raise ValueError(
        "h_from_logits must be 'neg_mean' (default), 'log', "
        "or an explicit legacy mode: 'softplus'/'exp'"
    )
