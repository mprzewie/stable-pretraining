"""Cached immutable CUDA/CPU constants used by the directional operators."""
from __future__ import annotations

import torch


_MARGINALS = {}
_GRID_LEVELS = {}


def uniform_marginal(n: int, reference: torch.Tensor) -> torch.Tensor:
    if torch.compiler.is_compiling():
        return torch.full((n,), 1.0 / n, dtype=reference.dtype, device=reference.device)
    key = (n, reference.dtype, reference.device)
    value = _MARGINALS.get(key)
    if value is None:
        value = torch.full((n,), 1.0 / n, dtype=reference.dtype, device=reference.device)
        _MARGINALS[key] = value
    return value


def uniform_grid_levels(n: int, reference: torch.Tensor):
    if torch.compiler.is_compiling():
        marginal = torch.full((n,), 1.0 / n, dtype=reference.dtype, device=reference.device)
        grid = torch.arange(n, dtype=reference.dtype, device=reference.device)
        levels = torch.cumsum(marginal, 0)[:-1].clamp(1e-9, 1 - 1e-9)
        return grid, levels
    key = (n, reference.dtype, reference.device)
    value = _GRID_LEVELS.get(key)
    if value is None:
        marginal = uniform_marginal(n, reference)
        grid = torch.arange(n, dtype=reference.dtype, device=reference.device)
        levels = torch.cumsum(marginal, 0)[:-1].clamp(1e-9, 1 - 1e-9)
        value = (grid, levels)
        _GRID_LEVELS[key] = value
    return value
