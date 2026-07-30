from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Literal

import torch
from torch import nn

from stable_pretraining.methods.lejepa import CWReg


def _helmert_contrasts(
    n: int,
    *,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Return an orthonormal basis orthogonal to the all-ones vector.

    Shape: [n - 1, n]

    For n=2 this is, up to sign:
        [1, -1] / sqrt(2)
    """
    if n < 1:
        raise ValueError(f"n must be positive, got {n=}")

    if n == 1:
        return torch.empty(0, 1, dtype=dtype)

    basis = torch.zeros(n - 1, n, dtype=dtype)

    for k in range(1, n):
        scale = math.sqrt(k * (k + 1))
        basis[k - 1, :k] = 1.0 / scale
        basis[k - 1, k] = -k / scale

    return basis


class MultiViewCWDecomposition(nn.Module):
    """Whiten exchangeable global and local views along the view axis.

    The target has ``n_global`` exchangeable global views, ``n_local``
    exchangeable local views, and correlations ``rho_gg``, ``rho_gl``, and
    ``rho_ll``.

    Input tensors have shape ``[B, n_global + n_local, D]``. Every returned
    block has a standard isotropic Gaussian target.
    """

    def __init__(
        self,
        *,
        n_global: int = 2,
        n_local: int = 6,
        rho_gg: float = 0.88,
        rho_gl: float = 0.72,
        rho_ll: float = 0.61,
        eps: float = 1e-8,
    ):
        super().__init__()

        if n_global < 1:
            raise ValueError(f"n_global must be positive, got {n_global=}")
        if n_local < 1:
            raise ValueError(f"n_local must be positive, got {n_local=}")
        if not math.isfinite(eps) or eps <= 0.0:
            raise ValueError(f"eps must be finite and positive, got {eps=}")
        if not -1.0 < rho_gg < 1.0:
            raise ValueError(f"Invalid {rho_gg=}")
        if not -1.0 < rho_gl < 1.0:
            raise ValueError(f"Invalid {rho_gl=}")
        if not -1.0 < rho_ll < 1.0:
            raise ValueError(f"Invalid {rho_ll=}")
        if n_global > 1 and 1.0 - rho_gg <= eps:
            raise ValueError(
                "Global residual eigenvalue is too small for stable whitening: "
                f"{1.0 - rho_gg}"
            )
        if n_local > 1 and 1.0 - rho_ll <= eps:
            raise ValueError(
                "Local residual eigenvalue is too small for stable whitening: "
                f"{1.0 - rho_ll}"
            )

        self.n_global = n_global
        self.n_local = n_local
        self.rho_gg = rho_gg
        self.rho_gl = rho_gl
        self.rho_ll = rho_ll
        self.eps = eps

        # Residual subspaces within the global and local view groups.
        q_global = _helmert_contrasts(n_global)
        q_local = _helmert_contrasts(n_local)

        self.register_buffer(
            "q_global",
            q_global,
            persistent=False,
        )
        self.register_buffer(
            "q_local",
            q_local,
            persistent=False,
        )

        global_collective_var = 1.0 + (n_global - 1) * rho_gg
        local_collective_var = 1.0 + (n_local - 1) * rho_ll
        collective_cross_cov = math.sqrt(n_global * n_local) * rho_gl

        collective_cov = torch.tensor(
            [
                [global_collective_var, collective_cross_cov],
                [collective_cross_cov, local_collective_var],
            ],
            dtype=torch.float64,
        )

        eigenvalues, eigenvectors = torch.linalg.eigh(collective_cov)

        # Put the higher-variance mode first. With positive rho_gl,
        # this is the same-sign shared global/local mode.
        order = torch.argsort(eigenvalues, descending=True)
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]

        if torch.any(eigenvalues <= eps):
            raise ValueError(
                "The requested multiview covariance is not positive definite. "
                f"Collective eigenvalues: {eigenvalues.tolist()}"
            )

        # Canonicalize arbitrary eigenvector signs for reproducibility.
        for column in range(eigenvectors.shape[1]):
            largest_index = eigenvectors[:, column].abs().argmax()
            if eigenvectors[largest_index, column] < 0:
                eigenvectors[:, column] *= -1

        self.register_buffer(
            "collective_eigenvalues",
            eigenvalues,
            persistent=False,
        )
        self.register_buffer(
            "collective_eigenvectors",
            eigenvectors,
            persistent=False,
        )

    def forward(
        self,
        z: torch.Tensor | Sequence[torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        if not isinstance(z, torch.Tensor):
            if not isinstance(z, Sequence):
                raise TypeError(
                    "Expected a tensor or a sequence of tensors, "
                    f"got {type(z).__name__}"
                )
            if not z:
                raise ValueError("Expected at least one view tensor.")
            z = torch.stack(list(z), dim=1)

        if z.ndim != 3:
            raise ValueError(f"Expected [B, V, D], got shape {tuple(z.shape)}")
        if not torch.is_floating_point(z):
            raise TypeError(f"Expected floating-point embeddings, got {z.dtype}")
        if z.shape[-1] == 0:
            raise ValueError("Feature dimension must be positive.")

        batch_size, view_count, _ = z.shape
        expected_views = self.n_global + self.n_local

        if view_count != expected_views:
            raise ValueError(f"Expected {expected_views} views, got {view_count}")
        if batch_size < 2:
            raise ValueError("CW requires at least two independent images.")

        if z.dtype in (torch.float16, torch.bfloat16):
            z = z.float()

        globals_ = z[:, : self.n_global]
        locals_ = z[:, self.n_global :]

        # Normalized collective coordinates:
        # sum / sqrt(number of views).
        global_mean_mode = globals_.sum(dim=1) / math.sqrt(self.n_global)
        local_mean_mode = locals_.sum(dim=1) / math.sqrt(self.n_local)

        collective = torch.stack(
            [global_mean_mode, local_mean_mode],
            dim=1,
        )  # [B, 2, D]

        eigenvectors = self.collective_eigenvectors.to(
            device=z.device,
            dtype=z.dtype,
        )
        eigenvalues = self.collective_eigenvalues.to(
            device=z.device,
            dtype=z.dtype,
        )

        # U^T collective, then divide by sqrt(eigenvalue).
        collective_modes = torch.einsum(
            "vk,bvd->bkd",
            eigenvectors,
            collective,
        )
        collective_modes = collective_modes / eigenvalues.sqrt()[None, :, None]

        q_global = self.q_global.to(device=z.device, dtype=z.dtype)
        q_local = self.q_local.to(device=z.device, dtype=z.dtype)

        global_residual = torch.einsum(
            "rv,bvd->brd",
            q_global,
            globals_,
        )
        local_residual = torch.einsum(
            "rv,bvd->brd",
            q_local,
            locals_,
        )

        if global_residual.shape[1] > 0:
            global_residual = global_residual / math.sqrt(1.0 - self.rho_gg)

        if local_residual.shape[1] > 0:
            local_residual = local_residual / math.sqrt(1.0 - self.rho_ll)

        collective_major = collective_modes[:, 0:1]
        collective_minor = collective_modes[:, 1:2]

        full = torch.cat(
            [
                collective_major,
                collective_minor,
                global_residual,
                local_residual,
            ],
            dim=1,
        )

        return {
            "full": full,
            "collective": collective_modes,
            "collective_major": collective_major,
            "collective_minor": collective_minor,
            "global_residual": global_residual,
            "local_residual": local_residual,
            "residual": torch.cat(
                [global_residual, local_residual],
                dim=1,
            ),
        }


MultiViewCWVariant = Literal[
    "canonical",
    "blockwise",
    "coarse",
    "full_joint",
]


class MultiViewBlockCWLoss(nn.Module):
    """Compute multiview Joint-CW over structured canonical blocks.

    The full joint term identifies the complete correlated Gaussian target.
    When its weight is zero, block variants constrain only their selected
    marginals. ``blockwise`` combines the two collective weights, while
    ``coarse`` also combines the two residual weights. ``full_joint`` has no
    configurable block weights.

    Recommended initial weights correspond to:

        joint               = 0.60
        shared center       = 0.25
        global/local shift  = 0.05
        global residual     = 0.03
        local residual      = 0.07

    Total = 1.0
    """

    def __init__(
        self,
        *,
        gamma: float | None = 0.5,
        variant: MultiViewCWVariant = "canonical",
        n_global: int = 2,
        n_local: int = 6,
        rho_gg: float = 0.88,
        rho_gl: float = 0.72,
        rho_ll: float = 0.61,
        w_joint: float = 0.60,
        w_collective_major: float = 0.25,
        w_collective_minor: float = 0.05,
        w_global_residual: float = 0.03,
        w_local_residual: float = 0.07,
    ):
        super().__init__()

        valid_variants = {
            "canonical",
            "blockwise",
            "coarse",
            "full_joint",
        }
        if variant not in valid_variants:
            raise ValueError(f"Unknown {variant=}; expected one of {valid_variants}")
        if gamma is not None and (not math.isfinite(gamma) or gamma <= 0.0):
            raise ValueError(f"gamma must be finite and positive or None, got {gamma=}")

        weights = {
            "joint": w_joint,
            "collective_major": w_collective_major,
            "collective_minor": w_collective_minor,
            "global_residual": w_global_residual,
            "local_residual": w_local_residual,
        }
        default_weights = {
            "joint": 0.60,
            "collective_major": 0.25,
            "collective_minor": 0.05,
            "global_residual": 0.03,
            "local_residual": 0.07,
        }

        if any(weight < 0.0 for weight in weights.values()):
            raise ValueError(f"All weights must be nonnegative: {weights}")

        total_weight = sum(weights.values())
        if variant == "full_joint" and weights != default_weights:
            raise ValueError(
                "full_joint does not use block weights; leave all weight "
                "arguments at their defaults."
            )
        if variant != "full_joint" and not math.isclose(
            total_weight,
            1.0,
            abs_tol=1e-6,
        ):
            raise ValueError(f"Weights must sum to 1, got {total_weight}: {weights}")

        self.variant = variant
        self.cwreg = CWReg(gamma=gamma)

        self.decomposition = MultiViewCWDecomposition(
            n_global=n_global,
            n_local=n_local,
            rho_gg=rho_gg,
            rho_gl=rho_gl,
            rho_ll=rho_ll,
        )

        self.w_joint = w_joint
        self.w_collective_major = w_collective_major
        self.w_collective_minor = w_collective_minor
        self.w_global_residual = w_global_residual
        self.w_local_residual = w_local_residual

    @staticmethod
    def _flatten(block: torch.Tensor) -> torch.Tensor:
        return block.flatten(start_dim=1)

    @staticmethod
    def _mean_whitened_variance(block: torch.Tensor) -> torch.Tensor:
        if block.shape[1] == 0:
            return block.new_zeros(())

        return block.var(
            dim=0,
            unbiased=False,
        ).mean()

    def _cw(self, block: torch.Tensor) -> torch.Tensor:
        if block.shape[1] == 0:
            return block.new_zeros(())

        return self.cwreg(self._flatten(block))

    def forward(
        self,
        z: torch.Tensor | Sequence[torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        blocks = self.decomposition(z)

        raw: dict[str, torch.Tensor] = {}
        weighted: dict[str, torch.Tensor] = {}

        if self.variant == "full_joint":
            raw["joint"] = self._cw(blocks["full"])
            weighted["joint"] = raw["joint"]

        elif self.variant == "coarse":
            raw["joint"] = self._cw(blocks["full"])
            raw["collective"] = self._cw(blocks["collective"])
            raw["residual"] = self._cw(blocks["residual"])

            w_collective = self.w_collective_major + self.w_collective_minor
            w_residual = self.w_global_residual + self.w_local_residual

            weighted["joint"] = self.w_joint * raw["joint"]
            weighted["collective"] = w_collective * raw["collective"]
            weighted["residual"] = w_residual * raw["residual"]

        elif self.variant == "blockwise":
            raw["joint"] = self._cw(blocks["full"])
            raw["collective"] = self._cw(blocks["collective"])
            raw["global_residual"] = self._cw(blocks["global_residual"])
            raw["local_residual"] = self._cw(blocks["local_residual"])

            w_collective = self.w_collective_major + self.w_collective_minor

            weighted["joint"] = self.w_joint * raw["joint"]
            weighted["collective"] = w_collective * raw["collective"]
            weighted["global_residual"] = (
                self.w_global_residual * raw["global_residual"]
            )
            weighted["local_residual"] = self.w_local_residual * raw["local_residual"]

        elif self.variant == "canonical":
            raw["joint"] = self._cw(blocks["full"])
            raw["collective_major"] = self._cw(blocks["collective_major"])
            raw["collective_minor"] = self._cw(blocks["collective_minor"])
            raw["global_residual"] = self._cw(blocks["global_residual"])
            raw["local_residual"] = self._cw(blocks["local_residual"])

            weighted["joint"] = self.w_joint * raw["joint"]
            weighted["collective_major"] = (
                self.w_collective_major * raw["collective_major"]
            )
            weighted["collective_minor"] = (
                self.w_collective_minor * raw["collective_minor"]
            )
            weighted["global_residual"] = (
                self.w_global_residual * raw["global_residual"]
            )
            weighted["local_residual"] = self.w_local_residual * raw["local_residual"]

        else:
            raise RuntimeError(f"Unhandled variant: {self.variant}")

        loss = torch.stack(list(weighted.values())).sum()

        diag: dict[str, torch.Tensor] = {
            f"cw/{name}": value.detach() for name, value in raw.items()
        }
        diag.update(
            {f"weighted/{name}": value.detach() for name, value in weighted.items()}
        )

        # These directly show whether each whitened mode reaches variance 1.
        for name in (
            "collective_major",
            "collective_minor",
            "global_residual",
            "local_residual",
        ):
            diag[f"variance/{name}"] = self._mean_whitened_variance(
                blocks[name]
            ).detach()

        return loss, diag
