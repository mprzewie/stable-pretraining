"""Deterministic regression audit for the fused CUDA Laphorn path.

The old identity-based full-plan construction is retained here only as a
small-instance oracle. Production ``coupling_cuda`` applies the strip directly
to ``W.T`` and never materialises the q x q strip matrix.
"""
from __future__ import annotations

import torch

from directional import coupling as coupling_torch


def _identity_reference(logits, s, t, h):
    """Former CUDA construction, used only as a small numerical oracle."""
    from directional_cuda import _core, _stripT_cuda

    n, m = logits.shape
    if n < m:
        return _identity_reference(logits.t(), t, s, h).t()
    W, gridq, bcol = _core(logits, s, t, h)
    eye = torch.eye(m, dtype=logits.dtype, device=logits.device)
    Bt = _stripT_cuda(eye, gridq, bcol, h)
    return s.unsqueeze(1) * (W @ Bt)


def _audit_case(n, m, d, seed):
    from directional_cuda import apply_cuda, coupling_cuda

    torch.manual_seed(seed)
    device = torch.device("cuda")
    dtype = torch.float32

    logits0 = torch.randn(n, m, device=device, dtype=dtype)
    s = torch.softmax(torch.randn(n, device=device, dtype=dtype), dim=0)
    t = torch.softmax(torch.randn(m, device=device, dtype=dtype), dim=0)
    X = torch.randn(m, d, device=device, dtype=dtype)
    G = torch.randn(n, m, device=device, dtype=dtype)

    logits = logits0.detach().requires_grad_(True)
    h = torch.tensor(0.35, device=device, dtype=dtype, requires_grad=True)
    pi = coupling_cuda(logits, s=s, t=t, h=h)
    pi_identity = _identity_reference(logits, s, t, h)
    pi_torch = coupling_torch(logits, s=s, t=t, h=h)
    y = apply_cuda(logits, X, s=s, t=t, h=h)

    metrics = {
        "structural_vs_identity": (pi - pi_identity).abs().max(),
        "cuda_vs_torch": (pi - pi_torch).abs().max(),
        "apply_vs_plan": (y - pi @ X).abs().max(),
        "row_marginal": (pi.sum(1) - s).abs().max(),
        "column_marginal": (pi.sum(0) - t).abs().max(),
    }

    grad_cuda = torch.autograd.grad((pi * G).sum(), (logits, h))
    logits_ref = logits0.detach().requires_grad_(True)
    h_ref = h.detach().requires_grad_(True)
    pi_ref = coupling_torch(logits_ref, s=s, t=t, h=h_ref)
    grad_torch = torch.autograd.grad((pi_ref * G).sum(), (logits_ref, h_ref))
    metrics["grad_logits_cuda_vs_torch"] = (grad_cuda[0] - grad_torch[0]).abs().max()
    metrics["grad_h_cuda_vs_torch"] = (grad_cuda[1] - grad_torch[1]).abs()

    # Automatic log_h=-mean(logits): value/apply and the extra log-width
    # gradient channel must agree with the pure-PyTorch implementation.
    logits_auto = logits0.detach().requires_grad_(True)
    pi_auto = coupling_cuda(logits_auto, s=s, t=t)
    pi_auto_log = coupling_cuda(logits_auto, s=s, t=t, log_h=-logits_auto.mean())
    y_auto = apply_cuda(logits_auto, X, s=s, t=t)
    logits_auto_ref = logits0.detach().requires_grad_(True)
    pi_auto_ref = coupling_torch(logits_auto_ref, s=s, t=t)
    grad_auto_cuda, = torch.autograd.grad((pi_auto * G).sum(), (logits_auto,))
    grad_auto_torch, = torch.autograd.grad((pi_auto_ref * G).sum(), (logits_auto_ref,))
    metrics["auto_vs_explicit_log_h"] = (pi_auto - pi_auto_log).abs().max()
    metrics["auto_cuda_vs_torch"] = (pi_auto - pi_auto_ref).abs().max()
    metrics["auto_apply_vs_plan"] = (y_auto - pi_auto @ X).abs().max()
    metrics["auto_grad_logits_cuda_vs_torch"] = (grad_auto_cuda - grad_auto_torch).abs().max()

    # Independent, learnable log_h tensor: both its gradient and the logits
    # gradient must survive the exp(log_h) reparameterisation.
    logits_log = logits0.detach().requires_grad_(True)
    log_h = torch.tensor(-0.2, device=device, dtype=dtype, requires_grad=True)
    pi_log = coupling_cuda(logits_log, s=s, t=t, log_h=log_h)
    y_log = apply_cuda(logits_log, X, s=s, t=t, log_h=log_h)
    grad_log_cuda = torch.autograd.grad((pi_log * G).sum(), (logits_log, log_h))
    logits_log_ref = logits0.detach().requires_grad_(True)
    log_h_ref = log_h.detach().requires_grad_(True)
    pi_log_ref = coupling_torch(logits_log_ref, s=s, t=t, log_h=log_h_ref)
    grad_log_torch = torch.autograd.grad(
        (pi_log_ref * G).sum(), (logits_log_ref, log_h_ref)
    )
    metrics["log_h_cuda_vs_torch"] = (pi_log - pi_log_ref).abs().max()
    metrics["log_h_apply_vs_plan"] = (y_log - pi_log @ X).abs().max()
    metrics["log_h_grad_logits_cuda_vs_torch"] = (
        grad_log_cuda[0] - grad_log_torch[0]
    ).abs().max()
    metrics["grad_log_h_cuda_vs_torch"] = (grad_log_cuda[1] - grad_log_torch[1]).abs()

    # Fused kernels use float32 internally. These bounds catch orientation and
    # backward regressions while allowing normal GPU reduction-order differences.
    tolerances = {
        "structural_vs_identity": 3e-6,
        "cuda_vs_torch": 3e-6,
        "apply_vs_plan": 3e-6,
        "row_marginal": 3e-6,
        "column_marginal": 3e-6,
        "grad_logits_cuda_vs_torch": 2e-5,
        "grad_h_cuda_vs_torch": 2e-5,
        "auto_vs_explicit_log_h": 3e-6,
        "auto_cuda_vs_torch": 3e-6,
        "auto_apply_vs_plan": 3e-6,
        "auto_grad_logits_cuda_vs_torch": 3e-5,
        "log_h_cuda_vs_torch": 3e-6,
        "log_h_apply_vs_plan": 3e-6,
        "log_h_grad_logits_cuda_vs_torch": 3e-5,
        "grad_log_h_cuda_vs_torch": 3e-5,
    }
    tag = f"n{n}_m{m}"
    for name, value in metrics.items():
        print(f"{tag}_{name}={float(value.detach()):.3e}")

    failures = [
        name
        for name, value in metrics.items()
        if not torch.isfinite(value) or float(value.detach()) > tolerances[name]
    ]
    if failures:
        raise AssertionError(f"CUDA audit failed for {tag}: " + ", ".join(failures))


def main():
    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU is required for audit_cuda.py")

    _audit_case(n=8, m=6, d=4, seed=20260719)
    _audit_case(n=6, m=8, d=4, seed=20260720)
    print("cuda_audit=PASS")


if __name__ == "__main__":
    main()
