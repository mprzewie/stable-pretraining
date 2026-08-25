"""Small deterministic audit for the paper's structural and derivative claims."""
import torch

from directional import coupling, apply, _core
from strip import _stripT
from width import resolve_width


def main():
    torch.manual_seed(20260718)
    dtype = torch.float64
    device = "cuda" if torch.cuda.is_available() else "cpu"
    n, m, d = 6, 5, 3
    logits = torch.randn(n, m, dtype=dtype, device=device, requires_grad=True)
    s = torch.softmax(torch.randn(n, dtype=dtype, device=device), 0)
    t = torch.softmax(torch.randn(m, dtype=dtype, device=device), 0)
    h = 0.35

    # New structural full-plan path versus the former B-on-identity construction.
    pi = coupling(logits, s, t, h)
    W, grid, beta = _core(logits, s, t, h)
    Bt = _stripT(torch.eye(m, dtype=dtype, device=device), grid, beta, h)
    pi_naive = s[:, None] * (W @ Bt)
    X = torch.randn(m, d, dtype=dtype, device=device)
    y = apply(logits, X, s, t, h)

    print(f"device={device}")
    print(f"structural_vs_naive={float((pi-pi_naive).abs().max().detach()):.3e}")
    print(f"apply_vs_plan={float((y-pi@X).abs().max().detach()):.3e}")
    print(f"row_residual={float((pi.sum(1)-s).abs().max().detach()):.3e}")
    print(f"column_residual={float((pi.sum(0)-t).abs().max().detach()):.3e}")

    # New default width channel: log_h=-mean(logits), shared by coupling/apply.
    log_h = -logits.mean()
    h_auto = resolve_width(logits)
    pi_auto = coupling(logits, s, t)
    pi_log_h = coupling(logits, s, t, log_h=log_h)
    y_auto = apply(logits, X, s, t)
    shift = torch.as_tensor(0.7, dtype=dtype, device=device)
    h_shift = resolve_width(logits + shift)
    h_hard = resolve_width(torch.full_like(logits, 1e6))
    h_soft = resolve_width(torch.full_like(logits, -1e6))
    print(f"auto_h={float(h_auto.detach()):.6e}")
    print(f"auto_vs_explicit_log_h={float((pi_auto-pi_log_h).abs().max().detach()):.3e}")
    print(f"auto_apply_vs_plan={float((y_auto-pi_auto@X).abs().max().detach()):.3e}")
    print(f"shift_width_law={float((h_shift/h_auto-torch.exp(-shift)).abs().detach()):.3e}")
    print(f"finite_width_guard={bool(torch.isfinite(h_hard) & torch.isfinite(h_soft) & (h_hard > 0))}")

    # Central directional differences over a range of step sizes.
    G = torch.randn_like(pi)
    direction = torch.randn_like(logits)
    direction = direction / direction.norm()
    loss = (pi * G).sum()
    grad, = torch.autograd.grad(loss, logits, create_graph=False)
    analytic = (grad * direction).sum().detach()
    with torch.no_grad():
        for step in (1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 3e-6):
            plus = (coupling(logits + step * direction, s, t, h) * G).sum()
            minus = (coupling(logits - step * direction, s, t, h) * G).sum()
            fd = (plus - minus) / (2 * step)
            rel = (fd - analytic).abs() / analytic.abs().clamp_min(1e-30)
            print(f"directional_fd step={step:.0e} relerr={float(rel.detach()):.3e}")

    # PyTorch's elementwise finite-difference audit of the full local Jacobian.
    check_logits = logits.detach().clone().requires_grad_(True)
    ok = torch.autograd.gradcheck(
        lambda z: coupling(z, s, t, h),
        (check_logits,), eps=1e-6, atol=2e-5, rtol=2e-4,
        raise_exception=False, fast_mode=False,
    )
    print(f"coupling_gradcheck={ok}")

    # Includes the gradient path logits -> log_h -> h -> coupling.
    check_auto = logits.detach().clone().requires_grad_(True)
    ok_auto = torch.autograd.gradcheck(
        lambda z: coupling(z, s, t),
        (check_auto,), eps=1e-6, atol=3e-5, rtol=3e-4,
        raise_exception=False, fast_mode=False,
    )
    print(f"automatic_log_h_gradcheck={ok_auto}")
    check_logits_log = logits.detach().clone().requires_grad_(True)
    check_log_h = torch.tensor(-0.2, dtype=dtype, device=device, requires_grad=True)
    ok_log_h = torch.autograd.gradcheck(
        lambda z, lh: coupling(z, s, t, log_h=lh),
        (check_logits_log, check_log_h), eps=1e-6, atol=3e-5, rtol=3e-4,
        raise_exception=False, fast_mode=False,
    )
    print(f"explicit_log_h_gradcheck={ok_log_h}")
    if not (ok and ok_auto and ok_log_h):
        raise AssertionError("PyTorch coupling gradcheck failed")


if __name__ == "__main__":
    main()
