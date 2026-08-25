"""
demo.py -- directional-softmax Laphorn coupling: coupling (form Pi) + apply (Pi @ X), with ONE strip.

  (1) coupling(logits, s, t) -> Pi, an exact-marginal matrix in the transportation polytope U(s,t)
  (2) apply(logits, X)       -> Y = Pi @ X, matrix-free (Pi never formed) == coupling @ X
  (3) differentiable in one pass
  (4) the CUDA fused-kernel version (if a GPU is present): same answer, much faster at scale

Pure-torch part runs on CPU or GPU.  The CUDA kernels JIT-compile on first import (~1-2 min once,
then cached) and need a CUDA toolkit + a host C++ compiler.  torch BEFORE numpy.
Run:  python -u demo.py
"""
import torch  # before numpy

from directional import coupling, apply                 # pure PyTorch (CPU or GPU)

torch.manual_seed(0)
dev = "cuda" if torch.cuda.is_available() else "cpu"
f64 = torch.float64
print(f"[demo] device = {dev}\n")


# (1) COUPLING: form Pi in U(s,t) --------------------------------------------------
n, m = 200, 160
logits = torch.randn(n, m, device=dev, dtype=f64)
s = torch.softmax(torch.randn(n, device=dev, dtype=f64), 0)      # row marginal (positive, sums to 1)
t = torch.softmax(torch.randn(m, device=dev, dtype=f64), 0)      # column marginal (sum(s)=sum(t)=1)
Pi = coupling(logits, s, t)                              # default: log_h = -mean(logits)
print("(1) coupling -> Pi in the transportation polytope U(s,t)   [one directional softmax + one strip]")
print(f"    |Pi 1 - s|   = {(Pi.sum(1) - s).abs().max():.1e}   (row marginal)")
print(f"    |Pi^T 1 - t| = {(Pi.sum(0) - t).abs().max():.1e}   (column marginal)")
print(f"    min Pi       = {Pi.min():.1e}   (nonnegative)\n")


# (2) APPLY: Y = Pi @ X, matrix-free -----------------------------------------------
X = torch.randn(m, 32, device=dev, dtype=f64)
Y = apply(logits, X, s, t)                               # exactly the same automatic width
print("(2) apply -> Y = Pi @ X, matrix-free (Pi never formed)")
print(f"    |apply - coupling @ X| = {(Y - Pi @ X).abs().max():.1e}\n")


# (3) DIFFERENTIABLE ---------------------------------------------------------------
lg = logits.clone().requires_grad_(True)
log_h = torch.tensor(-0.2, device=dev, dtype=f64, requires_grad=True)
(apply(lg, X, s, t, log_h=log_h) * torch.randn_like(Y)).sum().backward()
print("(3) differentiable in one pass")
print(f"    grad finite: logits {torch.isfinite(lg.grad).all().item()}, log_h {torch.isfinite(log_h.grad).all().item()}\n")


# (4) CUDA FUSED KERNELS (if available): same answer, much faster at scale ----------
if dev == "cuda":
    torch.set_float32_matmul_precision("high")                  # TF32 tensor cores for the W@Z GEMM
    print("[demo] compiling CUDA kernels on first import (cached afterwards)...", flush=True)
    from directional_cuda import apply_cuda

    N = 4096
    lg = torch.randn(N, N, device=dev); Xb = torch.randn(N, 64, device=dev); ht = torch.tensor(0.3, device=dev)
    Yc, Yt = apply_cuda(lg, Xb, h=ht), apply(lg, Xb, h=ht)
    print(f"(4) CUDA apply_cuda == pure-torch apply:  |diff| = {(Yc - Yt).abs().max():.1e}")

    def ev(fn, reps=30):
        for _ in range(5):
            fn()
        torch.cuda.synchronize()
        a, b = torch.cuda.Event(True), torch.cuda.Event(True)
        a.record()
        for _ in range(reps):
            fn()
        b.record(); torch.cuda.synchronize()
        return a.elapsed_time(b) / reps

    t_c, t_t = ev(lambda: apply_cuda(lg, Xb, h=ht)), ev(lambda: apply(lg, Xb, h=ht))
    print(f"    forward n=m={N}, d=64:  CUDA {t_c:.2f} ms | pure-torch {t_t:.2f} ms  ->  {t_t / t_c:.1f}x faster")

print("\n[demo] done.")
