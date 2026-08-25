# Laphorn — directional coupling (one strip): `coupling` + `apply`, PyTorch & CUDA

Turn a score matrix `logits ∈ ℝ^{n×m}` into an **exact-marginal** doubly-stochastic / transportation-
polytope matrix `Π ∈ U(s,t)` (`Π 1 = s`, `Πᵀ 1 = t`, `Π ≥ 0`), Sinkhorn-free and in **one shot**.

This is the **directional** variant of the Laphorn coupling. Instead of a *global* softmax over all
`n·m` scores plus *two* marginal corrections, it normalises along **one** axis so that marginal is exact
by construction, and corrects only the other with **one** strip:

```
W  = softmax(logits, dim=1)      # row-wise: every row sums to 1  -> row marginal is "clean"
Π  = diag(s) · W · Bᵀ            # diag(s) fixes rows EXACTLY;  B = column strip transporting Wᵀs -> t
```

Because `B` is column-stochastic (`Bᵀ 1 = 1`), applying `Bᵀ` on the right does not change row sums, so
`diag(s)` (rows) and `B` (columns) don't interfere → **exact `U(s,t)` with ONE weighted barrier solve
and ONE strip**. (When `n<m` the strip is placed on the shorter axis by transposing.) This is morally
**doubly-stochastic attention**: softmax-attention plus a single step that fixes the second marginal.

Two entry points, in **two interchangeable implementations** (same API):

| | forms `Π` | applies `Π` to a batch |
|---|---|---|
| **pure PyTorch** (`directional.py`) | `coupling(logits, s, t, h)` | `apply(logits, X, s, t, h)` |
| **fused CUDA** (`directional_cuda.py`) | `coupling_cuda(...)` | `apply_cuda(...)` |

`apply` never forms `Π`: `Y = Π X = diag(s)·(W·(Bᵀ X))`. When the full plan is requested, both
`coupling` and `coupling_cuda` form it structurally as `diag(s)·(B Wᵀ)ᵀ`, applying the strip
directly to `Wᵀ` in `O(nm)` rather than materialising `B` on an identity matrix. This is the
reference code for the paper *"Laphorn: One-Pass Exact-Marginal Coupling Layers."*

By default, all four entry points use the redundant global logit shift as a
hardness channel:

```
log_h = -logits.mean()
h     = exp(log_h)
```

Adding `c` to every logit leaves the directional softmax `W` unchanged but
multiplies `h` by `exp(-c)`. Thus the centred logits control `W`, while their
mean independently controls strip hardness. The same rule is used by
`coupling`, `apply`, `coupling_cuda`, and `apply_cuda`.

---

## Why it's fast

Two orthogonal wins that **multiply**:

1. **Directional softmax** (`softmax(dim=1)`, the tuned attention kernel) replaces the *global* softmax
   over `n²` scores — `~12×` cheaper at `n=4096` and the dominant cost there.
2. **One strip + one barrier solve** instead of two, and `diag(s)` instead of a second strip.

Plus the CUDA version runs the barrier solve and the strip on fused kernels.

**`Y = Π X`, `n=m`, `d=64`, fp32, laptop RTX 5060** — forward time (ms):

| n | **directional CUDA** | directional torch | global CUDA | global torch |
|---|---|---|---|---|
| 1024 | **0.67** | 3.16 | 1.31 | 4.19 |
| 4096 | **1.54** | 3.84 | 6.97 | 10.0 |

The directional CUDA `apply` is `6.3×` (n=1024) to `6.5×` (n=4096) faster than the original global
pure-torch operator. Note the algorithmic change alone (directional torch, 3.84 ms) already beats the
*kernel-optimised global* operator (global CUDA, 6.97 ms) at `n=4096` — softmax was the bottleneck.

(`demo.py` prints the directional CUDA-vs-torch speedup, the two bold vs. two right columns; the
global-operator columns are from a separate benchmark and are not part of this package.)

On the same machine, `torch.compile(apply, fullgraph=False)` reaches 0.64 ms at `n=1024/2048`
and 1.33 ms at `n=4096`. For forward+backward, the hand-written CUDA path is faster from
`n=1024` upward. See `PERFORMANCE_RESULTS.md` for the complete timings, peak memory,
materialized-plan reuse break-even points, and methodology.

---

## Install & run

The pure-torch part needs only PyTorch (CPU or GPU). The CUDA kernels **JIT-compile on first import**
(`torch.utils.cpp_extension` → `nvcc`), so `directional_cuda` additionally needs, on the machine that
runs it: a **CUDA GPU**, the **CUDA toolkit** (`nvcc`) matching your torch wheel, and a **host C++
compiler** (Linux `gcc/g++`; Windows MSVC Build Tools).

```bash
pip install torch                        # a CUDA wheel; RTX 50-series -> --index-url .../whl/cu128
export TORCH_CUDA_ARCH_LIST=9.0          # your GPU: 8.0 A100, 9.0 H100, 8.9 Ada, 12.0 Blackwell
python -u demo.py                        # pure-torch always; CUDA section compiles once (~1-2 min), then cached
python audit_validation.py               # fp64 structural and autograd audit
python audit_cuda.py                     # compiled CUDA audit; checks both matrix orientations
```

On RTX 50-series/Blackwell, set `TORCH_CUDA_ARCH_LIST=12.0` explicitly before the first CUDA import.
Some Windows Conda activation scripts populate a broad architecture list containing `10.1`, which
current PyTorch extension tooling may reject even though the actual `12.0` target is supported.

The CUDA compile flags in `strip_cuda.py` target the author's setup (Linux `nvcc`; Windows + MSVC 14.39,
auto-skipped if absent). If the build fails on your toolchain, edit `_NVCC_FLAGS` at the top of
`strip_cuda.py` (a plain list).

## Files

```
laphorn_directional/
├── README.md, requirements.txt, demo.py   ← start with demo.py
├── audit_validation.py   ← deterministic value, marginal, and gradient audit
├── audit_cuda.py         ← compiled-CUDA structural and gradient regression audit
├── TEST_RESULTS.md       ← recorded validation environment and numerical results
├── benchmark_profile.py  ← eager / compile / CUDA / materialized benchmark and stage profile
├── PERFORMANCE_RESULTS.md ← current RTX 5060 timings, memory, and profiling conclusions
├── directional.py        ← pure-torch:  coupling, apply           (import this for CPU/GPU torch)
├── directional_cuda.py   ← fused CUDA:  coupling_cuda, apply_cuda  (import this for the kernels)
├── width.py              ← shared automatic/explicit log-width parameterisation
├── strip.py              ← pure-torch strip-mass primitives (_strip, _stripT, _cdf_matvec_shared)
├── barriers.py           ← pure-torch barrier solver (shared: the torch path + the CUDA _states)
├── strip_cuda.py         ← CUDA strip cdf/pdf-matvec kernels + differentiable cdf_matvec_auto
└── barrier_cuda.py       ← CUDA barrier kernels + lapsum_barriers_cuda
```
The pure-torch path is `directional.py` → `strip.py`, `barriers.py`; the CUDA path is
`directional_cuda.py` → `strip_cuda.py`, `barrier_cuda.py` → `barriers.py`. Nothing else is needed.

---

## Quick start

```python
import torch
from directional import coupling, apply                 # pure torch (CPU or GPU)
# from directional_cuda import coupling_cuda, apply_cuda  # same API, fused CUDA kernels

logits = torch.randn(1024, 1024, device="cuda")
X      = torch.randn(1024, 64,   device="cuda")

Pi = coupling(logits)                                    # exact doubly-stochastic matrix (s,t uniform)
Y  = apply(logits, X)                                    # Y = Pi @ X, matrix-free
# automatic default: log_h = -logits.mean()
# alternatively pass h=0.3 or a differentiable log_h tensor:
log_h = torch.tensor(-1.0, device="cuda", requires_grad=True)
Y_log = apply(logits, X, log_h=log_h)
# custom marginals + differentiable end to end:
s = torch.softmax(torch.randn(1024, device="cuda"), 0)   # row marginal (sum = 1)
t = torch.softmax(torch.randn(1024, device="cuda"), 0)   # column marginal (sum = 1)
logits.requires_grad_(True)
(apply(logits, X, s, t) * cost).sum().backward()         # exact gradient, no Pi
```

## API

```python
coupling(logits, s=None, t=None, h=None, h_from_logits="neg_mean", *, log_h=None) -> Pi
apply(logits, X, s=None, t=None, h=None, h_from_logits="neg_mean", *, log_h=None) -> Y
#   *_cuda variants in directional_cuda have the identical signature.
```

- **`logits`** `(n, m)`, **`X`** `(m, d)` → **`Y`** `(n, d)`.
- **`s, t`** target marginals `(n,)`, `(m,)` — positive distributions with `sum(s)=sum(t)=1`; `None` →
  uniform (doubly-stochastic). Differentiable. (Not validated at runtime — pass normalised marginals.)
- **Automatic width.** If neither `h` nor `log_h` is supplied, all implementations use
  `log_h = -mean(logits)` and `h = exp(log_h)`.
- **`log_h`.** Optional scalar float/tensor in log scale. A tensor is differentiable. Its exponential
  is clipped to `[finfo.eps, 1/finfo.eps]`, preventing numerical zero/infinity in finite precision.
- **`h`.** Optional direct positive strip width, retained for compatibility and controlled experiments.
  `h` and `log_h` are mutually exclusive. Neither is numerically equivalent to Sinkhorn's entropic `ε`.
- **Legacy modes.** Explicit `h_from_logits="softplus"` and `"exp"` remain available for old callers;
  the default is now `"neg_mean"`.

---

## Notes

- **`coupling` vs `apply`.** Both PyTorch and fused-CUDA `coupling` form the `n×m` matrix by a
  structural `O(nm)` strip application (use when you need `Π` itself). The corresponding `apply`
  paths avoid storing that additional matrix (use for `Π X` at scale — doubly-stochastic attention,
  transport costs — where `d ≪ n`).
- **Hardness channel.** A global shift of the logits deliberately changes `h` while leaving the
  directional softmax unchanged. This uses an otherwise redundant degree of freedom and does not
  restrict which row-stochastic `W` or positive width can be selected.
- **Axis.** The strip goes on the shorter axis (fewer barriers); for square `n=m` the softmax is `dim=1`
  (the contiguous, faster reduction). Handled automatically.
- **Exactness.** The marginal on the `diag` axis is exact to machine precision; the other (the single
  strip) is exact to fp rounding. For `n≥m` that means rows (`s`) are machine-exact and columns (`t`)
  the strip; for `n<m` (transposed) it is the reverse. `Π ≥ 0` either way.
- **`torch.compile` / Blackwell.** On the tested RTX 5060 with PyTorch 2.11+cu128, the pure-PyTorch
  path is numerically correct under `torch.compile`. Use `fullgraph=False`: `strip.py` deliberately
  leaves only `searchsorted` eager because Inductor's native-Windows lowering can fail on a `View`
  for common `d=64` graphs. For a sweep over many static shapes, raise Dynamo's recompilation limit
  or compile one fixed-shape model per workload; the default limit can otherwise silently fall back
  to eager. The hand-written CUDA path remains the fastest backward path at larger sizes and does
  not depend on Triton/Inductor.
- **fp32 `grad_h`.** The width gradient is a scalar reduction over grid coords `0..n`, fp32-fragile
  (affects the pure-torch path too); the tensor gradients (`grad_logits`, `grad_X`) are accurate.
- **Windows.** `import torch` before `numpy`.

Paper: *"Laphorn: One-Pass Exact-Marginal Coupling Layers"*
(Ł. Struski, B. Wójcik, M. Sendera, J. Tabor). Questions → Jacek.
