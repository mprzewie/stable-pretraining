# Validation results (2026-07-19)

Test machine:

- NVIDIA GeForce RTX 5060 Laptop GPU (`sm_120`)
- PyTorch 2.11.0+cu128
- CUDA toolkit/runtime 12.8
- fused extensions compiled from this source archive

## Fused CUDA audit

`python audit_cuda.py` checks the new structural full-plan path against both
the former identity-based construction and the pure-PyTorch reference. It also
checks `apply_cuda`, exact marginals, and reverse-mode gradients. Both automatic
matrix orientations are covered.

| case | structural vs identity | CUDA vs torch | apply vs plan | row residual | column residual | logits-gradient error | h-gradient error |
|---|---:|---:|---:|---:|---:|---:|---:|
| `n=8, m=6` | 1.583e-08 | 1.490e-08 | 5.960e-08 | 1.490e-08 | 8.941e-08 | 1.304e-08 | 5.215e-07 |
| `n=6, m=8` | 4.284e-08 | 9.313e-09 | 1.192e-07 | 1.043e-07 | 5.960e-08 | 1.537e-08 | 7.078e-07 |

Result: `cuda_audit=PASS`.

### Automatic log-width path

The same CUDA audit additionally used the new default
`log_h=-mean(logits)` in both `coupling_cuda` and `apply_cuda`:

| case | automatic vs explicit `log_h` | automatic CUDA vs torch | automatic apply vs plan | automatic logits-gradient error |
|---|---:|---:|---:|---:|
| `n=8, m=6` | 0.000e+00 | 2.328e-08 | 2.235e-08 | 1.735e-08 |
| `n=6, m=8` | 0.000e+00 | 6.706e-08 | 1.043e-07 | 5.588e-09 |

An independent learnable `log_h` tensor was also checked:

| case | explicit `log_h` CUDA vs torch | explicit `log_h` apply vs plan | logits-gradient error | `log_h`-gradient error |
|---|---:|---:|---:|---:|
| `n=8, m=6` | 1.490e-08 | 2.980e-08 | 2.980e-08 | 2.049e-07 |
| `n=6, m=8` | 2.980e-08 | 7.451e-08 | 2.049e-08 | 8.196e-08 |

## Pure-PyTorch audit on the GPU

`python audit_validation.py` in fp64 reported:

- structural vs identity-based reference: 4.163e-17;
- `apply` vs materialised plan: 1.388e-16;
- row residual: 2.776e-17;
- column residual: 2.914e-16;
- best central directional-difference relative error: 2.017e-09;
- full `coupling` gradcheck: `True`.
- automatic `log_h=-mean(logits)` equals explicit `log_h`: 0.000e+00;
- automatic `apply` vs materialised automatic plan: 1.110e-16;
- multiplicative global-shift width law residual: 0.000e+00;
- automatic-width full `coupling` gradcheck: `True`;
- independent explicit-`log_h` full gradcheck: `True`;
- extreme-logit finite-width guard: `True`.

## Pure-PyTorch CPU smoke test

- automatic `apply` vs materialised plan: 6.939e-17;
- row residual: 2.776e-17;
- column residual: 1.110e-16;
- automatic-width logit gradient finite: `True`.

These are correctness and derivative audits, not performance benchmarks. The
identity-based construction is retained only as a small-instance oracle.
