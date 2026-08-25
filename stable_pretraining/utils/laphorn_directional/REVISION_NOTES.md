# Revision notes (2026-07-19)

- `directional.coupling()` now forms the full plan structurally as
  `diag(s) * (B @ W.T).T`.  It no longer materialises `B` on an identity matrix
  and then multiplies `W @ B.T`; full-plan construction is therefore `O(nm)`.
- `directional_cuda.coupling_cuda()` now uses exactly the same structural
  identity and applies the fused CUDA strip directly to `W.T`. The old CUDA
  identity materialisation has also been removed from the production path.
- The automatic width in `coupling`, `apply`, `coupling_cuda`, and
  `apply_cuda` is now parameterised consistently as
  `log_h = -mean(logits)`, `h = exp(log_h)`. The global softmax-invariant logit
  shift is thereby used as an independent hardness channel.
- Added the keyword-only `log_h` argument and shared `width.py` resolver.
  Direct `h` remains supported; passing both raises an error. Finite-precision
  exponentiation is guarded to keep `h` in `[eps, 1/eps]`.
- Added `audit_validation.py`, a deterministic fp64 audit of:
  - structural versus identity-based reference values;
  - `apply()` versus `coupling() @ X`;
  - row and column marginals;
  - central directional differences; and
  - full PyTorch `gradcheck`.
- Added `audit_cuda.py`, which checks the compiled CUDA implementation against
  both its former identity-based construction and the pure-PyTorch reference,
  including marginals, `apply_cuda`, and reverse-mode gradients.
- Both audits now cover the automatic log-width value and gradient paths, the
  explicit `log_h` equivalence, and the multiplicative global-shift law.
- Updated the README to distinguish the strip width `h` from Sinkhorn's entropic
  regularisation and to document the structural full-plan path.
- Removed CUDA host synchronizations caused by converting the device-resident
  width to a Python float. Barrier and strip kernels now read scalar `h`
  directly from device memory in both forward and backward.
- Added a probability-native CUDA barrier path with a fused probability
  preparation kernel and a direct probability VJP. This removes redundant
  logit/sigmoid and log/log-softmax round trips.
- Cached immutable uniform marginals, grids, and levels outside compiled graphs.
- Made the common `d=64` pure-PyTorch path usable with `torch.compile` by
  isolating the current native-Windows Inductor `searchsorted(View)` failure in
  a tiny eager island.
- Fused the CUDA barrier's interval search into its inverse/root kernel,
  eliminating a separate `searchsorted`, int64 index tensor, and int32 cast.
- Removed a redundant barrier sort in the probability-native CUDA backward:
  cumulative target levels and their inverse-CDF barriers are already ordered.
- Propagated the same ordered-level invariant into the probability-native pure
  PyTorch backward, including the graph used by `torch.compile`; the general
  arbitrary-level barrier API still sorts as before.
- Added `benchmark_profile.py` and `PERFORMANCE_RESULTS.md`, covering eager,
  compiled, CUDA, materialized, backward, memory, stage profiles, and reuse of
  prebuilt transport plans.

The former identity-based constructions remain only inside the audits as
small-size value references; they are not performance baselines.
