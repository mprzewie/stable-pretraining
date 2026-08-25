# Performance and profiling results (2026-07-19)

Test machine:

- NVIDIA GeForce RTX 5060 Laptop GPU (`sm_120`, 8 GB)
- PyTorch 2.11.0+cu128, CUDA 12.8
- `triton-windows` 3.7.0.post26
- fp32, square score matrices, `d=64`
- automatic width `log_h=-mean(logits)`
- `torch.set_float32_matmul_precision("high")`

Times below are medians from CUDA events after warm-up. The compiled path is
`torch.compile(..., fullgraph=False, dynamic=False)`. A tiny eager island is
used only for `searchsorted`, due to an Inductor `View` lowering failure for
typical `d=64` graphs on this Windows installation.

## Forward: one matrix-free apply or build-plan-and-apply

| n | PyTorch eager | PyTorch compiled | CUDA apply | PyTorch materialized | CUDA materialized |
|---:|---:|---:|---:|---:|---:|
| 512  | 3.390 ms | **0.699 ms** | 0.756 ms | 3.489 ms | 0.850 ms |
| 1024 | 3.159 ms | **0.636 ms** | 0.666 ms | 3.533 ms | 0.784 ms |
| 2048 | 3.243 ms | **0.644 ms** | 0.673 ms | 9.875 ms | 1.140 ms |
| 4096 | 3.836 ms | **1.329 ms** | 1.541 ms | 31.452 ms | 5.726 ms |

Peak extra allocated memory (MiB):

| n | PyTorch eager | PyTorch compiled | CUDA apply | PyTorch materialized | CUDA materialized |
|---:|---:|---:|---:|---:|---:|
| 512  | 3.9 | **1.9** | 1.4 | 23.9 | 4.0 |
| 1024 | 9.8 | 5.8 | **4.8** | 92.0 | 16.0 |
| 2048 | 27.6 | 19.5 | **17.5** | 368.1 | 64.0 |
| 4096 | 87.1 | 71.1 | **67.0** | 1472.2 | 256.1 |

## Forward plus backward

The loss is a fixed linear probe of the output; gradients are taken with
respect to both logits and values.

| n | PyTorch eager | PyTorch compiled | CUDA apply | PyTorch materialized | CUDA materialized |
|---:|---:|---:|---:|---:|---:|
| 512  | 10.168 ms | **3.389 ms** | 3.535 ms | 10.163 ms | 4.003 ms |
| 1024 | 9.245 ms | 3.351 ms | **2.584 ms** | 12.097 ms | 3.150 ms |
| 2048 | 8.739 ms | 3.012 ms | **2.720 ms** | 44.980 ms | 4.997 ms |

The compiled backward silently fell back to eager in an earlier multi-shape
sweep after reaching Dynamo's default specialization limit of 8. The benchmark
raises that limit to 64 because it deliberately tests four static shapes. A
normal fixed-shape model does not need this adjustment.

## Reusing a materialized plan

The table separates plan construction from multiplication by a new `X`.
`break-even` is the minimum number of applications to the same logits at which
building the plan once becomes faster than repeated matrix-free calls. This is
an inference/reuse scenario; training through repeated backward passes has
different graph-lifetime constraints.

| n | builder | build | prebuilt `P @ X` | plan storage | break-even |
|---:|---|---:|---:|---:|---:|
| 512  | PyTorch | 3.391 ms | 0.041 ms | 1 MiB | 2 applies |
| 512  | CUDA | 0.741 ms | 0.042 ms | 1 MiB | 2 applies |
| 1024 | PyTorch | 3.641 ms | 0.041 ms | 4 MiB | 2 applies |
| 1024 | CUDA | 0.788 ms | 0.047 ms | 4 MiB | 2 applies |
| 2048 | PyTorch | 8.444 ms | 0.062 ms | 16 MiB | 3 applies |
| 2048 | CUDA | 1.110 ms | 0.045 ms | 16 MiB | 2 applies |
| 4096 | PyTorch | 31.172 ms | 0.212 ms | 64 MiB | 9 applies |
| 4096 | CUDA | 5.612 ms | 0.214 ms | 64 MiB | 5 applies |

## Profiling conclusions and implemented optimizations

At `n=2048`, the PyTorch eager path spends about 58% of GPU stage time in the
barrier and 33% in the strip. The optimized matrix-free CUDA path spends about
48% in the barrier and 17% in the strip; softmax itself is only about 6%.

The following changes were implemented and validated:

1. Removed GPU-to-CPU synchronizations caused by `float(h)` in the CUDA barrier
   and strip wrappers. Kernels now read the scalar width directly from device
   memory. This is especially important for automatic `h`, which depends on
   the logits and is produced on the GPU.
2. Added a probability-native barrier path. It avoids redundant
   `logit -> sigmoid` and `log -> log_softmax` round trips, fuses
   `log(weight)` with `logit(level)` in one CUDA kernel, and returns the VJP
   directly with respect to probabilities.
3. Cached immutable uniform marginals, integer grids, and uniform CDF levels
   for eager/CUDA calls. Compilation deliberately keeps tensor construction in
   the graph instead of consulting the cache.
4. Added a small eager `searchsorted` island so that `torch.compile` works for
   the common `d=64` graph despite the current Inductor Windows lowering bug.
5. Fused the CUDA barrier's binary interval search into the closed-form inverse
   kernel, removing two launches and the temporary index tensor.
6. Reused the monotonicity of cumulative target levels in the CUDA backward,
   avoiding a redundant barrier sort and gather.
7. Applied the same ordered-level fast path to the pure-PyTorch custom
   backward. The general barrier entry point retains sorting for arbitrary
   level order.

### Barrier follow-up profile

For `n=2048`, the pre-optimization CUDA barrier forward split into roughly
`0.026 ms` probability preparation, `0.183 ms` for the four `_states`
log-scans, `0.032 ms` for node logits, `0.034 ms` for search+cast, and
`0.023 ms` for the closed-form inverse. Fusing search into inverse removes the
standalone `0.034 ms` region. The ordered-level backward optimization removes
another `0.10--0.13 ms` sort+gather region.

A focused rerun after both changes measured CUDA `forward+backward` at
`2.54 ms` (`n=1024`) and `2.55 ms` (`n=2048`, `d=64`). The latter was about
6% faster than the earlier `2.72 ms` run. Timing differences below a few
hundredths of a millisecond should still be treated as clock/warm-up noise.

For PyTorch, a same-process A/B comparison of the directional apply graph
showed `7.13 -> 6.99 ms` eager and `2.08 -> 1.99 ms` compiled at `n=1024`.
At `n=2048` it showed `7.18 -> 7.11 ms` eager and approximately
`2.08 -> 1.97--1.99 ms` compiled. Values were bit-identical and the maximum
gradient difference was `4.5e-13`.

The main remaining barrier-forward opportunity is `_states`. Compiling it
alone reduced its time from about `0.21 ms` to `0.10 ms`, but changed barrier
positions by up to `1.3e-3` at `n=4096`; this variant was therefore rejected.
A future safe speedup would require a dedicated, numerically audited CUDA
affine/log-scan for the regular directional grid rather than relying on the
current Inductor lowering.

The CUDA value/gradient audit passes in both automatic matrix orientations;
the pure PyTorch fp64 structural and gradcheck audit also passes. Runtime
differences against eager are at normal fp32 rounding scale (roughly `1e-7`).

Run the benchmark with:

```bash
python -u benchmark_profile.py
```
