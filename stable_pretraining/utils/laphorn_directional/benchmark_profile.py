"""Benchmark and profile the directional Laphorn implementations on CUDA.

Compared paths:
  torch-eager       pure-PyTorch matrix-free apply
  torch-compile     the same apply under torch.compile(fullgraph=False)
  cuda              hand-written CUDA matrix-free apply
  materialized      pure-PyTorch coupling(logits) @ X
  cuda-materialized CUDA coupling_cuda(logits) @ X (extra diagnostic)
"""
from __future__ import annotations

import argparse
import math
import statistics
import time

import torch

from directional import apply, coupling
from directional_cuda import apply_cuda, coupling_cuda
from barriers import lapsum_barriers_probs
from barrier_cuda import lapsum_barriers_probs_cuda
from strip import _strip, _stripT
from directional_cuda import _strip_cuda, _stripT_cuda
from width import resolve_width
from constants import uniform_marginal, uniform_grid_levels


def _median_cuda_ms(fn, *, warmup: int, repeats: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]
    for start, end in zip(starts, ends):
        start.record()
        fn()
        end.record()
    torch.cuda.synchronize()
    return statistics.median(start.elapsed_time(end) for start, end in zip(starts, ends))


def _peak_extra_mib(fn) -> float:
    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    fn()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    return (peak - baseline) / (1024**2)


def _make_inputs(n: int, d: int):
    logits = torch.randn(n, n, device="cuda", dtype=torch.float32)
    values = torch.randn(n, d, device="cuda", dtype=torch.float32)
    probe = torch.randn(n, d, device="cuda", dtype=torch.float32)
    return logits, values, probe


def _forward_functions(compiled_apply, logits, values):
    return {
        "torch-eager": lambda: apply(logits, values),
        "torch-compile": lambda: compiled_apply(logits, values),
        "cuda": lambda: apply_cuda(logits, values),
        "materialized": lambda: coupling(logits) @ values,
        "cuda-materialized": lambda: coupling_cuda(logits) @ values,
    }


def _backward_function(name, compiled_apply, logits0, values0, probe):
    logits = logits0.detach().requires_grad_(True)
    values = values0.detach().requires_grad_(True)
    forward = _forward_functions(compiled_apply, logits, values)[name]

    def step():
        output = forward()
        return torch.autograd.grad((output * probe).sum(), (logits, values))

    return step


def _stage_once(kind, logits, values):
    n = logits.shape[0]
    events = [torch.cuda.Event(enable_timing=True) for _ in range(7)]
    events[0].record()
    h = resolve_width(logits, h=None, log_h=None, h_from_logits="neg_mean")
    marginal = uniform_marginal(n, logits)
    events[1].record()
    W = torch.softmax(logits, dim=1)
    events[2].record()
    grid, levels = uniform_grid_levels(n, logits)
    measure = W.t() @ marginal
    measure = measure / measure.sum()
    events[3].record()
    barrier_fn = lapsum_barriers_probs_cuda if kind.startswith("cuda") else lapsum_barriers_probs
    beta = barrier_fn(
        grid,
        levels,
        measure,
        h=h,
        presorted=True,
    )
    events[4].record()
    if kind == "torch-eager":
        transformed = _stripT(values, grid, beta, h)
        events[5].record()
        output = marginal.unsqueeze(1) * (W @ transformed)
    elif kind == "cuda":
        transformed = _stripT_cuda(values, grid, beta, h)
        events[5].record()
        output = marginal.unsqueeze(1) * (W @ transformed)
    elif kind == "materialized":
        plan = marginal.unsqueeze(1) * _strip(W.t(), grid, beta, h).t()
        events[5].record()
        output = plan @ values
    elif kind == "cuda-materialized":
        plan = marginal.unsqueeze(1) * _strip_cuda(W.t(), grid, beta, h).t()
        events[5].record()
        output = plan @ values
    else:
        raise ValueError(kind)
    events[6].record()
    return events, output


def _profile_stages(kind, logits, values, repeats=20):
    for _ in range(3):
        _stage_once(kind, logits, values)
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        events, output = _stage_once(kind, logits, values)
        samples.append(events)
    torch.cuda.synchronize()
    names = ("width+marginals", "softmax", "measure", "barrier", "strip/plan", "final-matmul")
    medians = [
        statistics.median(events[index].elapsed_time(events[index + 1]) for events in samples)
        for index in range(len(names))
    ]
    total = sum(medians)
    print(f"\nSTAGE_PROFILE {kind} total_stages_ms={total:.6f}")
    for name, elapsed in zip(names, medians):
        print(f"STAGE_ROW\t{kind}\t{name}\t{elapsed:.6f}ms\t{100.0 * elapsed / total:.2f}%")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", type=int, nargs="+", default=[512, 1024, 2048, 4096])
    parser.add_argument("--d", type=int, default=64)
    parser.add_argument("--forward-repeats", type=int, default=30)
    parser.add_argument("--backward-repeats", type=int, default=12)
    parser.add_argument("--profile-size", type=int, default=2048)
    parser.add_argument("--skip-profile", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.manual_seed(20260719)
    torch.set_float32_matmul_precision("high")
    # This benchmark intentionally sweeps several static shapes.  The pure
    # path has a small searchsorted eager island, so the default limit of 8
    # specializations can otherwise trigger a silent eager fallback late in
    # the sweep and mislabel it as compiled performance.
    torch._dynamo.config.recompile_limit = max(torch._dynamo.config.recompile_limit, 64)
    print("torch", torch.__version__)
    print("cuda", torch.version.cuda)
    print("gpu", torch.cuda.get_device_name(0))
    print("automatic width: log_h=-mean(logits)")

    def torch_apply(logits, values):
        return apply(logits, values)

    compiled_apply = torch.compile(torch_apply, fullgraph=False, dynamic=False)

    # Compile-time diagnostic on a shape not used by the correctness smoke test.
    logits_c, values_c, _ = _make_inputs(args.sizes[0], args.d)
    torch.cuda.synchronize()
    start = time.perf_counter()
    compiled_apply(logits_c, values_c)
    torch.cuda.synchronize()
    print(f"compile_first_call_ms {1000.0 * (time.perf_counter() - start):.3f}")
    del logits_c, values_c

    # Correctness at a compact size, for all paths.
    logits_q, values_q, _ = _make_inputs(256, min(args.d, 32))
    reference = apply(logits_q, values_q)
    for name, fn in _forward_functions(compiled_apply, logits_q, values_q).items():
        error = (fn() - reference).abs().max().item()
        print(f"correctness {name} max_abs={error:.6e}")
    del logits_q, values_q, reference

    print("\nFORWARD_MS")
    print("n\tpath\tmedian_ms\tpeak_extra_MiB")
    forward_times = {}
    for n in args.sizes:
        logits, values, _ = _make_inputs(n, args.d)
        functions = _forward_functions(compiled_apply, logits, values)
        for name, fn in functions.items():
            with torch.no_grad():
                try:
                    ms = _median_cuda_ms(fn, warmup=5, repeats=args.forward_repeats)
                    memory = _peak_extra_mib(fn)
                    forward_times[(n, name)] = ms
                    print(f"{n}\t{name}\t{ms:.6f}\t{memory:.3f}", flush=True)
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    print(f"{n}\t{name}\tOOM\tOOM", flush=True)
        del logits, values, functions
        torch.cuda.empty_cache()

    print("\nREUSED_PLAN_FORWARD_MS")
    print("n\tpath\tbuild_ms\tprebuilt_matmul_ms\tplan_MiB\tbreak_even_applies")
    for n in args.sizes:
        logits, values, _ = _make_inputs(n, args.d)
        for name, builder, matrix_free_name in (
            ("materialized", lambda: coupling(logits), "torch-eager"),
            ("cuda-materialized", lambda: coupling_cuda(logits), "cuda"),
        ):
            with torch.no_grad():
                build_ms = _median_cuda_ms(builder, warmup=3, repeats=max(5, args.forward_repeats // 2))
                plan = builder()
                matmul_ms = _median_cuda_ms(lambda: plan @ values, warmup=5, repeats=args.forward_repeats)
            matrix_free_ms = forward_times.get((n, matrix_free_name), float("nan"))
            saving = matrix_free_ms - matmul_ms
            break_even = math.ceil(build_ms / saving) if saving > 0 else -1
            plan_mib = plan.numel() * plan.element_size() / (1024**2)
            print(
                f"{n}\t{name}\t{build_ms:.6f}\t{matmul_ms:.6f}\t"
                f"{plan_mib:.3f}\t{break_even}", flush=True
            )
            del plan
        del logits, values
        torch.cuda.empty_cache()

    backward_sizes = [n for n in args.sizes if n <= 2048]
    print("\nFORWARD_BACKWARD_MS")
    print("n\tpath\tmedian_ms\tpeak_extra_MiB")
    for n in backward_sizes:
        logits, values, probe = _make_inputs(n, args.d)
        forward_functions = _forward_functions(compiled_apply, logits, values)
        for name in forward_functions:
            step = _backward_function(name, compiled_apply, logits, values, probe)
            try:
                ms = _median_cuda_ms(step, warmup=3, repeats=args.backward_repeats)
                memory = _peak_extra_mib(step)
                print(f"{n}\t{name}\t{ms:.6f}\t{memory:.3f}", flush=True)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                print(f"{n}\t{name}\tOOM\tOOM", flush=True)
        del logits, values, probe, forward_functions
        torch.cuda.empty_cache()

    if not args.skip_profile:
        n = args.profile_size
        logits, values, _ = _make_inputs(n, args.d)
        for name in ("torch-eager", "cuda", "materialized", "cuda-materialized"):
            _profile_stages(name, logits, values)


if __name__ == "__main__":
    main()
