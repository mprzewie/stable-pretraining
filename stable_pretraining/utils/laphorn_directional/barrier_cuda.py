"""
barrier_cuda.py -- fused CUDA kernels for the Laphorn barrier FORWARD (step 3 of the operator).

Current B=1 profiling at n=512..4096 shows a roughly 0.30--0.38 ms barrier.
The four `_states` log-scans dominate at about 0.18--0.22 ms; node logits and
the fused interval-search/root kernels are much smaller. Since compiling the
scans changes barrier values measurably on Blackwell, only the algebraic
elementwise chains and interval search are fused here.

Step 3a (this file, first kernel): `inverse_tail` -- fuse interval binary search, boundary gather,
the closed-form root `_stable_G`, and the barrier density into ONE per-level kernel
(grid over B*m, one thread per level).
Ports barriers.py::_stable_G faithfully (softplus/expm1/log stabilisation, -inf sentinels).
Validated against `_inverse_multi(..., return_density=True)`.  Reuses strip_cuda's toolchain.
torch BEFORE numpy.
"""
from __future__ import annotations
import math
import os
import sys

import torch

import strip_cuda                       # toolchain flags + strip kernels (reused by the barrier backward)
from strip_cuda import _NVCC_FLAGS, load_inline, cdf_pdf_matvec_cuda

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))                  # flat package: barriers.py sibling
from barriers import _states                                                    # noqa: E402  (torch log-scans)


_BAR_SRC = r"""
#include <torch/extension.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Prepare normalized log-weights and level logits in one launch for the
// probability-native directional fast path.
__global__ void prepare_probs_k(
    const float* __restrict__ weight, const float* __restrict__ alpha,
    float* __restrict__ log_w, float* __restrict__ y, int nw, int na)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < nw) log_w[i] = logf(fmaxf(weight[i], 1e-30f));
    if (i < na) {
        float p = fminf(fmaxf(alpha[i], 1e-7f), 1.f - 1e-7f);
        y[i] = logf(p) - log1pf(-p);
    }
}

std::vector<torch::Tensor> prepare_probs_fwd(torch::Tensor weight, torch::Tensor alpha)
{
    auto log_w = torch::empty_like(weight), y = torch::empty_like(alpha);
    int nw = weight.numel(), na = alpha.numel(), total = nw > na ? nw : na;
    int blk = 256, grid = (total + blk - 1) / blk;
    prepare_probs_k<<<grid, blk>>>(weight.data_ptr<float>(), alpha.data_ptr<float>(),
        log_w.data_ptr<float>(), y.data_ptr<float>(), nw, na);
    C10_CUDA_CHECK(cudaGetLastError());
    return {log_w, y};
}

// Closed-form root of the interval equation (port of barriers.py::_stable_G).  All -inf sentinels
// follow IEEE (expf(-inf)=0, expm1f(-inf)=-1, fmaxf(-inf,x)=x); c_ext/d_ext are never both -inf at
// the same level, so no -inf - -inf = NaN arises (see the python note).
__device__ __forceinline__ float dstable_G(
    float alpha, float beta, float xl, float xr, float h, float gamma, float delta, float z)
{
    float p = delta + z;
    bool  hi = p > gamma;
    float Amax    = hi ? p : gamma;                 // maximum(delta+z, gamma)
    float negdiff = hi ? (gamma - p) : (p - gamma); // Amin - Amax   (<= 0)
    float r = -expm1f(negdiff);                      // 1 - e^{negdiff} in [0,1]
    float log_den = fmaxf(z, 0.f) + log1pf(expf(-fabsf(z)));   // softplus(z), stable
    float P = Amax - log_den;
    float A = alpha + beta + (xl - xr) / h;
    float nn = fmaxf(P, 0.5f * A);                   // stabiliser (G invariant to it)
    float u  = expf(P - nn) * r;
    float vv = expf(A - 2.f * nn);
    float G  = nn + logf(u + sqrtf(u * u + vv));
    return hi ? (xr + h * (G - alpha)) : (xl + h * (beta - G));
}

// One thread per (batch, level). Locate the interval and evaluate the root + density.
// Doing the lower_bound here removes a separate searchsorted launch, an int64 index
// allocation, an int64->int32 conversion launch, and the resulting index traffic.
__global__ void inverse_tail(
    const float* __restrict__ a,   const float* __restrict__ b_st,
    const float* __restrict__ c,   const float* __restrict__ d,
    const float* __restrict__ x,   const float* __restrict__ node_logit,
    const float* __restrict__ y,   const float* __restrict__ hptr,
    float* __restrict__ bout, float* __restrict__ dens, int B, int n, int m)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= B * m) return;
    int bi = t / m, li = t % m;
    const float* ar = a    + (size_t)bi * n;
    const float* br = b_st + (size_t)bi * n;
    const float* cr = c    + (size_t)bi * n;
    const float* dr = d    + (size_t)bi * n;
    const float* xr = x    + (size_t)bi * n;
    const float* nr = node_logit + (size_t)bi * n;
    float z  = y[(size_t)bi * m + li];
    // torch.searchsorted(node_logit, y, right=False): first node >= z.
    int lo_i = 0, hi_i = n;
    while (lo_i < hi_i) {
        int mid = lo_i + ((hi_i - lo_i) >> 1);
        if (nr[mid] < z) lo_i = mid + 1;
        else hi_i = mid;
    }
    int id = lo_i;
    float h = hptr[0], margin = 80.f * h;
    const float NEG = -INFINITY;
    float alpha = (id < n) ? ar[id]     : NEG;       // a_ext  = [a, NEG]
    float beta  = (id > 0) ? br[id - 1] : NEG;       // b_ext  = [NEG, b_st]
    float gamma = (id > 0) ? cr[id - 1] : NEG;       // c_ext  = [NEG, c]
    float delta = (id < n) ? dr[id]     : NEG;       // d_ext  = [d, NEG]
    float xl = (id > 0) ? xr[id - 1] : (xr[0]     - margin);   // r_ext = [lo, x, hi]
    float xR = (id < n) ? xr[id]     : (xr[n - 1] + margin);
    float bb = dstable_G(alpha, beta, xl, xR, h, gamma, delta, z);
    bout[(size_t)bi * m + li] = bb;
    // barrier density d_i = 1/2 ( e^{beta+(xl-b)/h} + e^{alpha+(b-xr)/h} ), both exponents <= 0
    dens[(size_t)bi * m + li] = 0.5f * (expf(beta + (xl - bb) / h) + expf(alpha + (bb - xR) / h));
}

// One thread per (batch, node).  node_logit[i] = logit(F(x_i)), a per-node stencil over the log-scan
// states a,b,c,d (port of barriers.py::_node_logits).  Dead (s=-inf) nodes -> -inf.
__global__ void node_logits_k(
    const float* __restrict__ x, const float* __restrict__ s,
    const float* __restrict__ a, const float* __restrict__ b_st,
    const float* __restrict__ c, const float* __restrict__ d,
    const float* __restrict__ hptr, float* __restrict__ out, int B, int n)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= B * n) return;
    int bi = t / n, i = t % n;
    float h = hptr[0];
    const float* xr = x + (size_t)bi * n; const float* sr = s + (size_t)bi * n;
    const float* ar = a + (size_t)bi * n; const float* br = b_st + (size_t)bi * n;
    const float* cr = c + (size_t)bi * n; const float* dr = d + (size_t)bi * n;
    const float LOG_HALF = -0.69314718055994531f;
    float log_f, log1mf;
    if (i == 0) {
        log_f = LOG_HALF + ar[0];
    } else {
        float gap = (xr[i - 1] - xr[i]) / h;
        float mm = fmaxf(cr[i - 1], ar[i]);
        log_f = mm + logf(expf(cr[i - 1] - mm) + 0.5f * expf(ar[i] - mm)
                          - 0.5f * expf(br[i - 1] - mm + gap));
    }
    if (i == n - 1) {
        log1mf = LOG_HALF + br[n - 1];
    } else {
        float gap = (xr[i] - xr[i + 1]) / h;
        float mm = fmaxf(dr[i + 1], br[i]);
        log1mf = mm + logf(expf(dr[i + 1] - mm) + 0.5f * expf(br[i] - mm)
                           - 0.5f * expf(ar[i + 1] - mm + gap));
    }
    float nl = log_f - log1mf;
    out[t] = (isinf(sr[i]) && sr[i] < 0.f) ? -INFINITY : nl;    // dead node -> NEG
}

torch::Tensor node_logits_fwd(
    torch::Tensor x, torch::Tensor s, torch::Tensor a, torch::Tensor b_st,
    torch::Tensor c, torch::Tensor d, torch::Tensor h)
{
    int B = x.size(0), n = x.size(1);
    auto out = torch::empty({B, n}, x.options());
    int total = B * n, blk = 256, grid = (total + blk - 1) / blk;
    node_logits_k<<<grid, blk>>>(
        x.data_ptr<float>(), s.data_ptr<float>(), a.data_ptr<float>(), b_st.data_ptr<float>(),
        c.data_ptr<float>(), d.data_ptr<float>(), h.data_ptr<float>(), out.data_ptr<float>(), B, n);
    C10_CUDA_CHECK(cudaGetLastError());
    return out;
}

std::vector<torch::Tensor> inverse_tail_fwd(
    torch::Tensor a, torch::Tensor b_st, torch::Tensor c, torch::Tensor d,
    torch::Tensor x, torch::Tensor node_logit, torch::Tensor y, torch::Tensor h)
{
    int B = a.size(0), n = a.size(1), m = y.size(1);
    auto bout = torch::empty({B, m}, a.options());
    auto dens = torch::empty({B, m}, a.options());
    int total = B * m, blk = 256, grid = (total + blk - 1) / blk;
    inverse_tail<<<grid, blk>>>(
        a.data_ptr<float>(), b_st.data_ptr<float>(), c.data_ptr<float>(), d.data_ptr<float>(),
        x.data_ptr<float>(), node_logit.data_ptr<float>(), y.data_ptr<float>(), h.data_ptr<float>(),
        bout.data_ptr<float>(), dens.data_ptr<float>(), B, n, m);
    C10_CUDA_CHECK(cudaGetLastError());
    return {bout, dens};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("prepare_probs_fwd", &prepare_probs_fwd, "fused log(weight) + logit(alpha)");
    m.def("inverse_tail_fwd", &inverse_tail_fwd, "fused gather + stable_G root + barrier density");
    m.def("node_logits_fwd", &node_logits_fwd, "fused per-node logit(F(x_i)) stencil");
}
"""


print("Compiling laphorn_barrier extension...", end=" ", flush=True)
_bext = load_inline(name="laphorn_barrier_ext", cpp_sources=[""], cuda_sources=[_BAR_SRC],
                    extra_cuda_cflags=_NVCC_FLAGS, verbose=False)
print("OK")


def node_logits_cuda(x, s, a, b_st, c, d, h):
    """Fused CUDA per-node node_logit[i] = logit(F(x_i)).  Drop-in for barriers._node_logits."""
    f = lambda t: t.to(torch.float32).contiguous()
    return _bext.node_logits_fwd(f(x), f(s), f(a), f(b_st), f(c), f(d), f(h))


def prepare_probs_cuda(weight, alpha):
    f = lambda t: t.to(torch.float32).contiguous()
    return _bext.prepare_probs_fwd(f(weight), f(alpha))


def inverse_tail_cuda(a, b_st, c, d, x, node_logit, y, h):
    """Fused CUDA boundary gather + closed-form root + density.  a,b_st,c,d,x (B,n) from _states;
    node_logit (B,n); y (B,m) level logits; h scalar.  Returns (b, dens) (B,m).  Drop-in for the
    gather + _stable_G tail of barriers._inverse_multi(..., return_density=True)."""
    f = lambda t: t.to(torch.float32).contiguous()
    return _bext.inverse_tail_fwd(f(a), f(b_st), f(c), f(d), f(x), f(node_logit), f(y), f(h))


def _barrier_forward_cuda(x_sorted, s, y, h):
    """Fused barrier forward value+density: torch _states -> CUDA node_logits -> CUDA inverse_tail."""
    a, b_st, c, d = _states(x_sorted, s, h)
    nl = node_logits_cuda(x_sorted, s, a, b_st, c, d, h)
    return inverse_tail_cuda(a, b_st, c, d, x_sorted, nl, y, h)                  # (b, dens)


class _BarriersCuda(torch.autograd.Function):
    """CUDA Laphorn barriers: fused forward (above) + IFT-VJP backward that reuses the strip
    cdf/pdf matvec kernel.  Mirrors barriers._Barriers exactly, swapping only the compute:
    the backward's `_lr_sums(x, sort(b), g)` (density-normalised cotangent g=grad_b/d) is one
    cdf_pdf_matvec per batch row -> pdf gives gx = w*pdf (grad_x), cdf gives grad_logw = -h*w*cdf."""

    @staticmethod
    def forward(ctx, x, log_w, alpha, h, presorted=False):
        with torch.no_grad():
            if presorted:
                x_sorted, s = x, log_w
            else:
                x_sorted, order = x.sort(dim=-1)
                s = torch.gather(log_w, -1, order)
            a = alpha.clamp(1e-7, 1 - 1e-7)
            y = torch.log(a) - torch.log1p(-a)                                  # logit(alpha)
            b, dens = _barrier_forward_cuda(x_sorted, s, y, h)
        ctx.save_for_backward(x, log_w, b, h)
        ctx.dens = dens
        return b

    @staticmethod
    def backward(ctx, grad_b):
        x, log_w, b, h = ctx.saved_tensors
        need_x, need_logw, need_alpha, need_h = ctx.needs_input_grad[:4]
        w = torch.exp(log_w)                                                    # (B,n)
        d = ctx.dens.clamp_min(1e-30)
        g = (grad_b / d).to(torch.float32)                                      # (B,m)
        B = x.shape[0]
        xf, bf = x.to(torch.float32), b.to(torch.float32)
        bs, order = bf.sort(dim=-1)
        gs = torch.gather(g, -1, order)                                         # values in b-sorted order
        cdf = torch.empty_like(xf); pdf = torch.empty_like(xf)                  # (B,n)
        for bi in range(B):                                                     # strip kernel per batch row
            y_c, y_p = cdf_pdf_matvec_cuda(xf[bi], bs[bi], gs[bi].unsqueeze(-1), h)
            cdf[bi] = y_c.squeeze(-1); pdf[bi] = y_p.squeeze(-1)
        gx = w * pdf                                                            # gx_j = w_j * 1/2 ker(x,b) g
        grad_x = gx if need_x else None
        grad_logw = (-h * w * cdf) if need_logw else None                      # -h w cdfker(x,b) g
        grad_alpha = grad_b * (h / d) if need_alpha else None
        grad_h = None
        if need_h:
            # scalar reduction over grid coords 0..n (large) -> accumulate in fp64 to avoid fp32 cancellation
            _num = ((b.double() * grad_b.double()).sum(-1)
                    - (x.double() * gx.double()).sum(-1)).to(x.dtype)           # (B,)
            grad_h = (_num / h.reshape(-1)).reshape(h.shape) if (torch.is_tensor(h) and h.dim() >= 1) \
                else (_num / h).sum()
        return grad_x, grad_logw, grad_alpha, grad_h, None


class _BarriersCudaProbs(torch.autograd.Function):
    """Probability-native fast path for the presorted directional operator.

    The internal caller supplies cumulative target probabilities, hence alpha
    is ascending and its inverse-CDF barriers are ascending as well.  Backward
    can therefore feed them directly to the scan instead of sorting again.
    """

    @staticmethod
    def forward(ctx, x, weight, alpha, h):
        with torch.no_grad():
            log_w, y = prepare_probs_cuda(weight, alpha)
            b, dens = _barrier_forward_cuda(x, log_w, y, h)
        ctx.save_for_backward(x, weight, b, h)
        ctx.dens = dens
        return b

    @staticmethod
    def backward(ctx, grad_b):
        x, weight, b, h = ctx.saved_tensors
        need_x, need_weight, need_alpha, need_h = ctx.needs_input_grad
        d = ctx.dens.clamp_min(1e-30)
        g = (grad_b / d).to(torch.float32)
        xf, bs = x.to(torch.float32), b.to(torch.float32)
        gs = g
        cdf = torch.empty_like(xf)
        pdf = torch.empty_like(xf)
        for bi in range(x.shape[0]):
            y_c, y_p = cdf_pdf_matvec_cuda(xf[bi], bs[bi], gs[bi].unsqueeze(-1), h)
            cdf[bi] = y_c.squeeze(-1)
            pdf[bi] = y_p.squeeze(-1)
        gx = weight * pdf
        grad_x = gx if need_x else None
        # Existing log-weight VJP is -h*w*cdf; composing with log(weight)
        # cancels w and gives the direct probability gradient below.
        grad_weight = (-h * cdf) if need_weight else None
        grad_alpha = grad_b * (h / d) if need_alpha else None
        grad_h = None
        if need_h:
            _num = ((b.double() * grad_b.double()).sum(-1)
                    - (x.double() * gx.double()).sum(-1)).to(x.dtype)
            grad_h = (_num / h.reshape(-1)).reshape(h.shape) if h.dim() >= 1 else (_num / h).sum()
        return grad_x, grad_weight, grad_alpha, grad_h


def lapsum_barriers_cuda(x, alpha_logit, weight_logit=None, h=None, presorted=False):
    """CUDA Laphorn barriers -- drop-in for barriers.lapsum_barriers (scan backend), same
    value+gradient.  x (B,n)|(n,), alpha_logit (B,m)|(m,), weight_logit (B,n)|None (uniform), h req."""
    if h is None:
        raise ValueError("lapsum_barriers_cuda: h (kernel width) is required")
    squeeze = x.dim() == 1
    if squeeze:
        x = x.unsqueeze(0); alpha_logit = alpha_logit.unsqueeze(0)
        if weight_logit is not None:
            weight_logit = weight_logit.unsqueeze(0)
    B, n = x.shape
    log_w = x.new_full((B, n), -math.log(n)) if weight_logit is None else torch.log_softmax(weight_logit, -1)
    alpha = torch.sigmoid(alpha_logit)
    h_t = (h.to(device=x.device, dtype=x.dtype) if torch.is_tensor(h)
           else torch.as_tensor(float(h), device=x.device, dtype=x.dtype))
    b = _BarriersCuda.apply(x, log_w, alpha, h_t, presorted)
    return b.squeeze(0) if squeeze else b


def lapsum_barriers_probs_cuda(x, alpha, weight, h, presorted=False):
    """Fast internal CUDA path for already-normalized probabilities.

    With ``presorted=True`` this directional-only path also assumes ascending
    ``alpha`` (cumulative target probabilities), allowing backward to reuse the
    already ascending inverse-CDF barriers without another sort.
    """
    squeeze = x.dim() == 1
    if squeeze:
        x = x.unsqueeze(0)
        alpha = alpha.unsqueeze(0)
        weight = weight.unsqueeze(0)
    h_t = (h.to(device=x.device, dtype=x.dtype) if torch.is_tensor(h)
           else torch.as_tensor(float(h), device=x.device, dtype=x.dtype))
    if presorted:
        b = _BarriersCudaProbs.apply(x, weight, alpha, h_t)
    else:
        # The probability-native VJP assumes the directional grid is already
        # sorted. Keep the fully general implementation as a fallback.
        log_w = torch.log(weight.clamp_min(1e-30))
        b = _BarriersCuda.apply(x, log_w, alpha, h_t, presorted)
    return b.squeeze(0) if squeeze else b
