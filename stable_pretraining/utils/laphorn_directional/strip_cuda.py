"""
strip_cuda.py -- fused CUDA forward for the Laphorn strip primitive (step 1 of the CUDA operator).

The strip apply Y = Pi @ X = A(W(B^T X)) is built from ONE core kernel: the shared-anchor
Laplace-CDF matvec

    y[j, :] = sum_i  Lap((a_i - q_j)/h)  V[i, :]        Lap = Laplace(0,1) CDF

which is exactly the pure-torch reference ``_cdf_matvec_shared(q, a, V, h)`` in the sibling
``strip.py`` (stripT = X[-1] + cdf_matvec(dX); strip = sumV - cdf_matvec(V), telescoped).

Numerical model -- the DECAYED (anchored) scan.  Unlike the LAPLEX mixture kernel (LayerNorm-bounded
coordinates, plain exp), here the anchors are a *grid* a_i = i divided by h, so (a_i - q_j)/h has a
huge range (e^{~3000} at n=1024, h=0.3) and a plain running exp overflows.  We split the CDF as

    y[j] = 0.5*Lq_j + (Tq_j - 0.5*Rq_j)
    Lq_j = e^{(a[lo-1]-q_j)/h} * SL[lo-1]      SL[k] = sum_{i<=k} e^{(a_i - a_k)/h} V_i   (decayed prefix)
    Rq_j = e^{(q_j-a[lo])/h}   * SR[lo]        SR[k] = sum_{i>=k} e^{(a_k - a_i)/h} V_i   (decayed suffix)
    Tq_j =                       T[lo]         T[k]  = sum_{i>=k} V_i                     (plain suffix)

with lo_j = searchsorted(a, q_j, right=True).  Every stored exponent is <= 0 (bounded), and the two
decayed sums are AFFINE (leaky) scans with a per-step decay rho / sig <= 1:

    SL[k] = rho[k]*SL[k-1] + V[k],  rho[k] = e^{(a_{k-1}-a_k)/h}   (prefix)
    SR[k] = sig[k]*SR[k+1] + V[k],  sig[k] = e^{(a_k-a_{k+1})/h}   (suffix)

The affine scan takes SIGNED V natively -- no log / sign-split (the operator's V = W(B^T X) is signed),
so the kernel is simpler than the log-domain pure-torch path.  All exps are precomputed torch-side
(bounded); the kernel only runs the two affine scans + one plain suffix sum + the boundary gather.

Two kernels: ``cdf_matvec_seq`` (thread-0 sequential scans -- obviously correct, small na, for
isolation) and ``cdf_matvec`` (block-cooperative affine scan -- the fast path).  The kernels
JIT-compile on first import (nvcc + a host C++ compiler required).  torch BEFORE numpy.
"""
from __future__ import annotations
import os
import sys

import torch
from torch.utils.cpp_extension import load_inline

# --- CUDA toolchain.  By default we let torch find the toolkit matching your wheel (it must match your
# GPU's arch -- e.g. Blackwell sm_120 needs CUDA 12.8+).  Only if you EXPORT LAPHORN_CUDA_HOME to a
# toolkit dir do we override CUDA_HOME with it (opt-in, so we never silently pin an incompatible nvcc).
_LEGACY = os.environ.get("LAPHORN_CUDA_HOME")
if _LEGACY and os.path.isdir(_LEGACY):
    os.environ.setdefault("CUDA_HOME", _LEGACY)
    os.environ.setdefault("CUDA_PATH", _LEGACY)
    os.environ["PATH"] = os.path.join(_LEGACY, "bin") + ";" + os.environ.get("PATH", "")
    import torch.utils.cpp_extension as _cext
    _cext.CUDA_HOME = _LEGACY

_NVCC_FLAGS = ["-O3", "--use_fast_math", "--allow-unsupported-compiler",
               "-D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH"]
if sys.platform == "win32":
    _MSVC_1439 = (r"C:\Program Files (x86)\Microsoft Visual Studio\2022"
                  r"\BuildTools\VC\Tools\MSVC\14.39.33519\bin\Hostx64\x64")
    if os.path.isdir(_MSVC_1439):
        _NVCC_FLAGS += ["-ccbin", _MSVC_1439]
    _NVCC_FLAGS += ["-DUSE_CUDA", "-Xcompiler", "/Zc:preprocessor"]
else:
    _NVCC_FLAGS += ["-DUSE_CUDA"]


_CUDA_SRC = r"""
#include <torch/extension.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>

// Skew a length-na smem index by one pad word per 2^skew elements.  The block scans give thread tid
// the contiguous chunk [tid*per, ...), so a warp reads with stride `per`; when per is a multiple of 32
// (na>=2048) that is an up-to-32-way bank conflict (fully serialized at na=8192).  Storing logical i at
// physical i+(i>>skew) breaks the alignment -> conflict-free (skew=5) or ~2-way (skew=6, for the 99KB
// smem cap at na=8192).  Bit-identical to the linear layout; ~1.2-1.9x on the scans.
#define SK(i, skew) ((i) + ((i) >> (skew)))

// ======================================================================
// In-block inclusive SUFFIX scan (plain sum): buf[i] <- sum_{j>=i} buf[j].
// (Copied from the LAPLEX mixture kernel -- used for the plain suffix T.)
//   wsum : shared scratch >= blockDim.x/32 floats.
// ======================================================================
__device__ __forceinline__ void block_suffix_inc(float* buf, int len, float* wsum, int skew) {
    int tid = threadIdx.x, nb = blockDim.x, lane = tid & 31, warp = tid >> 5;
    int nw = (nb + 31) >> 5;
    int per = (len + nb - 1) / nb;
    int lo = tid * per;
    int hi = lo + per; if (hi > len) hi = len;
    float acc = 0.f;
    for (int i = hi - 1; i >= lo; --i) { acc += buf[SK(i, skew)]; buf[SK(i, skew)] = acc; }
    float v = acc;
    for (int off = 1; off < 32; off <<= 1) {
        float t = __shfl_down_sync(0xffffffff, v, off);
        if (lane + off < 32) v += t;
    }
    if (lane == 0) wsum[warp] = v;
    __syncthreads();
    if (warp == 0) {
        float wv = (lane < nw) ? wsum[lane] : 0.f;
        for (int off = 1; off < 32; off <<= 1) {
            float t = __shfl_down_sync(0xffffffff, wv, off);
            if (lane + off < 32) wv += t;
        }
        if (lane < nw) wsum[lane] = wv;
    }
    __syncthreads();
    float warp_excl = (warp + 1 < nw) ? wsum[warp + 1] : 0.f;
    float excl = warp_excl + (v - acc);
    for (int i = lo; i < hi; ++i) buf[SK(i, skew)] += excl;
    __syncthreads();
}

// ======================================================================
// In-block AFFINE (leaky) prefix scan:  out[k] = rho[k]*out[k-1] + V[k].
//   V     : read-only source (na)      rho : per-step decay (na), rho[0] unused
//   out   : destination (na)           sp  : shared scratch >= 2*(#warps) floats
// Two sequential chunk passes + one warp-cooperative affine scan of the
// per-chunk maps M_c(x) = D_c x + A_c (D = prod rho, A = local scan at SL_in=0),
// composed as (M_b o M_a)(x) = (D_b D_a) x + (D_b A_a + A_b).  The chunk's
// incoming SL_in is the A-part of the EXCLUSIVE composed map applied to 0.
// Exclusive is taken by a lane/warp SHIFT of the inclusive scan (never a
// division by D) -- D underflows to 0 for far-apart anchors, which is the
// physically-correct exponential decay, and a shift keeps it finite.
// ======================================================================
__device__ __forceinline__ void affine_prefix(const float* __restrict__ V,
                                               const float* __restrict__ rho,
                                               float* __restrict__ out, int len, float* sp, int skew) {
    int tid = threadIdx.x, nb = blockDim.x, lane = tid & 31, warp = tid >> 5;
    int nw = (nb + 31) >> 5;
    int per = (len + nb - 1) / nb;
    int lo = tid * per;
    int hi = lo + per; if (hi > len) hi = len;

    // pass 1: chunk map (D, A) with SL_in = 0    (V is skewed smem; rho is linear global)
    float D = 1.f, carry = 0.f;
    for (int i = lo; i < hi; ++i) { carry = rho[i] * carry + V[SK(i, skew)]; D *= rho[i]; }
    float A = carry;                                   // local scan value at chunk end

    // warp-inclusive affine scan (combine with EARLIER lanes)
    float d_ = D, a_ = A;
    for (int off = 1; off < 32; off <<= 1) {
        float dt = __shfl_up_sync(0xffffffff, d_, off);
        float at = __shfl_up_sync(0xffffffff, a_, off);
        if (lane >= off) { a_ = d_ * at + a_; d_ = d_ * dt; }   // MY o EARLIER
    }
    if (lane == 31) { sp[warp] = d_; sp[warp + nw] = a_; }      // per-warp total map
    __syncthreads();
    if (warp == 0) {                                           // warp 0 scans the per-warp maps
        float wd = (lane < nw) ? sp[lane] : 1.f;
        float wa = (lane < nw) ? sp[lane + nw] : 0.f;
        for (int off = 1; off < 32; off <<= 1) {
            float dt = __shfl_up_sync(0xffffffff, wd, off);
            float at = __shfl_up_sync(0xffffffff, wa, off);
            if (lane >= off) { wa = wd * at + wa; wd = wd * dt; }
        }
        if (lane < nw) { sp[lane] = wd; sp[lane + nw] = wa; }   // inclusive over warps [0..w]
    }
    __syncthreads();
    // within-warp EXCLUSIVE map = inclusive of (lane-1); identity for lane 0
    float ewd = __shfl_up_sync(0xffffffff, d_, 1);
    float ewa = __shfl_up_sync(0xffffffff, a_, 1);
    if (lane == 0) { ewd = 1.f; ewa = 0.f; }
    // earlier-warps inclusive map (identity for warp 0)
    float pd = (warp > 0) ? sp[warp - 1]      : 1.f;
    float pa = (warp > 0) ? sp[warp - 1 + nw] : 0.f;
    // total exclusive map = (within-warp-excl) o (earlier-warps); apply to 0 -> A-part only
    float SLin = ewd * pa + ewa;
    __syncthreads();                                           // sp reused next call

    // pass 2: real scan with the resolved incoming SL_in
    float c2 = SLin;
    for (int i = lo; i < hi; ++i) { c2 = rho[i] * c2 + V[SK(i, skew)]; out[SK(i, skew)] = c2; }
    __syncthreads();
}

// In-block AFFINE SUFFIX scan: out[k] = sig[k]*out[k+1] + V[k].  Mirror of the
// prefix (down-shuffles, later lanes/warps).  sig[na-1] unused.
__device__ __forceinline__ void affine_suffix(const float* __restrict__ V,
                                              const float* __restrict__ sig,
                                              float* __restrict__ out, int len, float* sp, int skew) {
    int tid = threadIdx.x, nb = blockDim.x, lane = tid & 31, warp = tid >> 5;
    int nw = (nb + 31) >> 5;
    int per = (len + nb - 1) / nb;
    int lo = tid * per;
    int hi = lo + per; if (hi > len) hi = len;

    float D = 1.f, carry = 0.f;
    for (int i = hi - 1; i >= lo; --i) { carry = sig[i] * carry + V[SK(i, skew)]; D *= sig[i]; }
    float A = carry;                                           // local scan value at chunk start

    float d_ = D, a_ = A;
    for (int off = 1; off < 32; off <<= 1) {
        float dt = __shfl_down_sync(0xffffffff, d_, off);
        float at = __shfl_down_sync(0xffffffff, a_, off);
        if (lane + off < 32) { a_ = d_ * at + a_; d_ = d_ * dt; }   // MY o LATER
    }
    if (lane == 0) { sp[warp] = d_; sp[warp + nw] = a_; }
    __syncthreads();
    if (warp == 0) {
        float wd = (lane < nw) ? sp[lane] : 1.f;
        float wa = (lane < nw) ? sp[lane + nw] : 0.f;
        for (int off = 1; off < 32; off <<= 1) {
            float dt = __shfl_down_sync(0xffffffff, wd, off);
            float at = __shfl_down_sync(0xffffffff, wa, off);
            if (lane + off < 32) { wa = wd * at + wa; wd = wd * dt; }
        }
        if (lane < nw) { sp[lane] = wd; sp[lane + nw] = wa; }   // inclusive over warps [w..nw-1]
    }
    __syncthreads();
    float ewd = __shfl_down_sync(0xffffffff, d_, 1);
    float ewa = __shfl_down_sync(0xffffffff, a_, 1);
    if (lane == 31) { ewd = 1.f; ewa = 0.f; }
    float pd = (warp + 1 < nw) ? sp[warp + 1]      : 1.f;
    float pa = (warp + 1 < nw) ? sp[warp + 1 + nw] : 0.f;
    float SRin = ewd * pa + ewa;
    __syncthreads();

    float c2 = SRin;
    for (int i = hi - 1; i >= lo; --i) { c2 = sig[i] * c2 + V[SK(i, skew)]; out[SK(i, skew)] = c2; }
    __syncthreads();
}

// ======================================================================
// FAST forward : grid = (d), block = BLK.  One block per COLUMN of V (the
// anchors/queries/boundaries are shared across columns).  smem: 3*na + BLK.
// ======================================================================
__global__ void cdf_matvec(
    const float* __restrict__ V,      // (na, d)
    const float* __restrict__ rho,    // (na,)
    const float* __restrict__ sig,    // (na,)
    const int*   __restrict__ lo,     // (nq,)
    const float* __restrict__ gL,     // (nq,)
    const float* __restrict__ gR,     // (nq,)
    float* __restrict__ y,            // (nq, d)  CDF matvec  = 0.5 Lq + Tq - 0.5 Rq
    float* __restrict__ yp,           // (nq, d)  PDF matvec  = 0.5 (Lq + Rq)  (backward), if want_pdf
    int want_pdf, int na, int nq, int d, int skew, int pitch)
{
    int col = blockIdx.x, tid = threadIdx.x, nb = blockDim.x;
    extern __shared__ float sm[];
    float* Vc = sm;             // skewed, pitch words  (V column, then overwritten by plain suffix T)
    float* SL = Vc + pitch;     // skewed  decayed prefix
    float* SR = SL + pitch;     // skewed  decayed suffix
    float* sp = SR + pitch;     // linear warp-scan scratch

    for (int i = tid; i < na; i += nb) Vc[SK(i, skew)] = V[(size_t)i * d + col];
    __syncthreads();

    affine_prefix(Vc, rho, SL, na, sp, skew);  // SL[k] = sum_{i<=k} e^{(a_i-a_k)/h} V_i
    affine_suffix(Vc, sig, SR, na, sp, skew);  // SR[k] = sum_{i>=k} e^{(a_k-a_i)/h} V_i
    block_suffix_inc(Vc, na, sp, skew);        // Vc[k] = sum_{i>=k} V_i   (plain suffix T)

    for (int j = tid; j < nq; j += nb) {
        int L = lo[j];
        float Lq = (L > 0)  ? gL[j] * SL[SK(L - 1, skew)] : 0.f;
        float Rq = (L < na) ? gR[j] * SR[SK(L, skew)]     : 0.f;
        float Tq = (L < na) ? Vc[SK(L, skew)]             : 0.f;
        size_t o = (size_t)j * d + col;
        y[o] = 0.5f * Lq + (Tq - 0.5f * Rq);              // CDF matvec (forward)
        if (want_pdf) yp[o] = 0.5f * (Lq + Rq);           // PDF (Laplace-density) matvec (backward)
    }
}

// ======================================================================
// SEQUENTIAL forward (thread 0 does the 3 scans) -- obviously correct,
// for small na, to isolate scan bugs from gather/wrapper bugs.
// ======================================================================
__global__ void cdf_matvec_seq(
    const float* __restrict__ V, const float* __restrict__ rho, const float* __restrict__ sig,
    const int* __restrict__ lo, const float* __restrict__ gL, const float* __restrict__ gR,
    float* __restrict__ y, float* __restrict__ yp, int want_pdf, int na, int nq, int d)
{
    int col = blockIdx.x, tid = threadIdx.x, nb = blockDim.x;
    extern __shared__ float sm[];
    float* Vc = sm; float* SL = Vc + na; float* SR = SL + na;
    for (int i = tid; i < na; i += nb) Vc[i] = V[(size_t)i * d + col];
    __syncthreads();
    if (tid == 0) {
        float c = 0.f;
        for (int i = 0; i < na; ++i)      { c = rho[i] * c + Vc[i]; SL[i] = c; }
        c = 0.f;
        for (int i = na - 1; i >= 0; --i) { c = sig[i] * c + Vc[i]; SR[i] = c; }
        c = 0.f;
        for (int i = na - 1; i >= 0; --i) { c += Vc[i]; Vc[i] = c; }   // plain suffix T (last)
    }
    __syncthreads();
    for (int j = tid; j < nq; j += nb) {
        int L = lo[j];
        float Lq = (L > 0)  ? gL[j] * SL[L - 1] : 0.f;
        float Rq = (L < na) ? gR[j] * SR[L]     : 0.f;
        float Tq = (L < na) ? Vc[L]             : 0.f;
        size_t o = (size_t)j * d + col;
        y[o] = 0.5f * Lq + (Tq - 0.5f * Rq);
        if (want_pdf) yp[o] = 0.5f * (Lq + Rq);
    }
}

// ======================================================================
// FUSED precompute: rho/sig (anchor step-decays) + lo/gL/gR (per-query boundary via binary search),
// all from (q, a, h) in ONE launch -- replaces the ~10 launch-bound torch ops of _precompute.
//   thread i:  if i<na -> rho[i],sig[i];  if i<nq -> lo[i],gL[i],gR[i] (searchsorted a, right=True).
// ======================================================================
__global__ void precompute_k(
    const float* __restrict__ q, const float* __restrict__ a, const float* __restrict__ hptr,
    float* __restrict__ rho, float* __restrict__ sig,
    int* __restrict__ lo, float* __restrict__ gL, float* __restrict__ gR, int na, int nq)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float h = hptr[0];
    if (i < na) {
        rho[i] = (i > 0)      ? expf((a[i - 1] - a[i]) / h) : 1.f;   // rho[0] unused
        sig[i] = (i < na - 1) ? expf((a[i] - a[i + 1]) / h) : 1.f;   // sig[na-1] unused
    }
    if (i < nq) {
        float qq = q[i];
        int loI = 0, hiI = na;                                       // searchsorted(a, qq, right=True)
        while (loI < hiI) { int mid = (loI + hiI) >> 1; if (a[mid] <= qq) loI = mid + 1; else hiI = mid; }
        lo[i] = loI;
        gL[i] = (loI > 0)  ? expf((a[loI - 1] - qq) / h) : 0.f;
        gR[i] = (loI < na) ? expf((qq - a[loI]) / h)     : 0.f;
    }
}

std::vector<torch::Tensor> precompute_fwd(torch::Tensor q, torch::Tensor a, torch::Tensor h)
{
    int na = a.size(0), nq = q.size(0);
    auto fo = a.options();
    auto rho = torch::empty({na}, fo), sig = torch::empty({na}, fo);
    auto gL = torch::empty({nq}, fo), gR = torch::empty({nq}, fo);
    auto lo = torch::empty({nq}, fo.dtype(torch::kInt32));
    int total = (na > nq ? na : nq), blk = 256, grid = (total + blk - 1) / blk;
    precompute_k<<<grid, blk>>>(q.data_ptr<float>(), a.data_ptr<float>(), h.data_ptr<float>(),
        rho.data_ptr<float>(), sig.data_ptr<float>(), lo.data_ptr<int>(),
        gL.data_ptr<float>(), gR.data_ptr<float>(), na, nq);
    C10_CUDA_CHECK(cudaGetLastError());
    return {rho, sig, lo, gL, gR};
}

// ----------------------------------------------------------------------
static int BLK = 256;
// PER-KERNEL highwater: the MaxDynamicSharedMemorySize attribute is set per CUDA
// function, so cdf_matvec and cdf_matvec_seq each need their own opt-in (a shared
// highwater would skip the second kernel's opt-in and launch it over the 48KB
// default cap -> cudaErrorInvalidValue).
static void optin_once(const void* fn, int* cur, int smem_bytes) {
    if (smem_bytes > 48 * 1024 && smem_bytes > *cur) {
        cudaError_t st = cudaFuncSetAttribute((void*)fn,
            cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
        TORCH_CHECK(st == cudaSuccess, "cudaFuncSetAttribute(", smem_bytes,
                    " B smem) failed: ", cudaGetErrorString(st),
                    " -- device sharedMemPerBlockOptin too small; needs the tiled kernel");
        *cur = smem_bytes;
    }
}

std::vector<torch::Tensor> cdf_matvec_fwd(
    torch::Tensor V, torch::Tensor rho, torch::Tensor sig,
    torch::Tensor lo, torch::Tensor gL, torch::Tensor gR, int64_t seq, int64_t want_pdf)
{
    int na = V.size(0), d = V.size(1), nq = lo.size(0);
    auto y = torch::empty({nq, d}, V.options());
    auto yp = want_pdf ? torch::empty({nq, d}, V.options()) : torch::empty({0}, V.options());
    float* ypp = want_pdf ? yp.data_ptr<float>() : y.data_ptr<float>();   // dummy (unwritten) when 0
    static int g_smem_fast = 0, g_smem_seq = 0;
    if (seq) {                                        // oracle: linear smem, unchanged
        int smem = (3 * na + BLK) * (int)sizeof(float);
        optin_once((const void*)cdf_matvec_seq, &g_smem_seq, smem);
        cdf_matvec_seq<<<d, BLK, smem>>>(
            V.data_ptr<float>(), rho.data_ptr<float>(), sig.data_ptr<float>(),
            lo.data_ptr<int>(), gL.data_ptr<float>(), gR.data_ptr<float>(),
            y.data_ptr<float>(), ypp, (int)want_pdf, na, nq, d);
    } else {                                          // fast: skewed smem (bank-conflict-free scans)
        // smallest pad shift whose 3 skewed arrays + scratch fit the ~99KB opt-in (25344 floats), margin left
        int SP = 32;                                  // warp-scan scratch (needs only 2*BLK/32=16)
        int skew = 5, pitch = na + (na >> skew) + 1;
        while (3 * pitch + SP > 25088 && skew < 12) { skew++; pitch = na + (na >> skew) + 1; }
        int smem = (3 * pitch + SP) * (int)sizeof(float);
        optin_once((const void*)cdf_matvec, &g_smem_fast, smem);
        cdf_matvec<<<d, BLK, smem>>>(
            V.data_ptr<float>(), rho.data_ptr<float>(), sig.data_ptr<float>(),
            lo.data_ptr<int>(), gL.data_ptr<float>(), gR.data_ptr<float>(),
            y.data_ptr<float>(), ypp, (int)want_pdf, na, nq, d, skew, pitch);
    }
    C10_CUDA_CHECK(cudaGetLastError());   // surface a bad launch (e.g. smem over the limit) HERE
    return {y, yp};
}

int64_t smem_optin_limit() {
    int dev = 0; cudaGetDevice(&dev);
    cudaDeviceProp p; cudaGetDeviceProperties(&p, dev);
    return (int64_t)p.sharedMemPerBlockOptin;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("cdf_matvec_fwd", &cdf_matvec_fwd, "Laphorn strip CDF-matvec forward (decayed affine scan)");
    m.def("precompute_fwd", &precompute_fwd, "fused rho/sig + lo/gL/gR from (q,a,h) in one launch");
    m.def("smem_optin_limit", &smem_optin_limit, "device sharedMemPerBlockOptin (bytes)");
}
"""


print("Compiling laphorn_strip extension...", end=" ", flush=True)
_ext = load_inline(name="laphorn_strip_ext", cpp_sources=[""], cuda_sources=[_CUDA_SRC],
                   extra_cuda_cflags=_NVCC_FLAGS, verbose=False)
print("OK")


def _precompute(q, a, h):
    """Torch-side precompute (all bounded exps): searchsorted boundary + step decays + gather exps.
    a (na,) sorted, q (nq,), h scalar tensor/float.  Returns fp32 contiguous kernel inputs."""
    na = a.shape[0]
    a = a.to(torch.float32)
    q = q.to(torch.float32)
    hf = float(h)
    ed = torch.exp(-(a[1:] - a[:-1]) / hf)                       # (na-1,) = e^{(a_{k-1}-a_k)/h}
    rho = torch.ones(na, dtype=torch.float32, device=a.device); rho[1:] = ed
    sig = torch.ones(na, dtype=torch.float32, device=a.device); sig[:-1] = ed
    lo = torch.searchsorted(a.contiguous(), q.contiguous(), right=True).to(torch.int32)  # (nq,) in [0,na]
    loc = lo.long()
    a_lm1 = a[(loc - 1).clamp(min=0)]                            # a[lo-1]
    a_l = a[loc.clamp(max=na - 1)]                               # a[lo]
    z = torch.zeros_like(q)
    gL = torch.where(loc > 0, torch.exp((a_lm1 - q) / hf), z)
    gR = torch.where(loc < na, torch.exp((q - a_l) / hf), z)
    return rho.contiguous(), sig.contiguous(), lo.contiguous(), gL.contiguous(), gR.contiguous()


def _precompute_cuda(q, a, h):
    """Same values as _precompute, computed in ONE fused CUDA kernel (rho/sig + binary-search lo/gL/gR)
    -- replaces the ~10 launch-bound torch ops that dominated the strip call.  a must be sorted."""
    qf = q.to(torch.float32).contiguous()
    af = a.to(torch.float32).contiguous()
    hf = (h.to(device=q.device, dtype=torch.float32) if torch.is_tensor(h)
          else torch.as_tensor(float(h), device=q.device, dtype=torch.float32)).contiguous()
    return _ext.precompute_fwd(qf, af, hf)                  # (rho, sig, lo, gL, gR)


def cdf_matvec_cuda(q, a, V, h, seq=False):
    """CUDA forward of y[j,:] = sum_i Lap((a_i - q_j)/h) V[i,:].  Matches strip._cdf_matvec_shared.
    q (nq,), a (na,) sorted, V (na, d), h scalar.  seq=True uses the sequential reference kernel."""
    rho, sig, lo, gL, gR = _precompute_cuda(q, a, h)
    Vc = V.to(torch.float32).contiguous()
    return _ext.cdf_matvec_fwd(Vc, rho, sig, lo, gL, gR, 1 if seq else 0, 0)[0]


def cdf_pdf_matvec_cuda(q, a, V, h, seq=False):
    """Both matvecs in one kernel: CDF y[j] = sum_i Lap((a_i-q_j)/h) V_i  AND
    PDF p[j] = sum_i 0.5 e^{-|a_i-q_j|/h} V_i (Laplace density).  Returns (y, p)."""
    rho, sig, lo, gL, gR = _precompute_cuda(q, a, h)
    Vc = V.to(torch.float32).contiguous()
    y, p = _ext.cdf_matvec_fwd(Vc, rho, sig, lo, gL, gR, 1 if seq else 0, 1)
    return y, p


class _CdfMatvec(torch.autograd.Function):
    """Differentiable CUDA cdf-matvec: y = K V, K_ji = Lap((a_i - q_j)/h).  a AND q sorted (the operator
    always feeds sorted grids/barriers).  The whole VJP reuses the forward kernel:
      grad_V = (sum_j gy_j) - Kswap^T gy          [swapped-role cdf-matvec, complement of Lap]
      grad_q = -(1/h) rowdot(gy, P_V)             P_V   = density matvec (fwd direction)
      grad_a =  (1/h) rowdot(V , P_gy)            P_gy  = density matvec (swapped roles)
      grad_h = -(1/h)(sum_i a_i grad_a_i + sum_j q_j grad_q_j)
    (density matvec = 0.5(Lq+Rq), same decayed scans as the cdf, one extra gather -- see cdf_pdf_matvec_cuda)."""

    @staticmethod
    def forward(ctx, q, a, V, h):
        h_t = (h if torch.is_tensor(h)
               else torch.as_tensor(float(h), device=q.device, dtype=q.dtype))
        rho, sig, lo, gL, gR = _precompute_cuda(q, a, h_t)
        y = _ext.cdf_matvec_fwd(V.to(torch.float32).contiguous(), rho, sig, lo, gL, gR, 0, 0)[0]
        ctx.save_for_backward(q, a, V, h_t)
        return y

    @staticmethod
    def backward(ctx, gy):
        q, a, V, h = ctx.saved_tensors
        gy = gy.to(torch.float32).contiguous()
        Vf = V.to(torch.float32).contiguous()
        af, qf = a.to(torch.float32), q.to(torch.float32)
        # forward-direction density P_V = sum_i p((a_i-q_j)/h) V_i
        rho, sig, lo, gL, gR = _precompute_cuda(q, a, h)
        _, P_V = _ext.cdf_matvec_fwd(Vf, rho, sig, lo, gL, gR, 0, 1)
        # swapped call (anchor=q sorted, query=a, values=gy) -> G_cdf (for grad_V), P_gy (for grad_a)
        rho2, sig2, lo2, gL2, gR2 = _precompute_cuda(a, q, h)
        G_cdf, P_gy = _ext.cdf_matvec_fwd(gy, rho2, sig2, lo2, gL2, gR2, 0, 1)
        grad_V = gy.sum(0, keepdim=True) - G_cdf                    # (na, d)
        inv_h = 1.0 / h
        grad_q = -inv_h * (gy * P_V).sum(1)                         # (nq,)
        grad_a = inv_h * (Vf * P_gy).sum(1)                         # (na,)
        # grad_h is a scalar reduction over q,a; with grid coords 0..n (large) the fp32 sum cancels
        # badly -> accumulate it in fp64 (negligible cost, a length-(n+m) reduction).
        grad_h = (-inv_h * ((af.double() * grad_a.double()).sum()
                            + (qf.double() * grad_q.double()).sum())).to(torch.float32)
        ni = ctx.needs_input_grad
        return (grad_q.to(q.dtype) if ni[0] else None,
                grad_a.to(a.dtype) if ni[1] else None,
                grad_V.to(V.dtype) if ni[2] else None,
                (grad_h.to(h.dtype).reshape(h.shape) if (h is not None and ni[3]) else None))


def cdf_matvec_auto(q, a, V, h):
    """Differentiable cdf-matvec (CUDA fwd + analytic CUDA bwd).  h must be a tensor to receive its grad."""
    return _CdfMatvec.apply(q, a, V, h)


# --- dense fp64 oracles (validation only; O(na*nq*d)) --------------------------------
def dense_cdf(q, a, V, h):
    """y[j,:] = sum_i Lap((a_i - q_j)/h) V[i,:], formed densely (fp64 ground truth for grads)."""
    u = (a[None, :] - q[:, None]) / h                              # (nq, na)
    K = torch.where(u <= 0, 0.5 * torch.exp(u.clamp(max=0)), 1.0 - 0.5 * torch.exp((-u).clamp(max=0)))
    return K @ V


def dense_pdf(q, a, V, h):
    """p[j,:] = sum_i 0.5 e^{-|a_i-q_j|/h} V[i,:], formed densely (fp64 ground truth)."""
    u = (a[None, :] - q[:, None]) / h
    return (0.5 * torch.exp(-u.abs())) @ V
