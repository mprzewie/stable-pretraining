"""
directional_cuda.py -- CUDA version of the directional-softmax Laphorn coupling (one strip).

Same API and math as directional.py (coupling/apply), but on fused CUDA kernels:
  * directional softmax = torch's fast softmax(dim=1)   (kills the global-softmax bottleneck)
  * ONE weighted barrier solve on the CUDA barrier      (barrier_cuda.lapsum_barriers_cuda, fused fwd + IFT bwd)
  * ONE strip on the CUDA cdf-matvec kernel             (strip_cuda.cdf_matvec_auto, differentiable)
  * diag(s) is a plain elementwise scale
autograd composes the backward from each piece's analytic VJP.  The strip anchors/queries are always
sorted grids/barriers, so the CUDA strip backward is valid.  The kernels JIT-compile on first import.
torch BEFORE numpy.

    coupling_cuda(logits, s, t, h/log_h) -> Pi        forms the n x m matrix
    apply_cuda(logits, X, s, t, h/log_h) -> Y = Pi X  matrix-free

The shared default is log_h = -mean(logits); explicit h and log_h are also supported.
"""
from __future__ import annotations
import torch

from strip_cuda import cdf_matvec_auto                # differentiable CUDA shared-anchor Laplace-CDF matvec
from barrier_cuda import lapsum_barriers_probs_cuda   # CUDA barrier solve (fused fwd + IFT bwd)
from width import resolve_width                       # shared log-width parameterisation
from constants import uniform_marginal, uniform_grid_levels


# strip-mass apply (B^T) / adjoint (B), matrix-free, on the CUDA cdf-matvec (telescoped) -----------
def _stripT_cuda(X, centers, beta, h):
    """StripMass(beta,centers)^T @ X.  X (K,d) -> (nctr,d)."""
    return X[-1:] + cdf_matvec_auto(centers, beta, X[:-1] - X[1:], h)


def _strip_cuda(V, centers, beta, h):
    """StripMass(beta,centers) @ V for signed V.  V (K,d) -> (K,d)."""
    sumV = V.sum(0, keepdim=True)
    gi = sumV - cdf_matvec_auto(beta, centers, V, h)
    z = torch.zeros(1, V.shape[1], dtype=V.dtype, device=V.device)
    g = torch.cat([z, gi, sumV], 0)
    return g[1:] - g[:-1]


def _resolve(logits, s, t, h, h_from_logits, log_h=None):
    if not logits.is_cuda:                                   # the kernels read raw device pointers: a CPU
        raise ValueError(                                    # tensor would be an illegal memory access, not
            "directional_cuda requires CUDA tensors (got a CPU tensor); "  # a clean error -- guard it.
            "move inputs to CUDA, or use directional.py (pure PyTorch) for CPU.")
    n, m = logits.shape
    h = resolve_width(logits, h=h, log_h=log_h, h_from_logits=h_from_logits)
    s_uniform = s is None
    t_uniform = t is None
    if s is None:
        s = uniform_marginal(n, logits)
    if t is None:
        t = uniform_marginal(m, logits)
    return s, t, h, s_uniform, t_uniform


# --- canonical orientation: softmax(dim=1), rows -> diag(a), strip on columns (q-axis) ---
def _core(L, a, b, h, b_uniform=False):
    W = torch.softmax(L, dim=1)                              # (p,q) row-stochastic (fast directional softmax)
    q = L.shape[1]
    if b_uniform:
        gridq, lvl = uniform_grid_levels(q, L)
    else:
        gridq = torch.arange(q, dtype=L.dtype, device=L.device)
        lvl = torch.cumsum(b, 0)[:-1].clamp(1e-9, 1 - 1e-9)
    cmeas = W.t() @ a                                        # (q,) = W^T a
    cmeas = cmeas / cmeas.sum()
    bcol = lapsum_barriers_probs_cuda(gridq, lvl, cmeas, h=h, presorted=True)  # ONE weighted CUDA solve
    return W, gridq, bcol


def _canon_matrix(L, a, b, h, b_uniform=False):
    W, gridq, bcol = _core(L, a, b, h, b_uniform)
    # W B^T = (B W^T)^T. Applying the strip directly to W^T avoids
    # materialising B^T by acting on an identity matrix. This forms the full
    # p x q plan in O(pq), matching the structural pure-PyTorch path.
    WBt = _strip_cuda(W.t(), gridq, bcol, h).t()
    return a.unsqueeze(1) * WBt


def _canon_apply(L, X, a, b, h, b_uniform=False):            # Pi X
    W, gridq, bcol = _core(L, a, b, h, b_uniform)
    Z = _stripT_cuda(X, gridq, bcol, h)                     # B^T X  (q,d)
    return a.unsqueeze(1) * (W @ Z)


def _canon_applyT(L, X, a, b, h, b_uniform=False):           # Pi^T X = B W^T diag(a) X
    W, gridq, bcol = _core(L, a, b, h, b_uniform)
    tmp = W.t() @ (a.unsqueeze(1) * X)                       # W^T diag(a) X  (q,d)
    return _strip_cuda(tmp, gridq, bcol, h)                 # B tmp  (q,d)


# ===========================================================================
# Public API  (drop-in for directional.coupling / directional.apply)
# ===========================================================================
def coupling_cuda(logits, s=None, t=None, h=None, h_from_logits="neg_mean", *, log_h=None):
    """Form the n x m directional coupling Pi in U(s,t) on CUDA kernels."""
    s, t, h, s_uniform, t_uniform = _resolve(logits, s, t, h, h_from_logits, log_h)
    n, m = logits.shape
    if n < m:                                                # strip on the shorter axis (rows)
        return _canon_matrix(logits.t(), t, s, h, s_uniform).t()
    return _canon_matrix(logits, s, t, h, t_uniform)


def apply_cuda(logits, X, s=None, t=None, h=None, h_from_logits="neg_mean", *, log_h=None):
    """Y = Pi @ X, matrix-free, on CUDA kernels.  logits (n,m), X (m,d) -> Y (n,d)."""
    s, t, h, s_uniform, t_uniform = _resolve(logits, s, t, h, h_from_logits, log_h)
    n, m = logits.shape
    if n < m:
        return _canon_applyT(logits.t(), X, t, s, h, s_uniform)
    return _canon_apply(logits, X, s, t, h, t_uniform)
