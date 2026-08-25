"""
strip.py -- matrix-free strip-mass primitives for the directional coupling (pure PyTorch).

The one core op is the shared-anchor Laplace-CDF matvec y[j,:] = sum_i Lap((a_i - q_j)/h) V[i,:]; the
two strip-mass applies telescope over it:  _stripT = StripMass(beta,centers)^T @ X (the adjoint B^T),
_strip = StripMass(beta,centers) @ V (B, for signed V).  Grad-safe, autograd-differentiable.  These are
all directional.py needs from the Laphorn machinery.  torch BEFORE numpy.
"""
from __future__ import annotations
import torch  # before numpy


@torch.compiler.disable
def _searchsorted_1d(a, q):
    """Small eager island for an Inductor View-lowering bug on CUDA."""
    return torch.searchsorted(a.contiguous(), q.contiguous(), right=True)


def _cdf_matvec_shared(q, a, V, h):
    """y[j,:] = sum_i Lap((a_i - q_j)/h) V[i,:].  a (na,) sorted and q (nq,) are 1-D and SHARED across the
    d columns of V (na, d): the searchsorted is done ONCE and the decayed scans batch over d by broadcast.
    Grad-safe (clamp_min(tiny) keeps the log finite; masked sentinels keep the query/width gradient finite)."""
    na = a.shape[0]; tiny = torch.finfo(V.dtype).tiny
    Vt = V.transpose(0, 1)                                       # (d, na): scan the LAST dim (fast/coalesced)
    ah = a / h                                                   # (na,) broadcasts over (d, na)
    rev = lambda z: torch.flip(z, [-1])
    lg = torch.stack([Vt, -Vt], 0).clamp(min=0.0).clamp_min(tiny).log()   # (2,d,na): [pos,neg], grad-safe
    L = torch.logcumsumexp(torch.cat([lg + ah, rev(lg - ah)], 0), -1)     # SL/SR pos/neg in ONE launch
    Ef = torch.exp(L[:2] - ah); SL = Ef[0] - Ef[1]              # decayed prefix (pos - neg)
    Er = torch.exp(rev(L[2:]) + ah); SR = Er[0] - Er[1]         # decayed suffix
    Tsuf = rev(torch.cumsum(rev(Vt), -1))                       # (d, na) suffix sum of V
    d = Vt.shape[0]; z = torch.zeros(d, 1, dtype=V.dtype, device=V.device)
    SL_p = torch.cat([z, SL], -1); SR_s = torch.cat([SR, z], -1); Tsuf0 = torch.cat([Tsuf, z], -1)  # (d,na+1)
    inf = torch.full((1,), float("inf"), dtype=a.dtype, device=a.device)
    a_lop = torch.cat([-inf, a], 0); a_hip = torch.cat([a, inf], 0)
    # A tiny eager island avoids an Inductor CUDA View-lowering failure for
    # common d=64 graphs; all scan-heavy work around it remains compiled.
    lo = _searchsorted_1d(a, q)                                           # (nq,) in [0,na], ONCE
    a_left = torch.where(lo > 0, a_lop[lo], q)                  # grad-safe: dead sentinel -> q  (nq,)
    a_right = torch.where(lo < na, a_hip[lo], q)
    Lq = torch.where(lo > 0, torch.exp((a_left - q) / h) * SL_p[:, lo], torch.zeros_like(SL_p[:, lo]))  # (d,nq)
    Rq = torch.where(lo < na, torch.exp((q - a_right) / h) * SR_s[:, lo], torch.zeros_like(SR_s[:, lo]))
    return (0.5 * Lq + (Tsuf0[:, lo] - 0.5 * Rq)).transpose(0, 1)   # (nq, d)  cdf-kernel matvec


def _stripT(X, centers, beta, h):
    """StripMass(beta, centers)^T @ X, matrix-free.  X (K, d) indexed by strips -> (nctr, d) by centers.
    Telescoping: out[c] = X[-1] + sum_i Lap((beta_i - c)/h) (X[i] - X[i+1])  (query=centers, anchors=beta)."""
    return X[-1:] + _cdf_matvec_shared(centers, beta, X[:-1] - X[1:], h)   # (nctr, d)


def _strip(V, centers, beta, h):
    """StripMass(beta, centers) @ V for SIGNED V, matrix-free WITHOUT log(V) (so the barrier gradient stays
    finite).  strip(V)[i] = g_i - g_{i-1},  g_i = sum_c Lap((beta_i - c)/h) V[c] = sumV - cdf_matvec(beta,c,V)."""
    sumV = V.sum(0, keepdim=True)                               # (1, ncol)
    gi = sumV - _cdf_matvec_shared(beta, centers, V, h)         # (K-1, ncol) interior g_i (shared anchor)
    z = torch.zeros(1, V.shape[1], dtype=V.dtype, device=V.device)
    g = torch.cat([z, gi, sumV], 0)                            # (K+1, ncol): g_0..g_K
    return g[1:] - g[:-1]                                      # (K, ncol) strip masses
