"""
barriers.py -- weighted quantile "barriers" b_i = F^{-1}(alpha_i) of a kernel-smoothed
weighted Laplace CDF, with a matrix-free full gradient.  This is the differentiable
inverse-CDF primitive Laphorn (laphorn.py) uses to place its marginal-quantile grid.

Given points x_1..x_n with weights w_1..w_n (= softmax of logits), build

    F(b) = sum_j w_j * Lap((b - x_j)/h),     Lap = Laplace(0,1) CDF,

a continuous increasing bijection R -> (0, 1), and return, for target levels
alpha_1..alpha_m (passed as logits for stability), the barriers b_i = F^{-1}(alpha_i),
with a FULL gradient w.r.t. the coordinates x, the weight logits, and the width h.

The barrier Jacobian d b_i / d x_j is a dense (m x n) matrix
    J_{ij} = w_j f((b_i - x_j)/h) / d_i,   d_i = sum_l w_l f((b_i - x_l)/h),  f = Laplace pdf,
which we NEVER materialise: the forward is a closed-form analytical inverse (no Newton,
no iteration) and the backward is the implicit-function-theorem (IFT) VJP -- every
J v / J^T v is a single O((n+m) log) Laplace-kernel prefix scan whose only divisor is
the density d_i > 0, so it is robust even at the symmetric doubly-stochastic point.
Pure PyTorch: runs on CPU or GPU, no CUDA, and is gradcheckable.

Two interchangeable backends (flag `dense`), same value + gradient to ~1e-15:
  * scan  (default)      -- `_Barriers`      : the prefix-scan inverse + scan IFT VJP.
  * dense (`dense=True`) -- `_BarriersDense` : masked-logsumexp inverse + a dense (m x n)
                            IFT VJP; fewer kernel launches at small n.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F

_LOG_HALF = math.log(0.5)
_NEG = float("-inf")


# ===========================================================================
# Fast exp-kernel matvec.  Pure-PyTorch build: the prefix/suffix scan below (``_lr_sums`` over
# ``_decayed_prefix``/``_decayed_suffix``) is what the barrier VJP uses -- O((n+m) log(n+m)),
# matrix-free (no CUDA).  ``laplace_{kernel,cdf}_matvec`` wrap it as a standalone primitive and
# are exercised by ``_self_test``.  Wrap hot code in ``torch.compile`` for speed.
# ===========================================================================

# ===========================================================================
# Laplace CDF (clamped exp; safe in both torch.where branches)
# ===========================================================================
def _lap_cdf(u: torch.Tensor) -> torch.Tensor:
    return torch.where(u <= 0,
                       0.5 * torch.exp(u.clamp(max=0.0)),
                       1.0 - 0.5 * torch.exp((-u).clamp(max=0.0)))


# ===========================================================================
# Stable decayed prefix / suffix sums and the fast Laplace-kernel matvec.
#
#   For sorted anchors a_(1) <= ... <= a_(m) (per row) and values v, define
#     SL_k = sum_{i<=k} v_i e^{(a_i - a_k)/h}      (decayed prefix)
#     SR_k = sum_{i>=k} v_i e^{(a_k - a_i)/h}      (decayed suffix)
#   computed stably via logcumsumexp split by the sign of v.  Then for a query
#   q with `lo` = #{a_i <= q} (searchsorted, right):
#     Lq = sum_{a_i<=q} e^{(a_i-q)/h} v_i = e^{(a_(lo) - q)/h} * SL_(lo)
#     Rq = sum_{a_i> q} e^{(q-a_i)/h} v_i = e^{(q - a_(lo+1))/h} * SR_(lo+1)
#   both prefactors <= 1, so no overflow.  Also a plain suffix sum
#     Tq = sum_{a_i> q} v_i.
# ===========================================================================
def _decayed_prefix(a: torch.Tensor, v: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    """SL_k = sum_{i<=k} v_i e^{(a_i-a_k)/h}, batched on last dim (a ascending)."""
    ah = a / h
    tiny = torch.finfo(v.dtype).tiny
    pos = torch.log(v.clamp(min=0.0).clamp_min(tiny)) + ah      # clamp_min(tiny): finite log (no -inf)
    neg = torch.log((-v).clamp(min=0.0).clamp_min(tiny)) + ah   #   so the query/width gradient stays finite
    sl_pos = torch.exp(torch.logcumsumexp(pos, dim=-1) - ah)
    sl_neg = torch.exp(torch.logcumsumexp(neg, dim=-1) - ah)
    return sl_pos - sl_neg


def _decayed_suffix(a: torch.Tensor, v: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    """SR_k = sum_{i>=k} v_i e^{(a_k-a_i)/h}, batched (a ascending)."""
    rev = lambda t: torch.flip(t, dims=[-1])
    nah = -a / h
    tiny = torch.finfo(v.dtype).tiny
    pos = torch.log(v.clamp(min=0.0).clamp_min(tiny)) + nah     # clamp_min(tiny): finite log (no -inf)
    neg = torch.log((-v).clamp(min=0.0).clamp_min(tiny)) + nah  #   so the query/width gradient stays finite
    sr_pos = torch.exp(rev(torch.logcumsumexp(rev(pos), dim=-1)) - nah)
    sr_neg = torch.exp(rev(torch.logcumsumexp(rev(neg), dim=-1)) - nah)
    return sr_pos - sr_neg


def _lr_sums(q: torch.Tensor, a: torch.Tensor, v: torch.Tensor, h: torch.Tensor):
    """Given queries q (B,nq) and anchors a (B,na) ASCENDING with values v (B,na),
    return per-query (Lq, Rq, Tq):
        Lq = sum_{a_i<=q} e^{(a_i-q)/h} v_i,
        Rq = sum_{a_i> q} e^{(q-a_i)/h} v_i,
        Tq = sum_{a_i> q} v_i.
    All stable (decay prefactors <= 1)."""
    B, na = a.shape
    SL = _decayed_prefix(a, v, h)                    # (B,na)
    SR = _decayed_suffix(a, v, h)                    # (B,na)
    Tsuf = torch.flip(torch.cumsum(torch.flip(v, [-1]), dim=-1), [-1])  # suffix sum of v
    # pad sentinels so index lo-1 / lo are always valid
    zeroB = torch.zeros(B, 1, device=a.device, dtype=a.dtype)
    SL_p = torch.cat([zeroB, SL], dim=-1)            # SL_p[:,k] = SL_(k-1); SL_p[:,0]=0
    NEGv = torch.full((B, 1), _NEG, device=a.device, dtype=a.dtype)
    a_lop = torch.cat([NEGv, a], dim=-1)             # a at lo (1-based) -> a_lop[:,lo]
    SR_s = torch.cat([SR, zeroB], dim=-1)            # SR_s[:,lo] = SR_(lo+1) after shift below
    a_hip = torch.cat([a, torch.full((B, 1), float("inf"), device=a.device, dtype=a.dtype)], dim=-1)
    Tsuf0 = torch.cat([Tsuf, zeroB], dim=-1)

    lo = torch.searchsorted(a, q, right=True)        # (B,nq) in [0,na]
    g = lambda T, idx: torch.gather(T, -1, idx)

    # left: last index <= q is lo-1 (0-based) -> SL_p[:,lo] (=SL_(lo-1)); anchor a_lop[:,lo]
    a_left = g(a_lop, lo)                             # a_(lo)  (1-based) = a_(lo-1) 0-based
    a_left = torch.where(lo > 0, a_left, q)          # dead (lo=0) sentinel is -inf: use q so the masked
    SL_lo = g(SL_p, lo)                              #   exp arg is finite -> gradient w.r.t. q,h is safe
    Lq = torch.where(lo > 0, torch.exp((a_left - q) / h) * SL_lo, torch.zeros_like(q))

    # right: first index > q is lo (0-based) -> SR[:,lo]; anchor a_hip[:,lo]
    a_right = g(a_hip, lo)                            # a_(lo) 0-based (first > q), inf if none
    a_right = torch.where(lo < na, a_right, q)       # dead (lo=na) sentinel is +inf: use q (grad-safe)
    SR_lo = g(SR_s, lo)
    Rq = torch.where(lo < na, torch.exp((q - a_right) / h) * SR_lo, torch.zeros_like(q))

    Tq = g(Tsuf0, lo)
    return Lq, Rq, Tq


def laplace_kernel_matvec(q, a, v, h, *, a_sorted=False):
    """y_j = sum_i e^{-|q_j - a_i|/h} v_i   (the fast scan-based ker(q,a) v).
    q (B,nq), a (B,na), v (B,na).  Returns (B,nq).  Matrix-free, O((n+m)log)."""
    if not a_sorted:
        a, order = a.sort(dim=-1)
        v = torch.gather(v, -1, order)
    Lq, Rq, _ = _lr_sums(q, a, v, h)
    return Lq + Rq


def laplace_cdf_matvec(q, a, v, h, *, a_sorted=False):
    """y_j = sum_i Lap((a_i - q_j)/h) v_i   (CDF-kernel matvec).  Same cost."""
    if not a_sorted:
        a, order = a.sort(dim=-1)
        v = torch.gather(v, -1, order)
    Lq, Rq, Tq = _lr_sums(q, a, v, h)
    # Lap((a_i-q)/h): a_i<=q -> 1/2 e^{(a_i-q)/h};  a_i>q -> 1 - 1/2 e^{(q-a_i)/h}
    return 0.5 * Lq + (Tq - 0.5 * Rq)


# ===========================================================================
# Analytical weighted multi-level inverse of  F(b)=sum_j w_j Lap((b-x_j)/h).
# (log-domain; port of the stable LapFlow / soft_topk_stable inverse, batched
#  over multiple target levels.)
# ===========================================================================
def _states(x_sorted, s, h):
    """a,b,c,d log-scans for sorted coords x and normalised log-weights s
    (logsumexp(s)=0).  The four scans are fused into ONE batched logcumsumexp
    (stack the two prefix and two reversed-suffix arguments) -- fewer kernel launches,
    identical result."""
    xh = x_sorted / h
    rev = lambda t: torch.flip(t, dims=[-1])
    L = torch.logcumsumexp(torch.stack([s + xh, s, rev(s - xh), rev(s)], 0), dim=-1)
    b = -xh + L[0]
    c = L[1]
    a = xh + rev(L[2])
    d = rev(L[3])
    return a, b, c, d


def _node_logits(x_sorted, s, h, a, b_st, c, d):
    """node_logit[i] = logit(F(x_(i))).  Dead (zero-weight) nodes -> -inf."""
    log_half = _LOG_HALF
    gap = (x_sorted[:, :-1] - x_sorted[:, 1:]) / h
    log_f = torch.empty_like(x_sorted)
    log_f[:, 0] = log_half + a[:, 0]
    m = torch.maximum(c[:, :-1], a[:, 1:])
    log_f[:, 1:] = m + torch.log(torch.exp(c[:, :-1] - m)
                                 + 0.5 * torch.exp(a[:, 1:] - m)
                                 - 0.5 * torch.exp(b_st[:, :-1] - m + gap))
    log1mf = torch.empty_like(x_sorted)
    log1mf[:, -1] = log_half + b_st[:, -1]
    m = torch.maximum(d[:, 1:], b_st[:, :-1])
    log1mf[:, :-1] = m + torch.log(torch.exp(d[:, 1:] - m)
                                   + 0.5 * torch.exp(b_st[:, :-1] - m)
                                   - 0.5 * torch.exp(a[:, 1:] - m + gap))
    node_logit = log_f - log1mf
    dead = torch.isneginf(s)
    return torch.where(dead, torch.full_like(node_logit, _NEG), node_logit)


def _stable_G(alpha, beta, xl, xr, h, gamma, delta, z):
    # Root of the interval equation.  Value-identical (~1e-16) to the plain
    #     log_num = Amax + log1p(-exp(Amin-Amax)) = log|e^{delta+z} - e^gamma|,
    # but with no degenerate log(0) = -inf intermediate at the coincident-mass point
    # delta+z == gamma (the symmetric doubly-stochastic case): (i) the mass factor
    # r = 1 - e^{Amin-Amax} is formed via expm1, so nothing reaches log(0); (ii) the max/exponent
    # are oriented by the SAME boolean `hi_branch` that picks the output branch, so if this map is
    # ever differentiated directly the tie yields the ACTIVE branch's one-sided limit (= the true
    # gradient, b being smooth) not an abs'(0)=0 subgradient.  NOTE: nothing in the tree
    # differentiates this map -- every barrier gradient is an IFT VJP (_Barriers / _BarriersDense)
    # and the inverse is called only under no_grad -- so the above is forward -inf hygiene, plus a
    # safe gradient if this map is ever differentiated directly.
    p = delta + z
    hi_branch = p > gamma
    Amax = torch.where(hi_branch, p, gamma)                  # == maximum(delta+z, gamma)
    negdiff = torch.where(hi_branch, gamma - p, p - gamma)   # == Amin - Amax  (<= 0)
    r = -torch.expm1(negdiff)                                # 1 - e^{Amin-Amax} in [0,1]
    log_den = F.softplus(z)
    P = Amax - log_den                                       # log(max-mass / denom); log_w = P+log r
    A = alpha + beta + (xl - xr) / h
    n = torch.maximum(P, A / 2)                              # stabiliser (G is invariant to it)
    u = torch.exp(P - n) * r                                 # = exp(log_w - n), no singular log
    vv = torch.exp(A - 2 * n)
    G = n + torch.log(u + torch.sqrt(u * u + vv))
    return torch.where(hi_branch, xr + h * (G - alpha), xl + h * (beta - G))


def _inverse_multi(y, x_sorted, s, h, return_density=False):
    """Solve logit(F(b))=y_i for each level.  y (B,m); x_sorted,s (B,n).  -> b (B,m).

    With ``return_density`` also returns the weighted Laplace density at each
    barrier, d_i = sum_j w_j f((b_i-x_j)/h), computed for free from the same
    interval state (alpha,beta,xl,xr): the left/right kernel masses are exactly
    the prefix/suffix log-scans already gathered for stable_G, so this is O(m)
    and lets the backward skip a full O(n log n) kernel matvec for d."""
    B, n = x_sorted.shape
    a, b_st, c, d = _states(x_sorted, s, h)
    node_logit = _node_logits(x_sorted, s, h, a, b_st, c, d)            # (B,n)
    margin = 80.0 * h
    NEG = torch.full((B, 1), _NEG, device=x_sorted.device, dtype=x_sorted.dtype)
    lo = x_sorted[:, :1] - margin
    hi = x_sorted[:, -1:] + margin
    a_ext = torch.cat([a, NEG], dim=-1)
    b_ext = torch.cat([NEG, b_st], dim=-1)
    c_ext = torch.cat([NEG, c], dim=-1)
    d_ext = torch.cat([d, NEG], dim=-1)
    r_ext = torch.cat([lo, x_sorted, hi], dim=-1)                  # (B,n+2)
    idx = torch.searchsorted(node_logit, y)                            # (B,m) in [0,n]
    g = lambda T, i: torch.gather(T, -1, i)
    alpha = g(a_ext, idx); beta = g(b_ext, idx)                   # suffix/prefix log-mass
    xl = g(r_ext, idx); xr = g(r_ext, idx + 1)                    # barrier interval [xl,xr)
    b = _stable_G(alpha, beta, xl, xr, h, g(c_ext, idx), g(d_ext, idx), y)
    if not return_density:
        return b
    # left mass (x_j<=b): e^{beta+(xl-b)/h};  right mass (x_j>b): e^{alpha+(b-xr)/h}
    # both exponents <=0 (xl<=b<xr); NEG sentinels give 0 at the two ends.
    dens = 0.5 * (torch.exp(beta + (xl - b) / h) + torch.exp(alpha + (b - xr) / h))
    return b, dens


# ---------------------------------------------------------------------------
# Small-n dense inverse: the four log-scans of _states as a masked logsumexp (no cumulative
# scan -> fewer kernels at small n), then the SAME closed-form bracket + _stable_G root.
# Used as the no_grad FORWARD of _BarriersDense (whose backward is the dense IFT VJP), so this
# map is not differentiated in the shipping paths; output is identical to _inverse_multi to
# ~1e-15.  (It stays autograd-differentiable on its own via the -inf-free _stable_G, but the IFT
# VJP is both faster -- ~1.3-1.5x fwd+bwd -- and robust at the symmetric doubly-stochastic point.)
# ---------------------------------------------------------------------------
def _states_dense(x_sorted, s, h):
    B, n = x_sorted.shape
    xh = x_sorted / h
    ar = torch.arange(n, device=x_sorted.device)
    zero = torch.zeros(n, n, dtype=x_sorted.dtype, device=x_sorted.device)
    pre = zero.masked_fill(ar[None, :] > ar[:, None], float("-inf"))   # 0 if j<=i else -inf
    suf = zero.masked_fill(ar[None, :] < ar[:, None], float("-inf"))   # 0 if j>=i else -inf
    red = lambda v, msk: torch.logsumexp(v.unsqueeze(1) + msk, dim=-1)  # (B,n) over j
    b = -xh + red(s + xh, pre)                                         # = _states b
    c = red(s, pre)                                                    # = _states c
    a = xh + red(s - xh, suf)                                          # = _states a
    d = red(s, suf)                                                    # = _states d
    return a, b, c, d


def _inverse_multi_dense(y, x_sorted, s, h):
    """Exact multi-level inverse, dense _states.  Bracket + _stable_G are the same exact
    algebra as _inverse_multi (no Newton, no iteration).  Supplies the value (no_grad forward of
    _BarriersDense); the shipping gradient is the dense IFT VJP, not autograd through this map."""
    B, n = x_sorted.shape
    a, b_st, c, d = _states_dense(x_sorted, s, h)
    node_logit = _node_logits(x_sorted, s, h, a, b_st, c, d)
    NEG = torch.full((B, 1), _NEG, device=x_sorted.device, dtype=x_sorted.dtype)
    margin = 80.0 * h
    lo = x_sorted[:, :1] - margin
    hi = x_sorted[:, -1:] + margin
    a_ext = torch.cat([a, NEG], dim=-1)
    b_ext = torch.cat([NEG, b_st], dim=-1)
    c_ext = torch.cat([NEG, c], dim=-1)
    d_ext = torch.cat([d, NEG], dim=-1)
    r_ext = torch.cat([lo, x_sorted, hi], dim=-1)
    idx = torch.searchsorted(node_logit, y)
    g = lambda T, i: torch.gather(T, -1, i)
    alpha = g(a_ext, idx); beta = g(b_ext, idx)
    xl = g(r_ext, idx); xr = g(r_ext, idx + 1)
    return _stable_G(alpha, beta, xl, xr, h, g(c_ext, idx), g(d_ext, idx), y)


# ===========================================================================
# Autograd Function: forward = barriers (no_grad);  backward = matrix-free VJP.
# Inputs: x (B,n), log_w (B,n; logsumexp=0), alpha (B,m) in (0,1), h scalar/(B,1).
# ===========================================================================
class _Barriers(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, log_w, alpha, h, presorted=False, levels_sorted=False):
        with torch.no_grad():
            if presorted:                                        # x already ascending (e.g. the
                x_sorted, s_sorted = x, log_w                    # integer pixel grid): skip the
            else:                                                # O(n log n) sort entirely
                x_sorted, order = x.sort(dim=-1)
                s_sorted = torch.gather(log_w, -1, order)
            a = alpha.clamp(1e-7, 1 - 1e-7)
            y = torch.log(a) - torch.log1p(-a)                   # logit(alpha)
            b, dens = _inverse_multi(y, x_sorted, s_sorted, h, return_density=True)
        ctx.save_for_backward(x, log_w, b, h)
        ctx.dens = dens                                          # (B,m) weighted density at barriers
        ctx.levels_sorted = levels_sorted
        return b

    @staticmethod
    def backward(ctx, grad_b):
        x, log_w, b, h = ctx.saved_tensors
        need_x, need_logw, need_alpha = ctx.needs_input_grad[:3]
        need_h = ctx.needs_input_grad[3]
        w = torch.exp(log_w)                                      # (B,n)

        # weighted Laplace density at each barrier d_i = sum_j w_j f((b_i-x_j)/h),
        # handed to us for free by the inverse (from the same interval state).
        d = ctx.dens.clamp_min(1e-30)
        g = grad_b / d                                           # (B,m)

        # gx (pdf kernel) and grad_logw (cdf kernel) are BOTH ker(x,b) matvecs with the SAME
        # (q=x, a=b, v=g): the sort of b and the two logcumsumexp scans (_lr_sums) are identical,
        # so run them ONCE and combine differently -- pdf: Lq+Rq;  cdf: 1/2 Lq + Tq - 1/2 Rq.
        # (Previously two full matvecs whenever need_h -- i.e. the h-from-logits layer default.)
        gx = grad_logw = None
        if need_x or need_h or need_logw:
            if ctx.levels_sorted:
                # The directional probability path supplies ascending cumulative
                # levels, so inverse-CDF barriers are already ascending.
                bs, gs = b, g
            else:
                bs, order = b.sort(dim=-1)                         # general arbitrary-level path
                gs = torch.gather(g, -1, order)
            Lq, Rq, Tq = _lr_sums(x, bs, gs, h)                   # one sort + two scans, reused below
            if need_x or need_h:                                  # gx_j = w_j * 1/2 * ker(x,b) g
                gx = w * (0.5 * (Lq + Rq))
            if need_logw:                                         # grad_(log w)_j = -h w_j * cdfker(x,b) g
                grad_logw = -h * w * (0.5 * Lq + (Tq - 0.5 * Rq))
        grad_x = gx if need_x else None
        # grad_alpha_i = grad_b_i * db_i/dalpha_i = grad_b_i * h / d_i
        grad_alpha = grad_b * (h / d) if need_alpha else None
        # db_i/dh = (1/(h d_i)) sum_j w_j f((b_i-x_j)/h)(b_i-x_j); contract with grad_b ->
        # grad_h = (<b, grad_b> - <x, gx>)/h per instance, reduced to h's shape (scalar | (B,1))
        grad_h = None
        if need_h:
            _num = (b * grad_b).sum(-1) - (x * gx).sum(-1)        # (B,)
            grad_h = (_num / h.reshape(-1)).reshape(h.shape) if (torch.is_tensor(h) and h.dim() >= 1) \
                else (_num / h).sum()

        return grad_x, grad_logw, grad_alpha, grad_h, None, None


# ===========================================================================
# Small-n path: exact analytical inverse (forward, no_grad) + a DENSE implicit-function-theorem
# VJP (backward).  Same gradient as _Barriers, but the three kernel matvecs are formed as explicit
# (m x n) Laplace kernels contracted with a GEMM, instead of the logcumsumexp scan -- fewer kernel
# launches at small n (the reason the dense path exists).  Crucially, unlike plain autograd through
# the closed-form root (_inverse_multi_dense -> _stable_G), this NEVER differentiates the root
# formula, so the coincident-mass 0/0 at the symmetric doubly-stochastic point cannot arise: the
# VJP's only divisor is the density d_i = sum_j w_j f((b_i-x_j)/h), a sum of positives (>0).
# ===========================================================================
class _BarriersDense(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, log_w, alpha, h, presorted=False):
        with torch.no_grad():
            if presorted:                                        # x already ascending (pixel grid)
                x_sorted, s_sorted = x, log_w
            else:
                x_sorted, order = x.sort(dim=-1)
                s_sorted = torch.gather(log_w, -1, order)
            a = alpha.clamp(1e-7, 1 - 1e-7)
            y = torch.log(a) - torch.log1p(-a)                   # logit(alpha)
            b = _inverse_multi_dense(y, x_sorted, s_sorted, h)   # value only (no autograd graph)
        ctx.save_for_backward(x, log_w, b, h)
        return b

    @staticmethod
    def backward(ctx, grad_b):
        x, log_w, b, h = ctx.saved_tensors                       # x (B,n), b (B,m)
        need_x, need_logw, need_alpha, need_h = ctx.needs_input_grad[:4]
        w = torch.exp(log_w)                                      # (B,n)
        h3 = h.unsqueeze(-1) if (torch.is_tensor(h) and h.dim() >= 1) else h   # (B,1,1) | scalar
        u = (b.unsqueeze(-1) - x.unsqueeze(-2)) / h3            # (B,m,n): (b_i - x_j)/h
        Kpdf = torch.exp(-u.abs())                              # e^{-|.|}  (dense Laplace pdf kernel)
        d = (0.5 * (Kpdf * w.unsqueeze(-2)).sum(-1)).clamp_min(1e-30)   # density d_i = 1/2 ker(b,x)w
        g = grad_b / d                                           # (B,m)
        # gx_j = w_j * 1/2 * sum_i Kpdf[i,j] g_i  (reused by grad_x and grad_h)
        gx = (w * (0.5 * (Kpdf * g.unsqueeze(-1)).sum(-2))) if (need_x or need_h) else None
        grad_x = gx if need_x else None
        grad_alpha = grad_b * (h / d) if need_alpha else None    # db/dalpha = h/d
        grad_logw = None
        if need_logw:
            Kcdf = _lap_cdf(u)                                   # Lap((b_i-x_j)/h)  (B,m,n)
            grad_logw = -h * w * (Kcdf * g.unsqueeze(-1)).sum(-2)
        grad_h = None
        if need_h:
            _num = (b * grad_b).sum(-1) - (x * gx).sum(-1)        # (B,)
            grad_h = (_num / h.reshape(-1)).reshape(h.shape) if (torch.is_tensor(h) and h.dim() >= 1) \
                else (_num / h).sum()
        return grad_x, grad_logw, grad_alpha, grad_h, None


# ===========================================================================
# Public API
# ===========================================================================
def lapsum_barriers(x: torch.Tensor,
                    alpha_logit: torch.Tensor,
                    weight_logit: Optional[torch.Tensor] = None,
                    h=None,
                    presorted: bool = False,
                    dense: bool = False) -> torch.Tensor:
    """Barriers b_i with F(b_i)=sigmoid(alpha_logit_i), F the weighted Laplace CDF.

    Args:
        x:            (B,n) or (n,)   point coordinates (the grid / anchors).
        alpha_logit:  (B,m) or (m,)   level logits; level alpha_i = sigmoid(.).
        weight_logit: (B,n) or None   None => uniform 1/n; else w = softmax(.).
        h:            kernel width -- REQUIRED; a python float or a tensor (scalar or
                      (B,1) per instance).  A tensor h keeps the width differentiable (grad_h).
        presorted:    True if x is already ascending (e.g. an integer grid) -> skip the sort.
        dense:        True -> the dense (m x n) IFT-VJP backend (fewer kernels at small n);
                      False (default) -> the O((n+m) log) scan backend.

    Returns:
        b: (B,m) or (m,) barriers, differentiable w.r.t. x, weight_logit, alpha_logit
           and h (full matrix-free gradient).
    """
    if h is None:
        raise ValueError("lapsum_barriers: h (kernel width) is required")
    squeeze = x.dim() == 1
    if squeeze:
        x = x.unsqueeze(0)
        alpha_logit = alpha_logit.unsqueeze(0)
        if weight_logit is not None:
            weight_logit = weight_logit.unsqueeze(0)
    B, n = x.shape

    if weight_logit is None:
        log_w = x.new_full((B, n), -math.log(n))                 # uniform, logsumexp=0
    else:
        log_w = torch.log_softmax(weight_logit, dim=-1)

    alpha = torch.sigmoid(alpha_logit)
    h_t = (h.to(device=x.device, dtype=x.dtype) if torch.is_tensor(h)  # tensor h stays differentiable
           else torch.as_tensor(float(h), device=x.device, dtype=x.dtype))

    if dense:                                                    # small-n: dense analytical inverse
        b = _BarriersDense.apply(x, log_w, alpha, h_t, presorted) # + dense IFT VJP (robust, few kernels)
    else:
        b = _Barriers.apply(x, log_w, alpha, h_t, presorted, False)  # arbitrary level order
    return b.squeeze(0) if squeeze else b


def lapsum_barriers_probs(x: torch.Tensor,
                          alpha: torch.Tensor,
                          weight: torch.Tensor,
                          h,
                          presorted: bool = False) -> torch.Tensor:
    """Fast internal path for callers that already have probabilities.

    ``alpha`` contains target CDF levels and ``weight`` contains positive,
    normalized point masses.  This avoids the redundant logit/sigmoid and
    log/log_softmax round trips in ``lapsum_barriers``. With
    ``presorted=True``, this directional fast path also assumes ascending
    ``alpha`` and skips a redundant barrier sort in backward.
    """
    squeeze = x.dim() == 1
    if squeeze:
        x = x.unsqueeze(0)
        alpha = alpha.unsqueeze(0)
        weight = weight.unsqueeze(0)
    log_w = torch.log(weight.clamp_min(1e-30))
    h_t = (h.to(device=x.device, dtype=x.dtype) if torch.is_tensor(h)
           else torch.as_tensor(float(h), device=x.device, dtype=x.dtype))
    b = _Barriers.apply(x, log_w, alpha, h_t, presorted, presorted)
    return b.squeeze(0) if squeeze else b


# ===========================================================================
# Self-test: matrix-free VJP vs gradcheck, kernel matvec vs dense.  (python barriers.py)
# ===========================================================================
def _self_test():
    torch.manual_seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"lapsum_barriers self-test  device={dev}")

    # ---- (1) fast kernel matvec vs dense ----
    B, n, m = 3, 200, 40
    x = torch.randn(B, n, device=dev, dtype=torch.float64)
    q = torch.randn(B, m, device=dev, dtype=torch.float64)
    v = torch.randn(B, n, device=dev, dtype=torch.float64)
    h = torch.tensor(0.3, device=dev, dtype=torch.float64)
    K = torch.exp(-(q.unsqueeze(-1) - x.unsqueeze(-2)).abs() / h)     # (B,m,n)
    dense = (K * v.unsqueeze(-2)).sum(-1)
    fast = laplace_kernel_matvec(q, x, v, h)
    print(f"  kernel matvec (scan vs dense) max err = {(dense-fast).abs().max():.2e}")
    Kc = _lap_cdf((x.unsqueeze(-2) - q.unsqueeze(-1)) / h)            # Lap((x_i-q_j)/h)
    dense_c = (Kc * v.unsqueeze(-2)).sum(-1)
    fast_c = laplace_cdf_matvec(q, x, v, h)
    print(f"  cdf matvec    max err = {(dense_c-fast_c).abs().max():.2e}")

    # ---- (2) forward: F(b_i) ~ alpha_i (exact mode) ----
    Bx, nn, mm = 2, 300, 16
    x = torch.randn(Bx, nn, device=dev, dtype=torch.float64)
    wl = torch.randn(Bx, nn, device=dev, dtype=torch.float64)
    al = torch.randn(Bx, mm, device=dev, dtype=torch.float64)
    hb = 0.2
    b = lapsum_barriers(x, al, wl, h=hb)
    w = torch.softmax(wl, dim=-1)
    Fb = (w.unsqueeze(-2) * _lap_cdf((b.unsqueeze(-1) - x.unsqueeze(-2)) / hb)).sum(-1)
    print(f"  |F(b)-alpha| max = {(Fb - torch.sigmoid(al)).abs().max():.2e}")

    # ---- (3) gradcheck the matrix-free VJP (exact mode, fp64) ----
    torch.manual_seed(1)
    x = torch.randn(2, 60, device=dev, dtype=torch.float64, requires_grad=True)
    wl = torch.randn(2, 60, device=dev, dtype=torch.float64, requires_grad=True)
    al = torch.randn(2, 8, device=dev, dtype=torch.float64, requires_grad=True)
    ok = torch.autograd.gradcheck(
        lambda xx, ww, aa: lapsum_barriers(xx, aa, ww, h=0.5),
        (x, wl, al), eps=1e-6, atol=1e-4, rtol=1e-3, raise_exception=False)
    print(f"  gradcheck (x, weight_logit, alpha_logit): {'PASS' if ok else 'FAIL'}")


if __name__ == "__main__":
    _self_test()
