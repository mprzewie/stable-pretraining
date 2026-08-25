"""Directional-softmax LaPHorn coupling in pure PyTorch: one strip, not two.

Instead of a GLOBAL softmax + two strip corrections (rows AND columns), normalise the score matrix
along ONE axis so that marginal is exact by construction, and correct only the OTHER with a single
strip:

    W  = softmax(logits, dim=1)          # row-wise: every row sums to 1  -> row marginal is "clean"
    Pi = diag(s) . W . B^T               # diag(s) fixes rows EXACTLY; B = column strip: W^T s -> t

Because B is column-stochastic (B^T 1 = 1), applying B^T on the right does not change row sums, so the
two corrections don't interfere -> exact U(s,t) in ONE shot with ONE weighted barrier solve.  When n<m
the strip is put on the shorter axis (rows) by transposing.  Two entry points:

    coupling(logits, s, t, h/log_h) -> Pi        forms the n x m transportation matrix
    apply(logits, X, s, t, h/log_h) -> Y = Pi X  matrix-free: Y = diag(s)(W (B^T X))

If neither width is supplied, log_h = -mean(logits). A global logit shift therefore leaves W
unchanged while controlling the strip hardness multiplicatively.

Autograd end to end (the barrier solve carries its own IFT VJP).  Runs on CPU or GPU.  torch BEFORE numpy.
See directional_cuda.py for the fused-kernel version with the same API.
"""
from __future__ import annotations

import torch  # before numpy

try:
    from .barriers import lapsum_barriers_probs
    from .constants import uniform_grid_levels, uniform_marginal
    from .strip import _strip, _stripT
    from .width import resolve_width
except ImportError:  # pragma: no cover - supports the bundled standalone audits
    from barriers import lapsum_barriers_probs
    from constants import uniform_grid_levels, uniform_marginal
    from strip import _strip, _stripT
    from width import resolve_width


def _resolve(logits, s, t, h, h_from_logits, log_h=None):
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
#     L (p,q); a (p,) row marginal; b (q,) column marginal.  Pi = diag(a) . softmax(L,1) . B^T.
def _core(L, a, b, h, b_uniform=False):
    W = torch.softmax(L, dim=1)                           # (p,q) row-stochastic
    q = L.shape[1]
    if b_uniform:
        gridq, lvl = uniform_grid_levels(q, L)
    else:
        gridq = torch.arange(q, dtype=L.dtype, device=L.device)
        lvl = torch.cumsum(b, 0)[:-1].clamp(1e-9, 1 - 1e-9)
    cmeas = W.t() @ a                                     # (q,) column measure of diag(a) W = W^T a
    cmeas = cmeas / cmeas.sum()
    bcol = lapsum_barriers_probs(gridq, lvl, cmeas, h=h, presorted=True)  # ONE weighted solve
    return W, gridq, bcol


def _canon_matrix(L, a, b, h, b_uniform=False):
    W, gridq, bcol = _core(L, a, b, h, b_uniform)
    # Form W B^T without first materialising B.  Since _strip(V,...) = B V,
    # applying it to W^T gives B W^T in O(pq), and transposing yields W B^T.
    WBt = _strip(W.t(), gridq, bcol, h).t()
    return a.unsqueeze(1) * WBt                           # diag(a) W B^T   (p,q)


def _canon_apply(L, X, a, b, h, b_uniform=False):         # Pi X,  X (q,d) -> (p,d)
    W, gridq, bcol = _core(L, a, b, h, b_uniform)
    Z = _stripT(X, gridq, bcol, h)                        # B^T X   (q,d)  matrix-free
    return a.unsqueeze(1) * (W @ Z)                       # diag(a) (W Z)


def _canon_applyT(L, X, a, b, h, b_uniform=False):        # Pi^T X, X (p,d) -> (q,d);  Pi^T = B W^T diag(a)
    W, gridq, bcol = _core(L, a, b, h, b_uniform)
    tmp = W.t() @ (a.unsqueeze(1) * X)                    # W^T diag(a) X   (q,d)
    return _strip(tmp, gridq, bcol, h)                    # B tmp   (q,d)  matrix-free


# ===========================================================================
# Public API
# ===========================================================================
def coupling(logits, s=None, t=None, h=None, h_from_logits="neg_mean", *, log_h=None):
    """Form the n x m directional-softmax coupling Pi in U(s,t).

    logits (n,m); s (n,), t (m,) positive
    distributions with sum(s)=sum(t)=1 (None -> uniform). If neither h nor log_h is supplied, the
    width is parameterised by log_h=-mean(logits). Differentiable. (s, t are not validated.)
    """
    s, t, h, s_uniform, t_uniform = _resolve(logits, s, t, h, h_from_logits, log_h)
    n, m = logits.shape
    if n < m:                                             # put the ONE strip on the shorter axis (rows)
        return _canon_matrix(logits.t(), t, s, h, s_uniform).t()
    return _canon_matrix(logits, s, t, h, t_uniform)


def apply(logits, X, s=None, t=None, h=None, h_from_logits="neg_mean", *, log_h=None):
    """Compute Y = Pi @ X without forming Pi.

    Pi = coupling(logits, s, t, h), logits (n,m), and X (m,d) -> Y (n,d).
    Differentiable in logits, X, h (and s, t).
    """
    s, t, h, s_uniform, t_uniform = _resolve(logits, s, t, h, h_from_logits, log_h)
    n, m = logits.shape
    if n < m:                                             # Pi = _canon_matrix(logits^T,t,s)^T ; Pi X = Pi_canon^T X
        return _canon_applyT(logits.t(), X, t, s, h, s_uniform)
    return _canon_apply(logits, X, s, t, h, t_uniform)
