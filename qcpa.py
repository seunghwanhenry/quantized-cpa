"""Minimum-variance (Harris) index under coarse output quantization.

Standard index, Sheppard-corrected index, lag-wise polychoric index, hybrid rule,
and the closed-form ceiling (Corollary 1) of the accompanying paper.
"""
import numpy as np
from scipy import optimize
from scipy.linalg import toeplitz
from scipy.stats import norm

_GLX, _GLW = np.polynomial.legendre.leggauss(48)

def quantize(y, q, offset=0.0):
    """Round y to a grid of step q shifted by `offset`."""
    return np.round((y + offset) / q) * q - offset

def acf(y, p):
    """Sample autocorrelation r_0..r_p."""
    y = np.asarray(y, float) - np.mean(y); v = y.var()
    return np.array([1.0] + [np.mean(y[:-k] * y[k:]) / v for k in range(1, p + 1)])

def pd_toeplitz(r, eps=1e-4, iters=50):
    """Nearest positive-definite Toeplitz correlation sequence (alternating projections)."""
    r = np.asarray(r, float).copy(); p = len(r)
    for _ in range(iters):
        w, V = np.linalg.eigh(toeplitz(r))
        if w.min() >= eps: break
        T = (V * np.maximum(w, eps)) @ V.T
        r = np.array([np.mean(np.diag(T, k)) for k in range(p)]); r /= r[0]
    return r

def harris_from_r(r, d):
    """Harris index from an autocorrelation sequence r_0..r_p via Yule-Walker AR(p)."""
    r = pd_toeplitz(r); p = len(r) - 1
    phi = np.linalg.solve(toeplitz(r[:p]), r[1:])
    s2 = 1.0 - phi @ r[1:]
    psi = np.zeros(d); psi[0] = 1.0
    for i in range(1, d):
        psi[i] = sum(phi[k] * psi[i - k - 1] for k in range(min(i, p)))
    return float(s2 * np.sum(psi ** 2))

def discretized_normal_ml(yq, q):
    """ML estimate of latent mean and s.d. from quantized values (bin width q)."""
    def nll(par):
        mu, ls = par; s = np.exp(ls)
        pr = norm.cdf((yq + q / 2 - mu) / s) - norm.cdf((yq - q / 2 - mu) / s)
        return -np.sum(np.log(np.maximum(pr, 1e-300)))
    o = optimize.minimize(nll, [np.mean(yq), np.log(max(np.std(yq), q / 10))], method='Nelder-Mead',
                          options=dict(xatol=1e-6, fatol=1e-6))
    return o.x[0], float(np.exp(o.x[1]))

def _rect_prob(lo1, hi1, lo2, hi2, rho):
    a = np.clip(lo1, -8, 8); b = np.clip(hi1, -8, 8)
    x = (b - a)[:, None] / 2 * _GLX[None, :] + (b + a)[:, None] / 2
    s = np.sqrt(max(1 - rho * rho, 1e-12))
    inner = norm.cdf((hi2[:, None] - rho * x) / s) - norm.cdf((lo2[:, None] - rho * x) / s)
    return ((b - a) / 2) * np.sum(_GLW[None, :] * norm.pdf(x) * inner, axis=1)

def polychoric_lag(yq, k, q, mu, s):
    """Lag-k latent correlation by bivariate discretized-normal maximum likelihood."""
    base = yq.min(); i = np.round((yq - base) / q).astype(int); lmax = i.max()
    pairs, counts = np.unique(np.stack([i[:-k], i[k:]]), axis=1, return_counts=True)
    lo = lambda j: (base + j * q - q / 2 - mu) / s
    hi = lambda j: (base + j * q + q / 2 - mu) / s
    lo1, hi1, lo2, hi2 = lo(pairs[0]), hi(pairs[0]), lo(pairs[1]), hi(pairs[1])
    lo1[pairs[0] == 0] = -np.inf; hi1[pairs[0] == lmax] = np.inf
    lo2[pairs[1] == 0] = -np.inf; hi2[pairs[1] == lmax] = np.inf
    f = lambda r: -np.sum(counts * np.log(np.maximum(_rect_prob(lo1, hi1, lo2, hi2, r), 1e-300)))
    return optimize.minimize_scalar(f, bounds=(-0.99999, 0.99999), method='bounded',
                                    options=dict(xatol=1e-5)).x

def ar_order(n, rho, pmin=5, pmax=20):
    """Order rule of the paper: p = clip(n / (150 rho), pmin, pmax).
    Each lag of the latent autocorrelation costs information that shrinks with rho;
    a larger order on a short or coarsely quantized record over-fits the noise of the
    polychoric estimates and biases the index downwards (Section 5.3)."""
    return int(np.clip(np.floor(n / (150.0 * max(rho, 1.0))), pmin, pmax))

def standard_index(yq, d, p=20):
    return harris_from_r(acf(yq, p), d)

def sheppard_index(yq, q, d, p=20):
    y = yq - yq.mean()
    g = np.array([y.var()] + [np.mean(y[:-k] * y[k:]) for k in range(1, p + 1)])
    g0 = g[0] - q * q / 12
    if g0 <= 0: return np.nan
    g[0] = g0
    return harris_from_r(g / g0, d)

def polychoric_index(yq, q, d, p=None):
    """Returns (index, estimated resolution ratio rho). p=None applies the order rule."""
    mu, s = discretized_normal_ml(yq, q); rho = q / s
    if p is None: p = ar_order(len(yq), rho)
    r = np.array([1.0] + [polychoric_lag(yq, k, q, mu, s) for k in range(1, p + 1)])
    return harris_from_r(r, d), rho

def corrected_index(yq, q, d, p=None):
    """Algorithm 1: Sheppard for rho<=1, polychoric for 1<rho<=3 (caution up to 4), NaN beyond."""
    mu, s = discretized_normal_ml(yq, q); rho = q / s
    if p is None: p = ar_order(len(yq), rho)
    if rho <= 1: return sheppard_index(yq, q, d, p), rho
    if rho <= 4: return polychoric_index(yq, q, d, p)[0], rho
    return np.nan, rho

def ceiling(eta_q, rho):
    """Corollary 1: upper bound on the true index (reliable for rho <~ 1 on industrial data)."""
    return (1 + rho ** 2 / 12) * eta_q - rho ** 2 / 12

# ======================================================================
# Revision: open-ended marginal, profile interval, minimum-distance AR fit
# ======================================================================
def codes_and_counts(yq, q):
    """Occupied codes (bin centres) and their counts; the grid is anchored at the smallest logged value."""
    yq = np.asarray(yq, float); base = yq.min()
    k, n = np.unique(np.round((yq - base) / q).astype(int), return_counts=True)
    return base + k * q, n

def marginal_ml_open(yq, q, profile=False):
    """Discretized-normal ML with the two extreme occupied codes open-ended.
    Returns mu, sigma and (optionally) the 95% profile-likelihood interval of sigma.
    With only two occupied codes sigma is not identified (interval is unbounded)."""
    c, n = codes_and_counts(yq, q)
    lo = c - q / 2; hi = c + q / 2; lo[0] = -np.inf; hi[-1] = np.inf
    def nll(mu, s):
        return -np.sum(n * np.log(np.maximum(norm.cdf((hi - mu) / s) - norm.cdf((lo - mu) / s), 1e-300)))
    lo_mu, hi_mu = c.min() - 10 * q, c.max() + 10 * q
    def prof(s): return optimize.minimize_scalar(lambda m: nll(m, s), bounds=(lo_mu, hi_mu), method='bounded').fun
    if not profile:
        y = np.asarray(yq, float); s0 = max(y.std(), q / 4)
        o = optimize.minimize(lambda z: nll(z[0], np.exp(z[1])), [y.mean(), np.log(s0)], method='Nelder-Mead',
                              options=dict(xatol=1e-7, fatol=1e-7, maxiter=2000))
        return o.x[0], float(np.exp(o.x[1]))
    grid = np.exp(np.linspace(np.log(q / 50), np.log(q * 10), 140)); pv = np.array([prof(s) for s in grid])
    i = int(np.argmin(pv)); s_hat = grid[i]
    s_hat = optimize.minimize_scalar(prof, bounds=(grid[max(i - 1, 0)], grid[min(i + 1, len(grid) - 1)]), method='bounded').x
    mu = optimize.minimize_scalar(lambda m: nll(m, s_hat), bounds=(lo_mu, hi_mu), method='bounded').x
    if not profile: return mu, s_hat
    if len(c) < 3: return mu, s_hat, (0.0, np.inf), len(c)
    target = prof(s_hat) + 1.92; g = lambda ls: prof(np.exp(ls)) - target
    from scipy.optimize import brentq
    ls = np.log(s_hat)
    try: lo_s = np.exp(brentq(g, np.log(grid[0]), ls)) if g(np.log(grid[0])) > 0 else 0.0
    except ValueError: lo_s = 0.0
    try: hi_s = np.exp(brentq(g, ls, np.log(grid[-1]))) if g(np.log(grid[-1])) > 0 else np.inf
    except ValueError: hi_s = np.inf
    return mu, s_hat, (lo_s, hi_s), len(c)

def polychoric_acf(yq, q, K, mu, s):
    return np.array([1.0] + [polychoric_lag(yq, k, q, mu, s) for k in range(1, K + 1)])

def _kappa_to(kap, K):
    p = len(kap); r = np.zeros(K + 1); r[0] = 1.0; phi = np.zeros(0)
    for k in range(1, p + 1):
        kk = kap[k - 1]
        r[k] = (phi @ r[k - 1:0:-1] if k > 1 else 0.0) + kk * (1 - (phi @ r[1:k] if k > 1 else 0.0))
        phi = np.r_[phi - kk * phi[::-1], kk]
    for k in range(p + 1, K + 1): r[k] = phi @ r[k - 1:k - 1 - p:-1]
    v = np.prod(1 - np.asarray(kap) ** 2)
    return phi, r, v

def ar_min_distance(rhat, p):
    """Stationary AR(p) whose autocorrelation is closest (least squares over lags 1..K) to rhat.
    Parameterized by partial autocorrelations in (-1,1): the result is always a valid autocorrelation."""
    rhat = np.asarray(rhat, float); K = len(rhat) - 1
    kap0 = np.zeros(p); phi = np.zeros(0); r = np.clip(rhat, -0.999, 0.999)
    for k in range(1, p + 1):
        den = 1 - (phi @ r[1:k] if k > 1 else 0.0)
        kk = np.clip(((r[k] - (phi @ r[k - 1:0:-1] if k > 1 else 0.0)) / den) if abs(den) > 1e-6 else 0.0, -0.98, 0.98)
        kap0[k - 1] = kk; phi = np.r_[phi - kk * phi[::-1], kk]
    f = lambda z: np.sum((_kappa_to(np.tanh(z), K)[1][1:] - rhat[1:]) ** 2)
    o = optimize.minimize(f, np.arctanh(kap0), method='L-BFGS-B', options=dict(maxiter=500))
    return _kappa_to(np.tanh(o.x), K)

def harris_ar(phi, v, d):
    p = len(phi); psi = np.zeros(d); psi[0] = 1.0
    for i in range(1, d): psi[i] = sum(phi[j] * psi[i - j - 1] for j in range(min(i, p)))
    return float(v * np.sum(psi ** 2))

def ar_order2(n, rho, pmin=5, pmax=20):
    """Order rule; rho=None (not identified) is treated as the least favourable admissible value 4."""
    rho = 4.0 if rho is None or not np.isfinite(rho) else rho
    return int(np.clip(np.floor(n / (150.0 * max(rho, 1.0))), pmin, pmax))

def corrected_index_v2(yq, q, d, p=None, K=20, switch_min=0.10, order_tol=0.10):
    """Algorithm 1 of the paper. Returns a dict with the reported index (None when flagged),
    the pair (corrected, standard), the latent autocorrelation and the diagnostics."""
    yq = np.asarray(yq, float)
    codes, _ = codes_and_counts(yq, q)
    if len(codes) < 2:
        return dict(eta=None, flags=['record constant: no assessment'])
    mu, s, (slo, shi), ncodes = marginal_ml_open(yq, q, profile=True)
    rho = q / s if ncodes >= 3 else np.nan
    switch = float(np.mean(np.diff(np.round((yq - yq.min()) / q)) != 0))
    if p is None: p = ar_order2(len(yq), rho)
    Kfit = max(K, p)
    if np.isfinite(rho) and rho <= 1:
        y = yq - yq.mean(); g = np.array([y.var()] + [np.mean(y[:-k] * y[k:]) for k in range(1, Kfit + 1)])
        g[0] = max(g[0] - q * q / 12, 1e-12); rhat = g / g[0]; est = 'Sheppard'
    else:
        rhat = polychoric_acf(yq, q, Kfit, mu, s); est = 'polychoric'
    def idx(r, pp):
        phi, _, v = ar_min_distance(r, pp); return harris_ar(phi, v, d)
    eta = idx(rhat, p); eta_half = idx(rhat, max(2, p // 2)); eta_std = idx(acf(yq, Kfit), p)
    flags = []
    if switch < switch_min: flags.append('low code switching')
    if abs(eta - eta_half) > order_tol: flags.append('order-unstable fit')
    return dict(eta=None if flags else eta, pair=(eta, eta_std), eta_half=eta_half, flags=flags, estimator=est,
                rho=rho, rho_profile_interval=(q / shi if np.isfinite(shi) and shi > 0 else 0.0, q / slo if slo > 0 else np.inf),
                codes=ncodes, switch_rate=switch, p=p, latent_acf=rhat)

def ar_fit_robust(rhat, p, starts=8, seed=0, rel_tol=0.25, maxiter=300):
    """Constrained AR(p) fit with the default start plus `starts` random starts.
    Returns the best fit and the range of indices (d supplied later) over near-optimal solutions,
    i.e. those whose objective is within rel_tol of the best (plus 1e-5)."""
    rhat = np.asarray(rhat, float); K = len(rhat) - 1
    f = lambda z: np.sum((_kappa_to(np.tanh(z), K)[1][1:] - rhat[1:]) ** 2)
    sols = []
    phi0, r0, v0 = ar_min_distance(rhat, p)
    kap = np.zeros(p); ph = np.zeros(0)
    # recover kappa of default solution via step-down from phi0
    a = phi0.copy()
    for k in range(p, 0, -1):
        kk = a[k - 1]; kap[k - 1] = kk
        if k > 1: a = (a[:k - 1] + kk * a[:k - 1][::-1]) / (1 - kk * kk)
    sols.append((f(np.arctanh(np.clip(kap, -0.999999, 0.999999))), kap))
    rng = np.random.default_rng(seed)
    for _ in range(starts):
        o = optimize.minimize(f, np.arctanh(rng.uniform(-0.9, 0.9, p)), method='L-BFGS-B', options=dict(maxiter=maxiter))
        sols.append((o.fun, np.tanh(o.x)))
    sols = [(fv if np.isfinite(fv) else np.inf, k) for fv, k in sols]
    best = min(sols, key=lambda t: t[0])
    near = [k for fv, k in sols if fv <= best[0] * (1 + rel_tol) + 1e-5] or [best[1]]
    return best[1], near

def harris_from_kappa(kap, d):
    phi, _, v = _kappa_to(np.asarray(kap), len(kap))
    return harris_ar(phi, v, d)
