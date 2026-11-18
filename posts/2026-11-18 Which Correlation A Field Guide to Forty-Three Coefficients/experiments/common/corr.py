"""A small zoo of association measures, all implemented on top of NumPy/SciPy.

Every function takes two 1-D arrays (or two blocks, for RV/CCA) and returns a
scalar.  Nothing here is a wrapper around a specialist package, so the
definitions in the article and the code in this file are the same thing.

Conventions
-----------
* `x`, `y` are float arrays of length n (except RV/CCA, which take n x p).
* Ordinal inputs are integer codes 0..k-1.
* Methods whose null hypothesis is independence are marked in the docstring.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

EPS = 1e-12


# ---------------------------------------------------------------- linear / moment

def pearson(x, y) -> float:
    """Linear correlation. H0: no linear association."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    return float(np.corrcoef(x, y)[0, 1])


def point_biserial(x, g) -> float:
    """Pearson between a continuous variable and a 0/1 group code."""
    return pearson(x, np.asarray(g, float))


def _resid(v, Z) -> np.ndarray:
    v = np.asarray(v, float)
    Z = np.asarray(Z, float)
    if Z.ndim == 1:
        Z = Z[:, None]
    Z = np.column_stack([np.ones(len(v)), Z])
    beta, *_ = np.linalg.lstsq(Z, v, rcond=None)
    return v - Z @ beta


def partial_corr(x, y, z) -> float:
    """Correlation of x and y after removing linear effects of z (one or more columns)."""
    return pearson(_resid(x, z), _resid(y, z))


def semi_partial(x, y, z) -> float:
    """Correlation of y with the part of x unexplained by z."""
    return pearson(_resid(x, z), np.asarray(y, float))


def multiple_r(y, X) -> float:
    """Multiple correlation of y with a block X (sqrt of R^2)."""
    y = np.asarray(y, float)
    r = y - _resid(y, X)
    ss_res = np.sum((y - r) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return float(np.sqrt(max(0.0, 1 - ss_res / ss_tot)))


# ---------------------------------------------------------------- rank

def spearman(x, y) -> float:
    """Monotone association on ranks. H0: no monotone association."""
    return float(stats.spearmanr(x, y).statistic)


def spearman_footrule(x, y) -> float:
    """Normalised Spearman footrule in [0, 1] (Diaconis-Graham)."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    n = len(x)
    rx = stats.rankdata(x)
    ry = stats.rankdata(y)
    return float(1 - 3 * np.abs(rx - ry).sum() / (n * n - 1))


def kendall_tau_a(x, y) -> float:
    return float(stats.kendalltau(x, y, variant="a").statistic)


def kendall_tau_b(x, y) -> float:
    return float(stats.kendalltau(x, y, variant="b").statistic)


def _pair_counts(x, y):
    """Concordant, discordant and tie counts over all i<j pairs."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    n = len(x)
    dx = np.sign(x[:, None] - x[None, :])
    dy = np.sign(y[:, None] - y[None, :])
    iu = np.triu_indices(n, 1)
    sx, sy = dx[iu], dy[iu]
    conc = np.sum((sx * sy) > 0)
    disc = np.sum((sx * sy) < 0)
    tie_x = np.sum((sx == 0) & (sy != 0))
    tie_y = np.sum((sy == 0) & (sx != 0))
    tie_both = np.sum((sx == 0) & (sy == 0))
    return conc, disc, tie_x, tie_y, tie_both


def gamma_gk(x, y) -> float:
    """Goodman-Kruskal gamma: ignores all ties. H0: independence of order."""
    c, d, _, _, _ = _pair_counts(x, y)
    return float((c - d) / max(c + d, EPS))


def somers_d(x, y) -> float:
    """Somers' D with y as the dependent variable (asymmetric)."""
    c, d, _, ty, _ = _pair_counts(x, y)
    return float((c - d) / max(c + d + ty, EPS))


def hoeffding_d(x, y) -> float:
    """Hoeffding's D in its empirical-copula (V-statistic) form, range [0, 1].

    D = 30 * mean_i (C_n(u_i, v_i) - u_i v_i)^2, with u, v mid-ranks / n.
    This is the population functional of Hoeffding (1948); the classical
    U-statistic formula differs in finite-sample scaling, so values from
    different implementations are not directly comparable.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    n = len(x)
    u = stats.rankdata(x) / n
    v = stats.rankdata(y) / n
    Cn = np.array([np.mean((u <= u[i]) & (v <= v[i])) for i in range(n)])
    return float(30 * np.mean((Cn - u * v) ** 2))


# ---------------------------------------------------------------- distance / kernel

def _dist(a) -> np.ndarray:
    a = np.asarray(a, float)[:, None]
    return np.abs(a - a.T)


def _u_center(D) -> np.ndarray:
    return D - D.mean(0)[None, :] - D.mean(1)[:, None] + D.mean()


def dcor(x, y) -> float:
    """Distance correlation. H0: x and y are independent (in distribution)."""
    A = _u_center(_dist(x))
    B = _u_center(_dist(y))
    dcov2 = (A * B).mean()
    dx = (A * A).mean()
    dy = (B * B).mean()
    return float(np.sqrt(max(dcov2, 0) / np.sqrt(max(dx * dy, EPS))))


def pdcor(x, y, z) -> float:
    """Partial distance correlation (Szekely-Rizzo 2014)."""
    A = _u_center(_dist(x))
    B = _u_center(_dist(y))
    C = _u_center(_dist(z))

    def ip(P, Q):
        return float((P * Q).mean())

    def nrm(P):
        return np.sqrt(max(ip(P, P), EPS))

    uxy = ip(A, B) / (nrm(A) * nrm(B))
    uxz = ip(A, C) / (nrm(A) * nrm(C))
    uyz = ip(B, C) / (nrm(B) * nrm(C))
    den = np.sqrt(max((1 - uxz ** 2) * (1 - uyz ** 2), EPS))
    return float((uxy - uxz * uyz) / den)


def _rbf(a) -> np.ndarray:
    a = np.asarray(a, float)[:, None]
    d = (a - a.T) ** 2
    med = np.median(d[d > 0]) if np.any(d > 0) else 1.0
    return np.exp(-d / max(med, EPS))


def hsic(x, y) -> float:
    """Normalised HSIC with RBF kernels and the median heuristic."""
    K = _u_center(_rbf(x))
    L = _u_center(_rbf(y))
    hxy = (K * L).mean()
    hxx = (K * K).mean()
    hyy = (L * L).mean()
    return float(hxy / np.sqrt(max(hxx * hyy, EPS)))


# ---------------------------------------------------------------- information / grid

def _mi_hist(x, y, bins) -> float:
    """Mutual information in nats from a 2-D histogram, Miller-Madow corrected."""
    if np.isscalar(bins):
        bx = by = int(bins)
    else:
        bx, by = bins
    c, _, _ = np.histogram2d(x, y, bins=[bx, by])
    n = c.sum()
    p = c / n
    px = p.sum(1, keepdims=True)
    py = p.sum(0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = p * np.log(p / (px * py))
    mi = np.nansum(t)
    k = np.sum(p > 0)
    return float(max(mi + (k - 1) / (2 * n), 0.0))


def mutual_info(x, y, bins=None) -> float:
    """Mutual information in nats. H0: independence."""
    n = len(x)
    if bins is None:
        bins = max(2, int(round(np.sqrt(n))))
    return _mi_hist(x, y, bins)


def nmi(x, y, bins=None) -> float:
    """MI / sqrt(Hx Hy), in [0, 1]."""
    n = len(x)
    if bins is None:
        bins = max(2, int(round(np.sqrt(n))))
    mi = _mi_hist(x, y, bins)
    cx, _ = np.histogram(x, bins=bins)
    cy, _ = np.histogram(y, bins=bins)
    px = cx / cx.sum()
    py = cy / cy.sum()
    hx = -np.sum(px[px > 0] * np.log(px[px > 0]))
    hy = -np.sum(py[py > 0] * np.log(py[py > 0]))
    return float(mi / np.sqrt(max(hx * hy, EPS)))


def variation_of_info(x, y, bins=None) -> float:
    """Variation of information in nats (a metric, not a correlation)."""
    n = len(x)
    if bins is None:
        bins = max(2, int(round(np.sqrt(n))))
    mi = _mi_hist(x, y, bins)
    cx, _ = np.histogram(x, bins=bins)
    cy, _ = np.histogram(y, bins=bins)
    px = cx / cx.sum()
    py = cy / cy.sum()
    hx = -np.sum(px[px > 0] * np.log(px[px > 0]))
    hy = -np.sum(py[py > 0] * np.log(py[py > 0]))
    return float(hx + hy - 2 * mi)


def mic(x, y, kmax: int = 10) -> float:
    """A grid approximation of the maximal information coefficient.

    This is *not* the Reshef et al. (2011) ApproxMaxMI algorithm: bins are
    equal-frequency and the search is over k x l grids with k*l <= B(n).
    It is labelled as an approximation everywhere it appears.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    n = len(x)
    B = max(4.0, n ** 0.6)
    best = 0.0
    for k in range(2, kmax + 1):
        for l in range(2, kmax + 1):
            if k * l > B:
                continue
            qx = np.quantile(x, np.linspace(0, 1, k + 1)[1:-1])
            qy = np.quantile(y, np.linspace(0, 1, l + 1)[1:-1])
            xb = np.searchsorted(qx, x)
            yb = np.searchsorted(qy, y)
            c = np.zeros((k, l))
            np.add.at(c, (xb, yb), 1)
            p = c / n
            px = p.sum(1, keepdims=True)
            py = p.sum(0, keepdims=True)
            with np.errstate(divide="ignore", invalid="ignore"):
                t = p * np.log(p / (px * py))
            mi = np.nansum(t)
            best = max(best, mi / np.log(min(k, l)))
    return float(min(best, 1.0))


def rdc(x, y, k: int = 20, s: float = 1 / 6, seed: int = 0) -> float:
    """Randomised dependence coefficient (Lopez-Paz et al. 2013)."""
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    x = (x - x.mean()) / (x.std() + EPS)
    y = (y - y.mean()) / (y.std() + EPS)
    sx, sy = np.sort(x), np.sort(y)

    def Phi(v, sort_v):
        return stats.norm.cdf((v - sort_v.mean()) / (sort_v.std() + EPS))

    vals = []
    for _ in range(k):
        a = rng.normal(0, s)
        b = rng.uniform(0, 2 * np.pi)
        fx = np.column_stack([x, np.sin(a * x + b)])
        fy = np.column_stack([y, np.sin(a * y + b)])
        # ridge regression of fy on fx, R^2 on the first canonical direction
        A = fx.T @ fx + 1e-3 * np.eye(2)
        W = np.linalg.solve(A, fx.T @ fy)
        pred = fx @ W
        ss_res = np.sum((fy - pred) ** 2)
        ss_tot = np.sum((fy - fy.mean(0)) ** 2)
        vals.append(1 - ss_res / (ss_tot + EPS))
    return float(np.clip(np.mean(vals), 0, 1))


def xi_chatterjee(x, y) -> float:
    """Chatterjee's xi. H0: independence; equals 1 for a noiseless function of x."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    n = len(x)
    order = np.argsort(x, kind="mergesort")
    y = y[order]
    r = stats.rankdata(y, method="max")
    l = r
    num = n * np.sum(np.abs(np.diff(r)))
    den = 2 * np.sum(l * (n - l))
    return float(1 - num / max(den, EPS))


def xi_perm(x, y, n_perm: int = 200, seed: int = 0) -> float:
    """Permutation-corrected xi: (xi - E0) / (1 - E0), clipped at 0."""
    rng = np.random.default_rng(seed)
    xi = xi_chatterjee(x, y)
    y = np.asarray(y, float)
    null = np.array([xi_chatterjee(x, rng.permutation(y)) for _ in range(n_perm)])
    e0 = null.mean()
    return float(max(0.0, (xi - e0) / max(1 - e0, EPS)))


def transfer_entropy(x, y, bins: int = 4, lag: int = 1) -> float:
    """TE(X -> Y) from a plug-in histogram estimator, in nats."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    xb = np.searchsorted(np.quantile(x, np.linspace(0, 1, bins + 1)[1:-1]), x)
    yb = np.searchsorted(np.quantile(y, np.linspace(0, 1, bins + 1)[1:-1]), y)
    x0, y0, y1 = xb[:-lag], yb[:-lag], yb[lag:]
    n = len(y1)
    # p(y1 | y0, x0) vs p(y1 | y0)
    joint = np.zeros((bins, bins, bins))
    np.add.at(joint, (x0, y0, y1), 1)
    p = joint / n
    pyx = p.sum(2, keepdims=True)      # p(x0, y0)
    py1y0 = p.sum(0)                   # p(y0, y1)
    py0 = p.sum((0, 2))                # p(y0)
    te = 0.0
    for i in range(bins):
        for j in range(bins):
            for k in range(bins):
                if p[i, j, k] <= 0:
                    continue
                cond_joint = p[i, j, k] / max(pyx[i, j, 0], EPS)
                cond_marg = py1y0[j, k] / max(py0[j], EPS)
                te += p[i, j, k] * np.log(cond_joint / max(cond_marg, EPS))
    return float(max(te, 0.0))


# ---------------------------------------------------------------- categorical / ordinal

def _chi2_table(a, b):
    tab = np.zeros((a.max() + 1, b.max() + 1))
    np.add.at(tab, (a, b), 1)
    chi2, _, _, _ = stats.chi2_contingency(tab, correction=False)
    return tab, chi2


def phi_coef(a, b) -> float:
    """Phi for two binary variables (signed by the 2x2 direction)."""
    a = np.asarray(a, int)
    b = np.asarray(b, int)
    tab, chi2 = _chi2_table(a, b)
    if tab.shape != (2, 2):
        raise ValueError("phi requires two binary variables")
    v = float(np.sqrt(chi2 / tab.sum()))
    return float(np.sign(tab[0, 0] * tab[1, 1] - tab[0, 1] * tab[1, 0]) * v)


def cramers_v(a, b) -> float:
    """Cramer's V for two categorical variables."""
    a = np.asarray(a, int)
    b = np.asarray(b, int)
    tab, chi2 = _chi2_table(a, b)
    n = tab.sum()
    return float(np.sqrt(chi2 / (n * (min(tab.shape) - 1))))


def tschuprow_t(a, b) -> float:
    a = np.asarray(a, int)
    b = np.asarray(b, int)
    tab, chi2 = _chi2_table(a, b)
    n = tab.sum()
    den = n * np.sqrt((tab.shape[0] - 1) * (tab.shape[1] - 1))
    return float(np.sqrt(chi2 / den))


def lambda_gk(a, b) -> float:
    """Goodman-Kruskal lambda: proportional reduction in error predicting b from a."""
    a = np.asarray(a, int)
    b = np.asarray(b, int)
    tab, _ = _chi2_table(a, b)
    n = tab.sum()
    err0 = n - tab.sum(0).max()
    err1 = n - tab.max(1).sum()
    return float((err0 - err1) / max(err0, EPS))


def theil_u(a, b) -> float:
    """Uncertainty coefficient U(b|a) in [0, 1]."""
    a = np.asarray(a, int)
    b = np.asarray(b, int)
    tab, _ = _chi2_table(a, b)
    tab = tab / tab.sum()

    def H(p):
        p = p[p > 0]
        return -np.sum(p * np.log(p))

    hb = H(tab.sum(0))
    hb_a = H(tab.ravel()) - H(tab.sum(1))
    return float((hb - hb_a) / max(hb, EPS))


def polychoric(a, b) -> float:
    """Two-step ML polychoric correlation for two ordinal variables."""
    a = np.asarray(a, int)
    b = np.asarray(b, int)
    ka, kb = a.max() + 1, b.max() + 1
    tab = np.zeros((ka, kb))
    np.add.at(tab, (a, b), 1)
    pa = tab.sum(1) / tab.sum()
    pb = tab.sum(0) / tab.sum()
    ta = stats.norm.ppf(np.cumsum(pa)[:-1])
    tb = stats.norm.ppf(np.cumsum(pb)[:-1])

    def nll(rho):
        rho = np.clip(rho, -0.999, 0.999)
        grid = np.array([[ta[i], tb[j]] for i in range(ka - 1) for j in range(kb - 1)])
        cum = stats.multivariate_normal.cdf(
            grid, mean=[0, 0], cov=[[1, rho], [rho, 1]]).reshape(ka - 1, kb - 1)
        F = np.zeros((ka + 1, kb + 1))
        F[1:-1, 1:-1] = cum
        F[-1, :] = 1.0
        F[:, -1] = 1.0
        cell = np.diff(np.diff(F, axis=0), axis=1)
        return -np.sum(tab * np.log(np.clip(cell, 1e-12, None)))

    from scipy.optimize import minimize_scalar
    res = minimize_scalar(nll, bounds=(-0.99, 0.99), method="bounded")
    return float(res.x)


def tetrachoric(a, b) -> float:
    return polychoric(a, b)


def polyserial(x, b) -> float:
    """ML polyserial correlation between a continuous x and an ordinal b."""
    x = np.asarray(x, float)
    b = np.asarray(b, int)
    k = b.max() + 1
    tau = stats.norm.ppf(np.cumsum(np.bincount(b) / len(b))[:-1])
    z = (x - x.mean()) / x.std(ddof=0)

    def nll(rho):
        rho = np.clip(rho, -0.99, 0.99)
        ll = 0.0
        for j in range(k):
            lo = tau[j - 1] if j > 0 else -np.inf
            hi = tau[j] if j < k - 1 else np.inf
            m = (rho * z)[b == j]
            sd = np.sqrt(max(1 - rho ** 2, EPS))
            ll += np.sum(np.log(np.clip(
                stats.norm.cdf((hi - m) / sd) - stats.norm.cdf((lo - m) / sd),
                1e-12, None)))
        return -ll

    from scipy.optimize import minimize_scalar
    res = minimize_scalar(nll, bounds=(-0.99, 0.99), method="bounded")
    return float(res.x)


# ---------------------------------------------------------------- agreement

def icc_1_1(y1, y2) -> float:
    """ICC(1,1): one-way random effects, single measurement."""
    Y = np.column_stack([y1, y2])
    n, k = Y.shape
    m = Y.mean()
    msb = k * np.sum((Y.mean(1) - m) ** 2) / (n - 1)
    msw = np.sum((Y - Y.mean(1, keepdims=True)) ** 2) / (n * (k - 1))
    return float((msb - msw) / (msb + (k - 1) * msw))


def icc_2_1(y1, y2) -> float:
    """ICC(2,1): two-way random effects, absolute agreement, single rater."""
    Y = np.column_stack([y1, y2])
    n, k = Y.shape
    m = Y.mean()
    msr = k * np.sum((Y.mean(1) - m) ** 2) / (n - 1)
    msc = n * np.sum((Y.mean(0) - m) ** 2) / (k - 1)
    mse = np.sum((Y - Y.mean(1, keepdims=True) - Y.mean(0, keepdims=True) + m) ** 2) \
        / ((n - 1) * (k - 1))
    return float((msr - mse) / (msr + (k - 1) * mse + k * (msc - mse) / n))


def ccc(x, y) -> float:
    """Lin's concordance correlation coefficient."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    sxy = np.cov(x, y, ddof=1)[0, 1]
    return float(2 * sxy / (x.var(ddof=1) + y.var(ddof=1) + (x.mean() - y.mean()) ** 2))


def bland_altman(y1, y2):
    d = np.asarray(y1, float) - np.asarray(y2, float)
    return float(d.mean()), float(d.mean() - 1.96 * d.std(ddof=1)), \
        float(d.mean() + 1.96 * d.std(ddof=1))


# ---------------------------------------------------------------- multivariate / time series

def rv(X, Y) -> float:
    """RV coefficient between two centred blocks (n x p) and (n x q)."""
    X = np.asarray(X, float)
    Y = np.asarray(Y, float)
    X = X - X.mean(0)
    Y = Y - Y.mean(0)
    Sxy = X @ X.T
    Syy = Y @ Y.T
    num = np.sum(Sxy * Syy)
    den = np.sqrt(np.sum(Sxy ** 2) * np.sum(Syy ** 2))
    return float(num / max(den, EPS))


def rv2(X, Y) -> float:
    """Mayer's modified RV, which removes the diagonal inflation."""
    X = np.asarray(X, float)
    Y = np.asarray(Y, float)
    X = X - X.mean(0)
    Y = Y - Y.mean(0)
    Sxy = X @ X.T
    Syy = Y @ Y.T
    dx = np.diag(np.diag(Sxy))
    dy = np.diag(np.diag(Syy))
    A = Sxy - dx
    B = Syy - dy
    return float(np.sum(A * B) / np.sqrt(max(np.sum(A * A) * np.sum(B * B), EPS)))


def cca_first(X, Y) -> float:
    """First canonical correlation between two blocks."""
    X = np.asarray(X, float)
    Y = np.asarray(Y, float)
    X = X - X.mean(0)
    Y = Y - Y.mean(0)
    Qx, _ = np.linalg.qr(X)
    Qy, _ = np.linalg.qr(Y)
    s = np.linalg.svd(Qx.T @ Qy, compute_uv=False)
    return float(np.clip(s[0], 0, 1))


def lagged_ccf(x, y, max_lag: int = 10) -> float:
    """Maximum absolute cross-correlation over lags -max_lag..max_lag."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    x = (x - x.mean()) / (x.std() + EPS)
    y = (y - y.mean()) / (y.std() + EPS)
    best = 0.0
    n = len(x)
    for k in range(-max_lag, max_lag + 1):
        if k >= 0:
            v = np.mean(x[:n - k] * y[k:])
        else:
            v = np.mean(x[-k:] * y[:n + k])
        best = max(best, abs(v))
    return float(best)


def lagged_dcor(x, y, max_lag: int = 10) -> float:
    """Lagged distance correlation, a proxy for cdCor; max over lags."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    best = 0.0
    n = len(x)
    for k in range(0, max_lag + 1):
        a, b = x[:n - k], y[k:]
        best = max(best, dcor(a, b))
    return float(best)


# ---------------------------------------------------------------- registry

def registry() -> dict:
    """All measures used in the article, with a short label."""
    return {
        "pearson": (pearson, "linear"),
        "spearman": (spearman, "monotone"),
        "kendall_b": (kendall_tau_b, "monotone"),
        "gamma": (gamma_gk, "ordinal"),
        "somers_d": (somers_d, "ordinal"),
        "hoeffding_d": (hoeffding_d, "omnibus"),
        "dcor": (dcor, "omnibus"),
        "hsic": (hsic, "omnibus"),
        "mic": (mic, "omnibus"),
        "rdc": (rdc, "omnibus"),
        "xi": (xi_chatterjee, "functional"),
        "mi": (mutual_info, "omnibus"),
        "nmi": (nmi, "omnibus"),
        "vi": (variation_of_info, "omnibus"),
        "cramers_v": (cramers_v, "categorical"),
        "theil_u": (theil_u, "categorical"),
        "polychoric": (polychoric, "ordinal"),
        "icc21": (icc_2_1, "agreement"),
        "ccc": (ccc, "agreement"),
    }
