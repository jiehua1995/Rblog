"""Study 7 - autocorrelation breaks the usual test.

Two independent AR(1) series produce sample correlations whose spread is much
wider than the iid formula assumes, so Fisher's interval and the |r| > 1.96/sqrt(n)
rule become anti-conservative.  The same series can still be told apart when the
coupling is lagged, and that is what cross-correlation and transfer entropy see.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import corr as C
from common.style import PAL, save, set_style
from common.utils import save_metrics, set_seed

N = 200
REPS = 400
PHI = 0.9
LAG = 5


def ar1(n, phi, rng, sd=1.0) -> np.ndarray:
    e = rng.normal(0, sd, size=n)
    return signal.lfilter([1.0], [1.0, -phi], e)


def fisher_ci(r, n):
    z = np.arctanh(np.clip(r, -0.999, 0.999))
    se = 1 / np.sqrt(n - 3)
    return np.tanh(z - 1.96 * se), np.tanh(z + 1.96 * se)


def main() -> None:
    set_seed(0)
    rng = np.random.default_rng(0)

    r_ar, r_iid, cover_ar, cover_iid = [], [], [], []
    for _ in range(REPS):
        x = ar1(N, PHI, rng)
        y = ar1(N, PHI, rng)
        u = rng.normal(size=N)
        v = rng.normal(size=N)
        for a, b, rs, cv in ((x, y, r_ar, cover_ar), (u, v, r_iid, cover_iid)):
            r = C.pearson(a, b)
            lo, hi = fisher_ci(r, N)
            rs.append(r)
            cv.append(bool(lo <= 0 <= hi))
    r_ar = np.array(r_ar)
    r_iid = np.array(r_iid)
    thr = 1.96 / np.sqrt(N)
    fp_ar = float(np.mean(np.abs(r_ar) > thr))
    fp_iid = float(np.mean(np.abs(r_iid) > thr))
    cov_ar = float(np.mean(cover_ar))
    cov_iid = float(np.mean(cover_iid))

    # lagged coupling
    x = ar1(1200, 0.8, rng)
    e = ar1(1200, 0.8, rng, sd=0.5)
    y = np.zeros_like(x)
    y[LAG:] = 0.6 * x[:-LAG]
    y += e
    lags = np.arange(0, 11)
    prof = []
    for k in lags:
        a, b = x[:len(x) - k], y[k:]
        prof.append(abs(C.pearson(a, b)))
    prof = np.array(prof)
    te_xy = C.transfer_entropy(x, y)
    te_yx = C.transfer_entropy(y, x)
    dc_lag = C.lagged_dcor(x, y)
    print(f"[timeseries] fp_ar={fp_ar:.3f} fp_iid={fp_iid:.3f} "
          f"peak_lag={lags[prof.argmax()]} te_xy={te_xy:.4f} te_yx={te_yx:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.6))
    ax = axes[0]
    bins = np.linspace(-0.8, 0.8, 41)
    ax.hist(r_iid, bins=bins, density=True, color=PAL["blue"], alpha=0.45, label="iid")
    ax.hist(r_ar, bins=bins, density=True, color=PAL["orange"], alpha=0.45,
            label="AR(1), phi=0.9")
    grid = np.linspace(-0.8, 0.8, 200)
    ax.plot(grid, np.exp(-grid ** 2 * (N - 3) / 2) * np.sqrt((N - 3) / (2 * np.pi)),
            color=PAL["dark"], lw=0.9, ls="--", label="Fisher-z null")
    ax.set_xlabel("sample Pearson r (true value 0)")
    ax.set_ylabel("density")
    ax.set_title("(a)  two independent series", loc="left")
    ax.legend(frameon=False, fontsize=6.6)

    ax = axes[1]
    ax.plot(lags, prof, marker="o", ms=3.2, color=PAL["teal"], label="|cross-correlation|")
    ax.axvline(LAG, color=PAL["orange"], lw=0.8, ls=":")
    ax.set_xlabel("lag k (y_t vs x_{t-k})")
    ax.set_ylabel("association")
    ax.set_title("(b)  lagged coupling at k = 5", loc="left")
    fig.tight_layout()
    save(fig, "fig08_timeseries")

    save_metrics("timeseries", {
        "n": N, "reps": REPS, "phi": PHI,
        "ar1_r_sd": float(r_ar.std()), "iid_r_sd": float(r_iid.std()),
        "naive_fp_ar1": fp_ar, "naive_fp_iid": fp_iid,
        "fisher_coverage_ar1": cov_ar, "fisher_coverage_iid": cov_iid,
        "coupling_lag": int(LAG), "ccf_profile": prof.tolist(),
        "ccf_peak_lag": int(lags[prof.argmax()]),
        "te_xy": float(te_xy), "te_yx": float(te_yx),
        "lagged_dcor": float(dc_lag),
    })


if __name__ == "__main__":
    main()
