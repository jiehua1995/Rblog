"""Article 1, study 1 — is LLM capability growth actually slowing?

Reads a curated table of published benchmark scores (data/raw/benchmarks.csv),
fits linear / exponential / logistic trends, computes bootstrap and OLS
confidence intervals, and looks for a change point.  The table is deliberately
small and every row has a source; missing cells are left missing rather than
interpolated.

Run:  python experiments/llm_progress/01_trends.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common.style import PAL, save, set_style
from common.utils import bootstrap_ci, save_metrics, set_seed

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "data" / "raw" / "benchmarks.csv"

EVENTS = [
    ("2012-09", "AlexNet", "architecture"),
    ("2014-06", "GAN", "architecture"),
    ("2015-12", "ResNet", "architecture"),
    ("2016-03", "AlphaGo", "reasoning"),
    ("2017-06", "Transformer", "architecture"),
    ("2018-10", "BERT", "scale"),
    ("2019-02", "GPT-2", "scale"),
    ("2020-05", "GPT-3", "scale"),
    ("2022-03", "Chinchilla", "scale"),
    ("2022-11", "ChatGPT", "product"),
    ("2023-03", "GPT-4", "product"),
    ("2024-09", "o1", "reasoning"),
    ("2025-01", "DeepSeek-R1", "reasoning"),
    ("2025-04", "o3", "reasoning"),
    ("2025-11", "Gemini 3", "product"),
    ("2026-01", "TH1.1 update", "evaluation"),
]
CAT_COL = {"architecture": PAL["navy"], "scale": PAL["teal"],
           "reasoning": PAL["orange"], "product": PAL["gold"],
           "evaluation": PAL["purple"]}


def load() -> pd.DataFrame:
    df = pd.read_csv(CSV)
    df["date"] = pd.to_datetime(df["date"])
    df["t"] = (df["date"] - pd.Timestamp("2020-01-01")).dt.days / 365.25
    return df


def ols_slope(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    n = len(x)
    b, a = np.polyfit(x, y, 1)
    resid = y - (a + b * x)
    sxx = np.sum((x - x.mean()) ** 2)
    se = np.sqrt(np.sum(resid ** 2) / max(n - 2, 1) / sxx) if sxx > 0 else np.nan
    return b, se


def fit_models(t, s):
    """Return dict of model predictions and AIC for linear/exp/logistic."""
    t = np.asarray(t, float)
    s = np.asarray(s, float)
    out = {}
    b, a = np.polyfit(t, s, 1)
    pred = a + b * t
    out["linear"] = (pred, b, a)

    def expm(x, A, k):
        return A * np.exp(k * x)

    try:
        p, _ = curve_fit(expm, t, np.maximum(s, 1e-3), p0=[10, 0.2], maxfev=20000)
        out["exponential"] = (expm(t, *p), *p)
    except Exception:
        out["exponential"] = (np.full_like(s, np.nan), np.nan, np.nan)

    def logi(x, L, k, x0):
        return L / (1 + np.exp(-k * (x - x0)))

    try:
        p, _ = curve_fit(logi, t, s, p0=[100, 1.0, np.median(t)], maxfev=40000)
        out["logistic"] = (logi(t, *p), *p)
    except Exception:
        out["logistic"] = (np.full_like(s, np.nan), np.nan, np.nan, np.nan)

    def aic(y, yhat, k):
        n = len(y)
        rss = np.sum((y - yhat) ** 2)
        return n * np.log(max(rss / n, 1e-12)) + 2 * k

    out_aic = {
        "linear": aic(s, out["linear"][0], 2),
        "exponential": aic(s, out["exponential"][0], 2),
        "logistic": aic(s, out["logistic"][0], 3),
    }
    return out, out_aic


def main() -> None:
    set_seed(0)
    set_style()
    df = load()

    # ---------- figure 1: timeline (two windows) ----------
    rows = {"architecture": 0, "scale": 1, "reasoning": 2, "product": 3,
            "evaluation": 4}
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.4),
                             gridspec_kw={"width_ratios": [0.9, 1.25]})
    for ax, (lo, hi) in zip(axes, [("2012-01-01", "2019-12-31"),
                                   ("2020-01-01", "2026-12-31")]):
        ev = [(pd.Timestamp(d + "-15"), name, cat)
              for d, name, cat in EVENTS
              if lo <= d + "-15" <= hi]
        for k, (t, name, cat) in enumerate(ev):
            yy = rows[cat]
            ax.plot([t], [yy], "o", ms=6, color=CAT_COL[cat], mec="white", mew=0.6)
            up = (k % 2 == 0)
            ax.annotate(name, (t, yy), xytext=(0, 8 if up else -9),
                        textcoords="offset points", ha="center",
                        va="bottom" if up else "top", fontsize=6.2,
                        color=CAT_COL[cat])
        ax.set_xlim(pd.Timestamp(lo), pd.Timestamp(hi))
        ax.set_ylim(-0.8, 4.8)
        ax.set_yticks(list(rows.values()))
        ax.set_yticklabels(list(rows.keys()), fontsize=6.6)
        ax.tick_params(axis="x", labelsize=6.4)
        ax.grid(axis="x")
        ax.set_xlabel("year")
    axes[0].set_title("(a)  2012–2019: architectures and scale", loc="left")
    axes[1].set_title("(b)  2020–2026: scale, reasoning, products", loc="left")
    for cat, name in [("architecture", "architecture"), ("scale", "scale/data"),
                      ("reasoning", "reasoning/RL"), ("product", "product"),
                      ("evaluation", "evaluation")]:
        axes[0].plot([], [], "o", color=CAT_COL[cat], label=name)
    fig.legend(loc="upper center", ncol=5, frameon=False, fontsize=6.8,
               bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save(fig, "fig01_timeline", 1)

    # ---------- figure 2: benchmark trajectories ----------
    from matplotlib.dates import DateFormatter, MonthLocator
    bench = ["MMLU", "GPQA", "SWE-bench", "HLE", "ARC-AGI", "FrontierMath"]
    ver_col = [PAL["navy"], PAL["orange"], PAL["teal"], PAL["purple"]]
    fig, axes = plt.subplots(2, 3, figsize=(11.4, 5.8), sharey=True)
    for ax, b in zip(axes.ravel(), bench):
        sub = df[df.benchmark == b]
        for k, (ver, g) in enumerate(sub.groupby("version")):
            g = g.sort_values("date").reset_index(drop=True)
            c = ver_col[k % len(ver_col)]
            ax.plot(g["date"], g["score"], marker="o", ms=3.6, color=c,
                    label=str(ver))
            for idx in sorted(set([0, len(g) - 1])):
                r = g.iloc[idx]
                ax.annotate(r["model"], (r["date"], r["score"]),
                            xytext=(-3, 4 if idx == 0 else -9),
                            textcoords="offset points", fontsize=4.8,
                            ha="right", color=PAL["grey"], clip_on=True)
        ax.set_title(b, loc="left")
        ax.set_ylim(-2, 100)
        ax.grid(True, alpha=0.18)
        ax.xaxis.set_major_locator(MonthLocator(interval=6))
        ax.xaxis.set_major_formatter(DateFormatter("%Y-%m"))
        ax.tick_params(axis="x", labelrotation=35, labelsize=6)
        if sub["version"].nunique() > 1:
            ax.legend(frameon=False, fontsize=5.4, loc="center right")
    for ax in axes[:, 0]:
        ax.set_ylabel("score (%)")
    fig.suptitle("(c)  Hard benchmarks are far from their ceiling; easy ones are not",
                 x=0.01, ha="left", fontsize=10)
    fig.tight_layout()
    save(fig, "fig03_benchmarks", 1)

    # ---------- figure 3: annualised slope with CI (within one version) ----------
    rows = []
    for (b, ver), sub in df.groupby(["benchmark", "version"]):
        if b not in bench:
            continue
        sub = sub.sort_values("date")
        if len(sub) < 3:
            continue
        bslope, se = ols_slope(sub["t"], sub["score"])
        label = b if df[df.benchmark == b]["version"].nunique() == 1 else f"{b}-{ver}"
        rows.append({"benchmark": label, "slope": bslope, "se": se,
                     "n": len(sub), "latest": sub["score"].iloc[-1]})
    tab = pd.DataFrame(rows).sort_values("slope")
    fig, ax = plt.subplots(figsize=(7.6, 3.8))
    ypos = np.arange(len(tab))
    ax.barh(ypos, tab["slope"], color=PAL["blue"], height=0.6, alpha=0.9)
    ax.errorbar(tab["slope"], ypos, xerr=1.96 * tab["se"], fmt="none",
                ecolor=PAL["dark"], capsize=3, lw=1)
    ax.set_yticks(ypos)
    ax.set_yticklabels([f"{b}  (n={n})" for b, n in zip(tab["benchmark"], tab["n"])],
                       fontsize=6.8)
    ax.set_xlabel("points per year (OLS slope, 95% CI)")
    ax.set_xlim(left=0)
    ax.set_title("(a)  Progress has not stopped; it is uneven across tests", loc="left")
    ax.grid(axis="x")
    save(fig, "fig04_growth_rates", 1)

    # ---------- figure 4: METR horizon ----------
    metr = df[df.benchmark == "METR"].sort_values("date")
    tt = metr["t"].values
    ll = np.log10(metr["score"].values)
    bslope, a = np.polyfit(tt, ll, 1)
    doubling_months = 12 * np.log10(2) / bslope if bslope > 0 else np.nan
    boots = []
    rng = np.random.default_rng(1)
    for _ in range(2000):
        idx = rng.integers(0, len(tt), len(tt))
        if len(np.unique(tt[idx])) < 2:
            continue
        bb, _ = np.polyfit(tt[idx], ll[idx], 1)
        if bb > 0:
            boots.append(12 * np.log10(2) / bb)
    lo, hi = np.quantile(boots, [0.025, 0.975])

    price = df[df.benchmark == "Inference price"].sort_values("date")
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.0))

    ax = axes[0]
    grid = np.linspace(tt.min() - 0.2, tt.max() + 0.8, 100)
    xgrid = pd.Timestamp("2020-01-01") + pd.to_timedelta(grid * 365.25, unit="D")
    ax.plot(xgrid, 10 ** (a + bslope * grid), color=PAL["navy"], lw=1.1,
            label=f"exponential fit: doubling {doubling_months:.1f} months\n"
                  f"bootstrap 95% CI {lo:.1f}–{hi:.1f} months")
    ax.scatter(metr["date"], metr["score"], color=PAL["orange"], s=34,
               zorder=3, edgecolor="white", linewidth=0.6, label="published estimates")
    ax.axhline(480, color=PAL["dark"], lw=0.8, ls="--")
    ax.annotate("one working day", (metr["date"].iloc[0], 480), xytext=(0, 3),
                textcoords="offset points", fontsize=6, color=PAL["dark"])
    ax.set_yscale("log")
    ax.set_ylabel("50% time horizon (minutes, log)")
    ax.set_xlabel("date")
    ax.set_title("(a)  Autonomous task length keeps doubling", loc="left")
    ax.legend(frameon=False, fontsize=6.2, loc="upper left")
    ax.grid(True, which="major")

    ax = axes[1]
    ax.plot(price["date"], price["score"], marker="o", color=PAL["teal"])
    for _, r in price.iterrows():
        ax.annotate(f"{r['model']}\n${r['score']:g}", (r["date"], r["score"]),
                    xytext=(4, 4), textcoords="offset points", fontsize=6.0)
    ax.set_yscale("log")
    ax.set_ylabel("output price (USD per million tokens, log)")
    ax.set_xlabel("date")
    ax.set_title("(b)  API list prices, flagship vs small model", loc="left")
    ax.set_xlim(price["date"].min() - pd.Timedelta(days=40),
                price["date"].max() + pd.Timedelta(days=40))
    ax.grid(True, which="major")
    fig.tight_layout()
    save(fig, "fig05_metr", 1)

    # ---------- figure 5: training compute of frontier models ----------
    comp = df[df.benchmark == "Training compute"].sort_values("date")
    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    ax.plot(comp["date"], comp["score"], marker="o", color=PAL["navy"], ms=4)
    for _, r in comp.iterrows():
        ax.annotate(f"{r['model']}\n{r['score']:.1e} FLOP", (r["date"], r["score"]),
                    xytext=(5, -2), textcoords="offset points", fontsize=6.2)
    g0 = (comp["date"].iloc[0], comp["score"].iloc[0])
    yrs = np.linspace(0, 5.2, 50)
    ax.plot(pd.Timestamp(g0[0]) + pd.to_timedelta(yrs * 365.25, unit="D"),
            g0[1] * 5 ** yrs, ls="--", lw=0.9, color=PAL["orange"],
            label="5× per year (Epoch AI trend)")
    ax.set_yscale("log")
    ax.set_ylabel("training compute (FLOP, log)")
    ax.set_xlabel("date")
    ax.set_title("Training compute of frontier models", loc="left")
    ax.legend(frameon=False, fontsize=6.6, loc="lower right")
    ax.grid(True, which="major")
    save(fig, "fig02_compute", 1)

    # ---------- figure 6: headroom and time-to-ceiling (latest version only) ----------
    latest, monthly = {}, {}
    for b in bench:
        sub = df[df.benchmark == b].sort_values("date")
        if sub.empty:
            continue
        v = sub["version"].iloc[-1]
        g = sub[sub["version"] == v].sort_values("date")
        latest[b] = g.iloc[-1]
        if len(g) >= 3:
            months = (g["date"].iloc[-1] - g["date"].iloc[0]).days / 30.44
            monthly[b] = (g["score"].iloc[-1] - g["score"].iloc[0]) / months
        else:
            monthly[b] = np.nan
    latest = pd.DataFrame(latest).T
    headroom = 100 - latest["score"].astype(float)
    months_to_ceiling = headroom / np.maximum(
        pd.Series(monthly)[headroom.index].astype(float), 1e-9)

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 3.8))
    ax = axes[0]
    ax.barh(range(len(headroom)), headroom.values[::-1], color=PAL["sky"])
    ax.set_yticks(range(len(headroom)))
    ax.set_yticklabels(headroom.index[::-1], fontsize=7)
    ax.set_xlabel("headroom to 100% (points)")
    ax.set_title("(a)  How much room is left", loc="left")
    ax.grid(axis="x")

    ax = axes[1]
    vals = np.array(months_to_ceiling, float)
    ax.barh(range(len(vals)), vals[::-1], color=PAL["orange"])
    ax.set_yticks(range(len(vals)))
    ax.set_yticklabels(list(months_to_ceiling.index)[::-1], fontsize=7)
    ax.set_xlabel("months to close the remaining headroom at current pace")
    ax.set_title("(b)  Saturation, not slowdown", loc="left")
    ax.grid(axis="x")
    fig.tight_layout()
    save(fig, "fig06_saturation", 1)

    # ---------- figure 7: model comparison ----------
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.8))
    aic_out = {}
    for ax, b in zip(axes, ["MMLU", "GPQA"]):
        sub = df[df.benchmark == b].sort_values("date")
        t, s = sub["t"].values, sub["score"].values
        out, aic = fit_models(t, s)
        aic_out[b] = aic
        grid = np.linspace(t.min() - 0.2, t.max() + 0.4, 100)
        ax.scatter(sub["date"], s, color=PAL["dark"], s=26, zorder=3)
        for name, c in [("linear", PAL["blue"]), ("exponential", PAL["teal"]),
                        ("logistic", PAL["orange"])]:
            pred = out[name][0]
            ax.plot(sub["date"], pred, color=c, lw=1.0)
        # extrapolate the logistic for intuition
        f = out["logistic"]
        if np.all(np.isfinite(f[0])):
            L, k, x0 = f[1], f[2], f[3]
            ax.plot(pd.Timestamp("2020-01-01") + pd.to_timedelta(grid * 365.25, unit="D"),
                    L / (1 + np.exp(-k * (grid - x0))), color=PAL["orange"],
                    lw=0.8, ls="--", alpha=0.6)
        ax.set_ylim(0, 105)
        ax.set_title(f"{b}: AIC " + ", ".join(f"{k2}={v:.0f}" for k2, v in aic.items()),
                     loc="left")
        ax.set_ylabel("score (%)")
        ax.grid(True)
    axes[0].set_xlabel("date")
    axes[1].set_xlabel("date")
    fig.tight_layout()
    save(fig, "fig07_fits", 1)

    # ---------- change point on GPQA ----------
    sub = df[df.benchmark == "GPQA"].sort_values("date")
    t, s = sub["t"].values, sub["score"].values

    def piecewise(params):
        a1, b1, a2, b2, tau = params
        return np.where(t < tau, a1 + b1 * t, a2 + b2 * (t - tau) + a1 + b1 * tau)

    best = None
    for tau in np.linspace(t.min() + 0.1, t.max() - 0.1, 40):
        X = np.column_stack([np.ones_like(t), np.minimum(t, tau),
                             np.maximum(t - tau, 0)])
        beta, *_ = np.linalg.lstsq(X, s, rcond=None)
        pred = X @ beta
        rss = np.sum((s - pred) ** 2)
        bic = len(t) * np.log(max(rss / len(t), 1e-12)) + 3 * np.log(len(t))
        if best is None or bic < best[0]:
            best = (bic, tau, beta)

    save_metrics("llm_progress", {
        "n_records": int(len(df)),
        "n_benchmarks": int(df["benchmark"].nunique()),
        "slopes": {r.benchmark: {"slope": float(r.slope), "se": float(r.se),
                                 "n": int(r.n), "latest": float(r.latest)}
                   for r in tab.itertuples()},
        "metr_doubling_months": float(doubling_months),
        "metr_doubling_ci": [float(lo), float(hi)],
        "metr_official_doubling_days_th11": 130.8,
        "metr_official_doubling_days_since_2024": 89.0,
        "metr_slope_per_year": float(bslope),
        "price_drop_factor": float(price["score"].iloc[0] / price["score"].iloc[-1]),
        "price_note": "not capability-matched: GPT-4 (2023) vs GPT-4o-mini (2024)",
        "headroom": {k: float(v) for k, v in headroom.items()},
        "months_to_ceiling": {k: float(v) for k, v in months_to_ceiling.items()
                              if np.isfinite(v)},
        "aic": aic_out,
        "gpqa_breakpoint_year": float(best[1] + 2020),
        "gpqa_slope_before": float(best[2][1]),
        "gpqa_slope_after": float(best[2][2]),
    })


if __name__ == "__main__":
    main()
