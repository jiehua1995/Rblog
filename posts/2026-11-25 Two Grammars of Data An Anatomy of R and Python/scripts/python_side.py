"""Python side of the comparison.

Reads data/bench.csv (written by scripts/00_data.R), performs the same tasks as
the R scripts, and writes

  outputs/python.json           every number quoted in the article
  outputs/py_sums_scale.json    exact references for the summation experiment
  figures/fig_py_*.png          the matplotlib and plotnine figures

Run from the post directory:  python scripts/python_side.py
"""

import json
import math
import platform
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotnine
import scipy
import scipy.stats as st
import statsmodels
import statsmodels.formula.api as smf

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "figures"
OUT = ROOT / "outputs"
FIG.mkdir(exist_ok=True)
OUT.mkdir(exist_ok=True)

res = {}


def jsonable(x):
    """json.dumps writes bare NaN, which R's jsonlite refuses to parse."""
    if isinstance(x, dict):
        return {k: jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, float) and not math.isfinite(x):
        return None
    return x


def as_chr(x):
    if isinstance(x, float) and math.isnan(x):
        return "NaN"
    return format(x, ".10g")


def bench(fn, times=5, warmup=1):
    """Fastest of `times` runs, in seconds, after one warm-up."""
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(times):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return min(ts)


df = pd.read_csv(ROOT / "data" / "bench.csv")
small = pd.read_csv(ROOT / "data" / "small.csv")
sums = pd.read_csv(ROOT / "data" / "sums.csv")["s"].to_numpy()

res["versions"] = {
    "python": platform.python_version(),
    "numpy": np.__version__,
    "pandas": pd.__version__,
    "scipy": scipy.__version__,
    "statsmodels": statsmodels.__version__,
    "plotnine": plotnine.__version__,
    "matplotlib": matplotlib.__version__,
    "machine": platform.processor(),
    "platform": platform.platform(),
}

# ---------------------------------------------------------------- semantics
# indexing: half-open slices, zero-based
res["indexing"] = {
    "py_second_to_fourth": df["x"].to_numpy()[1:4].round(6).tolist(),
    "py_first": float(df["x"].iloc[0]),
    "py_last": float(df["x"].iloc[-1]),
    "py_len": int(len(df)),
    # drop_first in Python is a slice, in R it is a negative index
    "py_drop_first_head": df["x"].to_numpy()[1:4].round(6).tolist(),
}

# missing values: numpy has one NaN, no separate NA
v = np.array([1.0, np.nan, 3.0])
res["missing"] = {
    "sum": float(np.sum(v)),
    "nansum": float(np.nansum(v)),
    "mean": float(np.mean(v)),
    "nanmean": float(np.nanmean(v)),
    "max": float(np.max(v)),
    "nanmax": float(np.nanmax(v)),
    "pandas_sum": float(pd.Series(v).sum()),
    "pandas_mean": float(pd.Series(v).mean()),
    "pandas_max": float(pd.Series(v).max()),
    "isna": [bool(z) for z in pd.isna(v)],
    "sort": [float(z) for z in np.sort(v)],
    "dropna_len": int(pd.Series(v).dropna().shape[0]),
}

# the same results as strings, so the article can print NA and NaN side by side
res["missing_chr"] = {
    "sum": as_chr(float(np.sum(v))),
    "nansum": as_chr(float(np.nansum(v))),
    "mean": as_chr(float(np.mean(v))),
    "nanmean": as_chr(float(np.nanmean(v))),
    "max": as_chr(float(np.max(v))),
    "nanmax": as_chr(float(np.nanmax(v))),
    "pandas_sum": as_chr(float(pd.Series(v).sum())),
    "pandas_mean": as_chr(float(pd.Series(v).mean())),
    "pandas_max": as_chr(float(pd.Series(v).max())),
}

# aliasing: a numpy slice is a view into the parent buffer
base = np.arange(5.0)
view = base[1:4]
view[0] = 99.0
res["alias_numpy"] = {"parent": base.tolist(), "is_view": bool(np.shares_memory(base, view))}

# pandas hands out a read-only array under copy-on-write
pdf = df.head(3).copy()
cols = pdf["x"].to_numpy()
try:
    cols[0] = -999.0
    pd_write = "allowed"
except Exception as exc:  # noqa: BLE001
    pd_write = type(exc).__name__ + ": " + str(exc)
res["alias_pandas"] = {
    "parent_after_column_write": float(pdf.loc[0, "x"]),
    "write_result": pd_write,
    "array_is_readonly": bool(getattr(cols, "flags", None) and cols.flags.writeable is False),
}

# recycling vs broadcasting: numpy refuses a ragged operation
rec = {}
try:
    a4 = np.arange(1.0, 5.0)
    b2 = np.array([1.0, 2.0])
    rec["result"] = (a4 + b2).tolist()
    rec["error"] = None
except Exception as exc:  # noqa: BLE001
    rec["result"] = None
    rec["error"] = type(exc).__name__ + ": " + str(exc)
res["recycle_numpy"] = rec

rec_vec = {"result": (a4 + np.array([10.0])).tolist()}
res["broadcast_scalar_vec"] = rec_vec

# boolean mask must match the length of the axis
try:
    mask = np.array([True, False, True, False])
    a_bad = np.arange(3.0)
    _ = a_bad[mask]
    res["bool_index"] = {"result": None, "error": None}
except Exception as exc:  # noqa: BLE001
    res["bool_index"] = {"result": None, "error": type(exc).__name__ + ": " + str(exc)}

res["bool_index_ok"] = np.arange(4.0)[np.array([True, False, True, False])].tolist()

# groupby: one row per observed group
gb_py = (df.groupby("grp", observed=True)["val"]
           .agg(["count", "mean", "std"])
           .reset_index())
res["groupby_rows"] = int(len(gb_py))

# ------------------------------------------------------------------- speed
loop_sum = lambda: sum(df["val"].tolist())
np_sum = lambda: float(np.sum(df["val"].to_numpy()))
g_py = lambda: df.groupby("grp", observed=True)["val"].mean()
sort_py = lambda: df["val"].sort_values()
lookup = pd.DataFrame({"grp": df["grp"].unique(),
                       "w": np.linspace(1.0, 2.0, df["grp"].nunique())})
merge_py = lambda: df.merge(lookup, on="grp", how="left")
pivot_py = lambda: (df.assign(bin=pd.cut(df["x"], 5))
                      .groupby(["cat", "bin"], observed=True)["val"]
                      .mean()
                      .unstack())

res["speed"] = {
    "n": int(len(df)),
    "loop_sum_s": bench(loop_sum),
    "np_sum_s": bench(np_sum),
    "groupby_mean_s": bench(g_py),
    "sort_s": bench(sort_py),
    "merge_s": bench(merge_py),
    "pivot_s": bench(pivot_py),
    "memory_bytes": int(df.memory_usage(deep=True).sum()),
    "memory_cat_bytes": int(df["cat"].memory_usage(deep=True)),
    "n_groups": int(df["grp"].nunique()),
}

# per-column memory: object dtype against category dtype
mem = {"n": int(len(df))}
for col in ["grp", "cat"]:
    mem[col + "_object"] = int(df[col].memory_usage(deep=True))
    mem[col + "_category"] = int(df[col].astype("category").memory_usage(deep=True))
mem["frame_object"] = int(df.memory_usage(deep=True).sum())
mem["frame_category"] = int(
    df.assign(grp=df["grp"].astype("category"), cat=df["cat"].astype("category"))
      .memory_usage(deep=True).sum())
res["memory"] = mem

gb_py.to_csv(OUT / "py_groupby.csv", index=False)

# ----------------------------------------------------------- numeric checks
ols = smf.ols("y ~ x + z", data=df).fit()
res["ols"] = {
    "names": list(ols.params.index),
    "coef": [float(c) for c in ols.params],
    "bse": [float(c) for c in ols.bse],
    "tvalues": [float(c) for c in ols.tvalues],
    "pvalues": [float(c) for c in ols.pvalues],
    "rsquared": float(ols.rsquared),
    "nobs": int(ols.nobs),
}

a = df.loc[df["cat"] == "alpha", "x"].to_numpy()
b = df.loc[df["cat"] == "beta", "x"].to_numpy()
tt = st.ttest_ind(a, b, equal_var=True)
res["ttest"] = {"t": float(tt.statistic), "p": float(tt.pvalue),
                "n_alpha": int(len(a)), "n_beta": int(len(b))}
res["ttest"]["mean_alpha"] = float(np.mean(a))
res["ttest"]["mean_beta"] = float(np.mean(b))


# summation order: naive loop vs numpy pairwise vs exact (fsum)
def kahan(x):
    total = 0.0
    c = 0.0
    for v_ in x.tolist():
        y = v_ - c
        t = total + y
        c = (t - total) - y
        total = t
    return total


s = sums
exact = math.fsum(s.tolist())
naive = 0.0
for v_ in s.tolist():
    naive += v_

res["sums"] = {
    "n": int(len(s)),
    "exact": exact,
    "naive": naive,
    "numpy": float(np.sum(s)),
    "kahan": kahan(s),
    "naive_rel": abs(naive - exact) / abs(exact),
    "numpy_rel": abs(float(np.sum(s)) - exact) / abs(exact),
    "kahan_rel": abs(kahan(s) - exact) / abs(exact),
}

# how the error grows with n, against an exact reference both languages can use.
# Twenty overlapping windows of the same file, median error at each length, so
# the curve is not the accident of one particular ordering.
ns = [1000, 3000, 10000, 30000, 100000, 300000]
REPS = 20
scale = {"n": ns, "exact": [], "naive_rel": [], "numpy_rel": [],
         "kahan_rel": [], "loop_s": [], "numpy_s": [], "fsum_s": []}
for n_ in ns:
    e_nv, e_np, e_ka, t_lp, t_np, t_fs, exs = [], [], [], [], [], [], []
    for r in range(REPS):
        pre = s[r * 20000:(r * 20000 + n_)]
        ex = math.fsum(pre.tolist())

        nv = 0.0
        t0 = time.perf_counter()
        for v_ in pre.tolist():
            nv += v_
        t_lp.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        sn = float(np.sum(pre))
        t_np.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        math.fsum(pre.tolist())
        t_fs.append(time.perf_counter() - t0)

        e_nv.append(abs(nv - ex) / abs(ex))
        e_np.append(abs(sn - ex) / abs(ex))
        e_ka.append(abs(kahan(pre) - ex) / abs(ex))
        exs.append(ex)
    scale["exact"].append(exs)
    scale["naive_rel"].append(float(np.median(e_nv)))
    scale["numpy_rel"].append(float(np.median(e_np)))
    scale["kahan_rel"].append(float(np.median(e_ka)))
    scale["loop_s"].append(float(np.median(t_lp)))
    scale["numpy_s"].append(float(np.median(t_np)))
    scale["fsum_s"].append(float(np.median(t_fs)))
scale["reps"] = REPS
(OUT / "py_sums_scale.json").write_text(json.dumps(scale, indent=2))

# catastrophic cancellation: exact answer is 1000, every algorithm that adds in
# the order given will lose the ones while the partial sum is ~1e17
cancel = pd.read_csv(ROOT / "data" / "cancel.csv")["t"].to_numpy()
ex_c = math.fsum(cancel.tolist())
nv_c = 0.0
for v_ in cancel.tolist():
    nv_c += v_
res["cancel"] = {
    "exact": ex_c,
    "naive": nv_c,
    "numpy": float(np.sum(cancel)),
    "kahan": kahan(cancel),
    "naive_abs": abs(nv_c - ex_c),
    "numpy_abs": abs(float(np.sum(cancel)) - ex_c),
    "kahan_abs": abs(kahan(cancel) - ex_c),
    "n": int(len(cancel)),
}

# ------------------------------------------------------------------- plots
def finish(fig, name):
    fig.patch.set_alpha(0)
    fig.savefig(FIG / name, dpi=200, transparent=True, bbox_inches="tight")
    plt.close(fig)


# 1) the default look of matplotlib, no styling at all
fig, ax = plt.subplots(figsize=(4.6, 3.4))
for c, g in small.groupby("cat", observed=True):
    ax.scatter(g["x"], g["y"], s=16, label=c)
    m, b0 = np.polyfit(g["x"], g["y"], 1)
    xs = np.linspace(g["x"].min(), g["x"].max(), 20)
    ax.plot(xs, m * xs + b0)
ax.legend(title="cat", frameon=True, fontsize=8, title_fontsize=8)
ax.set_xlabel("x")
ax.set_ylabel("y")
finish(fig, "fig_py_default.png")

# 2) the default look of plotnine
from plotnine import (aes, element_rect, element_text, facet_wrap, geom_point,
                      geom_smooth, ggplot, labs, theme, theme_gray)

p = (ggplot(small, aes("x", "y", colour="cat"))
     + geom_point()
     + geom_smooth(method="lm", se=False)
     + labs(x="x", y="y", colour="cat")
     + theme(plot_background=element_rect(fill="none", colour="none")))


def save_p9(plot, name, w=4.6, h=3.4):
    plot.save(FIG / name, width=w, height=h, dpi=200, verbose=False)


save_p9(p, "fig_py_plotnine.png")

# 3) faceting by hand (matplotlib) vs facet_wrap (plotnine)
small = small.assign(bin=pd.qcut(small["x"], 3, labels=["low", "mid", "high"]))
cats = ["alpha", "beta", "gamma"]
bins = ["low", "mid", "high"]
fig, axes = plt.subplots(3, 3, figsize=(7.0, 5.2), sharex=True, sharey=True)
for i, c in enumerate(cats):
    for j, bn in enumerate(bins):
        ax = axes[i, j]
        g = small[(small["cat"] == c) & (small["bin"] == bn)]
        ax.scatter(g["x"], g["y"], s=10)
        ax.set_title(f"{c} / {bn}", fontsize=7)
for ax in axes.flat:
    ax.tick_params(labelsize=6)
fig.tight_layout()
finish(fig, "fig_py_facet.png")

p2 = (ggplot(small, aes("x", "y"))
      + geom_point(size=1.0, colour="#4C6EF5")
      + facet_wrap("~ cat + bin", ncol=3)
      + theme_gray(base_size=8)
      + theme(strip_text=element_text(size=7),
              plot_background=element_rect(fill="none", colour="none")))
save_p9(p2, "fig_py_facet_plotnine.png", w=7.0, h=5.2)

# 4) render cost: build and save the same three-panel figure, both stacks
def draw_mpl():
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    for c, g in small.groupby("cat", observed=True):
        ax.scatter(g["x"], g["y"], s=16, label=c)
        m, b0 = np.polyfit(g["x"], g["y"], 1)
        xs = np.linspace(g["x"].min(), g["x"].max(), 20)
        ax.plot(xs, m * xs + b0)
    ax.legend(fontsize=8)
    fig.savefig(FIG / "_tmp_mpl.png", dpi=200, transparent=True)
    plt.close(fig)


def draw_p9():
    p.save(FIG / "_tmp_p9.png", width=4.6, height=3.4, dpi=200, verbose=False)


res["render"] = {"mpl_s": bench(draw_mpl), "plotnine_s": bench(draw_p9)}

(OUT / "python.json").write_text(json.dumps(jsonable(res), indent=2, sort_keys=True))
print("python side done")

