import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib.pyplot as plt
import numpy as np

from common.data import make_shapes
from common.style import PAL, set_style, save
from common.train import collapse_metrics, extract_features, probe, train_jepa
from common.utils import save_metrics, set_seed, timer

set_style()
set_seed(1)

N_TR, N_TE, EPOCHS = 3000, 800, 8
tr, ftr = make_shapes(N_TR, seed=0)
te, fte = make_shapes(N_TE, seed=1)

configs = [
    ("joint", dict(ema=0.0, stop_grad=False, reg=0.0)),
    ("joint + reg 0.1", dict(ema=0.0, stop_grad=False, reg=0.1)),
    ("joint + reg 1.0", dict(ema=0.0, stop_grad=False, reg=1.0)),
    ("EMA 0.99", dict(ema=0.99, stop_grad=True, reg=0.0)),
    ("EMA + reg 0.1", dict(ema=0.99, stop_grad=True, reg=0.1)),
    ("EMA 0.999 + reg 0.1", dict(ema=0.999, stop_grad=True, reg=0.1)),
]

rows = []
with timer("collapse regimes"):
    for k, (name, kw) in enumerate(configs):
        model, hist = train_jepa(tr, epochs=EPOCHS, seed=30 + k,
                                 mask_ratio=0.25, **kw)
        f_tr, f_te = extract_features(model, tr), extract_features(model, te)
        res = probe(f_tr, ftr, f_te, fte)
        col = collapse_metrics(f_tr)
        rows.append({
            "regime": name,
            "shape_acc": res["shape_acc"],
            "shape_acc_raw": res["shape_acc_raw"],
            "effective_rank": col["effective_rank"],
            "feature_std": col["feature_std"],
            "mean_cosine": col["mean_cosine"],
            "final_loss": float(hist[-1]),
        })
        print(name, rows[-1])

names = [r["regime"] for r in rows]
y = np.arange(len(rows))[::-1]
fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.4))
axes[0].barh(y, [r["effective_rank"] for r in rows], color=PAL["blue"])
axes[0].set_yticks(y, names)
axes[0].set_xlabel("effective rank")
axes[0].set_title("rank")

axes[1].barh(y, [r["feature_std"] for r in rows], color=PAL["vermillion"])
axes[1].set_yticks(y, [""] * len(rows))
axes[1].set_xscale("log")
axes[1].set_xlabel("mean feature std (log)")
axes[1].set_title("variance")

axes[2].barh(y, [r["shape_acc"] for r in rows], color=PAL["green"])
axes[2].axvline(1 / 3, color=PAL["grey"], linestyle="--", linewidth=0.8)
axes[2].set_yticks(y, [""] * len(rows))
axes[2].set_xlim(0, 1)
axes[2].set_xlabel("shape accuracy")
axes[2].set_title("probe")
save(fig, "fig06_collapse")

save_metrics("e02_collapse", {
    "n_train": N_TR, "epochs": EPOCHS, "rows": rows,
    "chance_shape": 1 / 3,
})
