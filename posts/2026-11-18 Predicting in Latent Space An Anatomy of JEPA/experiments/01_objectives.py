import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib.pyplot as plt
import numpy as np
import torch

from common.data import make_shapes
from common.style import PAL, set_style, save
from common.train import (collapse_metrics, extract_features, probe, train_jepa,
                          train_mae, train_simclr, train_supervised)
from common.utils import save_metrics, set_seed, timer

set_style()
set_seed(0)

N_TR, N_TE, EPOCHS = 6000, 1500, 15
tr, ftr = make_shapes(N_TR, seed=0)
te, fte = make_shapes(N_TE, seed=1)

runs = {}
with timer("JEPA"):
    runs["JEPA"], hist = train_jepa(tr, epochs=EPOCHS, seed=11,
                                    ema=0.999, reg=0.1, mask_ratio=0.25)
    runs["JEPA"].loss_hist = hist
with timer("MAE"):
    runs["MAE"], hist = train_mae(tr, epochs=EPOCHS, seed=12)
    runs["MAE"].loss_hist = hist
with timer("SimCLR"):
    runs["SimCLR"], hist = train_simclr(tr, epochs=EPOCHS, seed=13)
    runs["SimCLR"].loss_hist = hist
with timer("Supervised"):
    runs["Supervised"], hist = train_supervised(tr, ftr["shape"], epochs=EPOCHS, seed=14)
    runs["Supervised"].loss_hist = hist

rows = []
for name, model in runs.items():
    f_tr = extract_features(model, tr)
    f_te = extract_features(model, te)
    res = probe(f_tr, ftr, f_te, fte)
    col = collapse_metrics(f_tr)
    rows.append({
        "model": name,
        "shape_acc": res["shape_acc"],
        "colour_acc": res["colour_acc"],
        "position_rmse": res["position_rmse"],
        "effective_rank": col["effective_rank"],
        "feature_std": col["feature_std"],
        "mean_cosine": col["mean_cosine"],
        "final_loss": float(model.loss_hist[-1]),
    })
    enc = model.context if name == "JEPA" else model.encoder
    torch.save(enc.state_dict(), Path("experiments/outputs") / f"encoder_{name.lower()}.pt")

names = [r["model"] for r in rows]
fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.2))
axes[0].bar(names, [r["shape_acc"] for r in rows],
            color=[PAL["vermillion"], PAL["blue"], PAL["green"], PAL["orange"]])
axes[0].set_ylim(0, 1)
axes[0].set_ylabel("shape accuracy")
axes[0].set_title("linear probe: shape")

axes[1].bar(names, [r["position_rmse"] for r in rows],
            color=[PAL["vermillion"], PAL["blue"], PAL["green"], PAL["orange"]])
axes[1].set_ylabel("position RMSE")
axes[1].set_title("linear probe: position")

axes[2].bar(names, [r["effective_rank"] for r in rows],
            color=[PAL["vermillion"], PAL["blue"], PAL["green"], PAL["orange"]])
axes[2].set_ylabel("effective rank")
axes[2].set_title("collapse diagnostic")
for ax in axes:
    ax.tick_params(axis="x", rotation=20)
fig.suptitle("Same encoder, four objectives", y=1.02, fontsize=10)
save(fig, "fig05_objectives")

save_metrics("e01_objectives", {
    "n_train": N_TR,
    "n_test": N_TE,
    "epochs": EPOCHS,
    "rows": rows,
    "chance_shape": 1 / 3,
    "chance_colour": 1 / 5,
})
