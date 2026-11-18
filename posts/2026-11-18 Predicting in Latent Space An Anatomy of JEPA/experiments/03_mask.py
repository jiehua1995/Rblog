import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib.pyplot as plt

from common.data import make_shapes
from common.style import PAL, set_style, save
from common.train import collapse_metrics, extract_features, probe, train_jepa, train_mae
from common.utils import save_metrics, set_seed, timer

set_style()
set_seed(2)

N_TR, N_TE, EPOCHS = 3000, 800, 8
tr, ftr = make_shapes(N_TR, seed=0)
te, fte = make_shapes(N_TE, seed=1)
ratios = [0.25, 0.5, 0.75]

rows = []
with timer("mask ratio"):
    for r in ratios:
        for name, fn in [("JEPA", train_jepa), ("MAE", train_mae)]:
            model, _ = fn(tr, epochs=EPOCHS, seed=int(100 * r) + len(name), mask_ratio=r)
            f_tr, f_te = extract_features(model, tr), extract_features(model, te)
            res = probe(f_tr, ftr, f_te, fte)
            col = collapse_metrics(f_tr)
            rows.append({"ratio": r, "model": name, **res,
                         "effective_rank": col["effective_rank"]})
            print(r, name, round(res["shape_acc"], 3), round(col["effective_rank"], 2))

fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.2))
for name, colour in [("JEPA", PAL["vermillion"]), ("MAE", PAL["blue"])]:
    sub = [x for x in rows if x["model"] == name]
    axes[0].plot([x["ratio"] for x in sub], [x["shape_acc"] for x in sub],
                 marker="o", color=colour, label=name)
    axes[1].plot([x["ratio"] for x in sub], [x["effective_rank"] for x in sub],
                 marker="o", color=colour, label=name)
axes[0].axhline(1 / 3, color=PAL["grey"], linestyle="--", linewidth=0.8)
axes[0].set_xlabel("mask ratio")
axes[0].set_ylabel("shape accuracy")
axes[0].set_title("linear probe")
axes[1].set_xlabel("mask ratio")
axes[1].set_ylabel("effective rank")
axes[1].set_title("representation rank")
axes[0].legend()
save(fig, "fig07_mask")

save_metrics("e03_mask", {"rows": rows, "ratios": ratios,
                          "chance_shape": 1 / 3})
