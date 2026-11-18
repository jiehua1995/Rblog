import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib.pyplot as plt

from common.data import make_shapes
from common.style import PAL, set_style, save
from common.train import extract_features, probe, train_jepa
from common.utils import save_metrics, set_seed, timer

set_style()
set_seed(3)

N_TR, N_TE, EPOCHS = 3000, 800, 8
tr, ftr = make_shapes(N_TR, seed=0)
te, fte = make_shapes(N_TE, seed=1)
configs = [(1, 64), (2, 64), (3, 64), (2, 128), (3, 128), (2, 256)]

rows = []
with timer("predictor capacity"):
    for k, (depth, width) in enumerate(configs):
        model, _ = train_jepa(tr, epochs=EPOCHS, seed=200 + k, mask_ratio=0.25,
                              ema=0.999, reg=0.1,
                              predictor_depth=depth, predictor_width=width)
        f_tr, f_te = extract_features(model, tr), extract_features(model, te)
        res = probe(f_tr, ftr, f_te, fte)
        rows.append({"depth": depth, "width": width, **res})
        print(depth, width, round(res["shape_acc"], 3))

labels = [f"d{d}\nw{w}" for d, w in configs]
fig, ax = plt.subplots(figsize=(7.2, 3.2))
ax.bar(labels, [r["shape_acc"] for r in rows], color=PAL["purple"])
ax.axhline(1 / 3, color=PAL["grey"], linestyle="--", linewidth=0.8)
ax.set_ylabel("shape accuracy")
ax.set_xlabel("predictor depth / width")
ax.set_title("predictor capacity")
save(fig, "fig08_predictor")

save_metrics("e04_predictor", {"rows": rows, "configs": configs,
                               "chance_shape": 1 / 3})
