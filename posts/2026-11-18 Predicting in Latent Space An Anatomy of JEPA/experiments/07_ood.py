import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib.pyplot as plt
import numpy as np
import torch

from common.data import make_shapes
from common.models import TinyViT
from common.style import PAL, set_style, save
from common.train import extract_features, probe
from common.utils import save_metrics, set_seed

set_style()
set_seed(6)

tr, ftr = make_shapes(3000, seed=0)
idte, fid = make_shapes(800, seed=1)
ood, food = make_shapes(800, seed=2, ood=True)

rows = []
for name in ["jepa", "mae", "simclr", "supervised"]:
    enc = TinyViT()
    enc.load_state_dict(torch.load(Path(f"experiments/outputs/encoder_{name}.pt"),
                                  map_location="cpu"))
    enc.eval()
    f_tr = extract_features(enc, tr)
    f_id = extract_features(enc, idte)
    f_ood = extract_features(enc, ood)
    res_id = probe(f_tr, ftr, f_id, fid)
    res_ood = probe(f_tr, ftr, f_ood, food)
    rows.append({
        "model": name,
        "id_shape": res_id["shape_acc"],
        "ood_shape": res_ood["shape_acc"],
        "id_colour": res_id["colour_acc"],
        "ood_colour": res_ood["colour_acc"],
        "id_position": res_id["position_rmse"],
        "ood_position": res_ood["position_rmse"],
    })
    print(name, rows[-1])

names = ["JEPA", "MAE", "SimCLR", "Supervised"]
x = np.arange(len(rows))
width = 0.36
fig, ax = plt.subplots(figsize=(7.6, 3.2))
ax.bar(x - width / 2, [r["id_shape"] for r in rows], width,
       label="in-distribution", color=PAL["blue"])
ax.bar(x + width / 2, [r["ood_shape"] for r in rows], width,
       label="OOD", color=PAL["vermillion"])
ax.axhline(1 / 3, color=PAL["grey"], linestyle="--", linewidth=0.8)
ax.set_xticks(x, names)
ax.set_ylim(0, 1)
ax.set_ylabel("shape accuracy")
ax.legend()
save(fig, "fig11_ood")

save_metrics("e07_ood", {"rows": rows, "chance_shape": 1 / 3})
