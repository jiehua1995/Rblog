import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib.pyplot as plt
import torch

from common.data import make_video
from common.style import PAL, set_style, save
from common.train import rollout_jepa, rollout_pixel, train_world_jepa, train_world_pixel
from common.utils import save_metrics, set_seed, timer

set_style()
set_seed(4)

N_TR, N_TE, FRAMES, EPOCHS = 1600, 200, 8, 5
frames, actions, _ = make_video(N_TR, frames=FRAMES, seed=0)
te_frames, te_actions, _ = make_video(N_TE, frames=FRAMES, seed=1)

with timer("latent world model"):
    wm, hist_j = train_world_jepa(frames, actions, epochs=EPOCHS, seed=50)
with timer("pixel world model"):
    pm, hist_p = train_world_pixel(frames, actions, epochs=EPOCHS, seed=51)

horizons = list(range(1, FRAMES))
latent_err, pixel_err = [], []
for h in horizons:
    latent_err.append(sum(rollout_jepa(wm, te_frames, te_actions, h)) / h)
    pixel_err.append(sum(rollout_pixel(pm, te_frames, te_actions, h)) / h)

fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.2))
axes[0].plot(horizons, latent_err, marker="o", color=PAL["vermillion"])
axes[0].set_yscale("log")
axes[0].set_xlabel("rollout horizon")
axes[0].set_ylabel("latent MSE")
axes[0].set_title("JEPA world model")
axes[1].plot(horizons, pixel_err, marker="o", color=PAL["blue"])
axes[1].set_yscale("log")
axes[1].set_xlabel("rollout horizon")
axes[1].set_ylabel("pixel MSE")
axes[1].set_title("pixel world model")
save(fig, "fig09_rollout")

torch.save(wm.state_dict(), Path("experiments/outputs/world_jepa.pt"))
torch.save(pm.state_dict(), Path("experiments/outputs/world_pixel.pt"))
save_metrics("e05_rollout", {
    "n_train": N_TR, "n_test": N_TE, "frames": FRAMES, "epochs": EPOCHS,
    "horizons": horizons,
    "latent_mse": latent_err,
    "pixel_mse": pixel_err,
    "latent_loss_hist": hist_j,
    "pixel_loss_hist": hist_p,
})
