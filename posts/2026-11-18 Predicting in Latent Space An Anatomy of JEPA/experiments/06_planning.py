import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib.pyplot as plt
import numpy as np
import torch

from common.data import PALETTE, render_state, step_state
from common.models import WorldModelJEPA, WorldModelPixel
from common.style import PAL, set_style, save
from common.train import plan_latent, plan_pixel
from common.utils import save_metrics, set_seed, timer

set_style()
set_seed(5)

wj = WorldModelJEPA()
wj.load_state_dict(torch.load(Path("experiments/outputs/world_jepa.pt"), map_location="cpu"))
wj.eval()
wp = WorldModelPixel()
wp.load_state_dict(torch.load(Path("experiments/outputs/world_pixel.pt"), map_location="cpu"))
wp.eval()


def run_episode(policy: str, seed: int, steps: int = 12):
    rng = np.random.default_rng(seed)
    r = 0.15
    pos = rng.uniform(0.25, 0.75, size=2)
    state = np.array([pos[0], pos[1], 0.0, 0.0], dtype=np.float32)
    goal = rng.uniform(0.2, 0.8, size=2)
    goal_frame = render_state(np.array([goal[0], goal[1], 0, 0]), colour=PALETTE[1],
                              shape=0, r=r)[None]
    for t in range(steps):
        frame = render_state(state, colour=PALETTE[1], shape=0, r=r)[None]
        if policy == "latent":
            action = plan_latent(wj, frame, goal_frame, horizon=5,
                                 pop=24, iters=3, elite=6)[0].numpy()
        elif policy == "pixel":
            action = plan_pixel(wp, frame, goal_frame, horizon=5,
                                pop=24, iters=3, elite=6)[0].numpy()
        else:
            action = rng.uniform(-1, 1, size=2)
        state = step_state(state, action, r)
    d = float(np.linalg.norm(state[:2] - goal))
    return d, float(d < 0.08)


episodes = 10
with timer("planning"):
    res = {"latent": [], "pixel": [], "random": []}
    for i in range(episodes):
        for k in res:
            res[k].append(run_episode(k, 100 + i))

names = ["latent CEM", "pixel CEM", "random"]
keys = ["latent", "pixel", "random"]
success = [np.mean([x[1] for x in res[k]]) for k in keys]
dist = [np.mean([x[0] for x in res[k]]) for k in keys]
print({k: (success[i], dist[i]) for i, k in enumerate(keys)})

fig, axes = plt.subplots(1, 2, figsize=(7.8, 3.0))
cols = [PAL["blue"], PAL["green"], PAL["grey"]]
axes[0].bar(names, success, color=cols)
axes[0].set_ylim(0, 1)
axes[0].set_ylabel("success rate")
axes[0].set_title(f"{episodes} episodes")
axes[1].bar(names, dist, color=cols)
axes[1].set_ylabel("final distance to goal")
axes[1].set_title("lower is better")
save(fig, "fig10_planning")

save_metrics("e06_planning", {
    "episodes": episodes,
    "success_latent": success[0], "success_pixel": success[1],
    "success_random": success[2],
    "distance_latent": dist[0], "distance_pixel": dist[1],
    "distance_random": dist[2],
})
