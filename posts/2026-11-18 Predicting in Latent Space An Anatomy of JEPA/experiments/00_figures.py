import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from common.data import make_shapes, make_video
from common.style import PAL, set_style, save

set_style()
rng = np.random.default_rng(0)

# ---- data samples -----------------------------------------------------------
imgs, fac = make_shapes(8, seed=3)
fig, axes = plt.subplots(2, 4, figsize=(8.6, 4.4))
for ax, im, s, c in zip(axes.ravel(), imgs, fac["shape"], fac["colour"]):
    ax.imshow(im.transpose(1, 2, 0))
    ax.set_title(f"shape {s}, colour {c}", fontsize=8)
    ax.axis("off")
fig.suptitle("32 x 32 synthetic scenes with known factors", fontsize=10)
save(fig, "fig01_data")

# ---- video frames -----------------------------------------------------------
frames, acts, states = make_video(1, frames=6, seed=7)
fig, axes = plt.subplots(1, 6, figsize=(9.6, 2.0))
for i, ax in enumerate(axes):
    ax.imshow(frames[0, i].transpose(1, 2, 0))
    ax.set_title(f"t = {i}", fontsize=8)
    ax.axis("off")
fig.suptitle("bouncing ball: actions are accelerations", fontsize=10)
save(fig, "fig01b_video")

# ---- objective map ----------------------------------------------------------
fig, ax = plt.subplots(figsize=(6.8, 4.6))
ax.set_xlim(-0.05, 1.05)
ax.set_ylim(-0.05, 1.05)
ax.set_xlabel("prediction target: observation  ->  representation")
ax.set_ylabel("invariance: none  ->  learned")
for name, x, y, col in [
    ("diffusion", 0.02, 0.08, PAL["grey"]),
    ("MAE", 0.10, 0.18, PAL["blue"]),
    ("SimCLR", 0.22, 0.78, PAL["green"]),
    ("BYOL", 0.25, 0.90, PAL["sky"]),
    ("DINO", 0.34, 0.84, PAL["orange"]),
    ("JEPA", 0.82, 0.55, PAL["vermillion"]),
    ("LeJEPA", 0.92, 0.85, PAL["purple"]),
    ("world model", 0.86, 0.30, PAL["dark"]),
]:
    ax.scatter([x], [y], s=70, color=col, zorder=3)
    ax.text(x + 0.02, y + 0.02, name, fontsize=8)
ax.annotate("", xy=(0.98, 0.02), xytext=(0.02, 0.02),
            arrowprops=dict(arrowstyle="->", color=PAL["grey"], lw=0.8))
ax.annotate("", xy=(0.02, 0.98), xytext=(0.02, 0.02),
            arrowprops=dict(arrowstyle="->", color=PAL["grey"], lw=0.8))
ax.set_title("where the loss lives")
save(fig, "fig02_map")

# ---- architecture -----------------------------------------------------------
fig, ax = plt.subplots(figsize=(8.6, 4.2))
ax.set_xlim(0, 10.2)
ax.set_ylim(1.8, 5.8)
ax.axis("off")


def box(x, y, w, h, text, col):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08",
                                facecolor=col, edgecolor="none", alpha=0.85))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8)


def arrow(x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="->",
                                 mutation_scale=12, color=PAL["dark"], lw=0.8))


box(0.2, 2.6, 1.4, 1.0, "context\npatches", PAL["light"])
box(2.2, 2.6, 1.7, 1.0, "context\nencoder", PAL["blue"])
box(4.4, 2.6, 1.7, 1.0, "predictor", PAL["vermillion"])
box(6.8, 2.6, 1.7, 1.0, "prediction", PAL["orange"])
box(2.2, 4.4, 1.7, 1.0, "target\nencoder", PAL["green"])
box(6.8, 4.4, 1.7, 1.0, "target", PAL["green"])
arrow(1.6, 3.1, 2.2, 3.1)
arrow(3.9, 3.1, 4.4, 3.1)
arrow(6.1, 3.1, 6.8, 3.1)
arrow(3.05, 4.4, 3.05, 3.6)
arrow(3.9, 4.9, 6.8, 4.9)
arrow(8.5, 3.1, 9.0, 3.1)
ax.text(1.9, 3.42, "visible", fontsize=7, ha="center")
ax.text(3.18, 4.0, "EMA", fontsize=7, ha="left")
ax.text(5.35, 5.02, "masked patches", fontsize=7, ha="center")
ax.text(7.65, 4.18, "stop-gradient", fontsize=7, ha="center")
ax.add_patch(plt.Circle((9.3, 3.1), 0.24, color=PAL["purple"], alpha=0.85))
ax.text(9.3, 3.1, "L", color="white", ha="center", va="center", fontsize=8)
ax.text(9.3, 3.55, "loss", fontsize=8, ha="center")
ax.set_title("JEPA: predict the target representation, not the pixels")
save(fig, "fig03_arch")

# ---- timeline ---------------------------------------------------------------
events = [
    (2022.4, "position paper", PAL["grey"]),
    (2023.6, "I-JEPA", PAL["blue"]),
    (2024.6, "V-JEPA", PAL["sky"]),
    (2025.45, "V-JEPA 2", PAL["vermillion"]),
    (2025.85, "LeJEPA", PAL["purple"]),
    (2025.95, "LLM-JEPA", PAL["orange"]),
    (2026.2, "V-JEPA 2.1", PAL["vermillion"]),
    (2026.25, "LeWorldModel", PAL["green"]),
    (2026.75, "JEPA-Anything", PAL["dark"]),
    (2026.8, "Predictive Closure", PAL["grey"]),
]
fig, ax = plt.subplots(figsize=(6.8, 7.0))
ax.set_xlim(-1.2, 1.2)
ax.set_ylim(2022, 2027.2)
ax.axvline(0, color=PAL["grey"], lw=0.7)
for i, (year, label, col) in enumerate(events):
    side = 1 if i % 2 == 0 else -1
    ax.plot([0, side * 0.1], [year, year], color=col, lw=0.8)
    ax.scatter([0], [year], color=col, s=34, zorder=3)
    ax.text(side * 0.16, year, label, ha="left" if side == 1 else "right",
            va="center", fontsize=8, color=col)
ax.set_yticks(range(2022, 2028))
ax.set_xticks([])
ax.set_title("JEPA, 2022-2026")
save(fig, "fig04_timeline")
