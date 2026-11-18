"""Synthetic shapes and bouncing-ball videos with known generative factors.

Nothing is downloaded.  The point is not ImageNet realism but controlled factors:
shape, colour, position and size are known, so a linear probe can say what a
representation actually kept.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

PALETTE = np.array(
    [
        [0.84, 0.15, 0.12],  # vermillion
        [0.10, 0.42, 0.72],  # blue
        [0.10, 0.62, 0.35],  # green
        [0.92, 0.62, 0.05],  # orange
        [0.62, 0.35, 0.68],  # purple
    ],
    dtype=np.float32,
)
OOD_PALETTE = np.array(
    [
        [0.10, 0.70, 0.75],
        [0.85, 0.35, 0.70],
        [0.50, 0.50, 0.10],
        [0.90, 0.85, 0.20],
        [0.25, 0.25, 0.90],
    ],
    dtype=np.float32,
)


def _canvas(size: int, bg: float, rng: np.random.Generator, ood: bool) -> Image.Image:
    base = np.full((size, size, 3), bg, dtype=np.float32)
    if ood:
        base += rng.normal(0, 0.06, base.shape).astype(np.float32)
    else:
        base += rng.normal(0, 0.015, base.shape).astype(np.float32)
    base = np.clip(base, 0, 1)
    return Image.fromarray((base * 255).astype(np.uint8))


def _draw(img: Image.Image, shape: int, colour: np.ndarray, cx: float, cy: float,
          r: float, size: int) -> None:
    d = ImageDraw.Draw(img)
    col = tuple(int(255 * c) for c in colour)
    x, y, rr = cx * size, cy * size, r * size
    if shape == 0:
        d.ellipse([x - rr, y - rr, x + rr, y + rr], fill=col)
    elif shape == 1:
        d.rectangle([x - rr, y - rr, x + rr, y + rr], fill=col)
    else:
        d.polygon([(x, y - rr), (x - rr, y + rr), (x + rr, y + rr)], fill=col)


def make_shapes(n: int, seed: int = 0, size: int = 32, ood: bool = False):
    rng = np.random.default_rng(seed)
    imgs = np.zeros((n, 3, size, size), dtype=np.float32)
    shape = np.zeros(n, dtype=np.int64)
    colour = np.zeros(n, dtype=np.int64)
    pos = np.zeros((n, 2), dtype=np.float32)
    scale = np.zeros(n, dtype=np.float32)

    palette = OOD_PALETTE if ood else PALETTE
    for i in range(n):
        bg = rng.uniform(0.05, 0.25)
        if ood:
            bg = rng.uniform(0.35, 0.65)
        img = _canvas(size, bg, rng, ood)
        shape[i] = rng.integers(0, 3)
        colour[i] = rng.integers(0, len(palette))
        cx, cy = rng.uniform(0.25, 0.75, size=2)
        r = rng.uniform(0.20, 0.28) if ood else rng.uniform(0.14, 0.22)
        if ood:
            r *= rng.choice([0.6, 1.5])
        pos[i] = [cx, cy]
        scale[i] = r
        _draw(img, shape[i], palette[colour[i]], cx, cy, r, size)
        arr = np.asarray(img, dtype=np.float32).transpose(2, 0, 1) / 255.0
        imgs[i] = arr

    factors = {
        "shape": shape,
        "colour": colour,
        "position": pos,
        "scale": scale,
    }
    return imgs, factors


def _render_state(state: np.ndarray, size: int, colour: np.ndarray,
                  shape: int, r: float) -> np.ndarray:
    img = _canvas(size, 0.12, np.random.default_rng(0), False)
    _draw(img, shape, colour, float(state[0]), float(state[1]), r, size)
    return np.asarray(img, dtype=np.float32).transpose(2, 0, 1) / 255.0


def make_video(n: int, frames: int = 8, seed: int = 0, size: int = 32,
               ood: bool = False):
    """Bouncing ball driven by random accelerations.

    State is (x, y, vx, vy).  Actions are accelerations in [-1, 1]^2.
    """
    rng = np.random.default_rng(seed)
    palette = OOD_PALETTE if ood else PALETTE
    out = np.zeros((n, frames, 3, size, size), dtype=np.float32)
    acts = np.zeros((n, frames - 1, 2), dtype=np.float32)
    states = np.zeros((n, frames, 4), dtype=np.float32)

    for i in range(n):
        pos = rng.uniform(0.25, 0.75, size=2).astype(np.float32)
        vel = rng.uniform(-0.04, 0.04, size=2).astype(np.float32)
        colour = palette[rng.integers(0, len(palette))]
        shape = int(rng.integers(0, 3))
        r = float(rng.uniform(0.12, 0.18) * (rng.choice([0.7, 1.4]) if ood else 1.0))
        for t in range(frames):
            states[i, t] = np.concatenate([pos, vel])
            out[i, t] = _render_state(np.clip(pos, r, 1 - r), size, colour, shape, r)
            if t < frames - 1:
                a = rng.uniform(-1, 1, size=2).astype(np.float32)
                acts[i, t] = a
                vel = np.clip(vel + 0.02 * a, -0.08, 0.08)
                pos = pos + vel
                for k in range(2):
                    if pos[k] < r:
                        pos[k] = 2 * r - pos[k]
                        vel[k] = -vel[k]
                    if pos[k] > 1 - r:
                        pos[k] = 2 * (1 - r) - pos[k]
                        vel[k] = -vel[k]
    return out, acts, states


def step_state(state: np.ndarray, action: np.ndarray, r: float = 0.15):
    x, y, vx, vy = state
    vx = float(np.clip(vx + 0.02 * action[0], -0.08, 0.08))
    vy = float(np.clip(vy + 0.02 * action[1], -0.08, 0.08))
    x = x + vx
    y = y + vy
    if x < r:
        x = 2 * r - x
        vx = -vx
    if x > 1 - r:
        x = 2 * (1 - r) - x
        vx = -vx
    if y < r:
        y = 2 * r - y
        vy = -vy
    if y > 1 - r:
        y = 2 * (1 - r) - y
        vy = -vy
    return np.array([x, y, vx, vy], dtype=np.float32)


def render_state(state: np.ndarray, size: int = 32, colour: np.ndarray | None = None,
                 shape: int = 0, r: float = 0.15) -> np.ndarray:
    if colour is None:
        colour = PALETTE[1]
    return _render_state(np.clip(state[:2], r, 1 - r), size, colour, shape, r)
