"""Render the lattice figures that R/ggplot does badly.

The R pipeline draws every animation frame with ggplot and stitches it with
magick, which is fine for point clouds but wastes resolution on rasters.  Here
the lattice *is* the image: each cell is a solid block of pixels, so a 40 x 40
window can be redrawn at any size without anti-aliasing mush.

Outputs (written to ../animations/):
    gol.gif        Conway's Game of Life, r-pentomino, window follows the object
    gol-pair.gif   glider (periodic) next to r-pentomino (never repeats)
    eca.gif        rule 30 and rule 110 grown row by row
    eca-atlas.png  all 256 elementary rules, one contact sheet

Run from the post directory:  python python/render_animations.py
"""

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "animations"
OUT.mkdir(exist_ok=True)

INK = (31, 78, 121)        # deep blue, matches em_pal$deep_blue
PAPER = (255, 255, 255)
GREY = (102, 102, 102)
RED = (192, 80, 77)        # deep red, matches em_pal$deep_red


def load_font(size):
    for name in ("arial.ttf", "Arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


# ---------------------------------------------------------------------------
# Game of Life on a torus
# ---------------------------------------------------------------------------
def life_step(m):
    n = 0
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            n = n + np.roll(np.roll(m, dr, axis=0), dc, axis=1)
    return ((m == 0) & (n == 3)) | ((m == 1) & ((n == 2) | (n == 3)))


def window(m, cr, cc, side):
    """Square window centred on (cr, cc) with torus wrap-around."""
    size = m.shape[0]
    r0 = (np.arange(side) + int(round(cr)) - side // 2) % size
    c0 = (np.arange(side) + int(round(cc)) - side // 2) % size
    return m[np.ix_(r0, c0)]


def draw_lattice(dr, win, x0, y0, cell, colour=INK):
    """Paint the live cells as solid blocks; a 1 px white gap keeps the grid."""
    for i, j in zip(*np.nonzero(win)):
        xa, ya = x0 + j * cell, y0 + i * cell
        dr.rectangle([xa, ya, xa + cell - 2, ya + cell - 2], fill=colour)


def life_frames(pattern, gens, step, side, size=140, smooth=0.15, cell=8):
    """Yield (generation, live cells, window) with a smoothed tracked centre."""
    m = np.zeros((size, size), dtype=np.uint8)
    r0, c0 = size // 2, size // 2
    for i, row in enumerate(pattern):
        for j, ch in enumerate(row):
            m[r0 + i, c0 + j] = int(ch)
    cr, cc = float(r0), float(c0)
    for t in range(1, gens + 1):
        m = life_step(m).astype(np.uint8)
        if t % step:
            continue
        ys, xs = np.nonzero(m)
        if len(ys):   # follow the bounding-box centre: the whole object stays in frame
            cr = (1 - smooth) * cr + smooth * (ys.min() + ys.max()) / 2
            cc = (1 - smooth) * cc + smooth * (xs.min() + xs.max()) / 2
        yield t, int(m.sum()), window(m, cr, cc, side)


def render_life(cell=8, gens=160, step=2, side=54):
    """One window on the r-pentomino: five cells grow into a churning blob."""
    frames = []
    font = load_font(15)
    for t, live, win in life_frames(["011", "110", "010"], gens, step, side):
        img = Image.new("RGB", (side * cell, side * cell + 30), PAPER)
        dr = ImageDraw.Draw(img)
        draw_lattice(dr, win, 0, 30, cell)
        dr.text((8, 6), f"generation {t}      live cells {live}", font=font, fill=GREY)
        frames.append(img)
    frames[0].save(OUT / "gol.gif", save_all=True, append_images=frames[1:],
                   duration=90, loop=0, optimize=True)
    print(f"gol.gif: {len(frames)} frames, {frames[0].width}x{frames[0].height}")


def render_pair(cell=5, gens=120, step=2, side=52):
    """Glider versus r-pentomino: same rule, same window, different fate.

    Both windows follow the object, so the only thing that changes between the
    two panels is the shape of the state, not its position.
    """
    panels = [
        ("glider", ["010", "001", "111"], "5 cells, period 4"),
        ("r-pentomino", ["011", "110", "010"], "5 cells, never repeats"),
    ]
    gens_left = list(life_frames(panels[0][1], gens, step, side,
                                 size=160, smooth=0.30))
    gens_right = list(life_frames(panels[1][1], gens, step, side,
                                  size=200, smooth=0.10))
    font = load_font(15)
    small = load_font(13)
    wp = side * cell
    gap = 34
    width = 2 * wp + gap
    height = wp + 46 + 26
    frames = []
    for (t, live_a, wa), (_, live_b, wb) in zip(gens_left, gens_right):
        img = Image.new("RGB", (width, height), PAPER)
        dr = ImageDraw.Draw(img)
        draw_lattice(dr, wa, 0, 46, cell)
        draw_lattice(dr, wb, wp + gap, 46, cell)
        dr.text((0, 6), "A  glider", font=font, fill=INK)
        dr.text((0, 26), f"{panels[0][2]}   live {live_a}", font=small, fill=GREY)
        dr.text((wp + gap, 6), "B  r-pentomino", font=font, fill=RED)
        dr.text((wp + gap, 26), f"{panels[1][2]}   live {live_b}",
                font=small, fill=GREY)
        dr.text((0, 46 + wp + 5), f"generation {t}", font=small, fill=GREY)
        dr.text((wp + gap, 46 + wp + 5), f"generation {t}", font=small, fill=GREY)
        frames.append(img)
    frames[0].save(OUT / "gol-pair.gif", save_all=True, append_images=frames[1:],
                   duration=90, loop=0, optimize=True)
    print(f"gol-pair.gif: {len(frames)} frames, {frames[0].width}x{frames[0].height}")


# ---------------------------------------------------------------------------
# Elementary CA
# ---------------------------------------------------------------------------
def eca_rows(rule, n=201, steps=200, init="single", seed=1):
    """Open boundaries: a classic Wolfram triangle, no wrap-around copies."""
    bits = np.array([(rule >> k) & 1 for k in range(8)], dtype=np.uint8)
    out = np.zeros((steps, n), dtype=np.uint8)
    if init == "single":
        out[0, n // 2] = 1
    else:
        rng = np.random.default_rng(seed)
        out[0] = (rng.random(n) < 0.5).astype(np.uint8)
    for t in range(1, steps):
        prev = out[t - 1]
        left = np.concatenate([[0], prev[:-1]])
        right = np.concatenate([prev[1:], [0]])
        out[t] = bits[left * 4 + prev * 2 + right]
    return out


def render_eca(cell=3, step=2):
    rules = [30, 110]
    imgs = [eca_rows(r) for r in rules]
    steps, n = imgs[0].shape
    font = load_font(18)
    frames = []
    for t in range(2, steps + 1, step):
        img = Image.new("RGB", (2 * n * cell + 30, steps * cell + 34), PAPER)
        dr = ImageDraw.Draw(img)
        for k, (rule, arr) in enumerate(zip(rules, imgs)):
            x0 = 10 + k * (n * cell + 10)
            dr.text((x0, 4), f"rule {rule}", font=font, fill=GREY)
            for i in range(t):
                for j in np.nonzero(arr[i])[0]:
                    dr.rectangle([x0 + j * cell, 28 + i * cell,
                                  x0 + j * cell + cell - 1,
                                  28 + i * cell + cell - 1], fill=INK)
        frames.append(img)
    frames[0].save(OUT / "eca.gif", save_all=True, append_images=frames[1:],
                   duration=60, loop=0, optimize=True)
    print(f"eca.gif: {len(frames)} frames, {frames[0].width}x{frames[0].height}")


def render_atlas(cell=2, n=52, steps=52, cols=16, gap=7):
    """All 256 rules from the same random row: a portrait of the rule space."""
    font = load_font(11)
    pw, ph = n * cell, steps * cell
    rows = (256 + cols - 1) // cols
    width = cols * (pw + gap) + gap
    height = rows * (ph + gap + 14) + gap
    img = Image.new("RGB", (width, height), PAPER)
    dr = ImageDraw.Draw(img)
    for rule in range(256):
        r, c = divmod(rule, cols)
        x0 = gap + c * (pw + gap)
        y0 = gap + r * (ph + gap + 14)
        arr = eca_rows(rule, n=n, steps=steps, init="random", seed=7)
        block = np.kron(arr, np.ones((cell, cell), dtype=np.uint8))
        panel = Image.fromarray(
            np.where(block[..., None] > 0, np.array(INK), np.array(PAPER)).astype(np.uint8))
        img.paste(panel, (x0, y0))
        dr.text((x0, y0 + ph + 2), f"{rule}", font=font, fill=GREY)
    img.save(OUT / "eca-atlas.png", optimize=True)
    print(f"eca-atlas.png: {width}x{height}")


if __name__ == "__main__":
    render_life()
    render_pair()
    render_eca()
    render_atlas()
