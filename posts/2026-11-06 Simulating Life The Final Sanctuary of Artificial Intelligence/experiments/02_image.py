"""Experiment 2 — what a generative image model actually learns.

A small VAE is trained on Fashion-MNIST with a two-dimensional latent space so
the geometry can be inspected directly.  The experiment has three parts:

  1. does the model reconstruct and sample plausible images?
  2. is the latent space organised (class clusters, smooth interpolation)?
  3. what happens to inputs that are *visually* close but structurally off
     (inverted, translated, pure noise)?

Run:  python experiments/02_image.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common.style import PAL, save, set_style
from common.utils import DATA, save_metrics, set_seed, timer

LATENT = 2
CLASSES = ["T-shirt", "Trouser", "Pullover", "Dress", "Coat",
           "Sandal", "Shirt", "Sneaker", "Bag", "Boot"]
DATASET_NAME = "Fashion-MNIST"


class VAE(nn.Module):
    def __init__(self, latent=LATENT, hidden=256):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(784, hidden), nn.ReLU())
        self.mu = nn.Linear(hidden, latent)
        self.logvar = nn.Linear(hidden, latent)
        self.dec = nn.Sequential(nn.Linear(latent, hidden), nn.ReLU(),
                                 nn.Linear(hidden, 784))

    def encode(self, x):
        h = self.enc(x)
        return self.mu(h), self.logvar(h)

    def decode(self, z):
        return torch.sigmoid(self.dec(z))

    def forward(self, x):
        mu, logvar = self.encode(x)
        std = torch.exp(0.5 * logvar)
        z = mu + std * torch.randn_like(std)
        return self.decode(z), mu, logvar, z


def loss_fn(recon, x, mu, logvar, beta=1.0):
    bce = nn.functional.binary_cross_entropy(recon, x, reduction="none").sum(1).mean()
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
    return bce + beta * kld, bce, kld


def load_digits_fallback(n_train, n_test):
    """Offline substitute when torchvision or the Fashion-MNIST download fails.

    sklearn's 8x8 digits are upsampled to 28x28 so the same VAE runs unchanged.
    The figure labels and the text of the article change with it, so this path
    is documented as a fallback rather than an equivalent dataset.
    """
    global CLASSES, DATASET_NAME
    from sklearn.datasets import load_digits

    d = load_digits()
    x = torch.tensor(d.images.reshape(-1, 1, 8, 8), dtype=torch.float32) / 16.0
    x = torch.nn.functional.interpolate(x, size=(28, 28), mode="bilinear",
                                        align_corners=False)
    y = torch.tensor(d.target, dtype=torch.long)
    g = torch.Generator().manual_seed(0)
    perm = torch.randperm(len(x), generator=g)
    n_tr = min(n_train, int(0.8 * len(x)))
    n_te = min(n_test, len(x) - n_tr)
    CLASSES = [str(i) for i in range(10)]
    DATASET_NAME = "sklearn digits (8x8, upsampled)"
    print(f"[warn] using offline fallback: {DATASET_NAME}")
    return (TensorDataset(x[perm[:n_tr]], y[perm[:n_tr]]),
            TensorDataset(x[perm[n_tr:n_tr + n_te]], y[perm[n_tr:n_tr + n_te]]))


def load_data(n_train=20000, n_test=5000):
    try:
        from torchvision import datasets, transforms

        tf = transforms.ToTensor()
        tr = datasets.FashionMNIST(DATA, train=True, download=True, transform=tf)
        te = datasets.FashionMNIST(DATA, train=False, download=True, transform=tf)
    except Exception as exc:                    # no torchvision, or no network
        print(f"[warn] Fashion-MNIST unavailable ({type(exc).__name__}: {exc})")
        return load_digits_fallback(n_train, n_test)
    g = torch.Generator().manual_seed(0)
    tr = Subset(tr, torch.randperm(len(tr), generator=g)[:n_train].tolist())
    te_idx = torch.randperm(len(te), generator=g)[:n_test].tolist()
    return tr, Subset(te, te_idx)


def flatten(loader, limit=None):
    xs, ys = [], []
    for x, y in loader:
        xs.append(x)
        ys.append(y)
        if limit and sum(len(a) for a in xs) >= limit:
            break
    return torch.cat(xs)[:limit], torch.cat(ys)[:limit]


@torch.no_grad()
def recon_error(model, x):
    mu, _ = model.encode(x.view(-1, 784))
    rec = model.decode(mu)
    return ((rec - x.view(-1, 784)) ** 2).mean(1)


def main():
    set_seed(0)
    set_style()
    torch.set_num_threads(4)

    train_set, test_set = load_data()
    dl_train = DataLoader(train_set, batch_size=128, shuffle=True, drop_last=True)
    dl_test = DataLoader(test_set, batch_size=512)
    x_test, y_test = flatten(dl_test, 5000)

    model = VAE()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    hist = {"train": [], "val": []}
    epochs = 15
    with timer("train VAE"):
        for ep in range(epochs):
            model.train()
            run = 0.0
            for x, _ in dl_train:
                x = x.view(-1, 784)
                rec, mu, logvar, _ = model(x)
                loss, _, _ = loss_fn(rec, x, mu, logvar)
                opt.zero_grad()
                loss.backward()
                opt.step()
                run += loss.item() * len(x)
            # validation ELBO with the mean (no sampling noise)
            model.eval()
            tot = 0.0
            with torch.no_grad():
                for x, _ in dl_test:
                    x = x.view(-1, 784)
                    mu, logvar = model.encode(x)
                    rec = model.decode(mu)
                    l, _, _ = loss_fn(rec, x, mu, logvar)
                    tot += l.item() * len(x)
            hist["train"].append(run / len(train_set))
            hist["val"].append(tot / len(test_set))
            print(f"epoch {ep + 1:2d}  train {hist['train'][-1]:.1f}  val {hist['val'][-1]:.1f}")

    # ---- reconstruction quality -------------------------------------------
    model.eval()
    with torch.no_grad():
        mu, _ = model.encode(x_test.view(-1, 784))
        rec = model.decode(mu).view(-1, 1, 28, 28)
        z_prior = torch.randn(12, LATENT)
        samples = model.decode(z_prior).view(-1, 1, 28, 28)

    mse_in = float(((rec.view(-1, 784) - x_test.view(-1, 784)) ** 2).mean())

    # ---- out-of-distribution probes ---------------------------------------
    probes = {
        "in distribution": x_test,
        "vertically flipped": torch.flip(x_test, dims=[2]),
        "shifted by 4 px": torch.roll(x_test, shifts=4, dims=3),
        "inverted": 1.0 - x_test,
        "uniform noise": torch.rand_like(x_test),
    }
    ood = {}
    with torch.no_grad():
        for name, xp in probes.items():
            err = recon_error(model, xp)
            ood[name] = {"mse": float(err.mean()), "sd": float(err.std())}
            print(f"  {name:20s} recon MSE {ood[name]['mse']:.4f}")

    # ---- latent geometry ---------------------------------------------------
    with torch.no_grad():
        z = model.encode(x_test.view(-1, 784))[0].numpy()

    # interpolation between two test images of the same class
    idx_a = int(torch.where(y_test == 5)[0][0])          # sandal
    idx_b = int(torch.where(y_test == 5)[0][1])
    with torch.no_grad():
        za = model.encode(x_test[idx_a].view(1, 784))[0].numpy()[0]
        zb = model.encode(x_test[idx_b].view(1, 784))[0].numpy()[0]
        ts = np.linspace(0, 1, 9)
        zs = torch.tensor(np.stack([(1 - t) * za + t * zb for t in ts]),
                          dtype=torch.float32)
        interp = model.decode(zs).view(-1, 28, 28).numpy()

    save_metrics(
        "image",
        {
            "epochs": epochs,
            "latent_dim": LATENT,
            "n_train": len(train_set),
            "train_loss": [round(v, 1) for v in hist["train"]],
            "val_loss": [round(v, 1) for v in hist["val"]],
            "final_val_elbo": round(hist["val"][-1], 1),
            "recon_mse_test": round(mse_in, 5),
            "ood": {k: {kk: round(vv, 5) for kk, vv in v.items()} for k, v in ood.items()},
            "latent_mean_abs": float(np.mean(np.abs(z))),
            "latent_sd": float(np.std(z)),
        },
    )

    # ---- figures -----------------------------------------------------------
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.0),
                             gridspec_kw={"width_ratios": [1, 1.05]})
    ax = axes[0]
    ax.plot(range(1, epochs + 1), hist["train"], color=PAL["blue"], label="train")
    ax.plot(range(1, epochs + 1), hist["val"], color=PAL["green"], label="held-out")
    ax.set_xlabel("epoch")
    ax.set_ylabel("negative ELBO (nats/image)")
    ax.set_title("(a) variational bound")
    ax.legend(loc="upper right")

    ax = axes[1]
    names = list(probes)
    means = [ood[n]["mse"] for n in names]
    sds = [ood[n]["sd"] for n in names]
    colors = [PAL["blue"]] + [PAL["sky"]] * 3 + [PAL["vermillion"]]
    ax.barh(range(len(names)), means, xerr=sds, color=colors,
            error_kw={"lw": 0.8, "ecolor": PAL["grey"]}, height=0.62)
    ax.set_yticks(range(len(names)), names, fontsize=7.5)
    ax.invert_yaxis()
    ax.set_xlabel("reconstruction MSE (mean $\\pm$ sd)")
    ax.set_title("(b) what counts as out of distribution")
    fig.tight_layout()
    save(fig, "fig-image-ood")

    # reconstruction / sample grid
    fig, axes = plt.subplots(4, 8, figsize=(8.4, 4.6))
    for j in range(8):
        axes[0, j].imshow(x_test[j, 0], cmap="gray_r", vmin=0, vmax=1)
        axes[1, j].imshow(rec[j, 0], cmap="gray_r", vmin=0, vmax=1)
        axes[2, j].imshow(x_test[j + 8, 0], cmap="gray_r", vmin=0, vmax=1)
        axes[3, j].imshow(samples[j, 0], cmap="gray_r", vmin=0, vmax=1)
        for i in range(4):
            axes[i, j].axis("off")
    for i, lab in enumerate(["input", "reconstruction", "input", "sample from prior"]):
        axes[i, 0].set_ylabel(lab, rotation=90, fontsize=7, labelpad=12)
    fig.text(0.5, 0.99, f"{DATASET_NAME}, 2-D latent space", ha="center", fontsize=10)
    fig.text(0.5, 0.01,
             f"test reconstruction MSE {mse_in:.4f};  28 x 28 pixels",
             ha="center", fontsize=8, color=PAL["grey"])
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    save(fig, "fig-image-recon")

    # latent space + interpolation
    fig, axes = plt.subplots(2, 1, figsize=(7.0, 5.6),
                             gridspec_kw={"height_ratios": [2.3, 1]})
    ax = axes[0]
    palette = [PAL["blue"], PAL["green"], PAL["sky"], PAL["orange"], PAL["purple"],
               PAL["vermillion"], "#7f7f7f", "#17becf", "#bcbd22", "#e377c2"]
    rng = np.random.default_rng(0)
    sub = rng.choice(len(z), 3000, replace=False)
    for c in range(10):
        m = y_test.numpy()[sub] == c
        ax.scatter(z[sub][m, 0], z[sub][m, 1], s=3, color=palette[c],
                   alpha=0.55, lw=0, label=CLASSES[c])
    ax.set_xlabel("latent $z_1$")
    ax.set_ylabel("latent $z_2$")
    ax.set_title("(a) the latent space is organised by garment type")
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=7,
              markerscale=2.5, handletextpad=0.2)

    ax = axes[1]
    strip = np.concatenate([np.repeat(np.repeat(interp[i], 2, 0), 2, 1)
                            for i in range(len(ts))], axis=1)
    ax.imshow(strip, cmap="gray_r", vmin=0, vmax=1)
    ax.set_xticks([56 * i + 27 for i in range(len(ts))], [f"{t:.1f}" for t in ts],
                  fontsize=7)
    ax.set_yticks([])
    ax.set_xlabel(f"interpolation coefficient between two {CLASSES[5].lower()}s")
    ax.set_title("(b) decoding a straight line in latent space")
    fig.tight_layout()
    save(fig, "fig-image-latent")


if __name__ == "__main__":
    main()
