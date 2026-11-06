"""Experiment 5 — from observation to intervention.

The simulator from experiment 4 is the ground truth.  Four models are asked to
predict what happens after a gene is knocked out:

  A  persistence        x(t+1) = x(t)
  B  MLP                the same inputs, no knowledge of the network
  C  graph network      message passing over the true adjacency
  D  mechanistic fit    the ODE family itself, with a handful of free parameters

Everything is trained on single knockouts of genes 0-4 under two environments.
The tests move outwards: unseen genes, double knockouts, unseen environments,
and long horizons.  A final negative experiment removes 30% of the edges from
the graph model to see what the structure was actually worth.

Run:  python experiments/05_perturbation.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common.grn import C_TRUE, N, W_TRUE, hill, rhs, step
from common.style import PAL, save, set_style
from common.utils import save_metrics, set_seed, timer

DT = 0.05
HORIZON_TRAIN = 20
TRAIN_ENVS = [0.0, 1.0]
TEST_ENVS = [0.0, 1.0, 0.5, 1.5]
TRAIN_KO_GENES = [0, 1, 2]
HELD_OUT_GENES = [3, 4]
DOUBLE_KO_GENES = [0, 1, 2, 3, 4]


# ---------------------------------------------------------------------------
# 1. Data
# ---------------------------------------------------------------------------
def simulate_traj(x0, u, ko_mask, steps):
    xs = np.empty((steps + 1, N))
    xs[0] = x0
    x = x0.copy()
    for i in range(steps):
        x = step(x, u, ko_mask)
        xs[i + 1] = x
    return xs


def make_scenarios(n, envs, ko_pool, steps=HORIZON_TRAIN, seed=0, n_ko=1):
    """Random initial states near attractors, random knockouts from ko_pool."""
    rng = np.random.default_rng(seed)
    from common.grn import find_attractors
    atts = find_attractors(n_starts=120, steps=900, u=0.0, seed=11)
    X, KO, U, Y = [], [], [], []
    for _ in range(n):
        a = atts[rng.integers(len(atts))]
        x0 = np.maximum(0.0, a + rng.normal(0, 0.12, N))
        ko = np.zeros(N, dtype=bool)
        if n_ko > 0:
            genes = rng.choice(ko_pool, size=n_ko, replace=False)
            ko[list(genes)] = True
        u = float(rng.choice(envs))
        traj = simulate_traj(x0, u, ko, steps)
        X.append(x0)
        KO.append(ko)
        U.append(u)
        Y.append(traj)
    return np.array(X), np.array(KO), np.array(U), np.array(Y)


def tensors(X, KO, U):
    return (torch.tensor(X, dtype=torch.float32),
            torch.tensor(KO, dtype=torch.float32),
            torch.tensor(U, dtype=torch.float32).unsqueeze(1))


# ---------------------------------------------------------------------------
# 2. Models
# ---------------------------------------------------------------------------
class Persistence(nn.Module):
    def step(self, x, ko, u):
        return x


class MLP(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(N + N + 1, hidden), nn.Tanh(),
                                 nn.Linear(hidden, hidden), nn.Tanh(),
                                 nn.Linear(hidden, N))

    def step(self, x, ko, u):
        inp = torch.cat([x, ko, u], dim=-1)
        return torch.clamp(x + self.net(inp), min=0.0)


class GraphNet(nn.Module):
    """Shared per-gene update that reads the neighbourhood through W."""

    def __init__(self, W, hidden=32):
        super().__init__()
        self.register_buffer("W", torch.tensor(W, dtype=torch.float32))
        self.register_buffer("Wabs", torch.tensor(np.abs(W), dtype=torch.float32))
        self.net = nn.Sequential(nn.Linear(5, hidden), nn.Tanh(),
                                 nn.Linear(hidden, hidden), nn.Tanh(),
                                 nn.Linear(hidden, 1))

    def step(self, x, ko, u):
        drive = x @ self.W.T
        drive_abs = x @ self.Wabs.T
        feats = torch.stack([x, drive, drive_abs, ko, u.expand(-1, N)], dim=-1)
        delta = self.net(feats).squeeze(-1)
        return torch.clamp(x + delta, min=0.0)


class Mechanistic(nn.Module):
    """The ODE family with fitted constants; W is fixed to the true graph."""

    def __init__(self, W):
        super().__init__()
        self.register_buffer("W", torch.tensor(W, dtype=torch.float32))
        self.b = nn.Parameter(torch.full((N,), 0.2))
        self.c = nn.Parameter(torch.zeros(N))
        self.log_k = nn.Parameter(torch.tensor(0.0))
        self.w_scale = nn.Parameter(torch.tensor(1.0))
        self.deg = nn.Parameter(torch.tensor(1.0))

    def step(self, x, ko, u):
        k = torch.exp(self.log_k)
        h = x ** 2 / (k ** 2 + x ** 2)
        prod = self.w_scale * (h @ self.W.T) + self.b + u * self.c
        prod = torch.where(ko > 0, torch.zeros_like(prod), prod)
        return torch.clamp(x + DT * (-self.deg * x + prod), min=0.0)


def rollout(model, x0, ko, u, steps):
    x = x0
    for _ in range(steps):
        x = model.step(x, ko, u)
    return x


def train_model(model, X, KO, U, Y, epochs=30, bs=64, horizon=6, lr=2e-3, seed=0):
    torch.manual_seed(seed)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=lr)
    x_t, ko_t, u_t = tensors(X, KO, U)
    y_t = torch.tensor(Y[:, 1:], dtype=torch.float32)
    loss_hist = []
    for ep in range(epochs):
        perm = torch.randperm(len(X))
        run = 0.0
        for i in range(0, len(X), bs):
            idx = perm[i:i + bs]
            x = x_t[idx]
            ko, u = ko_t[idx], u_t[idx]
            loss = 0.0
            for h in range(horizon):
                x = model.step(x, ko, u)
                loss = loss + ((x - y_t[idx, h]) ** 2).mean()
            loss = loss / horizon
            opt.zero_grad()
            loss.backward()
            opt.step()
            run += loss.item() * len(idx)
        loss_hist.append(run / len(X))
    return loss_hist


@torch.no_grad()
def predict(model, X, KO, U, steps):
    x = torch.tensor(X, dtype=torch.float32)
    ko = torch.tensor(KO, dtype=torch.float32)
    u = torch.tensor(U, dtype=torch.float32).unsqueeze(1)
    return rollout(model, x, ko, u, steps).numpy()


def main():
    set_seed(0)
    set_style()
    torch.set_num_threads(4)

    with timer("simulate training set"):
        Xtr, KOtr, Utr, Ytr = make_scenarios(3000, TRAIN_ENVS, TRAIN_KO_GENES,
                                             seed=1, n_ko=1)
        Xva, KOva, Uva, Yva = make_scenarios(500, TRAIN_ENVS, TRAIN_KO_GENES,
                                             seed=2, n_ko=1)
    var = float(np.var(Ytr))
    print(f"training scenarios {len(Xtr)}, response variance {var:.3f}")

    models = {
        "persistence": Persistence(),
        "MLP": MLP(),
        "graph net": GraphNet(W_TRUE),
        "mechanistic": Mechanistic(W_TRUE),
    }
    hist = {}
    with timer("train four models"):
        for name, m in models.items():
            if name == "persistence":
                hist[name] = [0.0]
                continue
            hist[name] = train_model(m, Xtr, KOtr, Utr, Ytr)
            print(f"  {name:12s} final rollout loss {hist[name][-1]:.5f}")

    # ---- test scenarios ---------------------------------------------------
    tests = {}
    tests["normal, seen env"] = make_scenarios(300, TRAIN_ENVS, TRAIN_KO_GENES,
                                               seed=3, n_ko=0)
    tests["held-out genes"] = make_scenarios(300, TRAIN_ENVS, HELD_OUT_GENES,
                                             seed=4, n_ko=1)
    tests["double knockout"] = make_scenarios(300, TRAIN_ENVS, DOUBLE_KO_GENES,
                                              seed=5, n_ko=2)
    tests["unseen env 0.5"] = make_scenarios(200, [0.5], TRAIN_KO_GENES, seed=6, n_ko=1)
    tests["unseen env 1.5"] = make_scenarios(200, [1.5], TRAIN_KO_GENES, seed=7, n_ko=1)

    horizon_eval = HORIZON_TRAIN
    results = {}
    for tname, (Xt, KOt, Ut, Yt) in tests.items():
        row = {}
        for mname, model in models.items():
            pred = predict(model, Xt, KOt, Ut, horizon_eval)
            true = Yt[:, -1]
            mse = float(np.mean((pred - true) ** 2) / var)
            # direction accuracy: does each gene move the right way?
            delta_true = true - Xt
            delta_pred = pred - Xt
            mask = np.abs(delta_true) > 0.1
            agree = (np.sign(delta_true[mask]) == np.sign(delta_pred[mask]))
            row[mname] = {"mse": mse, "dir": float(agree.mean()) if mask.any() else np.nan}
        results[tname] = row
        print(f"{tname:18s} " + "  ".join(f"{k} {v['mse']:.3f}" for k, v in row.items()))

    # ---- long horizon -----------------------------------------------------
    Xl, KOl, Ul, Yl = make_scenarios(150, TRAIN_ENVS, TRAIN_KO_GENES, steps=200,
                                     seed=8, n_ko=1)
    long_err = {}
    for mname, model in models.items():
        pred = predict(model, Xl, KOl, Ul, 200)
        long_err[mname] = float(np.mean((pred - Yl[:, -1]) ** 2) / var)

    # ---- negative experiment: damage the graph ----------------------------
    rng = np.random.default_rng(0)
    W_bad = W_TRUE.copy()
    nz = np.argwhere(np.abs(W_bad) > 0)
    drop = rng.choice(len(nz), int(0.3 * len(nz)), replace=False)
    for k in drop:
        i, j = nz[k]
        W_bad[i, j] = 0.0
    damaged = GraphNet(W_bad)
    train_model(damaged, Xtr, KOtr, Utr, Ytr)
    Xd, KOd, Ud, Yd = tests["double knockout"]
    pred_bad = predict(damaged, Xd, KOd, Ud, horizon_eval)
    bad_mse = float(np.mean((pred_bad - Yd[:, -1]) ** 2) / var)
    print(f"graph with 30% of edges removed, double KO MSE {bad_mse:.3f}")

    save_metrics(
        "perturbation",
        {
            "n_train": len(Xtr),
            "response_variance": round(var, 4),
            "horizon": horizon_eval,
            "test": {t: {m: {"mse": round(v["mse"], 4),
                             "dir": None if np.isnan(v["dir"]) else round(v["dir"], 3)}
                         for m, v in row.items()} for t, row in results.items()},
            "long_horizon": {k: round(v, 4) for k, v in long_err.items()},
            "damaged_graph_mse": round(bad_mse, 4),
            "train_loss": {k: round(v[-1], 5) for k, v in hist.items() if k != "persistence"},
            "n_edges": int(np.count_nonzero(W_TRUE)),
            "n_edges_dropped": int(len(drop)),
        },
    )

    # ---- figures ----------------------------------------------------------
    import matplotlib.pyplot as plt

    order = list(models)
    colors = {"persistence": PAL["grey"], "MLP": PAL["vermillion"],
              "graph net": PAL["blue"], "mechanistic": PAL["green"]}
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.4))
    ax = axes[0]
    tnames = list(tests)
    xpos = np.arange(len(tnames))
    width = 0.2
    for j, m in enumerate(order):
        vals = [results[t][m]["mse"] for t in tnames]
        ax.bar(xpos + (j - 1.5) * width, vals, width=width, color=colors[m], label=m)
    ax.set_yscale("log")
    ax.set_xticks(xpos, [t.replace(", ", ",\n") for t in tnames], fontsize=6.5)
    ax.set_ylabel("MSE / response variance")
    ax.set_title("(a) 20-step prediction error")
    ax.legend(loc="upper left", fontsize=6.5, ncol=2)

    ax = axes[1]
    for j, m in enumerate(order):
        vals = [results[t][m]["dir"] for t in tnames]
        ax.bar(xpos + (j - 1.5) * width, vals, width=width, color=colors[m])
    ax.set_xticks(xpos, [t.replace(", ", ",\n") for t in tnames], fontsize=6.5)
    ax.axhline(0.5, color=PAL["grey"], ls=":", lw=0.8)
    ax.set_ylabel("direction agreement")
    ax.set_title("(b) does each gene move the right way?")
    fig.tight_layout()
    save(fig, "fig-cell-perturbation")

    # example: one double knockout, true vs predicted
    Xd, KOd, Ud, Yd = tests["double knockout"]
    pred_mlp = predict(models["MLP"], Xd, KOd, Ud, horizon_eval)
    pred_gr = predict(models["graph net"], Xd, KOd, Ud, horizon_eval)
    idx = int(np.argmax(np.abs(Yd[:, -1] - Xd).sum(1)))
    fig, ax = plt.subplots(figsize=(7.0, 3.2))
    xx = np.arange(N)
    ax.bar(xx - 0.26, Yd[idx, -1], width=0.26, color=PAL["dark"], label="truth")
    ax.bar(xx, pred_gr[idx], width=0.26, color=PAL["blue"], label="graph net")
    ax.bar(xx + 0.26, pred_mlp[idx], width=0.26, color=PAL["vermillion"], label="MLP")
    ko_genes = np.where(KOd[idx])[0]
    ax.set_xticks(xx, [f"g{i}" for i in range(N)], fontsize=7)
    ax.set_ylabel("expression after 20 steps")
    ax.set_title("Double knockout of " + " and ".join(f"g{i}" for i in ko_genes))
    ax.legend(loc="upper right", fontsize=7)
    fig.tight_layout()
    save(fig, "fig-cell-ko")

    # negative result
    fig, ax = plt.subplots(figsize=(5.4, 3.2))
    labels = ["graph net\n(full)", "graph net\n(70% of edges)", "MLP"]
    vals = [results["double knockout"]["graph net"]["mse"], bad_mse,
            results["double knockout"]["MLP"]["mse"]]
    ax.bar(labels, vals, color=[PAL["blue"], PAL["sky"], PAL["vermillion"]], width=0.6)
    for i, v in enumerate(vals):
        ax.text(i, v * 1.02, f"{v:.3f}", ha="center", fontsize=8)
    ax.set_ylabel("MSE / response variance")
    ax.set_title("Structure is only worth what the structure is worth")
    fig.tight_layout()
    save(fig, "fig-cell-negative")


if __name__ == "__main__":
    main()
