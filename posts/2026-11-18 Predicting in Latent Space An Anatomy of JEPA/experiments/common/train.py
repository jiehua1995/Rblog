"""Training loops, probes and collapse diagnostics."""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from .models import (JEPA, MAE, SimCLR, Supervised, WorldModelJEPA,
                     WorldModelPixel, block_mask, nt_xent, update_ema,
                     variance_loss)


def loader(images, labels=None, batch=128, shuffle=True):
    if isinstance(images, tuple):
        ds = TensorDataset(*[torch.from_numpy(a) for a in images])
    elif labels is None:
        ds = TensorDataset(torch.from_numpy(images))
    else:
        ds = TensorDataset(torch.from_numpy(images), torch.from_numpy(labels))
    return DataLoader(ds, batch_size=batch, shuffle=shuffle, drop_last=False)


def train_world_jepa(frames, actions, epochs=5, seed=0, batch=128, lr=1e-3,
                     ema=0.999):
    torch.manual_seed(seed)
    model = WorldModelJEPA()
    opt = torch.optim.AdamW(
        list(model.context.parameters()) + list(model.predictor.parameters()),
        lr=lr, weight_decay=0.05,
    )
    x = frames[:, :-1].reshape(-1, 3, 32, 32)
    y = frames[:, 1:].reshape(-1, 3, 32, 32)
    a = actions.reshape(-1, 2)
    hist = []
    for ep in range(epochs):
        model.train()
        total, n = 0.0, 0
        for xb, ab, yb in loader((x, a, y), batch=batch):
            loss, _ = model(xb, ab, yb)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(model.context.parameters()) + list(model.predictor.parameters()), 1.0)
            opt.step()
            update_ema(model.target, model.context, ema)
            total += loss.item() * xb.size(0)
            n += xb.size(0)
        hist.append(total / n)
    model.eval()
    return model, hist


def train_world_pixel(frames, actions, epochs=5, seed=0, batch=128, lr=1e-3):
    torch.manual_seed(seed)
    model = WorldModelPixel()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    x = frames[:, :-1].reshape(-1, 3, 32, 32)
    y = frames[:, 1:].reshape(-1, 3, 32, 32)
    a = actions.reshape(-1, 2)
    hist = []
    for ep in range(epochs):
        model.train()
        total, n = 0.0, 0
        for xb, ab, yb in loader((x, a, y), batch=batch):
            loss, _ = model(xb, ab, yb)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item() * xb.size(0)
            n += xb.size(0)
        hist.append(total / n)
    model.eval()
    return model, hist


@torch.no_grad()
def rollout_jepa(model, frames, actions, horizon=8):
    frames = torch.as_tensor(frames, dtype=torch.float32)
    actions = torch.as_tensor(actions, dtype=torch.float32)
    tokens = model.context(frames[:, 0], return_tokens=True)
    errs = []
    for t in range(horizon):
        pred = model.predictor(tokens, actions[:, t])
        tokens = torch.cat([tokens[:, :1], pred], dim=1)
        target = model.target(frames[:, t + 1], return_tokens=True)[:, 1:, :]
        errs.append(float(((pred - target) ** 2).mean()))
    return errs


@torch.no_grad()
def rollout_pixel(model, frames, actions, horizon=8):
    frames = torch.as_tensor(frames, dtype=torch.float32)
    actions = torch.as_tensor(actions, dtype=torch.float32)
    cur = frames[:, 0]
    errs = []
    for t in range(horizon):
        pred = model.predictor(model.encoder(cur, return_tokens=True), actions[:, t])
        pred = model.head(pred).reshape(-1, model.grid, model.grid, 3,
                                         model.patch, model.patch)
        pred = pred.permute(0, 3, 1, 4, 2, 5).reshape(-1, 3, 32, 32)
        errs.append(float(((pred - frames[:, t + 1]) ** 2).mean()))
        cur = pred
    return errs


@torch.no_grad()
def plan_latent(model, start_frame, goal_frame, horizon=5, pop=24, iters=3,
                elite=6):
    start_frame = torch.as_tensor(start_frame, dtype=torch.float32)
    goal_frame = torch.as_tensor(goal_frame, dtype=torch.float32)
    start = model.context(start_frame, return_tokens=True)
    goal = model.target(goal_frame, return_tokens=True)[:, 1:, :]
    mean = torch.zeros(horizon, 2)
    std = torch.ones(horizon, 2) * 0.8
    for _ in range(iters):
        acts = (mean + std * torch.randn(pop, horizon, 2)).clamp(-1, 1)
        tok = start.expand(pop, -1, -1)
        for t in range(horizon):
            pred = model.predictor(tok, acts[:, t])
            tok = torch.cat([tok[:, :1], pred], dim=1)
        cost = ((tok[:, 1:, :] - goal) ** 2).mean(dim=(1, 2))
        idx = torch.topk(cost, elite, largest=False).indices
        best = acts[idx]
        mean = best.mean(0)
        std = best.std(0) + 0.05
    return mean


def plan_pixel(model, start_frame, goal_frame, horizon=5, pop=24, iters=3,
               elite=6):
    start_frame = torch.as_tensor(start_frame, dtype=torch.float32)
    goal_frame = torch.as_tensor(goal_frame, dtype=torch.float32)
    mean = torch.zeros(horizon, 2)
    std = torch.ones(horizon, 2) * 0.8
    for _ in range(iters):
        acts = (mean + std * torch.randn(pop, horizon, 2)).clamp(-1, 1)
        cur = start_frame.expand(pop, -1, -1, -1)
        for t in range(horizon):
            pred = model.predictor(model.encoder(cur, return_tokens=True), acts[:, t])
            pred = model.head(pred).reshape(-1, model.grid, model.grid, 3,
                                             model.patch, model.patch)
            cur = pred.permute(0, 3, 1, 4, 2, 5).reshape(-1, 3, 32, 32)
        cost = ((cur - goal_frame) ** 2).mean(dim=(1, 2, 3))
        idx = torch.topk(cost, elite, largest=False).indices
        best = acts[idx]
        mean = best.mean(0)
        std = best.std(0) + 0.05
    return mean


def train_jepa(images, epochs=8, mask_ratio=0.5, ema=0.99, reg=0.0,
               predictor_depth=2, predictor_width=96, seed=0, batch=128,
               lr=1e-3, weight_decay=0.05, stop_grad=True):
    torch.manual_seed(seed)
    model = JEPA(predictor_depth=predictor_depth, predictor_width=predictor_width)
    if ema <= 0:
        model.target = model.context
    params = list(model.context.parameters()) + list(model.predictor.parameters())
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    hist = []
    for ep in range(epochs):
        model.train()
        total, n = 0.0, 0
        for (x,) in loader(images, batch=batch):
            mask = block_mask(x.size(0), model.context.grid, mask_ratio, x.device)
            loss, pred = model(x, mask, stop_grad=stop_grad)
            if reg > 0:
                loss = loss + reg * variance_loss(pred.reshape(-1, pred.size(-1)))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            if ema > 0:
                update_ema(model.target, model.context, ema)
            total += loss.item() * x.size(0)
            n += x.size(0)
        hist.append(total / n)
    model.eval()
    return model, hist


def train_mae(images, epochs=8, mask_ratio=0.5, seed=0, batch=128, lr=1e-3):
    torch.manual_seed(seed)
    model = MAE()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    hist = []
    for ep in range(epochs):
        model.train()
        total, n = 0.0, 0
        for (x,) in loader(images, batch=batch):
            mask = block_mask(x.size(0), model.grid, mask_ratio, x.device)
            loss = model(x, mask)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item() * x.size(0)
            n += x.size(0)
        hist.append(total / n)
    model.eval()
    return model, hist


def _augment(x):
    b = x.size(0)
    x = F.pad(x, (4, 4, 4, 4), mode="reflect")
    i = torch.randint(0, 9, (1,)).item()
    j = torch.randint(0, 9, (1,)).item()
    x = x[:, :, i:i + 32, j:j + 32]
    if torch.rand(1).item() < 0.5:
        x = torch.flip(x, dims=[3])
    scale = (0.8 + 0.4 * torch.rand(b, 1, 1, 1)).to(x)
    shift = (-0.1 + 0.2 * torch.rand(b, 1, 1, 1)).to(x)
    x = torch.clamp(x * scale + shift, 0, 1)
    return x


def train_simclr(images, epochs=8, seed=0, batch=128, lr=1e-3):
    torch.manual_seed(seed)
    model = SimCLR()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    hist = []
    for ep in range(epochs):
        model.train()
        total, n = 0.0, 0
        for (x,) in loader(images, batch=batch):
            z1 = model(_augment(x))
            z2 = model(_augment(x))
            loss = nt_xent(z1, z2)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item() * x.size(0)
            n += x.size(0)
        hist.append(total / n)
    model.eval()
    return model, hist


def train_supervised(images, labels, epochs=8, seed=0, batch=128, lr=1e-3):
    torch.manual_seed(seed)
    model = Supervised(n_classes=int(labels.max()) + 1)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    hist = []
    for ep in range(epochs):
        model.train()
        total, n = 0.0, 0
        for x, y in loader(images, labels, batch=batch):
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * x.size(0)
            n += x.size(0)
        hist.append(total / n)
    model.eval()
    return model, hist


@torch.no_grad()
def extract_features(model, images, batch=256):
    model.eval()
    if isinstance(model, JEPA):
        enc = model.context
    elif isinstance(model, (MAE, SimCLR, Supervised)):
        enc = model.encoder
    else:
        enc = model
    out = []
    for (x,) in loader(images, batch=batch, shuffle=False):
        out.append(enc(x).cpu().numpy())
    return np.concatenate(out, axis=0)


def probe(features_tr, factors_tr, features_te, factors_te):
    scaler = StandardScaler().fit(features_tr)
    a, b = scaler.transform(features_tr), scaler.transform(features_te)
    res = {}
    for key in ("shape", "colour"):
        clf = LogisticRegression(max_iter=1000)
        clf.fit(a, factors_tr[key])
        res[f"{key}_acc"] = float(clf.score(b, factors_te[key]))
    raw = LogisticRegression(max_iter=1000).fit(features_tr, factors_tr["shape"])
    res["shape_acc_raw"] = float(raw.score(features_te, factors_te["shape"]))
    reg = Ridge(alpha=1.0)
    reg.fit(a, factors_tr["position"])
    pred = reg.predict(b)
    res["position_rmse"] = float(np.sqrt(np.mean((pred - factors_te["position"]) ** 2)))
    return res


def collapse_metrics(features):
    f = features.astype(np.float64)
    f = f - f.mean(0, keepdims=True)
    s = np.linalg.svd(f, compute_uv=False)
    p = s / (s.sum() + 1e-12)
    eff_rank = float(np.exp(-np.sum(p * np.log(p + 1e-12))))
    f_norm = f / (np.linalg.norm(f, axis=1, keepdims=True) + 1e-9)
    idx = np.random.default_rng(0).choice(len(f), min(300, len(f)), replace=False)
    cos = f_norm[idx] @ f_norm[idx].T
    off = cos[~np.eye(len(idx), dtype=bool)]
    return {
        "effective_rank": eff_rank,
        "feature_std": float(f.std(0).mean()),
        "mean_cosine": float(off.mean()),
    }
