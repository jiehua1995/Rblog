"""Tiny ViT encoders, JEPA, MAE and SimCLR for CPU-scale experiments."""

from __future__ import annotations

import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out")
        if m.bias is not None:
            nn.init.zeros_(m.bias)


class TinyViT(nn.Module):
    def __init__(self, dim: int = 96, depth: int = 3, heads: int = 4,
                 patch: int = 8, img: int = 32):
        super().__init__()
        self.patch = patch
        self.grid = img // patch
        self.n_tokens = self.grid * self.grid
        self.dim = dim
        self.patch_embed = nn.Conv2d(3, dim, patch, patch)
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos = nn.Parameter(torch.zeros(1, 1 + self.n_tokens, dim))
        layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads, dim_feedforward=2 * dim,
            dropout=0.0, activation="gelu", batch_first=True, norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(layer, depth, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(dim)
        self.apply(_init_weights)
        nn.init.trunc_normal_(self.cls, std=0.02)
        nn.init.trunc_normal_(self.pos, std=0.02)

    def patch_tokens(self, x):
        p = self.patch_embed(x).flatten(2).transpose(1, 2)
        return p + self.pos[:, 1:]

    @staticmethod
    def gather_visible(tokens, keep):
        n_keep = int(keep.sum(1)[0].item())
        idx = torch.argsort((~keep).int(), dim=1)[:, :n_keep]
        return torch.gather(tokens, 1, idx.unsqueeze(-1).expand(-1, -1, tokens.size(-1)))

    def forward(self, x, keep=None, return_tokens: bool = False):
        p = self.patch_tokens(x)
        if keep is not None:
            p = self.gather_visible(p, keep)
        cls = self.cls.expand(p.size(0), -1, -1) + self.pos[:, :1]
        t = torch.cat([cls, p], dim=1)
        t = self.norm(self.blocks(t))
        return t if return_tokens else t[:, 0]


def block_mask(batch: int, grid: int, ratio: float, device) -> torch.Tensor:
    """True means masked.  A contiguous rectangle whenever the ratio allows it."""
    n = grid * grid
    m = max(1, min(n - 1, int(round(ratio * n))))
    mask = torch.zeros(batch, n, dtype=torch.bool, device=device)
    if m >= n - 1:
        for b in range(batch):
            idx = torch.randperm(n, device=device)[:m]
            mask[b, idx] = True
        return mask
    best = None
    for h in range(1, grid + 1):
        for w in range(1, grid + 1):
            if h * w >= m and (best is None or h * w < best[0] * best[1]):
                best = (h, w)
    h, w = best
    for b in range(batch):
        top = torch.randint(0, grid - h + 1, (1,), device=device).item()
        left = torch.randint(0, grid - w + 1, (1,), device=device).item()
        cells = []
        for i in range(h):
            for j in range(w):
                cells.append((top + i) * grid + (left + j))
        cells = cells[:m]
        mask[b, torch.tensor(cells, device=device)] = True
    return mask


class Predictor(nn.Module):
    def __init__(self, dim: int = 96, depth: int = 2, heads: int = 4,
                 n_tokens: int = 16, width: int | None = None,
                 in_dim: int | None = None):
        super().__init__()
        width = width or dim
        in_dim = in_dim or dim
        self.in_proj = nn.Linear(in_dim, width) if in_dim != width else nn.Identity()
        self.out_proj = nn.Linear(width, in_dim) if in_dim != width else nn.Identity()
        self.mask_token = nn.Parameter(torch.zeros(1, 1, width))
        self.pos = nn.Parameter(torch.zeros(1, 1 + n_tokens, width))
        layer = nn.TransformerEncoderLayer(
            d_model=width, nhead=heads, dim_feedforward=2 * width,
            dropout=0.0, activation="gelu", batch_first=True, norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(layer, depth, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(width)
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        nn.init.trunc_normal_(self.pos, std=0.02)

    def forward(self, context_tokens, keep):
        """context_tokens: [B, 1+K, dim]; keep: [B, N] True for visible."""
        context_tokens = self.in_proj(context_tokens)
        B = context_tokens.size(0)
        n = keep.size(1)
        patches = self.mask_token.expand(B, n, -1).clone()
        patches[keep] = context_tokens[:, 1:, :].reshape(-1, context_tokens.size(-1))
        cls = context_tokens[:, :1, :]
        t = torch.cat([cls, patches], dim=1) + self.pos
        t = self.norm(self.blocks(t))
        return self.out_proj(t[:, 1:, :])


@torch.no_grad()
def update_ema(target: nn.Module, source: nn.Module, momentum: float) -> None:
    for pt, ps in zip(target.parameters(), source.parameters()):
        pt.data.mul_(momentum).add_(ps.data, alpha=1.0 - momentum)


def variance_loss(z: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    std = torch.sqrt(z.var(dim=0) + eps)
    return torch.relu(1.0 - std).mean()


class JEPA(nn.Module):
    def __init__(self, dim=96, depth=3, heads=4, patch=8, img=32,
                 predictor_depth=2, predictor_width=None):
        super().__init__()
        self.context = TinyViT(dim, depth, heads, patch, img)
        self.target = copy.deepcopy(self.context)
        for p in self.target.parameters():
            p.requires_grad = False
        self.predictor = Predictor(
            dim, predictor_depth, heads, self.context.n_tokens,
            width=predictor_width or dim, in_dim=dim,
        )

    def forward(self, x, mask, stop_grad: bool = True):
        keep = ~mask
        if stop_grad:
            with torch.no_grad():
                target_tokens = self.target(x, return_tokens=True)
        else:
            target_tokens = self.target(x, return_tokens=True)
        context_tokens = self.context(x, keep=keep, return_tokens=True)
        pred = self.predictor(context_tokens, keep)
        tgt = target_tokens[:, 1:, :][mask].reshape(x.size(0), -1, target_tokens.size(-1))
        prd = pred[mask].reshape(x.size(0), -1, pred.size(-1))
        return F.smooth_l1_loss(prd, tgt), prd


class MAE(nn.Module):
    def __init__(self, dim=96, depth=3, heads=4, patch=8, img=32,
                 decoder_depth=2):
        super().__init__()
        self.encoder = TinyViT(dim, depth, heads, patch, img)
        self.decoder = Predictor(dim, decoder_depth, heads, self.encoder.n_tokens)
        self.head = nn.Linear(dim, 3 * patch * patch)
        self.patch = patch
        self.grid = self.encoder.grid

    def forward(self, x, mask):
        keep = ~mask
        context_tokens = self.encoder(x, keep=keep, return_tokens=True)
        tokens = self.decoder(context_tokens, keep)
        pred = self.head(tokens[mask]).reshape(-1, 3, self.patch, self.patch)
        tgt = x.permute(0, 2, 3, 1).reshape(x.size(0), self.grid, self.patch,
                                            self.grid, self.patch, 3)
        tgt = tgt.permute(0, 1, 3, 2, 4, 5).reshape(x.size(0), -1, 3, self.patch,
                                                    self.patch)
        tgt = tgt[mask]
        return F.mse_loss(pred, tgt)


class ActionPredictor(nn.Module):
    """Predict next-frame patch tokens from current tokens plus an action."""

    def __init__(self, dim=96, depth=2, heads=4, n_tokens=16):
        super().__init__()
        self.action = nn.Sequential(nn.Linear(2, dim), nn.GELU(), nn.Linear(dim, dim))
        self.pos = nn.Parameter(torch.zeros(1, 1 + n_tokens + 1, dim))
        layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads, dim_feedforward=2 * dim,
            dropout=0.0, activation="gelu", batch_first=True, norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(layer, depth, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(dim)
        self.n_tokens = n_tokens
        nn.init.trunc_normal_(self.pos, std=0.02)

    def forward(self, tokens, action):
        a = self.action(action).unsqueeze(1)
        t = torch.cat([tokens, a], dim=1) + self.pos
        t = self.norm(self.blocks(t))
        return t[:, 1:1 + self.n_tokens, :]


class WorldModelJEPA(nn.Module):
    def __init__(self, dim=96, depth=3, heads=4, patch=8, img=32,
                 predictor_depth=2):
        super().__init__()
        self.context = TinyViT(dim, depth, heads, patch, img)
        self.target = copy.deepcopy(self.context)
        for p in self.target.parameters():
            p.requires_grad = False
        self.predictor = ActionPredictor(dim, predictor_depth, heads,
                                         self.context.n_tokens)

    def forward(self, x, action, next_x, stop_grad=True):
        if stop_grad:
            with torch.no_grad():
                target = self.target(next_x, return_tokens=True)[:, 1:, :]
        else:
            target = self.target(next_x, return_tokens=True)[:, 1:, :]
        tokens = self.context(x, return_tokens=True)
        pred = self.predictor(tokens, action)
        return F.smooth_l1_loss(pred, target), pred


class WorldModelPixel(nn.Module):
    def __init__(self, dim=96, depth=3, heads=4, patch=8, img=32,
                 predictor_depth=2):
        super().__init__()
        self.encoder = TinyViT(dim, depth, heads, patch, img)
        self.predictor = ActionPredictor(dim, predictor_depth, heads,
                                         self.encoder.n_tokens)
        self.head = nn.Linear(dim, 3 * patch * patch)
        self.patch = patch
        self.grid = self.encoder.grid

    def forward(self, x, action, next_x):
        tokens = self.encoder(x, return_tokens=True)
        pred = self.predictor(tokens, action)
        pred = self.head(pred).reshape(-1, self.grid, self.grid, 3, self.patch, self.patch)
        pred = pred.permute(0, 3, 1, 4, 2, 5).reshape(-1, 3, 32, 32)
        return F.mse_loss(pred, next_x), pred


class SimCLR(nn.Module):
    def __init__(self, dim=96, depth=3, heads=4, patch=8, img=32, proj=64):
        super().__init__()
        self.encoder = TinyViT(dim, depth, heads, patch, img)
        self.proj = nn.Sequential(
            nn.Linear(dim, dim), nn.ReLU(inplace=True), nn.Linear(dim, proj)
        )

    def forward(self, x):
        z = self.encoder(x)
        return F.normalize(self.proj(z), dim=-1)


class Supervised(nn.Module):
    def __init__(self, n_classes=3, dim=96, depth=3, heads=4, patch=8, img=32):
        super().__init__()
        self.encoder = TinyViT(dim, depth, heads, patch, img)
        self.head = nn.Linear(dim, n_classes)

    def forward(self, x):
        return self.head(self.encoder(x))


def nt_xent(z1, z2, temperature=0.2):
    z = torch.cat([z1, z2], dim=0)
    sim = z @ z.t() / temperature
    n = z1.size(0)
    labels = torch.arange(2 * n, device=z.device)
    labels = (labels + n) % (2 * n)
    mask = torch.eye(2 * n, dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(mask, -1e9)
    return F.cross_entropy(sim, labels)
