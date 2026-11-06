"""Experiment 1 — a toy language model on a synthetic grammatical language.

The language has two dependencies that a purely local model cannot satisfy:

  * determiner-noun number agreement ("a fox" / "the foxes"),
  * subject-verb agreement across a relative clause
    ("the cat that the dogs chase runs" — *runs* agrees with *cat*).

A GRU is trained only to predict the next token.  Two sizes are trained so the
article can separate "the data has structure" from "this model can use it".

Run:  python experiments/01_language.py
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common.style import PAL, save, set_style
from common.utils import save_metrics, set_seed, timer

# ---------------------------------------------------------------------------
# 1. A small grammar, plus an independent parser used only for evaluation
# ---------------------------------------------------------------------------
NOUNS = {"sg": ["cat", "dog", "bird", "fox"], "pl": ["cats", "dogs", "birds", "foxes"]}
NOUN_NUM = {w: num for num, ws in NOUNS.items() for w in ws}
VERBS = {
    "see": {"sg": "sees", "pl": "see"},
    "chase": {"sg": "chases", "pl": "chase"},
    "like": {"sg": "likes", "pl": "like"},
}
FORM_INFO = {(lemma, num): form for lemma, d in VERBS.items() for num, form in d.items()}
FORM_TO_LEMMA = {form: (lemma, num) for (lemma, num), form in FORM_INFO.items()}


def gen_np(rng: random.Random, depth: int, allow_rc: bool, rc_prob: float = 0.7):
    """Return (tokens, number) for a noun phrase, optionally with a relative clause."""
    num = rng.choice(["sg", "pl"])
    det = "a" if num == "sg" and rng.random() < 0.4 else "the"
    toks = [det, rng.choice(NOUNS[num])]
    if allow_rc and depth > 0 and rng.random() < rc_prob:
        toks.append("that")
        # the subject of the relative clause may itself carry a relative clause,
        # which is what makes the dependency distance grow
        np2, num2 = gen_np(rng, depth - 1, True, rc_prob)
        lemma = rng.choice(list(VERBS))
        clause = np2 + [VERBS[lemma][num2]]
        if rng.random() < 0.5:                      # relative clause with an object
            clause += gen_np(rng, 0, False)[0]
        toks += clause
    return toks, num


def gen_sentence(rng: random.Random, max_nested: int = 5):
    """Return (tokens, metadata) for one grammatical sentence."""
    np1, num1 = gen_np(rng, max_nested, True)
    lemma = rng.choice(list(VERBS))
    other = "pl" if num1 == "sg" else "sg"
    toks = np1 + [VERBS[lemma][num1]]
    meta = {
        "verb_pos": len(np1),
        "verb": VERBS[lemma][num1],
        "alt_verb": VERBS[lemma][other],
        "subject_number": num1,
        "distance": len(np1) - 2,                    # tokens between head noun and verb
    }
    if rng.random() < 0.8:
        toks += gen_np(rng, 1, True)[0]
    return toks, meta


# --- independent recursive-descent parser ----------------------------------
def is_grammatical(tokens: list[str]) -> bool:
    memo_np, memo_rc, memo_vp = {}, {}, {}

    def parse_np(i):
        if i in memo_np:
            return memo_np[i]
        out = set()
        if i + 1 < len(tokens) and tokens[i] in ("the", "a") and tokens[i + 1] in NOUN_NUM:
            num = NOUN_NUM[tokens[i + 1]]
            if tokens[i] == "a" and num != "sg":
                memo_np[i] = out
                return out
            j = i + 2
            out.add((j, num))
            # the relative clause may have a subject of either number; only the
            # clause's own subject-verb agreement is constrained
            for k, _ in parse_rc(j):
                out.add((k, num))
        memo_np[i] = out
        return out

    def parse_rc(i):
        if i in memo_rc:
            return memo_rc[i]
        out = set()
        if i < len(tokens) and tokens[i] == "that":
            for k, n in parse_np(i + 1):
                for m, n2 in parse_vp(k):
                    if n == n2:
                        out.add((m, n))
        memo_rc[i] = out
        return out

    def parse_vp(i):
        if i in memo_vp:
            return memo_vp[i]
        out = set()
        if i < len(tokens) and tokens[i] in FORM_TO_LEMMA:
            num = FORM_TO_LEMMA[tokens[i]][1]
            out.add((i + 1, num))
            for k, _ in parse_np(i + 1):
                out.add((k, num))
        memo_vp[i] = out
        return out

    for j, n in parse_np(0):
        for m, n2 in parse_vp(j):
            if n2 == n and m == len(tokens):
                return True
    return False


# ---------------------------------------------------------------------------
# 2. Vocabulary and data
# ---------------------------------------------------------------------------
PAD, BOS, EOS = 0, 1, 2
words = ["<pad>", "<bos>", "<eos>", "a", "the", "that"] + sorted(NOUN_NUM) + sorted(FORM_TO_LEMMA)
STOI = {w: i for i, w in enumerate(words)}
ITOS = {i: w for w, i in STOI.items()}
VOCAB = len(words)


def build_set(n: int, max_len: int, seed: int, min_len: int = 0, nested: int = 3):
    rng = random.Random(seed)
    items = []
    while len(items) < n:
        toks, meta = gen_sentence(rng, max_nested=nested)
        if min_len <= len(toks) <= max_len:
            items.append((toks, meta))
    return items


def encode(toks):
    return [BOS] + [STOI[t] for t in toks] + [EOS]


def to_tensor(items):
    """Pad a list of sentences to a rectangular (x, y) pair; PAD targets are -100."""
    seqs = [encode(t) for t, _ in items]
    width = max(len(s) for s in seqs) - 1
    x = torch.full((len(seqs), width), PAD, dtype=torch.long)
    y = torch.full((len(seqs), width), -100, dtype=torch.long)
    for i, s in enumerate(seqs):
        x[i, : len(s) - 1] = torch.tensor(s[:-1], dtype=torch.long)
        y[i, : len(s) - 1] = torch.tensor(s[1:], dtype=torch.long)
    return x, y


# ---------------------------------------------------------------------------
# 3. Model and training
# ---------------------------------------------------------------------------
class TinyLM(nn.Module):
    def __init__(self, hidden=96, emb=32, vocab=VOCAB):
        super().__init__()
        self.emb = nn.Embedding(vocab, emb, padding_idx=PAD)
        self.gru = nn.GRU(emb, hidden, batch_first=True)
        self.out = nn.Linear(hidden, vocab)

    def forward(self, x, h=None):
        e = self.emb(x)
        y, h = self.gru(e, h)
        return self.out(y), h


def evaluate(model, items, batch=256):
    model.eval()
    x, y = to_tensor(items)
    total, count = 0.0, 0
    with torch.no_grad():
        for i in range(0, len(items), batch):
            logits, _ = model(x[i : i + batch])
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, VOCAB), y[i : i + batch].reshape(-1), reduction="sum"
            )
            total += loss.item()
            count += int((y[i : i + batch] != -100).sum())
    return total / count, total / count / np.log(2)


@torch.no_grad()
def next_token_probs(model, prefix_tokens):
    ids = torch.tensor([[BOS] + [STOI[t] for t in prefix_tokens]], dtype=torch.long)
    logits, _ = model(ids)
    return torch.softmax(logits[0, -1], dim=-1)


@torch.no_grad()
def sample(model, temperature=0.8, max_new=40):
    ids = [BOS]
    h = None
    for _ in range(max_new):
        x = torch.tensor([[ids[-1]]], dtype=torch.long)
        logits, h = model(x, h)
        probs = torch.softmax(logits[0, -1] / temperature, dim=-1)
        nxt = int(torch.multinomial(probs, 1))
        if nxt == EOS:
            break
        if nxt in (PAD, BOS):
            continue
        ids.append(nxt)
    return [ITOS[i] for i in ids[1:]]


def train_model(hidden: int, train, val, epochs=10, seed=0):
    torch.manual_seed(seed)
    model = TinyLM(hidden=hidden)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    x_tr, y_tr = to_tensor(train)
    hist = {"train": [], "val": []}
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(train))
        run = 0.0
        for i in range(0, len(train), 128):
            idx = perm[i : i + 128]
            logits, _ = model(x_tr[idx])
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, VOCAB), y_tr[idx].reshape(-1)
            )
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            run += loss.item() * len(idx)
        va, _ = evaluate(model, val)
        hist["train"].append(run / len(train))
        hist["val"].append(va)
        print(f"  hidden {hidden:3d}  epoch {ep + 1:2d}  train {hist['train'][-1]:.4f}  "
              f"val {va:.4f}")
    return model, hist


def nll_by_bin(model, items, bins, min_n=25):
    """Mean bits/token per length bin (NaN when a bin has too few sentences)."""
    model.eval()
    buckets = {b: [] for b in bins}
    with torch.no_grad():
        for toks, _ in items:
            L = len(toks)
            b = next((bb for bb in bins if bb[0] <= L <= bb[1]), None)
            if b is None:
                continue
            enc = encode(toks)
            x = torch.tensor([enc[:-1]], dtype=torch.long)
            y = torch.tensor(enc[1:], dtype=torch.long)
            logits, _ = model(x)
            lp = nn.functional.cross_entropy(logits[0], y, reduction="sum") / len(y)
            buckets[b].append(float(lp) / np.log(2))
    return [float(np.mean(buckets[b])) if len(buckets[b]) >= min_n else float("nan")
            for b in bins]


def main():
    set_seed(0)
    set_style()
    torch.set_num_threads(4)

    with timer("generate corpora"):
        train = build_set(18000, max_len=14, seed=1)
        val = build_set(3000, max_len=14, seed=2)
        test_same = build_set(1200, max_len=14, seed=3, min_len=8)
        test_long = build_set(900, 45, seed=4, min_len=15, nested=6)
        pool = build_set(1200, 60, seed=5, nested=6)
        dbins = [(2, 4), (5, 7), (8, 10), (11, 14), (15, 30)]
        probe = []
        for lo, hi in dbins:
            probe += [s for s in pool if lo <= s[1]["distance"] <= hi][:70]

    lbins = [(8, 10), (11, 12), (13, 14), (15, 17), (18, 20), (21, 24), (25, 40)]

    print(f"train {len(train)} sentences, mean length "
          f"{np.mean([len(t) for t, _ in train]):.1f}, vocab {VOCAB}")

    sizes = {"small": 32, "large": 96}
    models, hists, bits = {}, {}, {}
    with timer("train both models"):
        for name, hidden in sizes.items():
            models[name], hists[name] = train_model(hidden, train, val)
            bits[name] = {
                "same": nll_by_bin(models[name], test_same, lbins),
                "long": nll_by_bin(models[name], test_long, lbins),
            }

    bins = [(0, 8), (9, 12), (13, 16), (17, 20), (21, 40)]
    per_model = {}
    for name, model in models.items():
        samples = [sample(model) for _ in range(400)]
        lengths = np.array([len(s) for s in samples])
        gram = np.array([is_grammatical(s) for s in samples])
        gram_by_bin, n_by_bin = [], []
        for lo, hi in bins:
            m = (lengths >= lo) & (lengths <= hi)
            n_by_bin.append(int(m.sum()))
            gram_by_bin.append(float(gram[m].mean()) if m.sum() else float("nan"))

        dis, correct, margin = [], [], []
        for toks, meta in probe:
            probs = next_token_probs(model, toks[: meta["verb_pos"]])
            p_ok = float(probs[STOI[meta["verb"]]])
            p_bad = float(probs[STOI[meta["alt_verb"]]])
            dis.append(meta["distance"])
            correct.append(p_ok > p_bad)
            margin.append(np.log(p_ok) - np.log(p_bad))
        dis, correct, margin = np.array(dis), np.array(correct), np.array(margin)
        agree_by_bin, margin_by_bin = [], []
        for lo, hi in dbins:
            m = (dis >= lo) & (dis <= hi)
            agree_by_bin.append(float(correct[m].mean()) if m.sum() else float("nan"))
            margin_by_bin.append(float(margin[m].mean()) if m.sum() else float("nan"))

        print(f"{name}: overall grammaticality {gram.mean():.3f}, agreement {correct.mean():.3f}")
        per_model[name] = {
            "hidden": sizes[name],
            "agreement_margin_by_distance": [round(v, 2) for v in margin_by_bin],
            "final_val_loss": round(hists[name]["val"][-1], 4),
            "final_val_bits_per_token": round(evaluate(model, val)[1], 3),
            "train_loss": [round(v, 5) for v in hists[name]["train"]],
            "val_loss": [round(v, 5) for v in hists[name]["val"]],
            "grammaticality_overall": round(float(gram.mean()), 3),
            "grammaticality_by_length": [round(v, 3) for v in gram_by_bin],
            "samples_by_length": n_by_bin,
            "agreement_overall": round(float(correct.mean()), 3),
            "agreement_by_distance": [round(v, 3) for v in agree_by_bin],
            "bits_by_length_in_dist": [None if np.isnan(v) else round(v, 3) for v in bits[name]["same"]],
            "bits_by_length_ood": [None if np.isnan(v) else round(v, 3) for v in bits[name]["long"]],
            "samples": [" ".join(s) for s in samples[:6]],
            "samples_grammatical": [bool(is_grammatical(s)) for s in samples[:6]],
            "generated_lengths": [int(x) for x in lengths],
        }

    save_metrics(
        "language",
        {
            "vocab": VOCAB,
            "n_train": len(train),
            "mean_train_length": round(float(np.mean([len(t) for t, _ in train])), 2),
            "length_bins": bins,
            "compression_bins": lbins,
            "distance_bins": dbins,
            "models": per_model,
        },
    )

    # ---- figures ----------------------------------------------------------
    import matplotlib.pyplot as plt

    colors = {"small": PAL["sky"], "large": PAL["blue"]}
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.1))
    ax = axes[0]
    for name in sizes:
        ax.plot(range(1, 11), hists[name]["val"], color=colors[name],
                label=f"validation, hidden {sizes[name]}")
    ax.set_xlabel("epoch")
    ax.set_ylabel("cross-entropy (nats/token)")
    ax.set_title("(a) next-token prediction loss")
    ax.legend(loc="upper right")

    ax = axes[1]
    xb = np.arange(len(lbins))
    for name in sizes:
        ax.plot(xb, bits[name]["same"], "o-", ms=3.5, color=colors[name],
                label=f"{name}, training lengths")
        ax.plot(xb, bits[name]["long"], "s--", ms=3.5, color=colors[name],
                alpha=0.55, label=f"{name}, longer")
    ax.axvspan(-0.5, 2.5, color=PAL["light"], alpha=0.5, lw=0)
    ax.set_xticks(xb, [f"{lo}-{hi}" for lo, hi in lbins], fontsize=7)
    ax.set_xlabel("sentence length (tokens)")
    ax.set_ylabel("bits per token")
    ax.set_title("(b) compression of unseen sentences")
    ax.legend(loc="upper left", ncol=1)
    fig.tight_layout()
    save(fig, "fig-language-curves")

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.0))
    ax = axes[0]
    maxlen = 26
    edges = np.arange(1, maxlen + 2) - 0.5
    for j, name in enumerate(sizes):
        ax.hist(np.clip(per_model[name]["generated_lengths"], 1, maxlen), bins=edges,
                color=colors[name], alpha=0.55,
                label=f"hidden {sizes[name]}, "
                      f"{per_model[name]['grammaticality_overall']:.2f} grammatical")
    top = ax.get_ylim()[1]
    ax.set_ylim(0, top * 1.5)
    ax.axvline(14.5, color=PAL["vermillion"], lw=1.0, ls="--")
    ax.text(15.0, top * 1.03, "longest training\nsentence",
            color=PAL["vermillion"], fontsize=7)
    ax.set_xlabel("length of the generated sentence (tokens)")
    ax.set_ylabel("count (400 samples)")
    ax.set_title("(a) generated lengths stay in-distribution")
    ax.legend(loc="upper left", fontsize=7, framealpha=0.95)

    ax = axes[1]
    for name in sizes:
        ax.plot(range(len(dbins)), per_model[name]["agreement_margin_by_distance"],
                "o-", ms=4, color=colors[name],
                label=f"hidden {sizes[name]} "
                      f"(acc {per_model[name]['agreement_overall']:.3f})")
    ax.axhline(0, color=PAL["grey"], lw=0.8, ls=":")
    ax.set_xticks(range(len(dbins)), [f"{lo}-{hi}" for lo, hi in dbins])
    ax.set_xlabel("tokens between subject noun and verb")
    ax.set_ylabel("log p(correct) - log p(wrong)  (nats)")
    ax.set_title("(b) agreement margin with distance")
    ax.legend(loc="upper right")
    fig.tight_layout()
    save(fig, "fig-language-generalization")

    for name in sizes:
        print(f"\n{name} samples:")
        for s, g in zip(per_model[name]["samples"], per_model[name]["samples_grammatical"]):
            print(f"  [{g}] {s}")


if __name__ == "__main__":
    main()
