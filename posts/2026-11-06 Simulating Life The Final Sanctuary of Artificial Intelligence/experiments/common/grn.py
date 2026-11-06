"""A ten-gene regulatory network, shared by experiments 4 and 5.

The network has two mutual-inhibition toggles, a hub gene that reads both
toggles, and five downstream readouts.  Dynamics are the standard
activation-degradation form

    dx_i/dt = -x_i + sum_j W_ij * h(x_j) + b_i + u * c_i

with a Hill function h(x) = x^2 / (K^2 + x^2).  A gene can be knocked out by
removing its production term, in which case it simply decays.

The point of the model is not realism: it is that the ground truth is known
exactly, so prediction, intervention and extrapolation can be scored.
"""

from __future__ import annotations

import numpy as np

N = 10
GENES = [f"g{i}" for i in range(N)]
K_HILL, N_HILL = 0.35, 2.0
DT = 0.05
BASE = 0.25


def make_W() -> np.ndarray:
    W = np.zeros((N, N))
    # toggle A
    W[0, 1], W[1, 0] = -3.0, -3.0
    W[0, 0], W[1, 1] = 1.6, 1.6
    # toggle B
    W[2, 3], W[3, 2] = -3.0, -3.0
    W[2, 2], W[3, 3] = 1.6, 1.6
    # hub: activated by either A branch, inhibited by either B branch
    W[4, 0], W[4, 1] = 2.2, 2.2
    W[4, 2], W[4, 3] = -1.6, -1.6
    # readouts
    W[5, 4], W[6, 4] = 1.8, 1.8
    W[5, 2] = 1.2
    W[7, 4] = 1.6
    W[8, 4], W[8, 0] = 1.6, -1.2
    W[9, 4], W[9, 1] = 1.6, -1.2
    return W


W_TRUE = make_W()
B_TRUE = np.full(N, BASE)
# environmental sensitivity: the signal relieves repression on the first toggle
C_TRUE = np.array([0.05, -2.8, 0.05, 0.05, 0.20, 0.10, 0.10, 0.10, 0.10, 0.10])


def hill(x):
    return x ** 2 / (K_HILL ** 2 + x ** 2)


def rhs(x, u, ko_mask, W=W_TRUE, b=B_TRUE, c=C_TRUE):
    prod = W @ hill(x) + b + u * c
    prod = np.where(ko_mask, 0.0, prod)
    return -x + prod


def step(x, u, ko_mask, dt=DT):
    """One RK4 step; concentrations are clipped at zero (they are amounts)."""
    k1 = rhs(x, u, ko_mask)
    k2 = rhs(x + 0.5 * dt * k1, u, ko_mask)
    k3 = rhs(x + 0.5 * dt * k2, u, ko_mask)
    k4 = rhs(x + dt * k3, u, ko_mask)
    return np.maximum(0.0, x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4))


def simulate(x0, u, ko_mask, steps=200):
    xs = np.empty((steps + 1, N))
    xs[0] = x0
    x = x0.copy()
    for i in range(steps):
        x = step(x, u, ko_mask)
        xs[i + 1] = x
    return xs


def simulate_schedule(x0, u_schedule, ko_mask, bounds=None):
    """u_schedule: array of length steps; ko applied from step `bounds` onwards."""
    steps = len(u_schedule)
    xs = np.empty((steps + 1, N))
    xs[0] = x0
    x = x0.copy()
    for i in range(steps):
        x = step(x, u_schedule[i], ko_mask)
        xs[i + 1] = x
    return xs


def find_attractors(n_starts=400, steps=1200, u=0.0, seed=0, tol=0.05):
    rng = np.random.default_rng(seed)
    ko = np.zeros(N, dtype=bool)
    finals = []
    for _ in range(n_starts):
        x0 = rng.uniform(0, 2.0, N)
        x = simulate(x0, u, ko, steps)[-1]
        finals.append(x)
    finals = np.array(finals)
    attractors = []
    for x in finals:
        if not any(np.max(np.abs(x - a)) < tol for a in attractors):
            attractors.append(x)
    return np.array(attractors)
