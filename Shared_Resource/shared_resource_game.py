"""
Pairwise-overhead shared-resource game (Section 3.4 / Appendix C.1) with m players:

    x_i in X_i = [0, 1],      u_i(x) = -(x_i - v_i)^2 / 2,      v_i ~ Uniform(0, 1) given g(v) > B,
    one shared capacity constraint
        g(x) = a'x + x'Qx / 2 <= B,    a = 1,  Q = (11' - I) / m,  B = m / 2,
    i.e. g(x) = S + (S^2 - ||x||^2) / (2m) with S = sum_i x_i.

Each utility depends only on the player's own strategy, so the game is an exact potential
game with potential (cost form) Psi(x) = sum_i (x_i - v_i)^2 / 2 and F(x) = grad Psi(x) = x - v,
hence L = 1. The coupling enters only through the constraint: Q is indefinite (eigenvalues
(m-1)/m and -1/m), so the feasible set C = {x in [0,1]^m : g(x) <= B} is not jointly convex,
but it is star-shaped w.r.t. 0 (Appendix C.1), so C' = C.

Q has a zero diagonal, so g is affine in each x_i with slope d_i(x_{-i}) = 1 + (S - x_i)/m >= 1.
The feasible-response sets are therefore intervals
    C_i(x_{-i}) = [0, h_i(x_{-i})],   h_i(x_{-i}) = min(1, x_i + (B - g(x)) / d_i(x_{-i})),
and the projection onto K(x) = C_1(x_{-1}) x ... x C_m(x_{-m}) is a coordinatewise clip.
"""

import numpy as np


# ----------------------------------------------------------------------------
# Game
# ----------------------------------------------------------------------------
class SharedResourceGame:
    def __init__(self, m, seed=None, v=None):
        """v ~ Uniform(0, 1)^m conditioned on g(v) > B (rejection sampling), so that the
        unconstrained equilibrium x = v is infeasible and the constraint binds.
        A v passed explicitly is used as is."""
        self.m = m
        self.a = np.ones(m)
        self.Q = (np.ones((m, m)) - np.eye(m)) / m
        self.B = m / 2.0
        self.L = 1.0                             # Lipschitz constant of grad Psi
        if v is None:
            rng = np.random.default_rng(seed)
            v = rng.uniform(0.0, 1.0, m)
            while self.constraint(v) <= self.B:
                v = rng.uniform(0.0, 1.0, m)
        self.v = np.asarray(v, float)

    # utility / potential ----------------------------------------------------
    def utility(self, i, x):
        return -0.5 * (x[i] - self.v[i]) ** 2

    def potential(self, x):
        return 0.5 * np.sum((x - self.v) ** 2)

    def F(self, x):
        """Game mapping F(x) = (grad_{x_i} f_i)_i = grad Psi(x), costs f_i = -u_i."""
        return x - self.v

    # constraint -------------------------------------------------------------
    def constraint(self, x):
        """Resource load g(x); feasible iff g(x) <= B."""
        return self.a @ x + 0.5 * x @ self.Q @ x

    def grad_constraint(self, x):
        return self.a + self.Q @ x

    def slack(self, x):
        return self.B - self.constraint(x)

    def is_feasible(self, x, tol=1e-12):
        in_box = np.all(x >= -tol) and np.all(x <= 1.0 + tol)
        return bool(in_box and self.slack(x) >= -tol)

    def upper(self, x):
        """Right end points h_i(x_{-i}) of the feasible-response intervals C_i(x_{-i}), all i."""
        d = self.grad_constraint(x)              # d_i = 1 + (S - x_i)/m, independent of x_i
        return np.minimum(1.0, x + self.slack(x) / d)

    def project_K(self, v, x):
        """Projection onto K(x) = C_1(x_{-1}) x ... x C_m(x_{-m}) (decomposes playerwise)."""
        return np.clip(v, 0.0, self.upper(x))

    # Nash gap (exact: each player's problem is a 1-D concave quadratic) ------
    def best_response(self, x):
        """Best responses of all players, each to x_{-i}."""
        return np.clip(self.v, 0.0, self.upper(x))

    def gap(self, x):
        # u_i depends only on x_i, so player i's deviation gain is u_i(z_i) - u_i(x_i)
        z = self.best_response(x)
        return max(0.0, np.max(0.5 * (x - self.v) ** 2 - 0.5 * (z - self.v) ** 2))


# ----------------------------------------------------------------------------
# FP-EG (Algorithm 1)
# ----------------------------------------------------------------------------
def fp_eg_step(game, x, eta, tol=1e-12):
    y = game.project_K(x - eta * game.F(x), x)           # predictor
    z = game.project_K(x - eta * game.F(y), x)           # corrector

    if game.is_feasible(z, tol):                         # C' = C (C is star-shaped)
        return z, y, z, True

    # potential-greedy unilateral fallback
    I_t = [i for i in range(len(x)) if z[i] != x[i]]
    candidates = []
    for i in I_t:
        xi = x.copy()
        xi[i] = z[i]
        candidates.append(xi)
    x_next = min(candidates, key=game.potential)
    return x_next, y, z, False


def run_fp_eg(game, x0=None, eta=0.9, T=1000, eps=1e-10):
    """FP-EG from a feasible x0 (default 0, which is in C) with stepsize eta < 1/L, until
    Gap(x_t) <= eps or T iterations. Returns the iterates x_0, ..., x_t."""
    if not eta < 1.0 / game.L:
        raise ValueError(f"eta={eta:g} violates eta < 1/L = {1 / game.L:g}")
    x = np.zeros(game.m) if x0 is None else np.array(x0, dtype=float)
    assert game.is_feasible(x), "x0 must be feasible"
    xs = [x]
    for _ in range(T):
        if game.gap(x) <= eps:
            break
        x = fp_eg_step(game, x, eta)[0]
        xs.append(x)
    return np.array(xs)
