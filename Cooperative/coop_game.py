"""
Cooperative game of Jordan & Kamgarpour (2026), Section 5:

    u1(x) = u2(x) = (x1 + x2) - a (x1 - x2)^2 - b (x1^2 + x2^2),   x in [0,1]^2
    shared constraint  c(x) = 1 - x1 x2 >= alpha   (i.e. x1 x2 <= cap = 1 - alpha)
with defaults a = 1, b = 1, alpha = 0.85 (cap = 0.15).

Identical-interest game  ->  potential (cost form)  Psi(x) = -u(x),  F = grad Psi.
Psi is a quadratic with Hessian [[2a+2b, -2a], [-2a, 2a+2b]], so L = 4a + 2b.

Feasible-response sets are intervals:
    C_i(x_j) = [0, h(x_j)],   h(x_j) = min(1, cap / x_j),
so the projection onto K(x) = C_1(x_2) x C_2(x_1) is a coordinatewise clip.

Two algorithms are implemented on this game:
  * FP-EG (Feasibility-Preserving ExtraGradient, Algorithm 1): predictor and
    corrector steps projected onto K(x), with a potential-greedy unilateral
    fallback when the corrector is infeasible.
  * Independent log-barrier regularized gradient ascent (Section 4.3): each
    player i simultaneously performs
        x_i^{t+1} = P_{X_i}[ x_i^t + gamma_t * grad_{x_i} B_i^mu(x^t) ],
        B_i^mu(x) = u_i(x) + mu * log(c(x) - alpha),
    with feasibility margin beta_t = c(x^t) - alpha and the (practical) stepsize
        gamma_t = beta_t / 72.
"""

import numpy as np


# ----------------------------------------------------------------------------
# Game
# ----------------------------------------------------------------------------
class CooperativeGame:
    def __init__(self, a=1.0, b=1.0, alpha=0.85):
        self.a, self.b, self.alpha = a, b, alpha
        self.cap = 1.0 - alpha               # x1 * x2 <= cap

    # utility / potential ----------------------------------------------------
    def utility(self, x):
        x1, x2 = x
        return (x1 + x2) - self.a * (x1 - x2) ** 2 - self.b * (x1 ** 2 + x2 ** 2)

    def potential(self, x):
        return -self.utility(x)

    def F(self, x):
        """Game mapping F(x) = (grad_{x1} f1, grad_{x2} f2) = grad Psi(x), costs f_i = -u_i."""
        x1, x2 = x
        g1 = -(1.0 - 2.0 * self.a * (x1 - x2) - 2.0 * self.b * x1)
        g2 = -(1.0 + 2.0 * self.a * (x1 - x2) - 2.0 * self.b * x2)
        return np.array([g1, g2])

    # constraint -------------------------------------------------------------
    def constraint(self, x):
        return 1.0 - x[0] * x[1]

    def is_feasible(self, x, tol=1e-12):
        in_box = np.all(x >= -tol) and np.all(x <= 1.0 + tol)
        return bool(in_box and x[0] * x[1] <= self.cap + tol)

    def upper(self, x_other):
        """Right end point h(x_j) of the feasible-response interval C_i(x_j)."""
        return 1.0 if x_other <= self.cap else self.cap / x_other

    def project_K(self, v, x):
        """Projection onto K(x) = C_1(x_2) x C_2(x_1) (decomposes playerwise)."""
        return np.array([np.clip(v[0], 0.0, self.upper(x[1])),
                         np.clip(v[1], 0.0, self.upper(x[0]))])

    # Nash gap (exact: each player's problem is a 1-D concave quadratic) ------
    def best_response(self, i, x):
        j = 1 - i
        unconstrained = (1.0 + 2.0 * self.a * x[j]) / (2.0 * self.a + 2.0 * self.b)
        return np.clip(unconstrained, 0.0, self.upper(x[j]))

    def gap(self, x):
        u = self.utility(x)
        g = []
        for i in range(2):
            dev = x.copy()
            dev[i] = self.best_response(i, x)
            g.append(self.utility(dev) - u)
        return max(0.0, max(g))

    # reference quantities ---------------------------------------------------
    def equilibrium_set(self, n=200001, tol=1e-10):
        """Sample the constrained NE set.

        Ignoring the shared constraint, the best-response fixed point on [0,1]^2 is
        x1 = x2 = min(1, 1/(2b)), which violates x1 x2 <= cap whenever
        min(1, 1/(4b^2)) > cap; then every equilibrium lies on the boundary
        x1 x2 = cap, and for this game the NE set is a whole arc of that curve.
        """
        s = np.linspace(self.cap, 1.0, n)
        pts = np.stack([s, self.cap / s], axis=1)
        mask = np.array([self.gap(p) <= tol for p in pts])
        return pts[mask]


# ----------------------------------------------------------------------------
# FP-EG (Algorithm 1)
# ----------------------------------------------------------------------------
def fp_eg_step(game, x, eta, tol=1e-12):
    y = game.project_K(x - eta * game.F(x), x)           # predictor
    z = game.project_K(x - eta * game.F(y), x)           # corrector

    if game.is_feasible(z, tol):                         # C' = C (connected)
        return z, y, z, True

    # potential-greedy unilateral fallback
    I_t = [i for i in range(2) if z[i] != x[i]]
    candidates = []
    for i in I_t:
        xi = x.copy()
        xi[i] = z[i]
        candidates.append(xi)
    x_next = min(candidates, key=game.potential)
    return x_next, y, z, False


# ----------------------------------------------------------------------------
# Log-barrier gradient ascent
# ----------------------------------------------------------------------------
def barrier_grad(game, x, mu):
    """(grad_{x_1} B_1^mu, grad_{x_2} B_2^mu) = grad u + mu grad c / (c - alpha)."""
    grad_u = -game.F(x)                      # F = grad Psi = -grad u
    grad_c = np.array([-x[1], -x[0]])        # c(x) = 1 - x1 x2
    return grad_u + mu * grad_c / (game.constraint(x) - game.alpha)


def stepsize(beta):
    return beta / 72.0


def lb_step(game, x, mu):
    """One simultaneous update; projection onto X = [0,1]^2 decomposes playerwise."""
    beta = game.constraint(x) - game.alpha
    return np.clip(x + stepsize(beta) * barrier_grad(game, x, mu), 0.0, 1.0)
