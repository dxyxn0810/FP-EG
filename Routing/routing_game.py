"""
Network routing game of Jordan & Kamgarpour (2026), Section 5 (Tatarenko 2017, Ex. 2.2):

Five players route flow x_i in [0, 10] through their own path R_i of a network with
links l1, ..., l5 (0-indexed in the code):
    R1 = (l1, l2, l4),  R2 = (l1, l3, l5),  R3 = (l2, l5),  R4 = (l3, l4),  R5 = (l4, l5).
Player i's utility is its linear profit minus the congestion costs of its links,
    u_i(x) = p x_i - sum_{k in R_i} P_k(s_k(x)),   s_k(x) = sum_{j : l_k in R_j} x_j,
with p = 42 and
    P1(s) = 5s^2 + s,  P2(s) = 4s^2 + 2s + 1,  P3(s) = 5s^2,  P4(s) = 2s^2 + s,  P5(s) = 10s.
Shared capacity constraints on links l2 and l5:
    c1(x) = s_2(x) <= alpha_1 = 2.7,   c2(x) = s_5(x) <= alpha_2 = 7.

Exact potential game  ->  potential (cost form)  Psi(x) = -p sum_i x_i + sum_k P_k(s_k(x)),
F = grad Psi. Psi is a quadratic with Hessian A^T diag(P_k'') A, where A is the
link-by-player incidence matrix, so L = lambda_max(A^T diag(P_k'') A).

The constraints are linear, so the feasible-response sets are intervals:
    C_i(x_{-i}) = [0, h_i(x_{-i})],   h_i(x_{-i}) = min(10, min_{capacitated l_k in R_i} (alpha_k - s_k(x) + x_i)),
and the projection onto K(x) = C_1(x_{-1}) x ... x C_5(x_{-5}) is a coordinatewise clip.

Two algorithms are implemented on this game:
  * FP-EG (Feasibility-Preserving ExtraGradient, Algorithm 1): predictor and
    corrector steps projected onto K(x), with a potential-greedy unilateral
    fallback when the corrector is infeasible.
  * Independent log-barrier regularized gradient ascent (Section 4.3): each
    player i simultaneously performs
        x_i^{t+1} = P_{X_i}[ x_i^t + gamma_t * grad_{x_i} B_i^mu(x^t) ],
        B_i^mu(x) = u_i(x) + mu * sum_k log(alpha_k - c_k(x)),
    with feasibility margin beta_t = min_k (alpha_k - c_k(x^t)) and the (practical)
    stepsize of the reference implementation
        gamma_t = beta_t / 100.
"""

import numpy as np


# ----------------------------------------------------------------------------
# Game
# ----------------------------------------------------------------------------
class RoutingGame:
    routes = [(0, 1, 3), (0, 2, 4), (1, 4), (2, 3), (3, 4)]
    # link cost P_k(s) = quad[k] s^2 + lin[k] s + const[k]
    quad = np.array([5.0, 4.0, 5.0, 2.0, 0.0])
    lin = np.array([1.0, 2.0, 0.0, 1.0, 10.0])
    const = np.array([0.0, 1.0, 0.0, 0.0, 0.0])

    def __init__(self, p=42.0, x_max=10.0, alpha=(2.7, 7.0), capped_links=(1, 4)):
        self.p, self.x_max = p, x_max
        self.alpha = np.array(alpha, dtype=float)    # capacity of each capped link
        self.capped = np.array(capped_links)         # links l2, l5
        self.m = len(self.routes)
        self.A = np.zeros((len(self.quad), self.m))  # A[k, i] = 1 if l_k in R_i
        for i, r in enumerate(self.routes):
            self.A[list(r), i] = 1.0
        self.A_cap = self.A[self.capped]
        H = self.A.T @ np.diag(2.0 * self.quad) @ self.A
        self.L = np.linalg.eigvalsh(H).max()         # Lipschitz constant of grad Psi

    # utility / potential ----------------------------------------------------
    def loads(self, x):
        return self.A @ x

    def link_costs(self, s):
        return self.quad * s ** 2 + self.lin * s + self.const

    def utility(self, i, x):
        return self.p * x[i] - self.link_costs(self.loads(x))[list(self.routes[i])].sum()

    def potential(self, x):
        return -self.p * np.sum(x) + self.link_costs(self.loads(x)).sum()

    def F(self, x):
        """Game mapping F(x) = (grad_{x_i} f_i)_i = grad Psi(x), costs f_i = -u_i."""
        marginal = 2.0 * self.quad * self.loads(x) + self.lin   # P_k'(s_k)
        return -self.p + self.A.T @ marginal

    # constraint -------------------------------------------------------------
    def constraint(self, x):
        """Loads (c_1, c_2) on the capped links; feasible iff c_k(x) <= alpha_k."""
        return self.A_cap @ x

    def slack(self, x):
        return self.alpha - self.constraint(x)

    def is_feasible(self, x, tol=1e-12):
        in_box = np.all(x >= -tol) and np.all(x <= self.x_max + tol)
        return bool(in_box and np.all(self.slack(x) >= -tol))

    def upper(self, i, x):
        """Right end point h_i(x_{-i}) of the feasible-response interval C_i(x_{-i})."""
        on_route = self.A_cap[:, i] > 0
        room = self.slack(x)[on_route] + x[i]
        return min(self.x_max, room.min()) if on_route.any() else self.x_max

    def project_K(self, v, x):
        """Projection onto K(x) = C_1(x_{-1}) x ... x C_m(x_{-m}) (decomposes playerwise)."""
        return np.array([np.clip(v[i], 0.0, self.upper(i, x)) for i in range(self.m)])

    # Nash gap (exact: each player's problem is a 1-D concave quadratic) ------
    def best_response(self, i, x):
        r = list(self.routes[i])
        s_other = self.loads(x)[r] - x[i]            # loads of R_i without player i
        a = 2.0 * self.quad[r]                       # P_k'(s) = a_k s + lin_k
        unconstrained = (self.p - np.sum(a * s_other + self.lin[r])) / np.sum(a)
        return np.clip(unconstrained, 0.0, self.upper(i, x))

    def gap(self, x):
        u = [self.utility(i, x) for i in range(self.m)]
        g = []
        for i in range(self.m):
            dev = x.copy()
            dev[i] = self.best_response(i, x)
            g.append(self.utility(i, dev) - u[i])
        return max(0.0, max(g))


# ----------------------------------------------------------------------------
# FP-EG (Algorithm 1)
# ----------------------------------------------------------------------------
def fp_eg_step(game, x, eta, tol=1e-12):
    y = game.project_K(x - eta * game.F(x), x)           # predictor
    z = game.project_K(x - eta * game.F(y), x)           # corrector

    if game.is_feasible(z, tol):                         # C' = C (C is convex)
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


# ----------------------------------------------------------------------------
# Log-barrier gradient ascent
# ----------------------------------------------------------------------------
def barrier_grad(game, x, mu):
    """(grad_{x_i} B_i^mu)_i = grad u - mu sum_k grad c_k / (alpha_k - c_k)."""
    grad_u = -game.F(x)                      # F = grad Psi = -grad u
    return grad_u - mu * game.A_cap.T @ (1.0 / game.slack(x))


def stepsize(beta):
    return beta / 100.0


def lb_step(game, x, mu):
    """One simultaneous update; projection onto X = [0,10]^5 decomposes playerwise."""
    beta = game.slack(x).min()
    return np.clip(x + stepsize(beta) * barrier_grad(game, x, mu), 0.0, game.x_max)
