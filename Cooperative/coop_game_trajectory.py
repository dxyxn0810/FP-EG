"""
Side-by-side trajectory plots on the cooperative game:
  left : practical log-barrier gradient ascent (coop_game.py), mu = 0.001 by default
  right: FP-EG (coop_game.py), eta = 0.1 by default

Usage:
    python coop_game_trajectory.py
    python coop_game_trajectory.py --mu 0.05 --eta 0.1 --T 500 --x0s 0.2 0.05 0.05 0.7
"""

import argparse
import os

import numpy as np
import matplotlib.pyplot as plt

from coop_game import CooperativeGame, fp_eg_step, lb_step


def trajectory(step, x0, T):
    """Iterates x_0, ..., x_T of x_{t+1} = step(x_t)."""
    xs = [np.array(x0, dtype=float)]
    for _ in range(T):
        xs.append(step(xs[-1]))
    return np.array(xs)


def draw_background(ax, game, E_pts):
    g = np.linspace(0, 1, 400)
    X1, X2 = np.meshgrid(g, g)
    ax.contourf(X1, X2, (X1 * X2 > game.cap).astype(float), levels=[0.5, 1.5],
                colors=["0.88"])
    s = np.linspace(game.cap, 1, 400)
    ax.plot(s, game.cap / s, color="0.45", lw=1,
            label=rf"$x_1x_2={game.cap:g}$ (boundary of $\mathcal{{C}}$)")
    ax.plot(E_pts[:, 0], E_pts[:, 1], color="#d95f02", lw=5, alpha=0.55,
            solid_capstyle="round", label="constrained NE set")


def arrival_iteration(xs, tol=1e-12):
    """Iteration after which the iterate no longer moves (len(xs)-1 if it never stops)."""
    moving = np.where(np.linalg.norm(np.diff(xs, axis=0), axis=1) > tol)[0]
    return int(moving[-1]) + 1 if len(moving) else 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--a", type=float, default=1.0)
    p.add_argument("--b", type=float, default=1.0)
    p.add_argument("--alpha", type=float, default=0.85)
    p.add_argument("--x0s", type=float, nargs="+",
                   default=[0.2, 0.05, 0.05, 0.7, 0.9, 0.05],
                   help="initial points as a flat list x1 x2 x1 x2 ...")
    p.add_argument("--mu", type=float, default=0.001)
    p.add_argument("--eta", type=float, default=0.1)
    p.add_argument("--T", type=int, default=300)
    p.add_argument("--out", type=str, default="results_trajectory")
    args = p.parse_args()

    game = CooperativeGame(args.a, args.b, args.alpha)
    os.makedirs(args.out, exist_ok=True)
    E_pts = game.equilibrium_set()
    x0s = np.array(args.x0s).reshape(-1, 2)
    for x0 in x0s:
        assert game.constraint(x0) - game.alpha > 0, f"x0={x0} not strictly feasible"

    algos = [
        (rf"Practical Log-Barrier Regularization ($\mu$={args.mu:g})", lambda x: lb_step(game, x, args.mu)),
        (rf"FP-EG ($\eta$={args.eta:g})", lambda x: fp_eg_step(game, x, args.eta)[0]),
    ]
    # one color per initial point, identical in both panels
    colors = ["#2c5aa0", "#1b9e77", "#7b3294", "#e7298a", "#66a61e", "#a6761d"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)
    # per-label offsets (points) to keep "t=..." off other lines: (panel, x0 index)
    label_offset = {(0, 0): (-20, 8)}
    for k_ax, (ax, (title, step)) in enumerate(zip(axes, algos)):
        draw_background(ax, game, E_pts)
        for k_x0, (x0, col) in enumerate(zip(x0s, colors)):
            xs = trajectory(step, x0, args.T)
            ax.plot(xs[:, 0], xs[:, 1], "-o", color=col, lw=1.5, ms=3, zorder=3)
            ax.plot(*x0, "*", color=col, mec="k", ms=15, zorder=4)
            ax.plot(*xs[-1], "o", ms=10, mfc="none", mec=col, mew=2, zorder=4)
            t_stop = arrival_iteration(xs)         # = T if the iterate never stopped
            ax.annotate(f"t={t_stop}", xs[-1],
                        xytext=label_offset.get((k_ax, k_x0), (8, 6)),
                        textcoords="offset points", fontsize=9, color="0.15", zorder=5)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.set_xlabel(r"$x_1$")
        ax.set_title(title)
        ax.grid(alpha=0.25, linewidth=0.6)
        ax.legend(fontsize=8, frameon=False, loc="upper right",
                  title=rf"★ $x_0$,  ○ final iterate (t: iterations until it stops, max {args.T})",
                  title_fontsize=8)
    axes[0].set_ylabel(r"$x_2$")
    fig.tight_layout()
    for ext in ("png", "pdf"):                    # pdf is vector, for the paper
        path = os.path.join(args.out, f"trajectory_compare.{ext}")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"saved: {path}")


if __name__ == "__main__":
    main()
