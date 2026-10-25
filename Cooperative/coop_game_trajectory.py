"""
Side-by-side trajectory plots on the cooperative game:
  left : practical log-barrier gradient ascent (coop_game.py), mu = 0.005 by default
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


def gap_label(game, xs, eps):
    """'t=...' at the first iteration with Gap <= eps ('T=...' if it is never reached)."""
    gaps = np.array([game.gap(x) for x in xs])
    hit = np.where(gaps <= eps)[0]
    return f"t={hit[0]}" if len(hit) else f"T={len(xs) - 1}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--a", type=float, default=1.0)
    p.add_argument("--b", type=float, default=1.0)
    p.add_argument("--alpha", type=float, default=0.85)
    p.add_argument("--x0s", type=float, nargs="+",
                   default=[0.2, 0.05, 0.05, 0.7, 0.9, 0.05],
                   help="initial points as a flat list x1 x2 x1 x2 ...")
    p.add_argument("--mu", type=float, default=0.005)
    p.add_argument("--eta", type=float, default=0.1)
    p.add_argument("--T", type=int, default=300)
    p.add_argument("--eps", type=float, default=1e-6, help="Nash-gap level for the t=... labels")
    p.add_argument("--out", type=str, default="results_trajectory")
    args = p.parse_args()

    plt.rcParams["font.size"] = 20               # 2x matplotlib's default (titles, labels, ticks)
    game = CooperativeGame(args.a, args.b, args.alpha)
    os.makedirs(args.out, exist_ok=True)
    E_pts = game.equilibrium_set()
    x0s = np.array(args.x0s).reshape(-1, 2)
    for x0 in x0s:
        assert game.constraint(x0) - game.alpha > 0, f"x0={x0} not strictly feasible"

    algos = [
        (rf"Log-Barrier ($\mu$={args.mu:g})", lambda x: lb_step(game, x, args.mu)),
        (rf"FP-EG ($\eta$={args.eta:g})", lambda x: fp_eg_step(game, x, args.eta)[0]),
    ]
    # one color per initial point, identical in both panels
    colors = ["#2c5aa0", "#1b9e77", "#7b3294", "#e7298a", "#66a61e", "#a6761d"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 7.5), sharex=True, sharey=True)
    # per-label offsets (points) to keep "t=..." off other lines: (panel, x0 index)
    label_offset = {(0, 0): (-80, -2), (0, 1): (16, 6),
                    (1, 0): (10, 12), (1, 1): (-62, -12), (1, 2): (16, -4)}
    for k_ax, (ax, (title, step)) in enumerate(zip(axes, algos)):
        draw_background(ax, game, E_pts)
        for k_x0, (x0, col) in enumerate(zip(x0s, colors)):
            xs = trajectory(step, x0, args.T)
            ax.plot(xs[:, 0], xs[:, 1], "-o", color=col, lw=1.5, ms=3, zorder=3)
            ax.plot(*x0, "*", color=col, mec="k", ms=15, zorder=4)
            ax.plot(*xs[-1], "o", ms=10, mfc="none", mec=col, mew=2, zorder=4)
            ax.annotate(gap_label(game, xs, args.eps), xs[-1],
                        xytext=label_offset.get((k_ax, k_x0), (8, 6)),
                        textcoords="offset points", fontsize=18, color=col, zorder=5)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.set_xlabel(r"$x_1$")
        ax.set_title(title, pad=16)                # default pad is 6 points
        ax.grid(alpha=0.25, linewidth=0.6)
    axes[0].set_ylabel(r"$x_2$")
    fig.tight_layout()
    # both panels share the same legend entries: one legend below the panels
    fig.legend(*axes[0].get_legend_handles_labels(), fontsize=16, frameon=False,
               loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=2,
               title=rf"★ $x_0$,  ○ final iterate (t: first iteration with Gap $\leq$ {args.eps:g}, T={args.T})",
               title_fontsize=16)
    for ext in ("png", "pdf"):                    # pdf is vector, for the paper
        path = os.path.join(args.out, f"trajectory_compare.{ext}")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"saved: {path}")


if __name__ == "__main__":
    main()
