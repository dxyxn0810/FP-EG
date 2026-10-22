"""
Nash gap over iterations on the routing game, one panel per initial point:
  practical log-barrier gradient ascent (routing_game.py), mu = 0.05 by default
  FP-EG (routing_game.py), eta = 0.025 by default (1/L ~ 0.027)

FP-EG is drawn only until it reaches a fixed point up to tolerance,
||x_{t+1} - x_t||_inf <= tol, and that iteration is marked "converged at t=...".

Usage:
    python routing_nash_gap.py
    python routing_nash_gap.py --mu 0.05 --eta 0.025 --T 500 --x0s 0.5 0.5 0.5 0.5 0.5
"""

import argparse
import os

import numpy as np
import matplotlib.pyplot as plt

from routing_game import RoutingGame, fp_eg_step, lb_step


def trajectory(step, x0, T):
    """Iterates x_0, ..., x_T of x_{t+1} = step(x_t)."""
    xs = [np.array(x0, dtype=float)]
    for _ in range(T):
        xs.append(step(xs[-1]))
    return np.array(xs)


def fixed_point_iteration(xs, tol):
    """First t with ||x_{t+1} - x_t||_inf <= tol (None if there is none)."""
    small = np.where(np.abs(np.diff(xs, axis=0)).max(axis=1) <= tol)[0]
    return int(small[0]) if len(small) else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--x0s", type=float, nargs="+",
                   default=[0.5, 0.5, 0.5, 0.5, 0.5,
                            1.5, 0.5, 1.0, 2.0, 2.0],
                   help="initial points as a flat list of 5-tuples x1 ... x5 x1 ... x5 ...")
    p.add_argument("--mu", type=float, default=0.05)
    p.add_argument("--eta", type=float, default=0.025)
    p.add_argument("--T", type=int, default=50)
    p.add_argument("--tol", type=float, default=0.0001, help="fixed-point tolerance for FP-EG")
    p.add_argument("--out", type=str, default="results_nash_gap")
    args = p.parse_args()

    game = RoutingGame()
    os.makedirs(args.out, exist_ok=True)
    x0s = np.array(args.x0s).reshape(-1, game.m)
    for x0 in x0s:
        assert game.slack(x0).min() > 0, f"x0={x0} not strictly feasible"
    if not args.eta < 1.0 / game.L:
        raise ValueError(f"eta={args.eta:g} violates eta < 1/L = {1 / game.L:.4f}")

    lb_label = rf"Practical Log-Barrier Regularization ($\mu$={args.mu:g})"
    fp_label = rf"FP-EG ($\eta$={args.eta:g})"
    lb_col, fp_col = "#2c5aa0", "#d95f02"

    fig, axes = plt.subplots(1, len(x0s), figsize=(5 * len(x0s), 4.2), squeeze=False)
    t = np.arange(args.T + 1)
    for ax, x0 in zip(axes[0], x0s):
        xs = trajectory(lambda x: lb_step(game, x, args.mu), x0, args.T)
        ax.plot(t, [game.gap(x) for x in xs], color=lb_col, lw=2, label=lb_label)

        xs = trajectory(lambda x: fp_eg_step(game, x, args.eta)[0], x0, args.T)
        t_conv = fixed_point_iteration(xs, args.tol)
        t_end = args.T if t_conv is None else t_conv
        gaps = [game.gap(x) for x in xs[:t_end + 1]]
        ax.plot(t[:t_end + 1], gaps, color=fp_col, lw=2, label=fp_label)
        if t_conv is not None:
            ax.plot(t_conv, gaps[-1], "o", color=fp_col, ms=7, zorder=4)
            ha = "right" if t_conv > 0.85 * args.T else "center"  # keep text inside the axes
            ax.annotate(f"converged at t={t_conv}", (t_conv, gaps[-1]), xytext=(0, 60),
                        textcoords="offset points", ha=ha, fontsize=9, color=fp_col,
                        arrowprops=dict(arrowstyle="->", color=fp_col, lw=1))

        ax.set_ylim(bottom=0)
        ax.set_xlabel(r"iteration $t$")
        ax.set_title(r"$x_0$ = (" + ", ".join(f"{v:g}" for v in x0) + ")", fontsize=10)
        ax.grid(alpha=0.25, linewidth=0.6)
    axes[0, 0].set_ylabel(r"Nash-Gap$(x_t)$")
    fig.tight_layout()
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="lower center",
               bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=9, frameon=False,
               title=rf"converged: $\|x_{{t+1}}-x_t\|_\infty \leq$ {args.tol:g}",
               title_fontsize=9)
    for ext in ("png", "pdf"):                    # pdf is vector, for the paper
        path = os.path.join(args.out, f"nash_gap_compare.{ext}")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"saved: {path}")


if __name__ == "__main__":
    main()
