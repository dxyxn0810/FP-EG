"""
Stepsize sensitivity of FP-EG on the shared-resource game (shared_resource_game.py) for a
fixed number of players m: iterations until convergence, number of fallback (unilateral)
steps, and wall-clock time versus eta in (0, 1/L) = (0, 1), as median and IQR band over
random instances (v drawn with g(v) > B), with each run shown as a dot.

Convergence: first t with Gap(x_t) <= eps (same rule as run_fp_eg). Wall-clock time counts
only the FP-EG steps, not the diagnostics (the Nash-gap checks included).

Usage:
    python fpeg_eta.py
    python fpeg_eta.py --m 10 --etas 0.1 0.3 0.5 0.7 0.9 --seeds 50
"""

import argparse
import os

import numpy as np

from shared_resource_game import SharedResourceGame
from fpeg_scaling import run, plot_stats


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--m", type=int, default=50)
    p.add_argument("--etas", type=float, nargs="+",
                   default=[0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95])
    p.add_argument("--seeds", type=int, default=45, help="random instances")
    p.add_argument("--T", type=int, default=5000)
    p.add_argument("--eps", type=float, default=1e-6, help="Nash-gap level for convergence")
    p.add_argument("--out", type=str, default="results_eta")
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    games = [SharedResourceGame(args.m, seed=seed) for seed in range(args.seeds)]
    stats = {}                                   # eta -> array (seeds, 3): iterations, fallbacks, time
    for eta in args.etas:
        if not eta < 1.0 / games[0].L:
            raise ValueError(f"eta={eta:g} violates eta < 1/L = {1 / games[0].L:g}")
        rows = []
        for seed, game in enumerate(games):
            it, fb, sec, gap = run(game, np.zeros(args.m), eta, args.T, args.eps)
            if it == args.T:
                print(f"  eta={eta:g} seed={seed}: not converged in T={args.T} (gap={gap:.1e})")
            rows.append((it, fb, 1e3 * sec))
        stats[eta] = np.array(rows)
        med = np.median(stats[eta], axis=0)
        print(f"eta={eta:5g}: median iterations {med[0]:.0f}, fallbacks {med[1]:.0f}, time {med[2]:.2f} ms")

    xticks = np.round(np.arange(0.0, 1.01, 0.1), 1)
    plot_stats(args.etas, stats, r"stepsize $\eta$",
               os.path.join(args.out, "fpeg_eta"), logx=False, logy=(False, False, False),
               xticks=xticks)
    plot_stats(args.etas, stats, r"stepsize $\eta$",          # iterations panel only
               os.path.join(args.out, "fpeg_eta_iterations"), logx=False, logy=(False, False, False),
               xticks=xticks, panels=(0,))


if __name__ == "__main__":
    main()
