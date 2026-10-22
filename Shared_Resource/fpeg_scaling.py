"""
Scaling of FP-EG on the shared-resource game (shared_resource_game.py) with the number of
players m: iterations until convergence, number of fallback (unilateral) steps, and wall-clock
time, as median and IQR band over random instances (v drawn with g(v) > B), with each run
shown as a dot.

Convergence: ||x_{t+1} - x_t||_inf <= tol (same rule as run_fp_eg). Wall-clock time counts
only the FP-EG steps, not the diagnostics.

Usage:
    python fpeg_scaling.py
    python fpeg_scaling.py --ms 2 5 10 20 50 100 --seeds 20 --eta 0.9
"""

import argparse
import os
import time

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from shared_resource_game import SharedResourceGame, fp_eg_step


def run(game, x0, eta, T, tol):
    """FP-EG from x0 until convergence; returns (iterations, fallbacks, seconds, final gap)."""
    x, n_fallback, elapsed = x0.copy(), 0, 0.0
    for t in range(1, T + 1):
        start = time.perf_counter()
        x_next, _, _, full = fp_eg_step(game, x, eta)
        elapsed += time.perf_counter() - start
        n_fallback += not full
        if np.abs(x_next - x).max() <= tol:
            return t, n_fallback, elapsed, game.gap(x_next)
        x = x_next
    return T, n_fallback, elapsed, game.gap(x)


def plot_stats(xs, stats, xlabel, title, path_stem, logx, logy=(False, False, True),
               xticks=None, logy_subs=(1.0,)):
    """1x3 panels (iterations, fallbacks, time): median, IQR band and individual runs vs xs.
    stats[x] is an array (runs, 3) of iterations, fallbacks, time in ms; logy per panel.
    xticks: tick positions on a linear x axis; logy_subs: labelled log ticks per decade."""
    xs = np.array(xs)
    panels = [("Iterations until convergence", "iterations"),
              ("Fallback (unilateral) steps", "fallback steps"),
              ("Wall-clock time", "time [ms]")]
    col = "#d95f02"
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for k, (ax, (panel, ylabel)) in enumerate(zip(axes, panels)):
        vals = [stats[x][:, k] for x in xs]
        q1, med, q3 = (np.array([np.percentile(v, q) for v in vals]) for q in (25, 50, 75))
        rng = np.random.default_rng(0)
        for x, v in zip(xs, vals):               # individual runs, jittered horizontally
            jit = rng.uniform(-0.08, 0.08, len(v))
            ax.scatter(x * np.exp(jit) if logx else x + 0.15 * np.diff(xs).min() * jit / 0.08,
                       v, s=8, color=col, alpha=0.25, lw=0)
        ax.fill_between(xs, q1, q3, color=col, alpha=0.2, lw=0)
        ax.plot(xs, med, "-o", color=col, lw=2, ms=5, label="median (band: IQR)")
        if logx:                                 # one tick per value of m
            ax.set_xscale("log")
            ax.minorticks_off()
            ax.set_xticks(xs)
            ax.set_xticklabels([f"{x:g}" for x in xs])
        elif xticks is not None:
            ax.set_xticks(xticks)
        if logy[k]:
            ax.set_yscale("log")
            if logy_subs != (1.0,):              # e.g. (1, 2, 5): label 1, 2, 5 x 10^k
                ax.yaxis.set_major_locator(ticker.LogLocator(subs=logy_subs))
                ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{y:g}"))
                ax.yaxis.set_minor_formatter(ticker.NullFormatter())
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(panel)
        ax.grid(alpha=0.25, linewidth=0.6)
    axes[0].legend(fontsize=9, frameon=False, loc="upper left")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    for ext in ("png", "pdf"):                    # pdf is vector, for the paper
        path = f"{path_stem}.{ext}"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"saved: {path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ms", type=int, nargs="+", default=[2, 5, 10, 20, 50, 100, 200, 500, 1000])
    p.add_argument("--seeds", type=int, default=50, help="random instances per m")
    p.add_argument("--eta", type=float, default=0.9)
    p.add_argument("--T", type=int, default=5000)
    p.add_argument("--tol", type=float, default=1e-10)
    p.add_argument("--out", type=str, default="results_scaling")
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    stats = {}                                   # m -> array (seeds, 3): iterations, fallbacks, time
    for m in args.ms:
        rows = []
        for seed in range(args.seeds):
            game = SharedResourceGame(m, seed=seed)
            it, fb, sec, gap = run(game, np.zeros(m), args.eta, args.T, args.tol)
            if it == args.T:
                print(f"  m={m} seed={seed}: not converged in T={args.T} (gap={gap:.1e})")
            rows.append((it, fb, 1e3 * sec))
        stats[m] = np.array(rows)
        med = np.median(stats[m], axis=0)
        print(f"m={m:4d}: median iterations {med[0]:.0f}, fallbacks {med[1]:.0f}, time {med[2]:.1f} ms")

    plot_stats(args.ms, stats, r"number of players $m$",
               rf"FP-EG on the shared-resource game ($\eta$={args.eta:g}, $x_0=0$, "
               rf"{args.seeds} instances per $m$, converged: $\|x_{{t+1}}-x_t\|_\infty \leq$ {args.tol:g})",
               os.path.join(args.out, "fpeg_scaling"), logx=True)


if __name__ == "__main__":
    main()
