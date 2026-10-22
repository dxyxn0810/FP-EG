# Beyond Joint Convexity: Existence and Feasibility-Preserving Computation of Constrained Nash Equilibria

Code for the numerical experiments in the paper, on **FP-EG** (Feasibility-Preserving ExtraGradient,
Algorithm 1). FP-EG finds Nash equilibria of potential games whose players share a
coupling constraint, and every iterate stays feasible.

Each FP-EG step does the following:

1. **Predictor / corrector.** An extragradient step in which each player's update is projected onto its
   feasible-response set given the other players' current strategies,
   `K(x) = C_1(x_{-1}) x ... x C_m(x_{-m})`.
2. **Fallback.** If the joint corrector point is infeasible, a single player moves unilaterally,
   chosen greedily to decrease the potential the most.

In all games below, every `C_i(x_{-i})` is an interval, so the projection onto `K(x)` is a
coordinatewise clip and the Nash gap can be computed exactly.

## Repository structure

```
.
├── Cooperative/            # 2-player identical-interest game, non-convex constraint x1*x2 <= 1-alpha
│   ├── coop_game.py                  # game, FP-EG, log-barrier gradient ascent
│   ├── coop_game_trajectory.py       # trajectory plot: log-barrier vs. FP-EG
│   └── results_trajectory/
├── Routing/                # 5-player network routing game with link capacities
│   ├── routing_game.py               # game, FP-EG, log-barrier gradient ascent
│   ├── routing_nash_gap.py           # Nash gap vs. iteration: log-barrier vs. FP-EG
│   └── results_nash_gap/
└── Shared_Resource/        # m-player game with a non-convex (star-shaped) shared capacity constraint
    ├── shared_resource_game.py       # game, FP-EG
    ├── fpeg_scaling.py               # scaling of FP-EG with the number of players m
    ├── fpeg_eta.py                   # sensitivity of FP-EG to the stepsize eta
    ├── results_scaling/
    └── results_eta/
```

Each `*_game.py` file is self-contained. Its module docstring defines the game (utilities,
constraints, potential, Lipschitz constant `L`, feasible-response sets). The other scripts produce
the figures.

## Experiments

| Folder | Game | Baseline | Script | Output |
|---|---|---|---|---|
| `Cooperative` | Cooperative game of Jordan & Kamgarpour (2026), Sec. 5 | Log-barrier gradient ascent, `mu = 0.001` | `coop_game_trajectory.py` | `results_trajectory/trajectory_compare.{png,pdf}` |
| `Routing` | Network routing game (Tatarenko 2017, Ex. 2.2), as in Jordan & Kamgarpour (2026), Sec. 5 | Log-barrier gradient ascent, `mu = 0.05` | `routing_nash_gap.py` | `results_nash_gap/nash_gap_compare.{png,pdf}` |
| `Shared_Resource` | Pairwise-overhead shared-resource game (Sec. 3.4 / App. C.1), `m` players, 50 random instances | — | `fpeg_scaling.py` | `results_scaling/fpeg_scaling.{png,pdf}` |
| `Shared_Resource` | Same, `m = 50` | — | `fpeg_eta.py` | `results_eta/fpeg_eta.{png,pdf}` |

Default stepsizes are `eta = 0.1` (Cooperative), `eta = 0.025 < 1/L ≈ 0.027` (Routing), and
`eta = 0.9 < 1/L = 1` (Shared-Resource scaling). An iteration counts as converged when
`||x_{t+1} - x_t||_inf <= tol`.

## Requirements

- Python 3.9+ (tested with Python 3.13)
- NumPy, Matplotlib

```bash
pip install -r requirements.txt
```

## Reproducing the figures

Run each script from inside its folder, because the scripts import the game module next to them:

```bash
cd Cooperative     && python coop_game_trajectory.py && cd ..
cd Routing         && python routing_nash_gap.py     && cd ..
cd Shared_Resource && python fpeg_scaling.py && python fpeg_eta.py && cd ..
```

With the default arguments, these commands reproduce the figures in the `results_*` folders.
Each figure is saved as PNG and as PDF (vector, for the paper). Every script accepts command-line
options, listed by `--help`. For example:

```bash
python coop_game_trajectory.py --mu 0.05 --eta 0.1 --T 500 --x0s 0.2 0.05 0.05 0.7
python routing_nash_gap.py --eta 0.02 --T 100 --x0s 0.5 0.5 0.5 0.5 0.5
python fpeg_scaling.py --ms 2 5 10 20 50 100 --seeds 20 --eta 0.9
python fpeg_eta.py --m 10 --etas 0.1 0.3 0.5 0.7 0.9 --seeds 50
```

The Shared-Resource instances are seeded (`seed = 0, ..., seeds-1`), so iteration and fallback
counts are deterministic. Wall-clock times depend on the machine. With the defaults
(`m` up to 1000, 50 instances each), `fpeg_scaling.py` takes noticeably longer than the other scripts.

## Using the code

```python
import numpy as np
from shared_resource_game import SharedResourceGame, run_fp_eg

game = SharedResourceGame(m=20, seed=0)
xs = run_fp_eg(game, eta=0.9, tol=1e-10)     # iterates x_0, ..., x_t (all feasible)
print(len(xs) - 1, "iterations, Nash gap:", game.gap(xs[-1]))
```

## References

- Jordan and Kamgarpour (2026). *Nash Equilibria in Games with Playerwise Concave Coupling Constraints: Existence and Computation*.
- T. Tatarenko (2017). *Game-Theoretic Learning and Distributed Optimization in Memoryless Multi-Agent Systems*.
