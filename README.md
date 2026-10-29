# FrozenLake testbed: Value Iteration vs Q-learning vs SARSA

A small, reproducible experiment comparing one **planning** method (Value Iteration) with two **learning** methods
(Q-learning and SARSA) on non-slippery FrozenLake (`gymnasium`, `is_slippery=False`), on the 4x4 and 8x8 maps.
To try slippery ice instead, set `SLIPPERY = True` in `frozenlake_rl/config.py` and recompute the results
(training budgets and test sizes change automatically).

## Install

Python 3.10 or newer.

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
jupyter notebook                   # then open the notebooks in order, "Restart & Run All"
```

| Notebook | Shows |
|---|---|
| `01_environment.ipynb` | Maps, shaped rewards, why not slippery, reward-model check |
| `02_value_iteration.ipynb` | Convergence, value heatmap, policy arrows, planning cost |
| `03_q_learning.ipynb` | Learning curves over 20 seeds, stable-performance point, policy and Q-values |
| `04_sarsa.ipynb` | Same as 03 for SARSA |
| `05_comparison.ipynb` | Side-by-side test metrics, costs in each method's own units, performance vs wall-clock time, policy differences, discussion |

Notebooks load cached results from `results/` if they exist and otherwise compute them. Recomputing everything takes
under a minute on a laptop (about 15 minutes with slippery ice). To recompute all results from the command line (e.g. after changing the config):

```bash
python run_all.py
```

or set `RERUN = True` in the first code cell of a notebook. Figures are also saved as PNGs in `figures/` for the
report and slides.

## Running the GUI

A small window that shows a penguin learning to cross the ice, using the same environment, rewards and algorithms
as the experiments:

```bash
python gui.py              # 4x4 map
python gui.py --map 8x8    # 8x8 map (--seed N picks the training seed, default 0)
```

| Key | Action |
|---|---|
| 1 / 2 / 3 | Value Iteration / Q-learning / SARSA (restarts that method) |
| Space | pause / resume |
| Up / Down | faster / slower |
| S | skip ahead 50 episodes without animating (Value Iteration: skip to the end of planning) |
| R | restart, Esc: quit |

- **Value Iteration**: the penguin waits on Start while each sweep tints the tiles from cold (looks bad) to warm
  (looks good), with arrows for the best move Once planning has converged, it plays
  one episode with the final policy.
- **Q-learning / SARSA**: every training episode is animated. Yellow footprints are best-looking (greedy) moves,
  pink ones are random (exploring) moves. The real `train_td` runs in a background thread and pauses after each
  step until it has been drawn.

The GUI uses [pygame-ce](https://pyga.me), a maintained fork of pygame with the same `import pygame` API. The original
`pygame` package has no build for Python 3.14 yet.

Run the tests:

```bash
pytest
```

## Layout

```
frozenlake_rl/
  config.py       all hyperparameters (rewards, gamma, maps, episode budgets, epsilon, seeds, ...)
  env.py          make_env(), the shaped-reward wrapper, the shaped transition model for Value Iteration
  agents.py       value_iteration(), train_td() for Q-learning / SARSA
  evaluate.py     frozen greedy-policy testing, "stable performance" detection, mean / 95% CI helpers
  experiments.py  run + cache experiments as CSV in results/
  plotting.py     shared plots (one colour per method everywhere)
tests/            reward wrapper vs model (deterministic and slippery), VI shortest path on deterministic 4x4, evaluation never changes Q, ...
results/          raw results (CSV)
figures/          PNG copies of the main plots
run_all.py        recompute all results
gui.py            pygame window that animates the three methods learning
```

## Key design points

- **Same rewards everywhere.** One function, `shaped_reward()`, sets the rewards (+1 goal, −1 hole, −0.01 per step).
  The gymnasium wrapper and the model given to Value Iteration both use it, and a test checks they agree for every (state, action).
  (`env.unwrapped.P` contains the *original* rewards, so it isn't used directly.)
- **Training and testing are separate.** Testing always uses a frozen greedy policy computed from a read-only view of the Q-table.
  A test checks that evaluating during training doesn't change what is learned.
- **Q-learning and SARSA differ only in the update target.** Exploration schedule, alpha, initial Q and budget are shared.
- **Own units for cost.** Sweeps/state updates (VI) and episodes/Q-updates (TD) are never put on the same axis.
  Wall-clock time is the only shared axis.
- **Seeds.** Q-learning and SARSA run with 20 seeds. Because the environment and test policy are deterministic, all spread
  comes from training randomness, not from repeated test episodes.
- **Stable performance.** A learning agent is "stable" once its greedy policy reaches the goal by the shortest route
  at 5 checkpoints in a row. (With slippery ice there is no fixed route, so it is "success rate within 5 percentage
  points of Value Iteration's" instead.)

## Adding a map

Add an entry to `CONFIG["maps"]` in `frozenlake_rl/config.py`, using either a built-in `map_name` or your own `desc`
(list of strings made of S/F/H/G), plus `max_steps`, `n_episodes` and `eps_decay_episodes`. The notebooks loop over all
configured maps, although figure sizes are tuned for 4x4 and 8x8.
