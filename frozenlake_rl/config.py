"""All experiment settings in one place.

Change a number here and every notebook / script picks it up. Results are cached
in ``results/``, so after changing a setting re-run with ``rerun=True`` (or delete
the matching CSV files) to regenerate them.
"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Slippery ice: the agent moves in the chosen direction only 1/3 of the time and slides to one of the two
# perpendicular directions otherwise. Several settings below depend on it; re-run everything after changing it.
SLIPPERY = False

CONFIG = {
    "slippery": SLIPPERY,

    # Shaped rewards. Used by BOTH the gymnasium wrapper (Q-learning, SARSA) and the
    # transition model given to Value Iteration, so every method sees the same rewards.
    "rewards": {
        "goal": 1.0,    # stepping onto G
        "hole": -1.0,   # stepping into H (episode ends in failure)
        "step": -0.01,  # every other step: a small cost that rewards short routes
    },

    # Discount factor, shared by all three methods.
    "gamma": 0.99,

    # Maps. Use a gymnasium built-in ``map_name`` or give your own ``desc``
    # (a list of strings using S, F, H, G), e.g.
    #   "5x5": {"desc": ["SFFFF", "FHFHF", "FFFHF", "HFFFF", "FFHFG"], "max_steps": 100, ...}
    "maps": {
        "4x4": {
            "map_name": "4x4",
            "max_steps": 100,          # episode cap (time limit)
            "n_episodes": 4000 if SLIPPERY else 1000,          # training budget for Q-learning and SARSA
            "eps_decay_episodes": 2000 if SLIPPERY else 500,   # epsilon reaches its minimum after this many episodes
        },
        "8x8": {
            "map_name": "8x8",
            "max_steps": 200,
            "n_episodes": 30000 if SLIPPERY else 3000,
            "eps_decay_episodes": 15000 if SLIPPERY else 1500,
        },
    },

    "value_iteration": {
        "theta": 1e-8,         # stop when the largest value change in a sweep is below this
        "timing_repeats": 7,   # planning is re-run this many times; we report the median time
    },

    # Shared by Q-learning and SARSA so the ONLY difference is the update rule.
    "td": {
        "alpha": 0.1,           # learning rate
        "epsilon_start": 1.0,   # exploration: probability of a random action at the start ...
        "epsilon_end": 0.01,    # ... decaying (exponentially) to this value
        "q_init": 0.0,          # initial value of every Q(state, action)
    },

    # On deterministic ice every test episode of a policy gives the same result, so 1 episode is enough.
    # On slippery ice the outcome is random, so a policy has to be tested on many episodes.
    "evaluation": {
        "eval_every": 250 if SLIPPERY else 50,       # snapshot + evaluate the greedy policy every N training episodes
        "eval_episodes": 100 if SLIPPERY else 1,     # test episodes per checkpoint
        "test_episodes": 1000 if SLIPPERY else 10,   # test episodes for the final, trained policy
        "stable_k": 5,        # "stable" = near-optimal in K consecutive checkpoints:
                              #   deterministic: 100% success by the shortest route
                              #   slippery: success rate within stable_tolerance of Value Iteration's
        "stable_tolerance": 5.0,  # percentage points (slippery only)
    },

    "seeds": list(range(20)),  # training seeds for Q-learning and SARSA

    "results_dir": PROJECT_ROOT / "results",
}

METHODS = ["Value Iteration", "Q-learning", "SARSA"]
