"""Testing a FROZEN greedy policy: no exploration, no learning.

The same functions are used for all three methods, so the testing metrics are identical.
"""
import warnings

import numpy as np
import pandas as pd


def greedy_policy(Q):
    """Best action per state. Ties go to the lowest action id, so the test policy is deterministic."""
    return np.argmax(Q, axis=1)


def run_policy(env, policy, n_episodes):
    """Play ``n_episodes`` with a fixed policy and return one row per episode.

    ``outcome`` is "goal", "hole" or "timeout" (hit the step limit without finishing).
    """
    policy = np.asarray(policy)
    tile_of = [c.decode() for c in env.unwrapped.desc.flatten()]
    rows = []
    for _ in range(n_episodes):
        s, _ = env.reset()
        total, steps = 0.0, 0
        while True:
            s, r, terminated, truncated, _ = env.step(int(policy[s]))
            total += r
            steps += 1
            if terminated or truncated:
                break
        outcome = {"G": "goal", "H": "hole"}.get(tile_of[s], "timeout") if terminated else "timeout"
        rows.append({"outcome": outcome, "success": outcome == "goal", "steps": steps, "return": total})
    return pd.DataFrame(rows)


def summarise(episodes):
    """Testing metrics from ``run_policy`` output.

    Path length is averaged over successful episodes only (NaN if there were none);
    failures are counted separately.
    """
    ok = episodes[episodes["success"]]
    return {
        "success_rate": 100.0 * episodes["success"].mean(),
        "path_length": ok["steps"].mean() if len(ok) else np.nan,
        "return": episodes["return"].mean(),
        "n_hole": int((episodes["outcome"] == "hole").sum()),
        "n_timeout": int((episodes["outcome"] == "timeout").sum()),
    }


def evaluate_q(env, Q, n_episodes):
    """Evaluate the greedy policy of ``Q`` without being able to change ``Q``.

    The policy is computed from a read-only view, so any attempt to write to the table during
    evaluation would raise an error.
    """
    frozen = Q.view()
    frozen.flags.writeable = False
    return summarise(run_policy(env, greedy_policy(frozen), n_episodes))


def first_stable(checkpoints, optimal_steps, k, min_success=100.0):
    """First checkpoint at which the greedy policy has been optimal for ``k`` checkpoints in a row.

    Optimal = success rate of at least ``min_success`` and, unless ``optimal_steps`` is None,
    a path of exactly ``optimal_steps`` steps. (On slippery ice there is no fixed route, so only
    the success rate is checked.)
    Returns a dict with ``reached`` False (and NaNs) if it never happens.
    """
    good = checkpoints["success_rate"] >= min_success
    if optimal_steps is not None:
        good &= checkpoints["path_length"] == optimal_steps
    good = good.to_numpy()
    run = 0
    for i, ok in enumerate(good):
        run = run + 1 if ok else 0
        if run == k:
            row = checkpoints.iloc[i]
            return {"reached": True, "episode": int(row["episode"]), "q_updates": int(row["q_updates"]),
                    "train_time_s": float(row["train_time_s"])}
    return {"reached": False, "episode": np.nan, "q_updates": np.nan, "train_time_s": np.nan}


def mean_and_ci(values, axis=0):
    """Mean, standard deviation and 95% confidence half-width (normal approximation) over seeds."""
    values = np.asarray(values, dtype=float)
    n = np.sum(~np.isnan(values), axis=axis)
    with warnings.catch_warnings():  # all-NaN columns (e.g. no successful seed yet) give NaN quietly
        warnings.simplefilter("ignore", RuntimeWarning)
        mean = np.nanmean(values, axis=axis)
        std = np.nanstd(values, axis=axis, ddof=1)
    ci = 1.96 * std / np.sqrt(np.maximum(n, 1))
    return mean, std, ci


def value_on_time_grid(times, values, grid):
    """Value of a step function (last observed value at or before each grid time).

    Used to average learning curves of different seeds on a shared wall-clock axis.
    Before the first observation the value is NaN.
    """
    idx = np.searchsorted(np.asarray(times), grid, side="right") - 1
    out = np.asarray(values, dtype=float)[np.clip(idx, 0, None)]
    out[idx < 0] = np.nan
    return out


def seeds_on_time_grid(checkpoints, column, grid):
    """Checkpoint ``column`` of every seed resampled onto a shared wall-clock ``grid``.

    Returns an array of shape (n_seeds, len(grid)), ready for ``mean_and_ci``.
    """
    return np.array([value_on_time_grid(g["train_time_s"], g[column], grid)
                     for _, g in checkpoints.sort_values("episode").groupby("seed")])
