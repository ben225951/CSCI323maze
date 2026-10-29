"""Run experiments and cache their raw results as CSV files in ``results/``.

``load_vi`` / ``load_td`` return cached results if the CSVs exist, otherwise they run the
experiment first. Pass ``rerun=True`` to force a fresh run (e.g. after changing the config).
"""
import json

import numpy as np
import pandas as pd

from . import agents
from .config import CONFIG
from .env import build_model, make_env, terminal_states
from .evaluate import first_stable, greedy_policy, run_policy, summarise

TD_METHODS = {"Q-learning": "q_learning", "SARSA": "sarsa"}


def _path(name):
    CONFIG["results_dir"].mkdir(parents=True, exist_ok=True)
    return CONFIG["results_dir"] / name


def _files(prefix, map_key, parts):
    return {part: _path(f"{prefix}_{map_key}_{part}.csv") for part in parts}


# --------------------------------------------------------------------------- Value Iteration
VI_PARTS = ["sweeps", "states", "test", "summary"]


def run_vi(map_key):
    env = make_env(map_key)
    env.reset(seed=0)  # fixes the slips during testing, so slippery results are reproducible
    P = build_model(env)
    n_states, n_actions = env.observation_space.n, env.action_space.n
    vi_cfg, gamma = CONFIG["value_iteration"], CONFIG["gamma"]
    args = (P, n_states, n_actions, gamma, vi_cfg["theta"], terminal_states(env))

    # Re-run planning several times; every run does identical work, only the timing varies.
    runs = [agents.value_iteration(*args) for _ in range(vi_cfg["timing_repeats"])]
    totals = [r.sweep_times[-1] for r in runs]
    result = runs[int(np.argsort(totals)[len(totals) // 2])]  # the run with the median total time
    median_sweep_times = np.median([r.sweep_times for r in runs], axis=0)

    # Policy after each sweep (evaluated afterwards, so not part of the planning time).
    rows = []
    for k, V in enumerate(result.V_history, start=1):
        policy = greedy_policy(agents.q_from_v(P, V, gamma, n_states, n_actions))
        summary = summarise(run_policy(env, policy, CONFIG["evaluation"]["eval_episodes"]))
        rows.append({"sweep": k, "max_delta": result.max_deltas[k - 1],
                     "state_updates": k * (n_states - len(args[-1])),
                     "planning_time_s": median_sweep_times[k - 1], **summary})
    sweeps = pd.DataFrame(rows)

    states = pd.DataFrame({"state": range(n_states), "value": result.V, "policy": result.policy,
                           **{f"q{a}": result.Q[:, a] for a in range(n_actions)}})
    test = run_policy(env, result.policy, CONFIG["evaluation"]["test_episodes"])
    summary = pd.DataFrame([{
        "sweeps": result.n_sweeps,
        "state_updates": result.n_state_updates,
        "planning_time_s_median": float(np.median(totals)),
        "planning_time_s_min": float(np.min(totals)),
        "planning_time_s_max": float(np.max(totals)),
        "timing_repeats": len(totals),
        "optimal_steps": summarise(test)["path_length"],
        "success_rate": summarise(test)["success_rate"],
    }])

    files = _files("vi", map_key, VI_PARTS)
    for part, df in zip(VI_PARTS, [sweeps, states, test, summary]):
        df.to_csv(files[part], index=False)
    return dict(zip(VI_PARTS, [sweeps, states, test, summary]))


def load_vi(map_key, rerun=False):
    files = _files("vi", map_key, VI_PARTS)
    if rerun or not all(f.exists() for f in files.values()):
        return run_vi(map_key)
    return {part: pd.read_csv(f) for part, f in files.items()}


# --------------------------------------------------------------------------- Q-learning / SARSA
TD_PARTS = ["episodes", "checkpoints", "qtables", "test", "stable"]


def run_td(method, map_key, seeds=None, progress=True):
    """Train ``method`` ("Q-learning" or "SARSA") on ``map_key`` once per seed."""
    seeds = CONFIG["seeds"] if seeds is None else seeds
    cfg, td, ev = CONFIG["maps"][map_key], CONFIG["td"], CONFIG["evaluation"]
    vi_summary = load_vi(map_key)["summary"].iloc[0]
    if CONFIG["slippery"]:  # no fixed route: "optimal" = about as successful as Value Iteration's policy
        optimal_steps, min_success = None, vi_summary["success_rate"] - ev["stable_tolerance"]
    else:
        optimal_steps, min_success = vi_summary["optimal_steps"], 100.0
    env, eval_env = make_env(map_key), make_env(map_key)

    parts = {p: [] for p in TD_PARTS}
    for i, seed in enumerate(seeds, start=1):
        eval_env.reset(seed=seed)  # fixes the slips during testing, so slippery results are reproducible
        res = agents.train_td(
            TD_METHODS[method], env, eval_env, n_episodes=cfg["n_episodes"], gamma=CONFIG["gamma"],
            alpha=td["alpha"], epsilon_start=td["epsilon_start"], epsilon_end=td["epsilon_end"],
            eps_decay_episodes=cfg["eps_decay_episodes"], q_init=td["q_init"], seed=seed,
            eval_every=ev["eval_every"], eval_episodes=ev["eval_episodes"],
        )
        parts["episodes"].append(res.episodes.assign(seed=seed))
        parts["checkpoints"].append(res.checkpoints.assign(seed=seed))
        q = pd.DataFrame(res.Q, columns=[f"q{a}" for a in range(res.Q.shape[1])])
        parts["qtables"].append(q.assign(state=np.arange(len(q)), seed=seed))
        parts["test"].append(run_policy(eval_env, greedy_policy(res.Q), ev["test_episodes"]).assign(seed=seed))
        stable = first_stable(res.checkpoints, optimal_steps, ev["stable_k"], min_success)
        parts["stable"].append(pd.DataFrame([{"seed": seed, **stable}]))
        if progress:
            print(f"\r{method} {map_key}: seed {i}/{len(seeds)}", end="", flush=True)
    if progress:
        print()

    out = {p: pd.concat(dfs, ignore_index=True) for p, dfs in parts.items()}
    files = _files(TD_METHODS[method], map_key, TD_PARTS)
    for p, df in out.items():
        df.to_csv(files[p], index=False)
    return out


def load_td(method, map_key, rerun=False):
    files = _files(TD_METHODS[method], map_key, TD_PARTS)
    if rerun or not all(f.exists() for f in files.values()):
        return run_td(method, map_key)
    return {p: pd.read_csv(f) for p, f in files.items()}


def q_table(qtables, seed):
    """Q-table (states x actions) for one seed from the ``qtables`` CSV."""
    df = qtables[qtables["seed"] == seed].sort_values("state")
    return df[[c for c in df.columns if c.startswith("q")]].to_numpy()


def save_config_snapshot():
    """Write the config used for the cached results (for the report's appendix)."""
    cfg = {k: v for k, v in CONFIG.items() if k != "results_dir"}
    _path("config_used.json").write_text(json.dumps(cfg, indent=2))
