from collections import deque

import numpy as np
import pandas as pd
import pytest

from frozenlake_rl import agents
from frozenlake_rl.config import CONFIG
from frozenlake_rl.env import build_model, follow_policy, make_env, start_state, terminal_states, tiles
from frozenlake_rl.evaluate import evaluate_q, first_stable

R = CONFIG["rewards"]


def step_from(env, state, action):
    env.reset()
    env.unwrapped.s = state
    return env.step(action)


def test_wrapper_reward_values():
    env = make_env("4x4", slippery=False)
    # 4x4 map: SFFF / FHFH / FFFH / HFFG  (states numbered row by row from 0)
    assert step_from(env, 0, 2)[1] == R["step"]    # 0 -> 1 (frozen)
    assert step_from(env, 4, 2)[1] == R["hole"]    # 4 -> 5 (hole)
    assert step_from(env, 14, 2)[1] == R["goal"]   # 14 -> 15 (goal)
    assert step_from(env, 0, 0)[1] == R["step"]    # bumping into the wall still costs a step


@pytest.mark.parametrize("map_key", list(CONFIG["maps"]))
def test_model_rewards_match_wrapper(map_key):
    """Value Iteration's model and the wrapper must give identical rewards for every (state, action)."""
    env = make_env(map_key, slippery=False)
    P = build_model(env)
    for s in range(env.observation_space.n):
        if s in terminal_states(env):
            continue
        for a in range(env.action_space.n):
            (_p, s2_model, r_model, done_model), = P[s][a]
            s2, r, terminated, _trunc, _info = step_from(env, s, a)
            assert (s2, r, terminated) == (s2_model, r_model, done_model)


def bfs_shortest_steps(env):
    """Independent check: fewest moves from S to G, avoiding holes."""
    t, (n_rows, n_cols) = tiles(env), env.unwrapped.desc.shape
    start = start_state(env)
    dist, queue = {start: 0}, deque([start])
    while queue:
        s = queue.popleft()
        if t[s] == "G":
            return dist[s]
        r, c = divmod(s, n_cols)
        for dr, dc in [(0, -1), (1, 0), (0, 1), (-1, 0)]:
            rr, cc = r + dr, c + dc
            s2 = rr * n_cols + cc
            if 0 <= rr < n_rows and 0 <= cc < n_cols and t[s2] != "H" and s2 not in dist:
                dist[s2] = dist[s] + 1
                queue.append(s2)


def test_slippery_model_rewards_match_wrapper():
    """On slippery ice each (state, action) has up to 3 outcomes; every reachable one must match the wrapper's reward."""
    env = make_env("4x4", slippery=True)
    P, t = build_model(env), tiles(env)
    for s in range(env.observation_space.n):
        for a in range(env.action_space.n):
            assert sum(p for p, *_ in P[s][a]) == pytest.approx(1.0)
            for _p, s2, r, done in P[s][a]:
                assert r == R[{"G": "goal", "H": "hole"}.get(t[s2], "step")]
                assert done == (t[s2] in "GH")


def test_value_iteration_finds_shortest_path_4x4():
    env = make_env("4x4", slippery=False)
    res = agents.value_iteration(build_model(env), 16, 4, CONFIG["gamma"],
                                 CONFIG["value_iteration"]["theta"], terminal_states(env))
    path = follow_policy(env, res.policy)
    assert tiles(env)[path[-1]] == "G"
    assert len(path) - 1 == bfs_shortest_steps(env) == 6
    assert res.max_deltas[-1] < CONFIG["value_iteration"]["theta"]


def test_evaluation_never_updates_q():
    env = make_env("4x4", slippery=False)
    Q = np.random.default_rng(0).normal(size=(16, 4))
    before = Q.copy()
    evaluate_q(env, Q, n_episodes=5)
    np.testing.assert_array_equal(Q, before)
    assert Q.flags.writeable  # the caller's table is untouched (only a read-only view was used)


def test_checkpoints_do_not_change_training():
    """Training with or without frequent evaluation must produce the same Q-table."""
    kwargs = dict(n_episodes=100, gamma=0.99, alpha=0.1, epsilon_start=1.0, epsilon_end=0.01,
                  eps_decay_episodes=50, q_init=0.0, seed=3, eval_episodes=1)
    a = agents.train_td("q_learning", make_env("4x4", slippery=False), make_env("4x4", slippery=False), eval_every=1, **kwargs)
    b = agents.train_td("q_learning", make_env("4x4", slippery=False), make_env("4x4", slippery=False), eval_every=1000, **kwargs)
    np.testing.assert_array_equal(a.Q, b.Q)


@pytest.mark.parametrize("method", ["q_learning", "sarsa"])
def test_recording_steps_does_not_change_training(method):
    """The animation recorder (step/episode callbacks) must not change what is learned, and its log must replay to Q."""
    kwargs = dict(n_episodes=100, gamma=0.99, alpha=0.1, epsilon_start=1.0, epsilon_end=0.01,
                  eps_decay_episodes=50, q_init=0.0, seed=3, eval_every=50, eval_episodes=1)
    steps = []
    a = agents.train_td(method, make_env("4x4", slippery=False), make_env("4x4", slippery=False), **kwargs)
    b = agents.train_td(method, make_env("4x4", slippery=False), make_env("4x4", slippery=False), callback=lambda *_: None,
                        step_callback=lambda step, _Q: steps.append(step), **kwargs)
    np.testing.assert_array_equal(a.Q, b.Q)
    replay = np.zeros_like(a.Q)
    for step in steps:
        replay[step["state"], step["action"]] = step["q_after"]
    np.testing.assert_array_equal(replay, a.Q)
    assert steps[0]["explored"]  # epsilon starts at 1.0, so the first move is random


def test_first_stable():
    cp = pd.DataFrame({"episode": [0, 50, 100, 150, 200], "q_updates": [0, 5, 10, 15, 20],
                       "train_time_s": [0, .1, .2, .3, .4],
                       "success_rate": [0, 100, 100, 100, 100], "path_length": [np.nan, 8, 6, 6, 6]})
    assert first_stable(cp, 6, 3)["episode"] == 200
    assert not first_stable(cp, 6, 4)["reached"]
