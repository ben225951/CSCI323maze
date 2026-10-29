"""Value Iteration (planning), Q-learning and SARSA (learning from experience)."""
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .env import tiles
from .evaluate import evaluate_q


# --------------------------------------------------------------------------- Value Iteration
@dataclass
class VIResult:
    V: np.ndarray              # converged state values
    Q: np.ndarray              # action values computed from V
    policy: np.ndarray         # greedy action per state
    max_deltas: list           # largest value change in each sweep
    sweep_times: list          # cumulative planning time (s) at the end of each sweep
    V_history: list            # copy of V after each sweep (for "policy after sweep k")
    n_sweeps: int
    n_state_updates: int


def q_from_v(P, V, gamma, n_states, n_actions):
    """Q(s, a) = sum over outcomes of probability * (reward + gamma * V(next state))."""
    Q = np.zeros((n_states, n_actions))
    for s in range(n_states):
        for a in range(n_actions):
            Q[s, a] = sum(p * (r + (0.0 if done else gamma * V[s2])) for p, s2, r, done in P[s][a])
    return Q


def value_iteration(P, n_states, n_actions, gamma, theta, terminal):
    """In-place Value Iteration. Terminal states keep value 0 and are not updated."""
    V = np.zeros(n_states)
    terminal = set(terminal)
    states = [s for s in range(n_states) if s not in terminal]
    max_deltas, sweep_times, V_history = [], [], []
    n_updates = 0
    start = time.perf_counter()
    while True:
        delta = 0.0
        for s in states:
            best = max(
                sum(p * (r + (0.0 if done else gamma * V[s2])) for p, s2, r, done in P[s][a])
                for a in range(n_actions)
            )
            delta = max(delta, abs(best - V[s]))
            V[s] = best
            n_updates += 1
        sweep_times.append(time.perf_counter() - start)
        max_deltas.append(delta)
        V_history.append(V.copy())
        if delta < theta:
            break
    Q = q_from_v(P, V, gamma, n_states, n_actions)
    return VIResult(V, Q, np.argmax(Q, axis=1), max_deltas, sweep_times, V_history,
                    len(max_deltas), n_updates)


# --------------------------------------------------------------------------- Q-learning / SARSA
def epsilon_at(episode, start, end, decay_episodes):
    """Exponential decay from ``start`` to ``end`` over ``decay_episodes``, then flat."""
    frac = min(episode, decay_episodes) / decay_episodes
    return start * (end / start) ** frac


def epsilon_greedy(Q, s, epsilon, rng):
    """Random action with probability epsilon, otherwise the best action (ties broken randomly)."""
    return epsilon_greedy_explored(Q, s, epsilon, rng)[0]


def epsilon_greedy_explored(Q, s, epsilon, rng):
    """``epsilon_greedy`` that also says whether the action was a random (exploring) one."""
    if rng.random() < epsilon:
        return int(rng.integers(Q.shape[1])), True
    q = Q[s]
    return int(rng.choice(np.flatnonzero(q == q.max()))), False


@dataclass
class TDResult:
    Q: np.ndarray
    episodes: pd.DataFrame      # one row per training episode
    checkpoints: pd.DataFrame   # one row per evaluation snapshot
    extra: dict = field(default_factory=dict)


def train_td(method, env, eval_env, n_episodes, gamma, alpha, epsilon_start, epsilon_end,
             eps_decay_episodes, q_init, seed, eval_every, eval_episodes, callback=None,
             step_callback=None):
    """Train Q-learning (``method="q_learning"``) or SARSA (``method="sarsa"``).

    Both share everything (exploration, alpha, initial Q, budget); only the update target differs:
      Q-learning: reward + gamma * max_a Q(next_state, a)
      SARSA:      reward + gamma * Q(next_state, next_action actually chosen)

    The wall-clock time recorded covers training only; checkpoint evaluations are not counted.

    ``callback(Q, episode_row, checkpoint_row_or_None)`` is called after every episode (outside the
    timed part), e.g. to show live progress. If it returns True, training stops early.

    ``step_callback(step, Q)`` is called after every Q-update with a dict describing the step
    (state, action, reward, next_state, explored, target, q_before, q_after, ...), e.g. to animate the
    agent. Time spent in it is not counted as training time.
    """
    if method not in ("q_learning", "sarsa"):
        raise ValueError(method)
    rng = np.random.default_rng(seed)
    n_states, n_actions = env.observation_space.n, env.action_space.n
    Q = np.full((n_states, n_actions), q_init, dtype=float)
    tile_of = tiles(env)
    env.reset(seed=seed)

    episode_rows, checkpoint_rows = [], []
    n_updates, train_time = 0, 0.0

    def checkpoint(episode):
        summary = evaluate_q(eval_env, Q, eval_episodes)
        checkpoint_rows.append({"episode": episode, "q_updates": n_updates, "train_time_s": train_time,
                                **summary})

    checkpoint(0)
    for episode in range(1, n_episodes + 1):
        epsilon = epsilon_at(episode - 1, epsilon_start, epsilon_end, eps_decay_episodes)
        t0 = time.perf_counter()
        s, _ = env.reset()
        a, explored = epsilon_greedy_explored(Q, s, epsilon, rng)
        ep_return, steps = 0.0, 0
        while True:
            s2, r, terminated, truncated, _ = env.step(a)
            ep_return += r
            steps += 1
            if method == "sarsa":
                a2, explored2 = epsilon_greedy_explored(Q, s2, epsilon, rng)
                target = r if terminated else r + gamma * Q[s2, a2]
            else:
                target = r if terminated else r + gamma * Q[s2].max()
            q_before = Q[s, a]
            Q[s, a] += alpha * (target - Q[s, a])
            n_updates += 1
            if step_callback is not None:
                paused = time.perf_counter()
                step_callback({"episode": episode, "step": steps, "epsilon": epsilon, "state": s, "action": a,
                               "reward": r, "next_state": s2, "explored": explored, "target": target,
                               "q_before": q_before, "q_after": Q[s, a],
                               "done": terminated or truncated}, Q)
                t0 += time.perf_counter() - paused
            if terminated or truncated:
                break
            if method == "q_learning":
                a2, explored2 = epsilon_greedy_explored(Q, s2, epsilon, rng)
            s, a, explored = s2, a2, explored2
        train_time += time.perf_counter() - t0
        outcome = {"G": "goal", "H": "hole"}[tile_of[s2]] if terminated else "timeout"
        episode_rows.append({"episode": episode, "epsilon": epsilon, "return": ep_return, "steps": steps,
                             "outcome": outcome, "q_updates": n_updates, "train_time_s": train_time})
        is_checkpoint = episode % eval_every == 0
        if is_checkpoint:
            checkpoint(episode)
        if callback is not None and callback(Q, episode_rows[-1], checkpoint_rows[-1] if is_checkpoint else None):
            break

    return TDResult(Q, pd.DataFrame(episode_rows), pd.DataFrame(checkpoint_rows))
