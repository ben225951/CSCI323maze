"""Environment creation, reward shaping and the transition model for Value Iteration."""
import gymnasium as gym
import numpy as np

from .config import CONFIG

ACTION_NAMES = ["left", "down", "right", "up"]  # action ids 0, 1, 2, 3


def map_config(map_key):
    return CONFIG["maps"][map_key]


def shaped_reward(tile, rewards=None):
    """Reward for stepping onto a tile ('S', 'F', 'H' or 'G').

    This single function defines the reward for both the wrapper and the model, so the
    two can never disagree.
    """
    rewards = rewards or CONFIG["rewards"]
    if tile == "G":
        return rewards["goal"]
    if tile == "H":
        return rewards["hole"]
    return rewards["step"]


def tiles(env):
    """Flat list of tile letters, indexed by state number."""
    return [c.decode() for c in env.unwrapped.desc.flatten()]


def grid_shape(env):
    return env.unwrapped.desc.shape


def terminal_states(env):
    return [s for s, t in enumerate(tiles(env)) if t in "GH"]


def start_state(env):
    return tiles(env).index("S")


class ShapedRewardWrapper(gym.Wrapper):
    """Replaces FrozenLake's reward (1 at the goal, else 0) with the shaped reward.

    The original reward is kept in ``info["original_reward"]`` for reference.
    """

    def __init__(self, env, rewards=None):
        super().__init__(env)
        self.rewards = dict(rewards or CONFIG["rewards"])
        self._tiles = tiles(env)

    def step(self, action):
        obs, original_reward, terminated, truncated, info = self.env.step(action)
        info["original_reward"] = original_reward
        return obs, shaped_reward(self._tiles[obs], self.rewards), terminated, truncated, info


def make_env(map_key, rewards=None, render_mode=None, slippery=None):
    """FrozenLake with shaped rewards and a step limit (slippery unless switched off in ``CONFIG``)."""
    cfg = map_config(map_key)
    kwargs = {"desc": cfg["desc"]} if "desc" in cfg else {"map_name": cfg["map_name"]}
    env = gym.make(
        "FrozenLake-v1",
        is_slippery=CONFIG["slippery"] if slippery is None else slippery,
        max_episode_steps=cfg["max_steps"],
        render_mode=render_mode,
        **kwargs,
    )
    return ShapedRewardWrapper(env, rewards)


def build_model(env, rewards=None):
    """Transition model with SHAPED rewards for Value Iteration.

    ``env.unwrapped.P`` holds the original rewards, so we copy its transitions and replace
    each reward with ``shaped_reward`` of the tile the agent lands on.

    Returns ``P[s][a] = [(probability, next_state, reward, terminated), ...]``.
    """
    tile_of = tiles(env)
    original = env.unwrapped.P
    return {
        s: {
            a: [(p, s2, shaped_reward(tile_of[s2], rewards), done) for p, s2, _r, done in original[s][a]]
            for a in original[s]
        }
        for s in original
    }


def follow_policy(env, policy):
    """States visited when following ``policy`` from S, assuming every move goes where it is aimed.

    On slippery ice this is the route the policy *aims* for, not one it is guaranteed to take.
    Stops at a terminal state, or once the path is longer than the number of states
    (the policy is going round in a loop).
    """
    s, tile_of = start_state(env), tiles(env)
    path = [s]
    while tile_of[s] not in "GH" and len(path) <= len(tile_of):
        outcomes = env.unwrapped.P[s][int(policy[s])]
        # slippery: outcomes are the moves (a - 1, a, a + 1) in that order, so the intended one is in the middle
        _p, s, _r, _d = outcomes[len(outcomes) // 2]
        path.append(s)
    return path


def play_episode(env, policy, seed=None):
    """States visited in one real episode following ``policy`` (on slippery ice the moves may slide)."""
    s, _ = env.reset(seed=seed)
    path = [s]
    while True:
        s, _r, terminated, truncated, _ = env.step(int(policy[s]))
        path.append(s)
        if terminated or truncated:
            return path


def describe_map(env):
    return "\n".join("".join(row) for row in np.array(tiles(env)).reshape(grid_shape(env)))
