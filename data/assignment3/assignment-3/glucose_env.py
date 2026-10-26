"""
BME1322 Assignment 3 - Blood Glucose Environment (provided, do NOT modify)
==========================================================================

Simplified discrete blood-glucose control environment for comparing
model-based RL (learn P/R + value iteration) with model-free Q-Learning.

Key design points
-----------------
* 9 discrete states B1..B9 (blood glucose bins), B1 is TERMINAL (severe hypoglycemia).
* 4 actions: w (wait), g (glucose), i1 (small insulin), i2 (large insulin).
* Stochastic transitions: meal events + metabolic noise + action effects.
* Reward = R(s') + c(a)  (reward on entering next state, plus action cost).
* Time step = 15 minutes, episode limit T = 200.
* Discount gamma = 0.9 (consistent with lecture).

Students may ONLY interact via reset()/step() and read self.actions.
Reading any internal transition table is forbidden (assignment rule).
"""

import math
import random
from typing import List, Optional, Tuple

STATE_NAMES: List[str] = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B9"]
ACTION_NAMES: List[str] = ["w", "g", "i1", "i2"]

# Blood-glucose bin edges (mmol/L).  B1..B9
BIN_EDGES = [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0 + 1e-9]

# Reward for *entering* state s' (R(s'))
REWARDS = {
    "B1": -10.0,
    "B2": -1.0,
    "B3": +1.0,
    "B4": 0.0,
    "B5": -1.0,
    "B6": -2.0,
    "B7": -4.0,
    "B8": -6.0,
    "B9": -8.0,
}

# Action cost c(a) added to the entering reward
ACTION_COST = {"w": 0.0, "g": -0.5, "i1": -0.2, "i2": -0.4}

# Action effects on glucose (mmol/L), mean shift; actual effect gets noise.
ACTION_EFFECT = {
    "w": 0.0,
    "g": +1.2,
    "i1": -1.5,
    "i2": -2.8,
}

GAMMA = 0.9
MAX_STEPS = 200
BASELINE = 6.5          # mmol/L, metabolic regression target
REGRESSION = 0.06       # per-step pull toward baseline
MEAL_PROB = 0.12        # probability of a meal event each step
MEAL_EFFECT = 2.4       # mmol/L mean glucose rise
MEAL_DECAY = 0.55       # meal remnant decays per step
NOISE_STD = 0.30        # metabolic noise (mmol/L)
ACTION_NOISE_STD = 0.25 # action-effect noise (mmol/L)


def glucose_to_bin(x: float) -> str:
    """Map continuous glucose (mmol/L) to discrete state name B1..B9."""
    x = max(BIN_EDGES[0], min(BIN_EDGES[-1] - 1e-9, x))
    for i in range(9):
        if BIN_EDGES[i] <= x < BIN_EDGES[i + 1]:
            return STATE_NAMES[i]
    return STATE_NAMES[-1]


class GlucoseEnv:
    """Discrete-state blood-glucose simulator.

    Usage
    -----
    env = GlucoseEnv(seed=42)
    s = env.reset()                # 'B3' etc.
    ns, r, done, info = env.step("i1")
    """

    def __init__(self, seed=None):
        self.rng = random.Random(seed)
        self.actions: List[str] = list(ACTION_NAMES)
        self._meal_remnant = 0.0
        self._true_glucose = None
        self._t = 0

    # ------------------------------------------------------------------ #
    # public API
    # ------------------------------------------------------------------ #
    def reset(self) -> str:
        self._t = 0
        self._meal_remnant = 0.0
        # initial glucose ~ N(6.5, 1.5) clipped to [2, 20]
        x = self.rng.gauss(BASELINE, 1.5)
        self._true_glucose = max(BIN_EDGES[0], min(BIN_EDGES[-1] - 1e-9, x))
        return glucose_to_bin(self._true_glucose)

    def step(self, action: str) -> Tuple[str, float, bool, dict]:
        if self._true_glucose is None:
            raise RuntimeError("Call reset() before step().")
        if action not in self.actions:
            raise ValueError(f"Unknown action: {action!r}")

        # 1) meal event (before action effect; meals keep acting via remnant)
        if self.rng.random() < MEAL_PROB:
            self._meal_remnant += MEAL_EFFECT
        meal_now = self._meal_remnant
        self._meal_remnant *= MEAL_DECAY

        # 2) metabolic regression toward baseline
        x = self._true_glucose
        x = x + REGRESSION * (BASELINE - x)

        # 3) meal + action effect (with noise)
        effect = meal_now + ACTION_EFFECT[action]
        effect += self.rng.gauss(0.0, ACTION_NOISE_STD)
        x += effect

        # 4) metabolic noise
        x += self.rng.gauss(0.0, NOISE_STD)

        # clip to supported range
        x = max(BIN_EDGES[0], min(BIN_EDGES[-1] - 1e-9, x))
        self._true_glucose = x

        ns = glucose_to_bin(x)
        reward = REWARDS[ns] + ACTION_COST[action]
        self._t += 1
        done = (ns == "B1") or (self._t >= MAX_STEPS)
        info = {
            "true_glucose": x,   # evaluation ONLY; students must not use in algorithms
            "t": self._t,
            "meal": meal_now,
        }
        return ns, reward, done, info

    # convenience for graders / student experiments ---------------------- #
    def get_true_glucose(self) -> Optional[float]:
        return self._true_glucose


# ---------------------------------------------------------------------- #
# Data collection helper (used by autograder and by main.py examples)
# ---------------------------------------------------------------------- #
def collect_random_episodes(n_episodes: int = 2000, seed: int = 2026,
                            env_seed_base: int = None) -> List[list]:
    """Collect episodes with a uniform random policy.

    The ``seed`` argument fully determines the dataset: by default the i-th
    episode uses ``GlucoseEnv(seed=seed + i)``. Passing ``env_seed_base``
    overrides the base seed (kept for backward compatibility).

    Returns a list of episodes; each episode is a list of 5-tuples
    (state, action, reward, next_state, done).
    """
    if env_seed_base is None:
        env_seed_base = seed
    episodes = []
    for ep in range(n_episodes):
        env = GlucoseEnv(seed=env_seed_base + ep)
        s = env.reset()
        traj = []
        done = False
        while not done:
            a = env.rng.choice(env.actions)
            ns, r, done, _ = env.step(a)
            traj.append((s, a, r, ns, done))
            s = ns
        episodes.append(traj)
    return episodes


if __name__ == "__main__":
    env = GlucoseEnv(seed=0)
    s = env.reset()
    total = 0.0
    for t in range(50):
        a = env.rng.choice(env.actions)
        s, r, done, info = env.step(a)
        total += r
        if done:
            break
    print(f"random episode: steps={t+1}, return={total:.2f}, final_state={s}")
