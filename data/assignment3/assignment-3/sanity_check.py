#!/usr/bin/env python3
"""
BME1322 Assignment 3 - Public Interface Checker (provided, do NOT modify)
=========================================================================

This checker validates ONLY the file names, class names, method signatures
and basic behavior of your submission. It contains NO reference answers and
NO grading thresholds. Passing it does NOT mean your implementation earns
points; it only means the grader will be able to load and call your code.

Official grading is performed by course staff with an internal grader
(not distributed to students).

Usage:  python sanity_check.py
"""

import importlib.util
import os
import sys

from glucose_env import ACTION_NAMES, STATE_NAMES, collect_random_episodes


def load_class(path, class_name):
    if not os.path.exists(path):
        print(f"  [MISS] file not found: {path}")
        return None
    spec = importlib.util.spec_from_file_location("_stu", path)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as exc:
        print(f"  [ERR ] import failed: {exc}")
        return None
    return getattr(mod, class_name, None)


def main():
    data = collect_random_episodes(n_episodes=30, seed=1)
    ok = True

    print("== sanity_check: assignment3/model_based.py ==")
    cls = load_class("assignment3/model_based.py", "ModelBasedAgent")
    if cls is None:
        ok = False
    else:
        try:
            ag = cls(states=list(STATE_NAMES), actions=list(ACTION_NAMES), gamma=0.9)
            ag.learn_model(data)
            dist = ag.get_transition("B3", "w")
            assert isinstance(dist, dict) and abs(sum(dist.values()) - 1.0) < 1e-6
            assert isinstance(ag.get_reward("B3", "w"), float)
            ag.run_value_iteration(max_iterations=50)
            assert isinstance(ag.get_value("B2"), float)
            assert ag.get_policy("B2") in ACTION_NAMES
            print("  [OK  ] ModelBasedAgent interface works")
        except Exception as exc:
            print(f"  [ERR ] {type(exc).__name__}: {exc}")
            ok = False

    print("== sanity_check: assignment3/q_learning.py ==")
    cls = load_class("assignment3/q_learning.py", "QLearningAgent")
    if cls is None:
        ok = False
    else:
        try:
            ag = cls(states=list(STATE_NAMES), actions=list(ACTION_NAMES),
                     gamma=0.9, alpha=0.5)
            ag.learn(data, mode="fixed")
            assert isinstance(ag.get_q_value("B3", "w"), float)
            assert ag.get_policy("B3") in ACTION_NAMES
            print("  [OK  ] QLearningAgent interface works")
        except Exception as exc:
            print(f"  [ERR ] {type(exc).__name__}: {exc}")
            ok = False

    print("\nsanity_check:", "PASS (interfaces OK)" if ok else "FAIL (fix errors above)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
