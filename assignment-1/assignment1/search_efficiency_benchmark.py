"""Measure completed search depth and cold leaf-evaluation cost."""

from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import sys
import time
from pathlib import Path


FIXTURES = (
    (
        "n9-k4-quiet-a",
        9,
        4,
        ((4, 4, 1), (4, 5, 2), (5, 4, 1), (3, 5, 2), (5, 6, 1), (2, 3, 2)),
    ),
    (
        "n9-k4-quiet-b",
        9,
        4,
        ((4, 4, 1), (3, 4, 2), (5, 5, 1), (3, 5, 2), (6, 3, 1), (2, 6, 2)),
    ),
    (
        "n15-k5-quiet-a",
        15,
        5,
        ((7, 7, 1), (7, 8, 2), (8, 7, 1), (6, 8, 2), (9, 9, 1), (5, 5, 2)),
    ),
    (
        "n15-k5-quiet-b",
        15,
        5,
        ((7, 7, 1), (8, 8, 2), (6, 6, 1), (8, 6, 2), (9, 7, 1), (5, 8, 2)),
    ),
    (
        "n15-k7-quiet-a",
        15,
        7,
        ((7, 7, 1), (7, 8, 2), (8, 7, 1), (6, 8, 2), (9, 9, 1), (5, 5, 2)),
    ),
    (
        "n15-k7-quiet-b",
        15,
        7,
        ((7, 7, 1), (8, 8, 2), (6, 6, 1), (8, 6, 2), (9, 7, 1), (5, 8, 2)),
    ),
)


def _load_module(ai_file: Path | None):
    if ai_file is None:
        import gomoku_ai

        return gomoku_ai
    spec = importlib.util.spec_from_file_location("_efficiency_target", ai_file)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _board(size, stones):
    result = [[0] * size for _ in range(size)]
    for row, col, player in stones:
        result[row][col] = player
    return result


def _measured_class(base, root_method):
    class Measured(base):
        def __init__(self, *args):
            super().__init__(*args)
            self._measured_completed_depth = 0

    original = getattr(base, root_method)

    def wrapper(self, board, depth, *args, **kwargs):
        result = original(self, board, depth, *args, **kwargs)
        self._measured_completed_depth = max(self._measured_completed_depth, depth)
        return result

    setattr(Measured, root_method, wrapper)
    return Measured


def run_benchmark(module, time_limit=0.5, repetitions=3):
    strategies = {
        "Enhanced Alpha-Beta": _measured_class(
            module.EnhancedAlphaBetaAI, "_search_root"
        ),
        "Hybrid Threat Search": _measured_class(
            module.HybridThreatSearchAI, "_hybrid_root"
        ),
    }
    result = {
        "time_limit_seconds": time_limit,
        "repetitions": repetitions,
        "fixtures": [],
        "strategies": {},
    }
    for name, cls in strategies.items():
        records = []
        for fixture_id, size, win, stones in FIXTURES:
            depths = []
            elapsed_values = []
            evaluator_values = []
            moves = []
            stats = []
            source = _board(size, stones)
            for _ in range(repetitions):
                ai = cls(1, size, win)
                start = time.perf_counter()
                move = ai.get_move(source, None, time_limit)
                elapsed_values.append(time.perf_counter() - start)
                moves.append(move)
                recorded = getattr(ai, "last_search_stats", {})
                depth = max(
                    getattr(ai, "_measured_completed_depth", 0),
                    int(recorded.get("completed_depth", 0)),
                )
                depths.append(depth)
                stats.append(recorded)

            evaluator = cls(1, size, win)
            evaluator._deadline = time.perf_counter() + 10.0
            for _ in range(20):
                if hasattr(evaluator, "_analysis_cache"):
                    evaluator._analysis_cache.clear()
                start = time.perf_counter()
                evaluator._evaluate_generalized(source, 1)
                evaluator_values.append(time.perf_counter() - start)
            records.append(
                {
                    "fixture_id": fixture_id,
                    "board_size": size,
                    "win_length": win,
                    "completed_depths": depths,
                    "median_completed_depth": statistics.median(depths),
                    "elapsed_seconds": elapsed_values,
                    "median_elapsed_seconds": statistics.median(elapsed_values),
                    "median_cold_evaluation_seconds": statistics.median(
                        evaluator_values
                    ),
                    "moves": moves,
                    "last_search_stats": stats,
                }
            )
        result["strategies"][name] = records
    result["fixtures"] = [fixture[0] for fixture in FIXTURES]
    return result


def compare_results(before, after):
    comparison = {}
    for name, after_records in after["strategies"].items():
        before_records = before["strategies"][name]
        before_depths = [record["median_completed_depth"] for record in before_records]
        after_depths = [record["median_completed_depth"] for record in after_records]
        before_eval = statistics.median(
            record["median_cold_evaluation_seconds"] for record in before_records
        )
        after_eval = statistics.median(
            record["median_cold_evaluation_seconds"] for record in after_records
        )
        gains = [new - old for old, new in zip(before_depths, after_depths)]
        comparison[name] = {
            "before_completed_depths": before_depths,
            "after_completed_depths": after_depths,
            "depth_gains": gains,
            "improved_fixture_count": sum(gain >= 1 for gain in gains),
            "regressed_fixture_count": sum(gain < 0 for gain in gains),
            "before_median_cold_evaluation_seconds": before_eval,
            "after_median_cold_evaluation_seconds": after_eval,
            "evaluation_speedup": before_eval / after_eval,
            "acceptance": {
                "evaluation_at_least_2x_faster": before_eval / after_eval >= 2.0,
                "at_least_4_of_6_depths_improved": sum(gain >= 1 for gain in gains) >= 4,
                "no_depth_regression": all(gain >= 0 for gain in gains),
            },
        }
    return comparison


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ai-file", type=Path)
    parser.add_argument("--time-limit", type=float, default=0.5)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument(
        "--before",
        type=Path,
        default=Path(__file__).with_name("experiments")
        / "search_benchmark_before.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("experiments")
        / "search_benchmark_after.json",
    )
    args = parser.parse_args()
    module = _load_module(args.ai_file)
    result = run_benchmark(module, args.time_limit, args.repetitions)
    if args.before.exists():
        before = json.loads(args.before.read_text(encoding="utf-8"))
        result["comparison"] = compare_results(before, result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    for name, records in result["strategies"].items():
        depths = [record["median_completed_depth"] for record in records]
        evaluations = [
            record["median_cold_evaluation_seconds"] for record in records
        ]
        print(
            f"{name}: depths={depths}, "
            f"median cold eval={statistics.median(evaluations):.6f}s"
        )
    print(f"Wrote: {args.output}")


if __name__ == "__main__":
    main()
