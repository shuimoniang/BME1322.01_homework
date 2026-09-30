import sys
import unittest
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from run_experiments import (  # noqa: E402
    ExperimentTask,
    build_tasks,
    play_game,
    stable_seed,
    summarize,
)


class ExperimentRunnerTests(unittest.TestCase):
    def test_full_matrix_has_expected_size_and_unique_keys(self):
        tasks = build_tasks()
        self.assertEqual(len(tasks), 1200)
        self.assertEqual(len({task.key for task in tasks}), 1200)
        self.assertEqual(sum(task.kind == "vs_random" for task in tasks), 900)
        self.assertEqual(sum(task.kind == "selfplay" for task in tasks), 300)

    def test_every_full_group_balances_black_and_white(self):
        groups = {}
        for task in build_tasks():
            group = (
                task.kind,
                task.strategy,
                task.opponent,
                task.board_size,
                task.win_length,
                task.time_limit,
            )
            groups.setdefault(group, []).append(task)
        self.assertEqual(len(groups), 24)
        for group, tasks in groups.items():
            with self.subTest(group=group):
                self.assertEqual(len(tasks), 50)
                if group[0] == "vs_random":
                    strategy = group[1]
                    self.assertEqual(sum(task.black_agent == strategy for task in tasks), 25)
                    self.assertEqual(sum(task.white_agent == strategy for task in tasks), 25)
                else:
                    self.assertEqual(
                        sum(task.black_agent == "Enhanced Alpha-Beta A" for task in tasks), 25
                    )
                    self.assertEqual(
                        sum(task.white_agent == "Enhanced Alpha-Beta A" for task in tasks), 25
                    )

    def test_random_seed_is_paired_across_strategies(self):
        matching = [
            task
            for task in build_tasks()
            if task.kind == "vs_random"
            and task.board_size == 9
            and task.win_length == 4
            and task.time_limit == 1.0
            and task.game_index == 7
        ]
        self.assertEqual(len(matching), 3)
        self.assertEqual(len({task.seed for task in matching}), 1)
        self.assertEqual(stable_seed("x"), stable_seed("x"))

    def test_smoke_matrix_uses_two_games_per_group(self):
        tasks = build_tasks(only="smoke")
        self.assertEqual(len(tasks), 48)
        self.assertEqual(len({task.key for task in tasks}), 48)

    def test_play_and_summarize_small_game(self):
        task = ExperimentTask(
            key="unit-game",
            kind="vs_random",
            strategy="Enhanced Alpha-Beta",
            opponent="Random AI",
            board_size=3,
            win_length=3,
            time_limit=0.05,
            game_index=0,
            seed=stable_seed("unit"),
            black_agent="Enhanced Alpha-Beta",
            white_agent="Random AI",
        )
        record = play_game(task)
        self.assertEqual(record["key"], "unit-game")
        self.assertGreater(record["total_moves"], 0)
        self.assertIn(record["winner_color"], (0, 1, 2))
        summary = summarize([record])
        self.assertEqual(summary["total_games"], 1)
        self.assertEqual(summary["unique_keys"], 1)
        self.assertEqual(summary["groups"][0]["games"], 1)


if __name__ == "__main__":
    unittest.main()
