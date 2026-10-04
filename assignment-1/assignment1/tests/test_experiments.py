import json
import sys
import tempfile
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
from advanced_match_benchmark import build_tasks as build_match_tasks  # noqa: E402
from final_experiments import (  # noqa: E402
    ensure_manifest as ensure_final_manifest,
    build_tasks as build_final_tasks,
)
from comprehensive_evaluation import (  # noqa: E402
    build_tasks as build_comprehensive_tasks,
    ensure_manifest as ensure_comprehensive_manifest,
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

    def test_advanced_match_matrix_is_unique_and_color_paired(self):
        tasks = build_match_tasks()
        self.assertEqual(len(tasks), 96)
        self.assertEqual(len({task.key for task in tasks}), 96)
        pairs = {}
        for task in tasks:
            key = (
                task.first_strategy,
                task.second_strategy,
                task.board_size,
                task.win_length,
                task.time_limit,
                task.opening_index,
            )
            pairs.setdefault(key, []).append(task)
        self.assertEqual(len(pairs), 48)
        for key, paired_tasks in pairs.items():
            with self.subTest(pair=key):
                self.assertEqual(len(paired_tasks), 2)
                self.assertEqual(paired_tasks[0].opening, paired_tasks[1].opening)
                first = paired_tasks[0].first_strategy
                self.assertEqual(
                    {task.black_agent == first for task in paired_tasks},
                    {False, True},
                )

    def test_final_matrix_has_1500_unique_games_in_30_groups(self):
        tasks = build_final_tasks()
        self.assertEqual(len(tasks), 1500)
        self.assertEqual(len({task.key for task in tasks}), 1500)
        groups = {}
        for task in tasks:
            key = (
                task.kind,
                task.strategy,
                task.opponent,
                task.board_size,
                task.win_length,
                task.time_limit,
            )
            groups.setdefault(key, []).append(task)
        self.assertEqual(len(groups), 30)
        for key, games in groups.items():
            with self.subTest(group=key):
                self.assertEqual(len(games), 50)
                if key[0] == "vs_random":
                    self.assertEqual(sum(game.black_agent == key[1] for game in games), 25)
                    self.assertTrue(all(not game.opening for game in games))
                    continue
                self.assertEqual(len({game.opening_index for game in games}), 25)
                for opening_index in range(25):
                    pair = [
                        game for game in games if game.opening_index == opening_index
                    ]
                    self.assertEqual(len(pair), 2)
                    self.assertEqual(pair[0].opening, pair[1].opening)
                    self.assertEqual({game.swap for game in pair}, {0, 1})
                    self.assertEqual(len(pair[0].opening), 4)
                    self.assertEqual(
                        len({(row, col) for row, col, _player in pair[0].opening}),
                        4,
                    )
                    self.assertEqual(
                        [player for _row, _col, player in pair[0].opening],
                        [1, 2, 1, 2],
                    )

    def test_final_manifest_rejects_a_different_source_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            ensure_final_manifest(path, games_per_group=2)
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["source_sha256"] = "0" * 64
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "manifest does not match"):
                ensure_final_manifest(path, games_per_group=2)

    def test_comprehensive_matrix_has_2400_balanced_games(self):
        tasks = build_comprehensive_tasks()
        self.assertEqual(len(tasks), 2400)
        self.assertEqual(len({task.key for task in tasks}), 2400)
        groups = {}
        for task in tasks:
            key = (
                task.kind,
                task.strategy,
                task.board_size,
                task.win_length,
                task.time_limit,
            )
            groups.setdefault(key, []).append(task)
        self.assertEqual(len(groups), 48)
        for key, games in groups.items():
            with self.subTest(group=key):
                self.assertEqual(len(games), 50)
                strategy = key[1]
                if key[0] == "vs_random":
                    self.assertEqual(sum(game.black_agent == strategy for game in games), 25)
                    self.assertEqual(sum(game.white_agent == strategy for game in games), 25)
                    self.assertTrue(all(not game.opening for game in games))
                    continue
                self.assertEqual(len({game.opening_index for game in games}), 25)
                for opening_index in range(25):
                    pair = [
                        game for game in games if game.opening_index == opening_index
                    ]
                    self.assertEqual(len(pair), 2)
                    self.assertEqual(pair[0].opening, pair[1].opening)
                    self.assertEqual({game.swap for game in pair}, {0, 1})
                    self.assertEqual(
                        {game.black_agent for game in pair},
                        {f"{strategy} A", f"{strategy} B"},
                    )

    def test_comprehensive_manifest_rejects_source_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            ensure_comprehensive_manifest(path, games_per_group=2)
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["source_sha256"] = "0" * 64
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "manifest does not match"):
                ensure_comprehensive_manifest(path, games_per_group=2)

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
