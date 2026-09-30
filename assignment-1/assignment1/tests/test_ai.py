import importlib.util
import sys
import threading
import time
import unittest
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from gomoku_ai import (  # noqa: E402
    BaselineAlphaBetaAI,
    EnhancedAlphaBetaAI,
    GomokuAI,
    MCTSAI,
)


AI_CLASSES = (BaselineAlphaBetaAI, EnhancedAlphaBetaAI, MCTSAI)


class GomokuAITests(unittest.TestCase):
    def assert_legal(self, move, board):
        self.assertIsInstance(move, tuple)
        self.assertEqual(len(move), 2)
        row, col = move
        self.assertIs(type(row), int)
        self.assertIs(type(col), int)
        self.assertTrue(0 <= row < len(board))
        self.assertTrue(0 <= col < len(board))
        self.assertEqual(board[row][col], 0)

    def call_with_watchdog(self, ai, board, last_move, limit):
        holder = {}

        def target():
            try:
                holder["move"] = ai.get_move(board, last_move, limit)
            except BaseException as exc:
                holder["error"] = exc

        start = time.perf_counter()
        thread = threading.Thread(target=target, daemon=True)
        thread.start()
        thread.join(limit)
        elapsed = time.perf_counter() - start
        self.assertFalse(thread.is_alive(), f"AI exceeded {limit}s")
        self.assertNotIn("error", holder)
        return holder["move"], elapsed

    def test_empty_board_uses_center_for_all_strategies(self):
        for size, win in ((3, 3), (9, 4), (20, 7)):
            board = [[0] * size for _ in range(size)]
            for cls in AI_CLASSES:
                with self.subTest(size=size, strategy=cls.__name__):
                    move, _ = self.call_with_watchdog(cls(1, size, win), board, None, 0.05)
                    self.assertEqual(move, (size // 2, size // 2))

    def test_immediate_win_and_block(self):
        for cls in AI_CLASSES:
            with self.subTest(strategy=cls.__name__, situation="win"):
                board = [[0] * 5 for _ in range(5)]
                board[2][1] = board[2][2] = 1
                original = [row[:] for row in board]
                move, _ = self.call_with_watchdog(cls(1, 5, 3), board, (0, 0), 0.1)
                self.assertIn(move, ((2, 0), (2, 3)))
                self.assertEqual(board, original)
            with self.subTest(strategy=cls.__name__, situation="block"):
                board = [[0] * 5 for _ in range(5)]
                board[3][1] = board[3][2] = 2
                move, _ = self.call_with_watchdog(cls(1, 5, 3), board, (3, 2), 0.1)
                self.assertIn(move, ((3, 0), (3, 3)))

    def test_near_full_board_returns_only_legal_move(self):
        board = [[1, 2, 1], [2, 1, 2], [2, 1, 0]]
        for cls in AI_CLASSES:
            with self.subTest(strategy=cls.__name__):
                move, _ = self.call_with_watchdog(cls(1, 3, 3), board, (2, 1), 0.05)
                self.assertEqual(move, (2, 2))

    def test_strict_time_limits_on_nonempty_board(self):
        board = [[0] * 15 for _ in range(15)]
        board[7][7], board[7][8], board[8][8] = 1, 2, 1
        for limit in (0.05, 0.1, 1.0):
            for cls in AI_CLASSES:
                with self.subTest(limit=limit, strategy=cls.__name__):
                    move, elapsed = self.call_with_watchdog(
                        cls(2, 15, 5), board, (8, 8), limit
                    )
                    self.assert_legal(move, board)
                    self.assertLess(elapsed, limit)

    def test_repeated_large_board_watchdog_calls(self):
        size = 40
        board = [[0] * size for _ in range(size)]
        board[size // 2][size // 2] = 1
        for repetition in range(5):
            for cls in AI_CLASSES:
                with self.subTest(repetition=repetition, strategy=cls.__name__):
                    move, elapsed = self.call_with_watchdog(
                        cls(2, size, 7), board, (size // 2, size // 2), 0.05
                    )
                    self.assert_legal(move, board)
                    self.assertLess(elapsed, 0.05)

    def test_protocol_file_loads_directly(self):
        path = PROJECT / "gomoku_ai.py"
        spec = importlib.util.spec_from_file_location("_protocol_test_ai", path)
        module = importlib.util.module_from_spec(spec)
        sys.modules["_protocol_test_ai"] = module
        spec.loader.exec_module(module)
        board = [[0] * 5 for _ in range(5)]
        move = module.GomokuAI(1, 5, 3).get_move(board, None, 0.05)
        self.assert_legal(move, board)
        self.assertTrue(issubclass(module.GomokuAI, EnhancedAlphaBetaAI) or module.GomokuAI.__name__ == "GomokuAI")

    def test_submission_entry_point_is_enhanced(self):
        self.assertTrue(issubclass(GomokuAI, EnhancedAlphaBetaAI))


if __name__ == "__main__":
    unittest.main()
