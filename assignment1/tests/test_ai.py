import importlib.util
import sys
import threading
import time
import unittest
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from gomoku_ai import (  # noqa: E402
    INF,
    TTEntry,
    TT_EXACT,
    TT_LOWER,
    TT_UPPER,
    WIN_SCORE,
    BaselineAlphaBetaAI,
    EnhancedAlphaBetaAI,
    GomokuAI,
    HybridThreatSearchAI,
    MCTSAI,
    ThreatResult,
)
from tactical_benchmark import (  # noqa: E402
    CATEGORIES,
    CONFIGURATIONS,
    build_tactical_cases,
)
from advanced_tactical_benchmark import (  # noqa: E402
    CATEGORIES as ADVANCED_CATEGORIES,
    CONFIGURATIONS as ADVANCED_CONFIGURATIONS,
    build_advanced_cases,
)


AI_CLASSES = (BaselineAlphaBetaAI, EnhancedAlphaBetaAI, MCTSAI, HybridThreatSearchAI)


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
        self.assertTrue(issubclass(module.GomokuAI, module.HybridThreatSearchAI))

    def test_submission_entry_point_is_hybrid(self):
        self.assertTrue(issubclass(GomokuAI, HybridThreatSearchAI))

    def test_tactical_case_generator_structure(self):
        cases = build_tactical_cases()
        self.assertEqual(len(cases), 100)
        self.assertEqual(len({case.case_id for case in cases}), 100)
        self.assertEqual({case.category for case in cases}, set(CATEGORIES))
        self.assertEqual(
            {(case.board_size, case.win_length) for case in cases},
            set(CONFIGURATIONS),
        )
        self.assertEqual({case.player for case in cases}, {1, 2})

    def test_advanced_tactical_cases_and_hybrid_answers(self):
        cases = build_advanced_cases()
        self.assertEqual(len(cases), 36)
        self.assertEqual(len({case.case_id for case in cases}), 36)
        self.assertEqual({case.category for case in cases}, set(ADVANCED_CATEGORIES))
        self.assertEqual(
            {(case.board_size, case.win_length) for case in cases},
            set(ADVANCED_CONFIGURATIONS),
        )
        self.assertEqual({case.player for case in cases}, {1, 2})
        for case in cases:
            with self.subTest(case=case.case_id):
                board = case.mutable_board()
                original = [row[:] for row in board]
                move, _ = self.call_with_watchdog(
                    HybridThreatSearchAI(
                        case.player, case.board_size, case.win_length
                    ),
                    board,
                    None,
                    0.1,
                )
                self.assertIn(move, case.allowed_moves)
                self.assertEqual(board, original)

    def test_open_k_minus_2_prevention_across_strategies(self):
        cases = [
            case
            for case in build_tactical_cases()
            if case.category == "open_k_minus_2_prevention"
        ]
        self.assertEqual(len(cases), len(CONFIGURATIONS) * 2)
        for case in cases:
            for cls in (EnhancedAlphaBetaAI, HybridThreatSearchAI, MCTSAI):
                with self.subTest(case=case.case_id, strategy=cls.__name__):
                    board = case.mutable_board()
                    original = [row[:] for row in board]
                    ai = cls(case.player, case.board_size, case.win_length)
                    if isinstance(ai, MCTSAI):
                        ai._random.seed(7)
                    move, _ = self.call_with_watchdog(ai, board, None, 0.15)
                    self.assertIn(move, case.allowed_moves)
                    self.assertEqual(board, original)

    def test_hybrid_single_winning_point_is_not_proven_for_defender(self):
        board = [[0] * 7 for _ in range(7)]
        board[3][1] = 2
        board[3][2] = board[3][3] = board[3][4] = 1
        ai = HybridThreatSearchAI(2, 7, 4)
        ai._deadline = time.perf_counter() + 0.5
        ai._threat_node_limit = 1000
        self.assertEqual(ai._winning_moves(board, 1), [(3, 5)])
        self.assertEqual(
            ai._threat_search(board, 2, 3, ai._board_hash(board)),
            ThreatResult.DISPROVEN,
        )

    def test_forcing_candidates_still_obey_branch_cap(self):
        board = [[0] * 7 for _ in range(7)]
        board[2][2] = board[4][4] = 2
        ai = EnhancedAlphaBetaAI(1, 7, 3)
        ai._deadline = time.perf_counter() + 1.0
        state_hash = ai._board_hash(board)
        moves = ai._ranked_tactical_moves(
            board, 1, preferred=None, cap=3, state_hash=state_hash
        )
        self.assertLessEqual(len(moves), 3)

    def test_unique_immediate_block_is_never_pruned(self):
        board = [[0] * 7 for _ in range(7)]
        board[3][0] = 1
        board[3][1] = board[3][2] = board[3][3] = 2
        ai = EnhancedAlphaBetaAI(1, 7, 4)
        ai._deadline = time.perf_counter() + 1.0
        state_hash = ai._board_hash(board)
        moves = ai._ranked_tactical_moves(
            board, 1, preferred=None, cap=1, state_hash=state_hash
        )
        self.assertEqual(moves, [(3, 4)])

    def test_quiescence_searches_forced_block_before_stand_pat(self):
        board = [[0] * 7 for _ in range(7)]
        board[3][0] = 1
        board[3][1] = board[3][2] = board[3][3] = 2
        ai = HybridThreatSearchAI(1, 7, 4)
        ai._reset_search_stats()
        ai._deadline = time.perf_counter() + 1.0
        state_hash = ai._board_hash(board)
        ai._quiescence(board, -INF, -INF + 1, 1, 0, 0, state_hash)
        self.assertGreaterEqual(ai.last_search_stats["quiescence_nodes"], 2)
        self.assertEqual(board[3][4], 0)

    def test_unknown_threat_result_is_not_cached_as_disproof(self):
        board = [[0] * 7 for _ in range(7)]
        board[3][2] = board[3][3] = board[3][4] = 1
        ai = HybridThreatSearchAI(2, 7, 4)
        ai._reset_search_stats()
        ai._deadline = time.perf_counter() + 1.0
        ai._threat_node_limit = 0
        state_hash = ai._board_hash(board)
        result = ai._threat_search(board, 2, 3, state_hash, 1)
        self.assertEqual(result, ThreatResult.UNKNOWN)
        self.assertFalse(ai._threat_table)

    def test_transposition_bounds_depth_and_mate_normalization(self):
        ai = EnhancedAlphaBetaAI(1, 7, 4)
        ai._reset_search_stats()
        move = (3, 3)
        key = (12345, 1)
        stored = ai._score_to_tt(WIN_SCORE - 5, 5)
        ai._table[key] = TTEntry(3, stored, TT_EXACT, move)
        score, _, _, preferred = ai._probe_tt(key, 3, -INF, INF, 2)
        self.assertEqual(score, WIN_SCORE - 2)
        self.assertEqual(preferred, move)

        score, _, _, preferred = ai._probe_tt(key, 4, -INF, INF, 2)
        self.assertIsNone(score)
        self.assertEqual(preferred, move)

        ai._table[key] = TTEntry(4, ai._score_to_tt(25, 0), TT_LOWER, move)
        score, alpha, _, _ = ai._probe_tt(key, 4, 0, 20, 0)
        self.assertEqual(score, 25)
        self.assertGreaterEqual(alpha, 25)

        ai._table[key] = TTEntry(4, ai._score_to_tt(-25, 0), TT_UPPER, move)
        score, _, beta, _ = ai._probe_tt(key, 4, -20, 0, 0)
        self.assertEqual(score, -25)
        self.assertLessEqual(beta, -25)

        ai._table[key] = TTEntry(5, 99, TT_LOWER, move)
        ai._store_tt(key, 4, 100, TT_EXACT, (2, 2), 0)
        self.assertEqual(ai._table[key], TTEntry(5, 99, TT_LOWER, move))

        ai._table[key] = TTEntry(5, 99, TT_EXACT, move)
        ai._store_tt(key, 5, 100, TT_UPPER, (2, 2), 0)
        self.assertEqual(ai._table[key], TTEntry(5, 99, TT_EXACT, move))

    def test_zobrist_hash_is_independent_of_move_order(self):
        ai = EnhancedAlphaBetaAI(1, 9, 5)
        ai._deadline = time.perf_counter() + 1.0
        stones = ((4, 4, 1), (4, 5, 2), (5, 4, 1), (3, 5, 2))
        first = 0
        for row, col, player in stones:
            first ^= ai._zobrist(row, col, player)
        second = 0
        for row, col, player in reversed(stones):
            second ^= ai._zobrist(row, col, player)
        self.assertEqual(first, second)

    def test_transposed_position_has_the_same_search_score(self):
        stones = ((3, 3, 1), (3, 4, 2), (4, 3, 1), (2, 4, 2))
        boards = []
        scores = []
        for ordering in (stones, tuple(reversed(stones))):
            board = [[0] * 7 for _ in range(7)]
            for row, col, player in ordering:
                board[row][col] = player
            boards.append(board)
            ai = EnhancedAlphaBetaAI(1, 7, 4)
            ai._reset_search_stats()
            ai._deadline = time.perf_counter() + 1.0
            state_hash = ai._board_hash(board)
            scores.append(ai._negamax(board, 2, -INF, INF, 1, 0, state_hash))
        self.assertEqual(boards[0], boards[1])
        self.assertEqual(scores[0], scores[1])

    def test_hybrid_detects_double_threat_candidate(self):
        board = [[0] * 7 for _ in range(7)]
        # The center completes two independent K=3 threats.
        board[3][2] = board[3][4] = 1
        board[2][3] = board[4][3] = 1
        ai = HybridThreatSearchAI(1, 7, 4)
        ai._deadline = time.perf_counter() + 0.2
        level = ai._move_level(board, 3, 3, 1)
        self.assertGreaterEqual(level, 4)

    def test_hybrid_move_level_preserves_existing_cell(self):
        board = [[0] * 7 for _ in range(7)]
        board[3][3] = 1
        ai = HybridThreatSearchAI(1, 7, 4)
        ai._deadline = time.perf_counter() + 0.2
        ai._move_level(board, 3, 3, 1)
        self.assertEqual(board[3][3], 1)

    def test_hybrid_does_not_modify_input(self):
        board = [[0] * 9 for _ in range(9)]
        board[4][4], board[4][5], board[5][4] = 1, 2, 1
        original = [row[:] for row in board]
        move, _ = self.call_with_watchdog(
            HybridThreatSearchAI(2, 9, 4), board, (5, 4), 0.1
        )
        self.assert_legal(move, board)
        self.assertEqual(board, original)


if __name__ == "__main__":
    unittest.main()
