import sys
import unittest
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from gomoku_game import BLACK, WHITE, GomokuGame  # noqa: E402


class GomokuGameTests(unittest.TestCase):
    def test_configuration_accepts_tic_tac_toe_and_large_board(self):
        self.assertEqual(GomokuGame(3, 3).board_size, 3)
        self.assertEqual(GomokuGame(20, 7).board_size, 20)
        with self.assertRaises(ValueError):
            GomokuGame(2, 3)
        with self.assertRaises(ValueError):
            GomokuGame(8, 9)

    def _play(self, size, win, moves):
        game = GomokuGame(size, win)
        for row, col in moves:
            game.place_move(row, col)
        return game

    def test_horizontal_win(self):
        game = self._play(5, 3, [(2, 1), (0, 0), (2, 2), (0, 1), (2, 3)])
        self.assertTrue(game.game_over)
        self.assertEqual(game.winner, BLACK)
        self.assertEqual(len(game.winning_line), 3)

    def test_vertical_win(self):
        game = self._play(5, 3, [(0, 2), (0, 0), (1, 2), (0, 1), (2, 2)])
        self.assertEqual(game.winner, BLACK)

    def test_both_diagonal_directions(self):
        descending = self._play(5, 3, [(0, 0), (0, 4), (1, 1), (1, 4), (2, 2)])
        self.assertEqual(descending.winner, BLACK)
        ascending = self._play(5, 3, [(2, 0), (0, 0), (1, 1), (0, 1), (0, 2)])
        self.assertEqual(ascending.winner, BLACK)

    def test_overline_wins(self):
        game = GomokuGame(7, 4)
        game.board[3][1:6] = [BLACK] * 5
        self.assertEqual(len(game.find_winning_line(3, 3, BLACK)), 5)

    def test_tic_tac_toe_draw(self):
        moves = [(0, 0), (1, 1), (0, 1), (0, 2), (2, 0), (1, 0), (1, 2), (2, 1), (2, 2)]
        game = self._play(3, 3, moves)
        self.assertTrue(game.game_over)
        self.assertTrue(game.is_draw)
        self.assertEqual(game.winner, 0)

    def test_invalid_move_and_board_copy(self):
        game = GomokuGame(3, 3)
        game.place_move(1, 1)
        with self.assertRaises(ValueError):
            game.place_move(1, 1)
        copy = game.board_copy()
        copy[1][1] = WHITE
        self.assertEqual(game.board[1][1], BLACK)


if __name__ == "__main__":
    unittest.main()

