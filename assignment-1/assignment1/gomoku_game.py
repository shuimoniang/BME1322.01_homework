"""Rules and mutable state for a generalized N-by-N, K-in-a-row game."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


EMPTY = 0
BLACK = 1
WHITE = 2
DIRECTIONS = ((0, 1), (1, 0), (1, 1), (1, -1))


@dataclass(frozen=True)
class Move:
    row: int
    col: int
    player: int


class GomokuGame:
    """Owns game state and enforces the generalized Gomoku rules."""

    def __init__(self, board_size: int = 15, win_length: int = 5) -> None:
        self._validate_configuration(board_size, win_length)
        self.board_size = board_size
        self.win_length = win_length
        self.reset()

    @staticmethod
    def _validate_configuration(board_size: int, win_length: int) -> None:
        if isinstance(board_size, bool) or not isinstance(board_size, int):
            raise TypeError("board_size must be an integer")
        if isinstance(win_length, bool) or not isinstance(win_length, int):
            raise TypeError("win_length must be an integer")
        if board_size < 3:
            raise ValueError("board_size must be at least 3")
        if not 3 <= win_length <= board_size:
            raise ValueError("win_length must satisfy 3 <= K <= N")

    def reset(self) -> None:
        self.board = [[EMPTY] * self.board_size for _ in range(self.board_size)]
        self.current_player = BLACK
        self.winner = EMPTY
        self.is_draw = False
        self.game_over = False
        self.winning_line: list[tuple[int, int]] = []
        self.history: list[Move] = []

    def board_copy(self) -> list[list[int]]:
        return [row[:] for row in self.board]

    @property
    def last_move(self) -> tuple[int, int] | None:
        if not self.history:
            return None
        move = self.history[-1]
        return move.row, move.col

    def validate_move(self, row: int, col: int) -> str | None:
        if self.game_over:
            return "棋局已经结束"
        if isinstance(row, bool) or isinstance(col, bool):
            return "坐标必须是整数"
        if not isinstance(row, int) or not isinstance(col, int):
            return "坐标必须是整数"
        if not (0 <= row < self.board_size and 0 <= col < self.board_size):
            return "坐标超出棋盘范围"
        if self.board[row][col] != EMPTY:
            return "该位置已有棋子"
        return None

    def place_move(self, row: int, col: int) -> Move:
        error = self.validate_move(row, col)
        if error:
            raise ValueError(error)

        player = self.current_player
        self.board[row][col] = player
        move = Move(row, col, player)
        self.history.append(move)

        winning_line = self.find_winning_line(row, col, player)
        if winning_line:
            self.winner = player
            self.game_over = True
            self.winning_line = winning_line
        elif len(self.history) == self.board_size * self.board_size:
            self.is_draw = True
            self.game_over = True
        else:
            self.current_player = WHITE if player == BLACK else BLACK
        return move

    def find_winning_line(
        self, row: int, col: int, player: int
    ) -> list[tuple[int, int]]:
        if not (0 <= row < self.board_size and 0 <= col < self.board_size):
            return []
        if self.board[row][col] != player:
            return []

        for dr, dc in DIRECTIONS:
            negative: list[tuple[int, int]] = []
            rr, cc = row - dr, col - dc
            while self._inside(rr, cc) and self.board[rr][cc] == player:
                negative.append((rr, cc))
                rr -= dr
                cc -= dc

            positive: list[tuple[int, int]] = []
            rr, cc = row + dr, col + dc
            while self._inside(rr, cc) and self.board[rr][cc] == player:
                positive.append((rr, cc))
                rr += dr
                cc += dc

            line = list(reversed(negative)) + [(row, col)] + positive
            if len(line) >= self.win_length:
                return line
        return []

    def legal_moves(self) -> Iterable[tuple[int, int]]:
        for row in range(self.board_size):
            for col in range(self.board_size):
                if self.board[row][col] == EMPTY:
                    yield row, col

    def _inside(self, row: int, col: int) -> bool:
        return 0 <= row < self.board_size and 0 <= col < self.board_size

