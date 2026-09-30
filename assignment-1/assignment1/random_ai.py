"""Simple protocol-compatible random opponent for local GUI testing."""

import random


NAME = "Random"


class GomokuAI:
    def __init__(self, player_id, board_size, win_length):
        self.player_id = player_id
        self.board_size = board_size
        self.win_length = win_length

    def get_move(self, board, last_opponent_move, time_limit):
        legal = [
            (row, col)
            for row in range(self.board_size)
            for col in range(self.board_size)
            if board[row][col] == 0
        ]
        return random.choice(legal)
