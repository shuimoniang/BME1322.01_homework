# -*- coding: utf-8 -*-
"""random_ai.py — BME1322 Assignment 1 参考实现：随机合法落子。"""

import random


class GomokuAI:
    def __init__(self, player_id, board_size, win_length):
        self.player_id = player_id
        self.board_size = board_size
        self.win_length = win_length

    def get_move(self, board, last_opponent_move, time_limit):
        empties = [(r, c) for r in range(self.board_size)
                          for c in range(self.board_size) if board[r][c] == 0]
        return random.choice(empties)
