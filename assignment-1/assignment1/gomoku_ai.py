"""Self-contained generalized Gomoku AIs implementing protocol v1.

The course arena loads this file directly, so it intentionally has no local
package imports.  ``GomokuAI`` is the submitted, enhanced alpha-beta player;
the baseline and MCTS classes are also exposed for GUI comparisons.
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field


NAME = "Adaptive-AlphaBeta"

EMPTY, BLACK, WHITE = 0, 1, 2
DIRECTIONS = ((0, 1), (1, 0), (1, 1), (1, -1))
WIN_SCORE = 10**12
INF = 10**15


class SearchTimeout(Exception):
    """Internal control-flow exception used to leave a search safely."""


class _BaseAI:
    def __init__(self, player_id: int, board_size: int, win_length: int):
        if player_id not in (BLACK, WHITE):
            raise ValueError("player_id must be 1 or 2")
        if isinstance(board_size, bool) or not isinstance(board_size, int):
            raise TypeError("board_size must be an integer")
        if isinstance(win_length, bool) or not isinstance(win_length, int):
            raise TypeError("win_length must be an integer")
        if board_size < 3 or not 3 <= win_length <= board_size:
            raise ValueError("configuration must satisfy N >= 3 and 3 <= K <= N")
        self.player_id = player_id
        self.board_size = board_size
        self.win_length = win_length
        self.opponent_id = 3 - player_id
        self._deadline = 0.0

    def _prepare_turn(
        self,
        board: list[list[int]],
        last_opponent_move: tuple[int, int] | None,
        time_limit: float,
    ) -> tuple[list[list[int]] | None, tuple[int, int], float]:
        start = time.perf_counter()
        try:
            limit = max(0.0, float(time_limit))
        except (TypeError, ValueError):
            limit = 0.0
        nominal_reserve = max(0.005, min(0.25, limit * 0.15))
        reserve = min(limit * 0.40, nominal_reserve)
        self._deadline = start + max(0.0, limit - reserve)

        if len(board) != self.board_size or any(
            len(row) != self.board_size for row in board
        ):
            raise ValueError("board dimensions do not match board_size")
        fallback = self._fallback_move(board, last_opponent_move)
        if not self._has_time(0.0005):
            return None, fallback, start
        work = [row[:] for row in board]
        return work, fallback, start

    def _fallback_move(
        self,
        board: list[list[int]],
        last_opponent_move: tuple[int, int] | None,
    ) -> tuple[int, int]:
        n = self.board_size
        center = n // 2
        if board[center][center] == EMPTY:
            return center, center

        if last_opponent_move is not None:
            row, col = last_opponent_move
            if isinstance(row, int) and isinstance(col, int):
                for radius in (1, 2):
                    for dr in range(-radius, radius + 1):
                        for dc in range(-radius, radius + 1):
                            rr, cc = row + dr, col + dc
                            if (
                                0 <= rr < n
                                and 0 <= cc < n
                                and board[rr][cc] == EMPTY
                            ):
                                return rr, cc

        for row, values in enumerate(board):
            try:
                return row, values.index(EMPTY)
            except ValueError:
                continue
        raise ValueError("no legal moves remain")

    def _check_time(self) -> None:
        if time.perf_counter() >= self._deadline:
            raise SearchTimeout

    def _has_time(self, minimum: float = 0.0) -> bool:
        return time.perf_counter() + minimum < self._deadline

    def _is_win(
        self, board: list[list[int]], row: int, col: int, player: int
    ) -> bool:
        n, k = self.board_size, self.win_length
        for dr, dc in DIRECTIONS:
            count = 1
            for sign in (-1, 1):
                rr, cc = row + sign * dr, col + sign * dc
                while 0 <= rr < n and 0 <= cc < n and board[rr][cc] == player:
                    count += 1
                    if count >= k:
                        return True
                    rr += sign * dr
                    cc += sign * dc
        return count >= k

    def _candidate_moves(
        self,
        board: list[list[int]],
        radius: int,
        cap: int | None,
    ) -> list[tuple[int, int]]:
        n = self.board_size
        occupied: list[tuple[int, int]] = []
        for row in range(n):
            if row & 3 == 0:
                self._check_time()
            for col, value in enumerate(board[row]):
                if value != EMPTY:
                    occupied.append((row, col))

        if not occupied:
            center = n // 2
            return [(center, center)]

        candidates: set[tuple[int, int]] = set()
        for index, (row, col) in enumerate(occupied):
            if index & 31 == 0:
                self._check_time()
            r0, r1 = max(0, row - radius), min(n, row + radius + 1)
            c0, c1 = max(0, col - radius), min(n, col + radius + 1)
            for rr in range(r0, r1):
                for cc in range(c0, c1):
                    if board[rr][cc] == EMPTY:
                        candidates.add((rr, cc))

        if not candidates:
            return []
        result = sorted(candidates)
        if cap is not None and len(result) > cap:
            center = (n - 1) / 2
            result.sort(key=lambda move: abs(move[0] - center) + abs(move[1] - center))
            result = result[:cap]
        return result

    def _tactical_move(
        self, board: list[list[int]], fallback: tuple[int, int]
    ) -> tuple[int, int] | None:
        try:
            candidates = self._candidate_moves(board, radius=1, cap=None)
            for player in (self.player_id, self.opponent_id):
                for index, (row, col) in enumerate(candidates):
                    if index & 15 == 0:
                        self._check_time()
                    board[row][col] = player
                    won = self._is_win(board, row, col, player)
                    board[row][col] = EMPTY
                    if won:
                        return row, col
        except SearchTimeout:
            return fallback
        return None

    def _evaluate_runs(
        self, board: list[list[int]], perspective: int, advanced: bool
    ) -> int:
        n, k = self.board_size, self.win_length
        score = 0
        for row in range(n):
            if row & 1 == 0:
                self._check_time()
            for col in range(n):
                player = board[row][col]
                if player == EMPTY:
                    continue
                sign = 1 if player == perspective else -1
                for dr, dc in DIRECTIONS:
                    previous_r, previous_c = row - dr, col - dc
                    if (
                        0 <= previous_r < n
                        and 0 <= previous_c < n
                        and board[previous_r][previous_c] == player
                    ):
                        continue
                    length = 1
                    rr, cc = row + dr, col + dc
                    while 0 <= rr < n and 0 <= cc < n and board[rr][cc] == player:
                        length += 1
                        rr += dr
                        cc += dc
                    if length >= k:
                        return sign * WIN_SCORE
                    open_ends = 0
                    if (
                        0 <= previous_r < n
                        and 0 <= previous_c < n
                        and board[previous_r][previous_c] == EMPTY
                    ):
                        open_ends += 1
                    if 0 <= rr < n and 0 <= cc < n and board[rr][cc] == EMPTY:
                        open_ends += 1
                    if open_ends == 0:
                        continue
                    if advanced:
                        progress = length / k
                        value = int(10 ** (1.0 + 6.0 * progress))
                        value *= 2 + 2 * open_ends
                        if length == k - 1:
                            value *= 40 if open_ends == 2 else 12
                    else:
                        value = (length + 1) ** 3 * open_ends
                    if sign < 0:
                        value = int(value * 1.12)
                    score += sign * value
        if advanced:
            score += self._evaluate_windows(board, perspective)
        return max(-WIN_SCORE + 1, min(WIN_SCORE - 1, score))

    def _evaluate_windows(self, board: list[list[int]], perspective: int) -> int:
        k = self.win_length
        opponent = 3 - perspective
        score = 0
        for line_index, line in enumerate(self._lines(board)):
            if line_index & 7 == 0:
                self._check_time()
            if len(line) < k:
                continue
            own = sum(value == perspective for value in line[:k])
            other = sum(value == opponent for value in line[:k])
            for start in range(len(line) - k + 1):
                if start:
                    leaving, entering = line[start - 1], line[start + k - 1]
                    own += (entering == perspective) - (leaving == perspective)
                    other += (entering == opponent) - (leaving == opponent)
                if own and not other:
                    value = 5**min(own, 10)
                    if own == k - 1:
                        value *= 30
                    score += value
                elif other and not own:
                    value = 5**min(other, 10)
                    if other == k - 1:
                        value *= 36
                    score -= value
        return score

    def _lines(self, board: list[list[int]]):
        n = self.board_size
        for row in board:
            yield row
        for col in range(n):
            yield [board[row][col] for row in range(n)]
        for start_col in range(n):
            yield [board[i][start_col + i] for i in range(n - start_col)]
            if start_col:
                yield [board[start_col + i][i] for i in range(n - start_col)]
        for start_col in range(n):
            yield [board[i][start_col - i] for i in range(start_col + 1)]
            if start_col < n - 1:
                yield [board[start_col + i][n - 1 - i] for i in range(n - start_col)]

    def _local_potential(
        self, board: list[list[int]], row: int, col: int, player: int
    ) -> int:
        n = self.board_size
        total = 0
        for dr, dc in DIRECTIONS:
            friendly = 0
            hostile = 0
            for sign in (-1, 1):
                for distance in range(1, min(self.win_length, 4)):
                    rr, cc = row + sign * distance * dr, col + sign * distance * dc
                    if not (0 <= rr < n and 0 <= cc < n):
                        break
                    value = board[rr][cc]
                    if value == player:
                        friendly += 1
                    elif value != EMPTY:
                        hostile += 1
                        break
            total += friendly * friendly * 8 - hostile
        center = (n - 1) / 2
        return total * 10 - int(abs(row - center) + abs(col - center))

    def _empty_count(self, board: list[list[int]]) -> int:
        count = 0
        for index, row in enumerate(board):
            if index & 7 == 0:
                self._check_time()
            count += row.count(EMPTY)
        return count


class BaselineAlphaBetaAI(_BaseAI):
    """Fixed-depth alpha-beta used as the non-advanced comparison."""

    def get_move(self, board, last_opponent_move, time_limit):
        work, fallback, _ = self._prepare_turn(board, last_opponent_move, time_limit)
        if work is None or not self._has_time(0.001):
            return fallback
        tactical = self._tactical_move(work, fallback)
        if tactical is not None:
            return tactical
        best = fallback
        depth = 4 if self.board_size <= 3 else 2
        try:
            moves = self._candidate_moves(work, radius=1, cap=14)
            alpha, beta = -INF, INF
            for row, col in moves:
                self._check_time()
                work[row][col] = self.player_id
                try:
                    if self._is_win(work, row, col, self.player_id):
                        score = WIN_SCORE
                    else:
                        score = -self._negamax(
                            work, depth - 1, -beta, -alpha, self.opponent_id, 1
                        )
                finally:
                    work[row][col] = EMPTY
                if score > alpha:
                    alpha, best = score, (row, col)
        except SearchTimeout:
            pass
        return best

    def _negamax(self, board, depth, alpha, beta, player, ply):
        self._check_time()
        if depth <= 0:
            return self._evaluate_runs(board, player, advanced=False)
        moves = self._candidate_moves(board, radius=1, cap=12)
        if not moves:
            return 0
        best = -INF
        for row, col in moves:
            self._check_time()
            board[row][col] = player
            try:
                if self._is_win(board, row, col, player):
                    score = WIN_SCORE - ply
                else:
                    score = -self._negamax(
                        board, depth - 1, -beta, -alpha, 3 - player, ply + 1
                    )
            finally:
                board[row][col] = EMPTY
            if score > best:
                best = score
            if score > alpha:
                alpha = score
            if alpha >= beta:
                break
        return best


class EnhancedAlphaBetaAI(_BaseAI):
    """Time-managed iterative-deepening alpha-beta player."""

    def __init__(self, player_id, board_size, win_length):
        super().__init__(player_id, board_size, win_length)
        self._table: dict[tuple[int, int, int], tuple[int, tuple[int, int] | None]] = {}

    def get_move(self, board, last_opponent_move, time_limit):
        work, fallback, _ = self._prepare_turn(board, last_opponent_move, time_limit)
        if work is None or not self._has_time(0.001):
            return fallback
        tactical = self._tactical_move(work, fallback)
        if tactical is not None:
            return tactical

        best = fallback
        preferred = fallback
        self._table.clear()
        try:
            root_hash = self._board_hash(work)
            max_depth = min(9 if self.board_size <= 4 else 7, self._empty_count(work))
        except SearchTimeout:
            return fallback
        for depth in range(1, max_depth + 1):
            if not self._has_time(0.003):
                break
            try:
                candidate, _ = self._search_root(work, depth, preferred, root_hash)
            except SearchTimeout:
                break
            if candidate is not None:
                best = preferred = candidate
        return best

    def _search_root(self, board, depth, preferred, state_hash):
        moves = self._ordered_moves(board, self.player_id, preferred, self._branch_cap(depth))
        if not moves:
            return None, 0
        best_move = moves[0]
        best_score = -INF
        alpha, beta = -INF, INF
        for row, col in moves:
            self._check_time()
            board[row][col] = self.player_id
            try:
                child_hash = state_hash ^ self._zobrist(row, col, self.player_id)
                if self._is_win(board, row, col, self.player_id):
                    score = WIN_SCORE
                else:
                    score = -self._negamax(
                        board,
                        depth - 1,
                        -beta,
                        -alpha,
                        self.opponent_id,
                        1,
                        child_hash,
                    )
            finally:
                board[row][col] = EMPTY
            if score > best_score:
                best_score, best_move = score, (row, col)
            alpha = max(alpha, score)
        return best_move, best_score

    def _negamax(self, board, depth, alpha, beta, player, ply, state_hash):
        self._check_time()
        if depth <= 0:
            return self._evaluate_runs(board, player, advanced=True)

        key = (state_hash, player, depth)
        cached = self._table.get(key)
        if cached is not None:
            return cached[0]
        preferred = None
        moves = self._ordered_moves(board, player, preferred, self._branch_cap(depth))
        if not moves:
            return 0

        best_score = -INF
        best_move = moves[0]
        cutoff = False
        for row, col in moves:
            self._check_time()
            board[row][col] = player
            try:
                child_hash = state_hash ^ self._zobrist(row, col, player)
                if self._is_win(board, row, col, player):
                    score = WIN_SCORE - ply
                else:
                    score = -self._negamax(
                        board,
                        depth - 1,
                        -beta,
                        -alpha,
                        3 - player,
                        ply + 1,
                        child_hash,
                    )
            finally:
                board[row][col] = EMPTY
            if score > best_score:
                best_score, best_move = score, (row, col)
            alpha = max(alpha, score)
            if alpha >= beta:
                cutoff = True
                break
        if not cutoff:
            self._table[key] = (best_score, best_move)
        return best_score

    def _ordered_moves(self, board, player, preferred, cap):
        candidates = self._candidate_moves(board, radius=2, cap=None)
        scores: list[tuple[int, tuple[int, int]]] = []
        opponent = 3 - player
        for index, (row, col) in enumerate(candidates):
            if index & 7 == 0:
                self._check_time()
            priority = self._local_potential(board, row, col, player)
            if preferred == (row, col):
                priority += WIN_SCORE
            board[row][col] = player
            if self._is_win(board, row, col, player):
                priority += WIN_SCORE // 2
            board[row][col] = opponent
            if self._is_win(board, row, col, opponent):
                priority += WIN_SCORE // 3
            board[row][col] = EMPTY
            scores.append((priority, (row, col)))
        scores.sort(key=lambda item: item[0], reverse=True)
        return [move for _, move in scores[:cap]]

    def _branch_cap(self, depth):
        remaining = max(0.0, self._deadline - time.perf_counter())
        if remaining < 0.03:
            return 6
        if depth >= 4:
            return 10
        return 22 if self.board_size <= 9 else 18

    def _board_hash(self, board):
        value = 0
        for row in range(self.board_size):
            if row & 7 == 0:
                self._check_time()
            for col, player in enumerate(board[row]):
                if player:
                    value ^= self._zobrist(row, col, player)
        return value

    @staticmethod
    def _zobrist(row, col, player):
        value = (row + 1) * 0x9E3779B185EBCA87
        value ^= (col + 1) * 0xC2B2AE3D27D4EB4F
        value ^= player * 0x165667B19E3779F9
        value &= (1 << 64) - 1
        value ^= value >> 30
        value = (value * 0xBF58476D1CE4E5B9) & ((1 << 64) - 1)
        value ^= value >> 27
        return value


@dataclass
class _MCTSNode:
    board: list[list[int]]
    player_to_move: int
    move: tuple[int, int] | None = None
    parent: "_MCTSNode | None" = None
    visits: int = 0
    value: float = 0.0
    children: list["_MCTSNode"] = field(default_factory=list)
    untried: list[tuple[int, int]] | None = None


class MCTSAI(_BaseAI):
    """UCT Monte Carlo tree search with deadline-bounded rollouts."""

    def __init__(self, player_id, board_size, win_length):
        super().__init__(player_id, board_size, win_length)
        self._random = random.Random()

    def get_move(self, board, last_opponent_move, time_limit):
        work, fallback, _ = self._prepare_turn(board, last_opponent_move, time_limit)
        if work is None or not self._has_time(0.001):
            return fallback
        tactical = self._tactical_move(work, fallback)
        if tactical is not None:
            return tactical

        root = _MCTSNode(work, self.player_id)
        try:
            root.untried = self._mcts_candidates(work, self.player_id)
            if not root.untried:
                return fallback
            while self._has_time(0.001):
                node = root
                while node.untried == [] and node.children:
                    self._check_time()
                    node = self._select_child(node)

                if node.untried:
                    move = node.untried.pop(self._random.randrange(len(node.untried)))
                    child_board = [row[:] for row in node.board]
                    child_board[move[0]][move[1]] = node.player_to_move
                    child = _MCTSNode(
                        child_board, 3 - node.player_to_move, move=move, parent=node
                    )
                    if self._is_win(
                        child_board, move[0], move[1], node.player_to_move
                    ):
                        child.untried = []
                    else:
                        child.untried = self._mcts_candidates(
                            child_board, child.player_to_move
                        )
                    node.children.append(child)
                    node = child

                result = self._rollout(node.board, node.player_to_move, node.move)
                while node is not None:
                    node.visits += 1
                    node.value += result
                    node = node.parent
        except SearchTimeout:
            pass

        if not root.children:
            return fallback
        return max(root.children, key=lambda child: child.visits).move

    def _select_child(self, node):
        log_parent = math.log(max(1, node.visits))
        maximizing_root = node.player_to_move == self.player_id

        def uct(child):
            if child.visits == 0:
                return INF
            exploit = child.value / child.visits
            if not maximizing_root:
                exploit = -exploit
            explore = math.sqrt(2.0 * log_parent / child.visits)
            return exploit + explore

        return max(node.children, key=uct)

    def _mcts_candidates(self, board, player):
        moves = self._candidate_moves(board, radius=1, cap=None)
        scored = []
        for index, (row, col) in enumerate(moves):
            if index & 7 == 0:
                self._check_time()
            scored.append((self._local_potential(board, row, col, player), (row, col)))
        scored.sort(reverse=True)
        cap = 16 if self.board_size <= 9 else 12
        return [move for _, move in scored[:cap]]

    def _rollout(self, source_board, player, last_move):
        board = [row[:] for row in source_board]
        if last_move is not None:
            previous = 3 - player
            if self._is_win(board, last_move[0], last_move[1], previous):
                return 1.0 if previous == self.player_id else -1.0

        max_steps = min(self._empty_count(board), max(10, self.win_length * 3))
        for _ in range(max_steps):
            self._check_time()
            moves = self._candidate_moves(board, radius=1, cap=18)
            if not moves:
                return 0.0
            move = self._rollout_move(board, moves, player)
            board[move[0]][move[1]] = player
            if self._is_win(board, move[0], move[1], player):
                return 1.0 if player == self.player_id else -1.0
            player = 3 - player

        score = self._evaluate_runs(board, self.player_id, advanced=False)
        if score > 0:
            return 0.2
        if score < 0:
            return -0.2
        return 0.0

    def _rollout_move(self, board, moves, player):
        opponent = 3 - player
        for candidate_player in (player, opponent):
            for row, col in moves:
                self._check_time()
                board[row][col] = candidate_player
                won = self._is_win(board, row, col, candidate_player)
                board[row][col] = EMPTY
                if won:
                    return row, col
        sample = moves[: min(6, len(moves))]
        weights = [
            max(1, self._local_potential(board, row, col, player) + 30)
            for row, col in sample
        ]
        return self._random.choices(sample, weights=weights, k=1)[0]


class GomokuAI(EnhancedAlphaBetaAI):
    """Course submission entry point."""
