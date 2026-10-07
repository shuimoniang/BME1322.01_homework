"""Self-contained generalized Gomoku AIs implementing protocol v1.

The course arena loads this file directly, so it intentionally has no local
package imports.  ``GomokuAI`` is the submitted hybrid threat-search player;
the baseline and MCTS classes are also exposed for GUI comparisons.
"""

from __future__ import annotations

import math
import random
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import IntEnum


NAME = "Hybrid-Threat-Search"

EMPTY, BLACK, WHITE = 0, 1, 2
DIRECTIONS = ((0, 1), (1, 0), (1, 1), (1, -1))
WIN_SCORE = 10**12
INF = 10**15
MATE_THRESHOLD = WIN_SCORE // 2

TT_EXACT, TT_UPPER, TT_LOWER = 0, -1, 1


class ThreatResult(IntEnum):
    UNKNOWN = -1
    DISPROVEN = 0
    PROVEN = 1


@dataclass(frozen=True)
class TTEntry:
    depth: int
    score: int
    bound: int
    best_move: tuple[int, int] | None


class SearchTimeout(Exception):
    """Internal control-flow exception used to leave a search safely."""


@dataclass(frozen=True)
class MoveThreatProfile:
    """Generalized tactical features created by one legal move.

    The representation deliberately depends on K-length windows and distinct
    completion squares, not Gomoku-specific names such as "open four".
    """

    immediate_win: bool
    winning_replies: frozenset[tuple[int, int]]
    near_windows: int
    development_windows: int
    progress_score: int

    @property
    def reply_count(self) -> int:
        return len(self.winning_replies)


@dataclass
class PositionThreatSummary:
    """All generalized tactical facts derived in one K-window scan."""

    winning_moves: tuple[set[tuple[int, int]], set[tuple[int, int]], set[tuple[int, int]]]
    winning_replies: tuple[
        dict[tuple[int, int], frozenset[tuple[int, int]]],
        dict[tuple[int, int], frozenset[tuple[int, int]]],
        dict[tuple[int, int], frozenset[tuple[int, int]]],
    ]
    near_windows: tuple[dict[tuple[int, int], int], dict[tuple[int, int], int], dict[tuple[int, int], int]]
    development_windows: tuple[dict[tuple[int, int], int], dict[tuple[int, int], int], dict[tuple[int, int], int]]
    move_progress: tuple[dict[tuple[int, int], int], dict[tuple[int, int], int], dict[tuple[int, int], int]]
    positional: tuple[int, int, int]

    def profile(self, move: tuple[int, int], player: int) -> MoveThreatProfile:
        return MoveThreatProfile(
            move in self.winning_moves[player],
            self.winning_replies[player].get(move, frozenset()),
            self.near_windows[player].get(move, 0),
            self.development_windows[player].get(move, 0),
            self.move_progress[player].get(move, 0),
        )


_WINDOW_GEOMETRY_CACHE: dict[
    tuple[int, int],
    tuple[tuple[int, int, int, int, int, int, int, int], ...],
] = {}


def _window_geometry(board_size: int, win_length: int):
    """Return compact board-independent descriptors for every K-window."""
    key = (board_size, win_length)
    cached = _WINDOW_GEOMETRY_CACHE.get(key)
    if cached is not None:
        return cached
    windows = []
    for row in range(board_size):
        for col in range(board_size):
            for dr, dc in DIRECTIONS:
                end_r = row + (win_length - 1) * dr
                end_c = col + (win_length - 1) * dc
                if 0 <= end_r < board_size and 0 <= end_c < board_size:
                    windows.append(
                        (
                            row,
                            col,
                            dr,
                            dc,
                            row - dr,
                            col - dc,
                            end_r + dr,
                            end_c + dc,
                        )
                    )
    result = tuple(windows)
    if len(_WINDOW_GEOMETRY_CACHE) >= 32:
        _WINDOW_GEOMETRY_CACHE.pop(next(iter(_WINDOW_GEOMETRY_CACHE)))
    _WINDOW_GEOMETRY_CACHE[key] = result
    return result


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
        self._progress_weights = tuple(
            0
            if stones == 0
            else min(
                WIN_SCORE // 10_000,
                round(10 ** (1.0 + 7.0 * stones / win_length)),
            )
            for stones in range(win_length + 1)
        )
        self._deadline = 0.0
        self._analysis_cache: dict[int, PositionThreatSummary] = {}
        self._analysis_cache_limit = 512 if board_size <= 15 else 128
        self.last_search_stats: dict[str, int | bool | float] = {}
        self._reset_search_stats()

    def _reset_search_stats(self) -> None:
        self._analysis_cache.clear()
        self.last_search_stats = {
            "completed_depth": 0,
            "nodes": 0,
            "evaluations": 0,
            "analysis_scans": 0,
            "analysis_cache_hits": 0,
            "tt_probes": 0,
            "tt_hits": 0,
            "tt_cutoffs": 0,
            "quiescence_nodes": 0,
            "threat_nodes": 0,
            "threat_proven": False,
            "threat_disproven": 0,
            "threat_unknown": 0,
            "mcts_iterations": 0,
            "mcts_max_depth": 0,
        }

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

    def _evaluate_baseline(
        self, board: list[list[int]], perspective: int
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
                    value = (length + 1) ** 3 * open_ends
                    if sign < 0:
                        value = int(value * 1.12)
                    score += sign * value
        return max(-WIN_SCORE + 1, min(WIN_SCORE - 1, score))

    def _progress_weight(self, stones: int) -> int:
        if stones <= 0:
            return 0
        return self._progress_weights[min(stones, self.win_length)]

    def _analyze_position(
        self, board: list[list[int]], state_hash: int | None = None
    ) -> PositionThreatSummary:
        if state_hash is not None:
            cached = self._analysis_cache.get(state_hash)
            if cached is not None:
                if self.last_search_stats:
                    self.last_search_stats["analysis_cache_hits"] += 1
                return cached

        if self.last_search_stats:
            self.last_search_stats["analysis_scans"] += 1
        n, k = self.board_size, self.win_length
        winning_moves = [set(), set(), set()]
        reply_sets = [defaultdict(set), defaultdict(set), defaultdict(set)]
        near_windows = [defaultdict(int), defaultdict(int), defaultdict(int)]
        development = [defaultdict(int), defaultdict(int), defaultdict(int)]
        progress = [defaultdict(int), defaultdict(int), defaultdict(int)]
        positional = [0, 0, 0]

        for index, descriptor in enumerate(_window_geometry(n, k)):
            if index & 31 == 0:
                self._check_time()
            row, col, dr, dc, before_r, before_c, after_r, after_c = descriptor
            counts = [0, 0, 0]
            empties: list[tuple[int, int]] = []
            for step in range(k):
                rr, cc = row + step * dr, col + step * dc
                value = board[rr][cc]
                counts[value] += 1
                if value == EMPTY:
                    empties.append((rr, cc))
            open_ends = int(
                0 <= before_r < n
                and 0 <= before_c < n
                and board[before_r][before_c] == EMPTY
            )
            open_ends += int(
                0 <= after_r < n
                and 0 <= after_c < n
                and board[after_r][after_c] == EMPTY
            )
            openness = 4 + open_ends

            for player in (BLACK, WHITE):
                opponent = 3 - player
                if counts[opponent]:
                    continue
                stones = counts[player]
                if stones:
                    positional[player] += self._progress_weight(stones) * openness // 4
                if not empties:
                    continue
                move_value = self._progress_weight(stones + 1) * openness // 4
                for move in empties:
                    progress[player][move] += move_value
                if stones == k - 1 and len(empties) == 1:
                    winning_moves[player].add(empties[0])
                elif stones == k - 2 and len(empties) == 2:
                    first, second = empties
                    reply_sets[player][first].add(second)
                    reply_sets[player][second].add(first)
                elif stones == k - 3 and len(empties) == 3:
                    for move in empties:
                        near_windows[player][move] += 1
                elif stones >= max(0, k - 4):
                    for move in empties:
                        development[player][move] += 1

        frozen_replies = []
        for player in range(3):
            frozen_replies.append(
                {move: frozenset(replies) for move, replies in reply_sets[player].items()}
            )
        summary = PositionThreatSummary(
            (winning_moves[0], winning_moves[1], winning_moves[2]),
            (frozen_replies[0], frozen_replies[1], frozen_replies[2]),
            (dict(near_windows[0]), dict(near_windows[1]), dict(near_windows[2])),
            (dict(development[0]), dict(development[1]), dict(development[2])),
            (dict(progress[0]), dict(progress[1]), dict(progress[2])),
            (positional[0], positional[1], positional[2]),
        )
        if state_hash is not None:
            if len(self._analysis_cache) >= self._analysis_cache_limit:
                self._analysis_cache.pop(next(iter(self._analysis_cache)))
            self._analysis_cache[state_hash] = summary
        return summary

    def _move_threat_profile(
        self,
        board: list[list[int]],
        row: int,
        col: int,
        player: int,
        state_hash: int | None = None,
        summary: PositionThreatSummary | None = None,
    ) -> MoveThreatProfile:
        """Return one move's profile from a shared position analysis."""
        if board[row][col] != EMPTY:
            return MoveThreatProfile(False, frozenset(), 0, 0, 0)
        if summary is None:
            summary = self._analyze_position(board, state_hash)
        return summary.profile((row, col), player)

    @staticmethod
    def _threat_tier(
        own: MoveThreatProfile, opponent: MoveThreatProfile
    ) -> int:
        if own.immediate_win:
            return 7
        if opponent.immediate_win:
            return 6
        if own.reply_count >= 2:
            return 5
        if opponent.reply_count >= 2:
            return 4
        if own.reply_count == 1:
            return 3
        if opponent.reply_count == 1:
            return 2
        return 1

    def _tactical_entries(
        self,
        board: list[list[int]],
        player: int,
        candidates: list[tuple[int, int]] | None = None,
        preferred: tuple[int, int] | None = None,
        state_hash: int | None = None,
        summary: PositionThreatSummary | None = None,
    ):
        if summary is None:
            summary = self._analyze_position(board, state_hash)
        if candidates is None:
            candidates = self._candidate_moves(board, radius=2, cap=None)
        opponent = 3 - player
        own_wins = summary.winning_moves[player]
        opponent_wins = summary.winning_moves[opponent]
        if own_wins:
            candidates = sorted(own_wins)
        elif opponent_wins:
            candidates = sorted(opponent_wins)
        entries = []
        center = (self.board_size - 1) / 2
        for index, (row, col) in enumerate(candidates):
            if index & 7 == 0:
                self._check_time()
            if board[row][col] != EMPTY:
                continue
            own = summary.profile((row, col), player)
            defensive = summary.profile((row, col), opponent)
            tier = self._threat_tier(own, defensive)
            positional = (
                own.progress_score
                + defensive.progress_score
                - int(abs(row - center) + abs(col - center))
            )
            dual = own.reply_count * 4 + defensive.reply_count * 3
            preferred_bonus = 1 if preferred == (row, col) else 0
            entries.append(
                (
                    (tier, preferred_bonus, dual, positional),
                    (row, col),
                    own,
                    defensive,
                )
            )
        entries.sort(key=lambda item: item[0], reverse=True)

        own_wins = [item for item in entries if item[2].immediate_win]
        if own_wins:
            return own_wins
        required_blocks = [item for item in entries if item[3].immediate_win]
        if required_blocks:
            return required_blocks
        return entries

    def _ranked_tactical_moves(
        self,
        board: list[list[int]],
        player: int,
        preferred: tuple[int, int] | None,
        cap: int | None,
        candidates: list[tuple[int, int]] | None = None,
        state_hash: int | None = None,
        summary: PositionThreatSummary | None = None,
    ) -> list[tuple[int, int]]:
        entries = self._tactical_entries(
            board,
            player,
            candidates,
            preferred,
            state_hash,
            summary,
        )
        if cap is None or len(entries) <= cap:
            return [item[1] for item in entries]
        return [item[1] for item in entries[: max(0, cap)]]

    def _advanced_tactical_move(
        self,
        board: list[list[int]],
        player: int,
        state_hash: int,
        summary: PositionThreatSummary | None = None,
    ) -> tuple[int, int] | None:
        """Return only root moves that are correct without deeper search."""
        if summary is None:
            summary = self._analyze_position(board, state_hash)
        opponent = 3 - player
        if summary.winning_moves[player]:
            return max(
                summary.winning_moves[player],
                key=lambda move: summary.move_progress[player].get(move, 0),
            )
        if len(summary.winning_moves[opponent]) == 1:
            return next(iter(summary.winning_moves[opponent]))
        if not summary.winning_moves[opponent]:
            forks = [
                move
                for move, replies in summary.winning_replies[player].items()
                if len(replies) >= 2 and board[move[0]][move[1]] == EMPTY
            ]
            if forks:
                return max(
                    forks,
                    key=lambda move: summary.move_progress[player].get(move, 0),
                )
        return None

    def _evaluate_generalized(
        self,
        board: list[list[int]],
        perspective: int,
        state_hash: int | None = None,
        summary: PositionThreatSummary | None = None,
    ) -> int:
        """Fast tempo-aware evaluation shared by non-baseline strategies."""
        if self.last_search_stats:
            self.last_search_stats["evaluations"] += 1
        if summary is None:
            summary = self._analyze_position(board, state_hash)
        opponent = 3 - perspective
        own_wins = len(summary.winning_moves[perspective])
        opponent_wins = len(summary.winning_moves[opponent])
        own_reply_counts = [
            len(replies) for replies in summary.winning_replies[perspective].values()
        ]
        opponent_reply_counts = [
            len(replies) for replies in summary.winning_replies[opponent].values()
        ]
        own_forks = sum(count >= 2 for count in own_reply_counts)
        opponent_forks = sum(count >= 2 for count in opponent_reply_counts)
        own_forcing = sum(count == 1 for count in own_reply_counts)
        opponent_forcing = sum(count == 1 for count in opponent_reply_counts)
        positional = summary.positional[perspective] - summary.positional[opponent]

        tactical = 0
        if own_wins:
            tactical += WIN_SCORE // 8 + min(own_wins, 4) * (WIN_SCORE // 256)
        if opponent_wins >= 2:
            tactical -= WIN_SCORE // 8
        elif opponent_wins == 1:
            tactical -= WIN_SCORE // 32
        tactical += min(own_forks, 4) * (WIN_SCORE // 64)
        tactical -= min(opponent_forks, 4) * (WIN_SCORE // 80)
        tactical += min(own_forcing, 8) * (WIN_SCORE // 4096)
        tactical -= min(opponent_forcing, 8) * (WIN_SCORE // 3072)
        score = tactical + max(
            -WIN_SCORE // 1000, min(WIN_SCORE // 1000, positional)
        )
        return max(-WIN_SCORE + 1, min(WIN_SCORE - 1, score))

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
            return self._evaluate_baseline(board, player)
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
        self._table: dict[tuple[int, int], TTEntry] = {}

    def get_move(self, board, last_opponent_move, time_limit):
        self._reset_search_stats()
        work, fallback, _ = self._prepare_turn(board, last_opponent_move, time_limit)
        if work is None or not self._has_time(0.001):
            return fallback
        self._table.clear()
        try:
            root_hash = self._board_hash(work)
            root_summary = self._analyze_position(work, root_hash)
            tactical = self._advanced_tactical_move(
                work, self.player_id, root_hash, root_summary
            )
            if tactical is not None:
                return tactical
            max_depth = min(9 if self.board_size <= 4 else 7, self._empty_count(work))
        except SearchTimeout:
            return fallback

        best = fallback
        preferred = fallback
        for depth in range(1, max_depth + 1):
            if not self._has_time(0.003):
                break
            try:
                candidate, _ = self._search_root(work, depth, preferred, root_hash)
            except SearchTimeout:
                break
            if candidate is not None:
                best = preferred = candidate
                self.last_search_stats["completed_depth"] = depth
        return best

    def _search_root(self, board, depth, preferred, state_hash):
        moves = self._ordered_moves(
            board,
            self.player_id,
            preferred,
            self._branch_cap(depth),
            state_hash,
        )
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
        self.last_search_stats["nodes"] += 1
        if depth <= 0:
            return self._evaluate_generalized(board, player, state_hash)

        alpha_original, beta_original = alpha, beta
        key = (state_hash, player)
        cached_score, alpha, beta, preferred = self._probe_tt(
            key, depth, alpha, beta, ply
        )
        if cached_score is not None:
            return cached_score
        moves = self._ordered_moves(
            board, player, preferred, self._branch_cap(depth), state_hash
        )
        if not moves:
            return 0

        best_score = -INF
        best_move = moves[0]
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
                break
        if best_score <= alpha_original:
            bound = TT_UPPER
        elif best_score >= beta_original:
            bound = TT_LOWER
        else:
            bound = TT_EXACT
        self._store_tt(key, depth, best_score, bound, best_move, ply)
        return best_score

    def _ordered_moves(self, board, player, preferred, cap, state_hash):
        return self._ranked_tactical_moves(
            board,
            player,
            preferred,
            cap,
            state_hash=state_hash,
        )

    def _branch_cap(self, depth):
        remaining = max(0.0, self._deadline - time.perf_counter())
        if remaining < 0.03:
            return 6
        if depth >= 4:
            return 10
        return 22 if self.board_size <= 9 else 18

    @staticmethod
    def _score_to_tt(score, ply):
        if score >= MATE_THRESHOLD:
            return score + ply
        if score <= -MATE_THRESHOLD:
            return score - ply
        return score

    @staticmethod
    def _score_from_tt(score, ply):
        if score >= MATE_THRESHOLD:
            return score - ply
        if score <= -MATE_THRESHOLD:
            return score + ply
        return score

    def _probe_tt(self, key, depth, alpha, beta, ply):
        self.last_search_stats["tt_probes"] += 1
        entry = self._table.get(key)
        if entry is None:
            return None, alpha, beta, None
        self.last_search_stats["tt_hits"] += 1
        preferred = entry.best_move
        if entry.depth < depth:
            return None, alpha, beta, preferred
        score = self._score_from_tt(entry.score, ply)
        if entry.bound == TT_EXACT:
            self.last_search_stats["tt_cutoffs"] += 1
            return score, alpha, beta, preferred
        if entry.bound == TT_LOWER:
            alpha = max(alpha, score)
        else:
            beta = min(beta, score)
        if alpha >= beta:
            self.last_search_stats["tt_cutoffs"] += 1
            return score, alpha, beta, preferred
        return None, alpha, beta, preferred

    def _store_tt(self, key, depth, score, bound, best_move, ply):
        current = self._table.get(key)
        if current is not None:
            if current.depth > depth:
                return
            if current.depth == depth and current.bound == TT_EXACT and bound != TT_EXACT:
                return
        self._table[key] = TTEntry(
            depth,
            self._score_to_tt(score, ply),
            bound,
            best_move,
        )


class HybridThreatSearchAI(EnhancedAlphaBetaAI):
    """Threat-space/VCF search combined with selective alpha-beta.

    This deliberately remains self-contained and protocol-compatible.  It
    uses the existing generalized evaluator, but gives forcing moves a much
    higher priority than quiet positional moves.  The threat search is a
    bounded VCF/VCT-style proof search; when it cannot prove a win, the
    regular iterative search takes over with quiescence and PVS.
    """

    def __init__(self, player_id, board_size, win_length):
        super().__init__(player_id, board_size, win_length)
        self._threat_table: dict[tuple[int, int, int, int, str], ThreatResult] = {}
        self._killers: dict[int, list[tuple[int, int]]] = {}
        self._history: dict[tuple[int, tuple[int, int]], int] = {}
        self._pv_move: tuple[int, int] | None = None
        self._threat_nodes = 0
        self._threat_hits = 0
        self._threat_node_limit = 0

    def get_move(self, board, last_opponent_move, time_limit):
        self._reset_search_stats()
        work, fallback, _ = self._prepare_turn(board, last_opponent_move, time_limit)
        if work is None or not self._has_time(0.001):
            return fallback
        try:
            self._table.clear()
            self._threat_table.clear()
            self._killers.clear()
            self._history.clear()
            self._pv_move = fallback
            self._threat_nodes = 0
            root_hash = self._board_hash(work)
            root_summary = self._analyze_position(work, root_hash)
            tactical = self._advanced_tactical_move(
                work, self.player_id, root_hash, root_summary
            )
            if tactical is not None:
                return tactical

            # First spend a genuinely independent, bounded slice proving
            # forcing wins.  Restoring the main deadline is essential: a
            # failed proof search must not consume the whole move budget and
            # force the positional search to return its fallback move.
            threat_depth = min(9, max(5, self.win_length + 2))
            main_deadline = self._deadline
            remaining = max(0.0, main_deadline - time.perf_counter())
            threat_budget = min(0.35, max(0.008, remaining * 0.12))
            self._threat_node_limit = 1600 if self.board_size <= 9 else 900
            forced = None
            self._deadline = min(main_deadline, time.perf_counter() + threat_budget)
            try:
                forced = self._threat_root(
                    work, threat_depth, root_hash, root_summary
                )
            except SearchTimeout:
                forced = None
                self.last_search_stats["threat_unknown"] += 1
            finally:
                self._deadline = main_deadline
                self.last_search_stats["threat_nodes"] = self._threat_nodes
            if forced is not None:
                self._threat_hits += 1
                self.last_search_stats["threat_proven"] = True
                return forced

            best, best_score = fallback, -INF
            max_depth = min(8 if self.board_size <= 9 else 6, self._empty_count(work))
            previous_score = 0
            for depth in range(1, max_depth + 1):
                if not self._has_time(0.004):
                    break
                # Aspiration windows reduce work after a stable iteration.
                if depth >= 3 and abs(previous_score) < WIN_SCORE // 2:
                    window = 25000 + depth * 5000
                    alpha, beta = previous_score - window, previous_score + window
                else:
                    alpha, beta = -INF, INF
                try:
                    move, score = self._hybrid_root(
                        work, depth, self._pv_move, root_hash, alpha, beta
                    )
                    if score <= alpha or score >= beta:
                        move, score = self._hybrid_root(
                            work, depth, self._pv_move, root_hash, -INF, INF
                        )
                except SearchTimeout:
                    break
                if move is not None:
                    best, best_score, self._pv_move = move, score, move
                    previous_score = score
                    self.last_search_stats["completed_depth"] = depth
            return best
        except SearchTimeout:
            return fallback

    def _winning_moves(self, board, player, cap=40, state_hash=None, summary=None):
        if summary is None:
            summary = self._analyze_position(board, state_hash)
        return sorted(summary.winning_moves[player])[:cap]

    def _forcing_moves(self, board, player, cap=18, state_hash=None, summary=None):
        """Return offensive forcing moves and tactically required defenses."""
        entries = self._tactical_entries(
            board,
            player,
            state_hash=state_hash,
            summary=summary,
        )
        return [item[1] for item in entries if item[0][0] >= 2][:cap]

    def _attacking_forcing_moves(
        self, board, player, cap=18, state_hash=None, summary=None
    ):
        entries = self._tactical_entries(
            board,
            player,
            state_hash=state_hash,
            summary=summary,
        )
        return [
            item[1]
            for item in entries
            if item[2].immediate_win or item[2].reply_count
        ][:cap]

    def _threat_root(self, board, plies_left, state_hash, summary=None):
        moves = self._attacking_forcing_moves(
            board,
            self.player_id,
            cap=14,
            state_hash=state_hash,
            summary=summary,
        )
        for row, col in moves:
            self._check_time()
            board[row][col] = self.player_id
            try:
                if self._is_win(board, row, col, self.player_id):
                    return row, col
                child_hash = state_hash ^ self._zobrist(row, col, self.player_id)
                result = self._threat_search(
                    board,
                    self.opponent_id,
                    plies_left - 1,
                    child_hash,
                    self.player_id,
                )
                if result == ThreatResult.PROVEN:
                    return row, col
            finally:
                board[row][col] = EMPTY
        return None

    def _threat_search(
        self, board, player, plies_left, state_hash, attacker=None
    ) -> ThreatResult:
        """Resolve a defender-to-move node without treating one threat as mate."""
        self._check_time()
        self._threat_nodes += 1
        self.last_search_stats["threat_nodes"] = self._threat_nodes
        if self._threat_nodes > self._threat_node_limit:
            self.last_search_stats["threat_unknown"] += 1
            return ThreatResult.UNKNOWN
        if plies_left <= 0:
            return ThreatResult.DISPROVEN
        if attacker is None:
            attacker = 3 - player
        key = (state_hash, player, attacker, plies_left, "defend")
        cached = self._threat_table.get(key)
        if cached is not None:
            return cached
        summary = self._analyze_position(board, state_hash)
        # A defender that can win immediately does not have to answer the threat.
        defensive_wins = self._winning_moves(
            board, player, cap=8, state_hash=state_hash, summary=summary
        )
        if defensive_wins:
            self._threat_table[key] = ThreatResult.DISPROVEN
            self.last_search_stats["threat_disproven"] += 1
            return ThreatResult.DISPROVEN
        attacker_wins = self._winning_moves(
            board, attacker, cap=8, state_hash=state_hash, summary=summary
        )
        if len(attacker_wins) >= 2:
            self._threat_table[key] = ThreatResult.PROVEN
            return ThreatResult.PROVEN
        if not attacker_wins:
            self._threat_table[key] = ThreatResult.DISPROVEN
            self.last_search_stats["threat_disproven"] += 1
            return ThreatResult.DISPROVEN

        # Exactly one winning square is a forced reply, not a proof.  After
        # blocking it, the attacker must create another concrete threat.
        row, col = attacker_wins[0]
        board[row][col] = player
        try:
            reply_hash = state_hash ^ self._zobrist(row, col, player)
            result = self._threat_attack(
                board, attacker, plies_left - 1, reply_hash
            )
        finally:
            board[row][col] = EMPTY
        if result != ThreatResult.UNKNOWN:
            self._threat_table[key] = result
        return result

    def _threat_attack(self, board, attacker, plies_left, state_hash):
        self._check_time()
        self._threat_nodes += 1
        self.last_search_stats["threat_nodes"] = self._threat_nodes
        if self._threat_nodes > self._threat_node_limit:
            self.last_search_stats["threat_unknown"] += 1
            return ThreatResult.UNKNOWN
        if plies_left <= 0:
            return ThreatResult.DISPROVEN
        defender = 3 - attacker
        key = (state_hash, attacker, attacker, plies_left, "attack")
        cached = self._threat_table.get(key)
        if cached is not None:
            return cached
        summary = self._analyze_position(board, state_hash)
        if summary.winning_moves[attacker]:
            self._threat_table[key] = ThreatResult.PROVEN
            return ThreatResult.PROVEN
        defender_wins = summary.winning_moves[defender]
        if len(defender_wins) >= 2:
            self._threat_table[key] = ThreatResult.DISPROVEN
            return ThreatResult.DISPROVEN
        moves = self._attacking_forcing_moves(
            board,
            attacker,
            cap=10,
            state_hash=state_hash,
            summary=summary,
        )
        if len(defender_wins) == 1:
            required = next(iter(defender_wins))
            moves = [move for move in moves if move == required]
        saw_unknown = False
        for row, col in moves:
            self._check_time()
            board[row][col] = attacker
            try:
                child_hash = state_hash ^ self._zobrist(row, col, attacker)
                if self._is_win(board, row, col, attacker):
                    result = ThreatResult.PROVEN
                else:
                    result = self._threat_search(
                        board,
                        defender,
                        plies_left - 1,
                        child_hash,
                        attacker,
                    )
            finally:
                board[row][col] = EMPTY
            if result == ThreatResult.PROVEN:
                self._threat_table[key] = result
                return result
            saw_unknown |= result == ThreatResult.UNKNOWN
        result = ThreatResult.UNKNOWN if saw_unknown else ThreatResult.DISPROVEN
        if result != ThreatResult.UNKNOWN:
            self._threat_table[key] = result
            self.last_search_stats["threat_disproven"] += 1
        return result

    def _hybrid_root(self, board, depth, preferred, state_hash, alpha, beta):
        moves = self._ordered_hybrid_moves(
            board, self.player_id, preferred, depth, state_hash
        )
        if not moves:
            return None, 0
        best_move, best_score = moves[0], -INF
        first = True
        for row, col in moves:
            self._check_time()
            board[row][col] = self.player_id
            try:
                child_hash = state_hash ^ self._zobrist(row, col, self.player_id)
                if self._is_win(board, row, col, self.player_id):
                    score = WIN_SCORE - 1
                elif first:
                    score = -self._hybrid_search(
                        board, depth - 1, -beta, -alpha, self.opponent_id, 1, child_hash
                    )
                else:
                    score = -self._hybrid_search(
                        board, depth - 1, -alpha - 1, -alpha,
                        self.opponent_id, 1, child_hash
                    )
                    if alpha < score < beta:
                        score = -self._hybrid_search(
                            board, depth - 1, -beta, -alpha,
                            self.opponent_id, 1, child_hash
                        )
            finally:
                board[row][col] = EMPTY
            first = False
            if score > best_score:
                best_score, best_move = score, (row, col)
            alpha = max(alpha, score)
            if alpha >= beta:
                self._record_cutoff((row, col), depth)
                break
        return best_move, best_score

    def _hybrid_search(self, board, depth, alpha, beta, player, ply, state_hash):
        self._check_time()
        self.last_search_stats["nodes"] += 1
        if depth <= 0:
            return self._quiescence(
                board, alpha, beta, player, 3, ply, state_hash
            )
        alpha_original, beta_original = alpha, beta
        key = (state_hash, player)
        cached_score, alpha, beta, preferred = self._probe_tt(
            key, depth, alpha, beta, ply
        )
        if cached_score is not None:
            return cached_score
        moves = self._ordered_hybrid_moves(
            board, player, preferred, depth, state_hash
        )
        if not moves:
            return 0
        best, best_move, original_alpha = -INF, moves[0], alpha
        for index, (row, col) in enumerate(moves):
            self._check_time()
            # LMR classification must inspect the position *before* the move
            # is placed. Calling _move_level after placement would let its
            # temporary probes erase the live child move before recursion.
            late_move = index >= 5 and depth >= 4
            move_level = (
                self._move_level(board, row, col, player, state_hash)
                if late_move
                else 6
            )
            is_killer = (row, col) in self._killers.get(depth, ())
            board[row][col] = player
            try:
                child_hash = state_hash ^ self._zobrist(row, col, player)
                if self._is_win(board, row, col, player):
                    score = WIN_SCORE - ply
                else:
                    reduction = 1 if late_move and move_level < 2 and not is_killer else 0
                    full_depth = depth - 1
                    if index == 0:
                        score = -self._hybrid_search(
                            board,
                            full_depth,
                            -beta,
                            -alpha,
                            3 - player,
                            ply + 1,
                            child_hash,
                        )
                    else:
                        score = -self._hybrid_search(
                            board,
                            max(0, full_depth - reduction),
                            -alpha - 1,
                            -alpha,
                            3 - player,
                            ply + 1,
                            child_hash,
                        )
                        if reduction and score > alpha:
                            score = -self._hybrid_search(
                                board,
                                full_depth,
                                -alpha - 1,
                                -alpha,
                                3 - player,
                                ply + 1,
                                child_hash,
                            )
                        if alpha < score < beta:
                            score = -self._hybrid_search(
                                board,
                                full_depth,
                                -beta,
                                -alpha,
                                3 - player,
                                ply + 1,
                                child_hash,
                            )
            finally:
                board[row][col] = EMPTY
            if score > best:
                best, best_move = score, (row, col)
            alpha = max(alpha, score)
            history_key = (player, (row, col))
            if score > original_alpha:
                self._history[history_key] = self._history.get(history_key, 0) + max(
                    1, depth * depth
                )
            if alpha >= beta:
                self._record_cutoff((row, col), depth)
                break
        if best <= alpha_original:
            bound = TT_UPPER
        elif best >= beta_original:
            bound = TT_LOWER
        else:
            bound = TT_EXACT
        self._store_tt(key, depth, best, bound, best_move, ply)
        return best

    def _quiescence(self, board, alpha, beta, player, depth, ply, state_hash):
        self._check_time()
        self.last_search_stats["quiescence_nodes"] += 1
        summary = self._analyze_position(board, state_hash)
        opponent = 3 - player
        if summary.winning_moves[player]:
            return WIN_SCORE - ply
        opponent_wins = summary.winning_moves[opponent]
        if len(opponent_wins) >= 2:
            return -WIN_SCORE + ply

        forced_defense = len(opponent_wins) == 1
        stand = self._evaluate_generalized(
            board, player, state_hash, summary
        )
        if depth < 0:
            return stand
        if forced_defense:
            forcing = [next(iter(opponent_wins))]
        else:
            if stand >= beta:
                return stand
            alpha = max(alpha, stand)
            if depth <= 0:
                return stand
            forcing = self._forcing_moves(
                board,
                player,
                cap=8,
                state_hash=state_hash,
                summary=summary,
            )
            if not forcing:
                return stand
        for row, col in forcing:
            self._check_time()
            board[row][col] = player
            try:
                child_hash = state_hash ^ self._zobrist(row, col, player)
                score = (
                    WIN_SCORE - ply
                    if self._is_win(board, row, col, player)
                    else -self._quiescence(
                        board,
                        -beta,
                        -alpha,
                        opponent,
                        depth - 1,
                        ply + 1,
                        child_hash,
                    )
                )
            finally:
                board[row][col] = EMPTY
            alpha = max(alpha, score)
            if alpha >= beta:
                break
        return alpha

    def _ordered_hybrid_moves(
        self, board, player, preferred, depth, state_hash
    ):
        entries = self._tactical_entries(
            board, player, preferred=preferred, state_hash=state_hash
        )
        scored = []
        for key, move, _own, _defensive in entries:
            score = key[0] * 2_000_000_000 + key[2] * 10_000_000 + key[3]
            score += self._history.get((player, move), 0)
            if move == preferred or move in self._killers.get(depth, ()):
                score += 500_000_000
            scored.append((score, move, key[0]))
        scored.sort(reverse=True)
        cap = 24 if self.board_size <= 9 and depth <= 2 else 18
        return [move for _, move, _tier in scored[:cap]]

    def _move_level(self, board, row, col, player, state_hash=None):
        summary = self._analyze_position(board, state_hash)
        own = summary.profile((row, col), player)
        defensive = summary.profile((row, col), 3 - player)
        return self._threat_tier(own, defensive)

    def _record_cutoff(self, move, depth):
        killers = self._killers.setdefault(depth, [])
        if move not in killers:
            killers.insert(0, move)
            del killers[2:]


@dataclass
class _MCTSNode:
    board: list[list[int]]
    player_to_move: int
    state_hash: int
    move: tuple[int, int] | None = None
    parent: "_MCTSNode | None" = None
    depth: int = 0
    visits: int = 0
    value: float = 0.0
    children: list["_MCTSNode"] = field(default_factory=list)
    untried: list[tuple[int, int]] | None = None
    forcing_count: int = 0


class MCTSAI(_BaseAI):
    """UCT Monte Carlo tree search with deadline-bounded rollouts."""

    def __init__(self, player_id, board_size, win_length):
        super().__init__(player_id, board_size, win_length)
        self._random = random.Random()

    def get_move(self, board, last_opponent_move, time_limit):
        self._reset_search_stats()
        work, fallback, _ = self._prepare_turn(board, last_opponent_move, time_limit)
        if work is None or not self._has_time(0.001):
            return fallback
        try:
            root_hash = self._board_hash(work)
            root_summary = self._analyze_position(work, root_hash)
            tactical = self._advanced_tactical_move(
                work, self.player_id, root_hash, root_summary
            )
            if tactical is not None:
                return tactical
        except SearchTimeout:
            return fallback

        root = _MCTSNode(work, self.player_id, root_hash)
        try:
            root.untried, root.forcing_count = self._mcts_candidates(
                work, self.player_id, root_hash, root_summary
            )
            if not root.untried:
                return fallback
            while self._has_time(0.001):
                node = root
                while True:
                    self._check_time()
                    if node.untried and len(node.children) < self._expansion_limit(node):
                        break
                    if not node.children:
                        break
                    node = self._select_child(node)

                if node.untried and len(node.children) < self._expansion_limit(node):
                    move = node.untried.pop(0)
                    child_board = [row[:] for row in node.board]
                    child_board[move[0]][move[1]] = node.player_to_move
                    child_hash = node.state_hash ^ self._zobrist(
                        move[0], move[1], node.player_to_move
                    )
                    child = _MCTSNode(
                        child_board,
                        3 - node.player_to_move,
                        child_hash,
                        move=move,
                        parent=node,
                        depth=node.depth + 1,
                    )
                    if self._is_win(
                        child_board, move[0], move[1], node.player_to_move
                    ):
                        child.untried = []
                    else:
                        child.untried, child.forcing_count = self._mcts_candidates(
                            child_board,
                            child.player_to_move,
                            child_hash,
                        )
                    node.children.append(child)
                    node = child

                result = self._rollout(
                    node.board, node.player_to_move, node.move, node.state_hash
                )
                self.last_search_stats["mcts_iterations"] += 1
                self.last_search_stats["mcts_max_depth"] = max(
                    self.last_search_stats["mcts_max_depth"], node.depth
                )
                while node is not None:
                    node.visits += 1
                    node.value += result
                    node = node.parent
        except SearchTimeout:
            pass

        if not root.children:
            return fallback
        root_tiers = {
            child.move: self._threat_tier(
                root_summary.profile(child.move, self.player_id),
                root_summary.profile(child.move, self.opponent_id),
            )
            for child in root.children
        }
        highest_tier = max(root_tiers.values())
        eligible = (
            [child for child in root.children if root_tiers[child.move] == highest_tier]
            if highest_tier >= 2
            else root.children
        )
        return max(eligible, key=lambda child: child.visits).move

    @staticmethod
    def _expansion_limit(node):
        total = len(node.children) + len(node.untried or ())
        quiet_limit = 1 + int(1.5 * math.sqrt(node.visits + 1))
        return min(total, max(node.forcing_count, quiet_limit))

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

    def _mcts_candidates(
        self, board, player, state_hash, summary=None
    ):
        moves = self._candidate_moves(board, radius=2, cap=None)
        cap = 16 if self.board_size <= 9 else 12
        if summary is None:
            summary = self._analyze_position(board, state_hash)
        entries = self._tactical_entries(
            board,
            player,
            candidates=moves,
            state_hash=state_hash,
            summary=summary,
        )
        selected = entries[:cap]
        forcing_count = sum(item[0][0] >= 2 for item in selected)
        return [item[1] for item in selected], forcing_count

    def _rollout(self, source_board, player, last_move, state_hash):
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
            summary = self._analyze_position(board, state_hash)
            move = self._rollout_move(
                board, moves, player, state_hash, summary
            )
            board[move[0]][move[1]] = player
            state_hash ^= self._zobrist(move[0], move[1], player)
            if self._is_win(board, move[0], move[1], player):
                return 1.0 if player == self.player_id else -1.0
            player = 3 - player

        score = self._evaluate_generalized(board, self.player_id, state_hash)
        if score > 0:
            return 0.2
        if score < 0:
            return -0.2
        return 0.0

    def _rollout_move(self, board, moves, player, state_hash, summary):
        entries = self._tactical_entries(
            board,
            player,
            candidates=moves,
            state_hash=state_hash,
            summary=summary,
        )
        if entries and entries[0][0][0] >= 2:
            return entries[0][1]
        sample = entries[: min(6, len(entries))]
        weights = [
            max(1, own.progress_score + defensive.progress_score)
            for _key, _move, own, defensive in sample
        ]
        return self._random.choices(
            [item[1] for item in sample], weights=weights, k=1
        )[0]


class GomokuAI(HybridThreatSearchAI):
    """Course entry point selected by tactical and paired-match benchmarks."""
