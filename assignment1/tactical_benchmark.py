"""Generated cross-N/K tactical benchmark for the three advanced AIs.

The positions are deliberately defined with K-length windows rather than
fixed Gomoku names. The result is the protocol/basic-tactics hard gate used
by the later multi-ply and paired-match selection stages.
"""

from __future__ import annotations

import argparse
import json
import statistics
import threading
import time
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path

from gomoku_ai import EnhancedAlphaBetaAI, HybridThreatSearchAI, MCTSAI


EMPTY, BLACK, WHITE = 0, 1, 2
DIRECTIONS = ((0, 1), (1, 0), (1, 1), (1, -1))
CONFIGURATIONS = ((5, 3), (7, 4), (9, 5), (11, 6), (15, 7))
CATEGORIES = (
    "immediate_win",
    "immediate_block",
    "open_k_minus_2_prevention",
    "own_double_threat",
    "broken_double_threat",
    "cross_direction_threat",
    "attack_defense",
    "attack_over_quiet_defense",
    "boundary_pseudo_threat",
    "unique_critical_move",
)
DEFENSE_CATEGORIES = {
    "immediate_block",
    "open_k_minus_2_prevention",
    "attack_defense",
    "boundary_pseudo_threat",
    "unique_critical_move",
}
MCTS_SEEDS = (7, 19, 41, 73, 101)


@dataclass(frozen=True)
class TacticalCase:
    case_id: str
    category: str
    board_size: int
    win_length: int
    player: int
    board: tuple[tuple[int, ...], ...]
    allowed_moves: tuple[tuple[int, int], ...]
    is_defense: bool
    description: str

    def mutable_board(self) -> list[list[int]]:
        return [list(row) for row in self.board]


def _in_bounds(n: int, row: int, col: int) -> bool:
    return 0 <= row < n and 0 <= col < n


def _is_win(board: list[list[int]], move: tuple[int, int], player: int, k: int) -> bool:
    row, col = move
    n = len(board)
    for dr, dc in DIRECTIONS:
        count = 1
        for sign in (-1, 1):
            rr, cc = row + sign * dr, col + sign * dc
            while _in_bounds(n, rr, cc) and board[rr][cc] == player:
                count += 1
                rr += sign * dr
                cc += sign * dc
        if count >= k:
            return True
    return False


def _windows_through(n: int, k: int, row: int, col: int):
    for dr, dc in DIRECTIONS:
        for offset in range(-(k - 1), 1):
            start_r, start_c = row + offset * dr, col + offset * dc
            end_r = start_r + (k - 1) * dr
            end_c = start_c + (k - 1) * dc
            if _in_bounds(n, start_r, start_c) and _in_bounds(n, end_r, end_c):
                yield tuple(
                    (start_r + step * dr, start_c + step * dc)
                    for step in range(k)
                )


def _winning_replies(
    board: list[list[int]], move: tuple[int, int], player: int, k: int
) -> frozenset[tuple[int, int]]:
    row, col = move
    if board[row][col] != EMPTY:
        return frozenset()
    board[row][col] = player
    replies: set[tuple[int, int]] = set()
    opponent = 3 - player
    try:
        for window in _windows_through(len(board), k, row, col):
            values = [board[rr][cc] for rr, cc in window]
            if opponent in values or values.count(player) != k - 1:
                continue
            empties = [window[i] for i, value in enumerate(values) if value == EMPTY]
            if len(empties) == 1:
                replies.add(empties[0])
    finally:
        board[row][col] = EMPTY
    return frozenset(replies)


def _immediate_moves(board: list[list[int]], player: int, k: int):
    result = set()
    for row in range(len(board)):
        for col in range(len(board)):
            if board[row][col] != EMPTY:
                continue
            board[row][col] = player
            won = _is_win(board, (row, col), player, k)
            board[row][col] = EMPTY
            if won:
                result.add((row, col))
    return result


def _fork_moves(board: list[list[int]], player: int, k: int):
    return {
        (row, col)
        for row in range(len(board))
        for col in range(len(board))
        if board[row][col] == EMPTY
        and len(_winning_replies(board, (row, col), player, k)) >= 2
    }


def _line(n: int, k: int, direction: tuple[int, int]):
    """Return a centered line of K+2 coordinates in one direction."""
    length = k + 2
    dr, dc = direction
    center = n // 2
    if (dr, dc) == (0, 1):
        start = (center, (n - length) // 2)
    elif (dr, dc) == (1, 0):
        start = ((n - length) // 2, center)
    elif (dr, dc) == (1, 1):
        start = ((n - length) // 2, (n - length) // 2)
    else:
        start = ((n - length) // 2, n - 1 - (n - length) // 2)
    return tuple((start[0] + i * dr, start[1] + i * dc) for i in range(length))


def _place(board, coords, player):
    for row, col in coords:
        if board[row][col] not in (EMPTY, player):
            raise ValueError("generated patterns overlap with opposite stones")
        board[row][col] = player


def _isolate_k3_open_prevention(board, center, direction, player):
    """Occupy off-axis neighbors without creating an immediate win."""
    row, col = center
    dr, dc = direction
    axis = {(row - dr, col - dc), (row + dr, col + dc)}
    neighbors = [
        (row + rr, col + cc)
        for rr in (-1, 0, 1)
        for cc in (-1, 0, 1)
        if (rr or cc)
        and _in_bounds(len(board), row + rr, col + cc)
        and (row + rr, col + cc) not in axis
    ]
    opponent = 3 - player
    for mask in range(1 << len(neighbors)):
        trial = [values[:] for values in board]
        for index, (rr, cc) in enumerate(neighbors):
            trial[rr][cc] = player if mask & (1 << index) else opponent
        if _immediate_moves(trial, player, 3) or _immediate_moves(trial, opponent, 3):
            continue
        forks = _fork_moves(trial, opponent, 3)
        if forks and forks.issubset(axis):
            for rr, cc in neighbors:
                board[rr][cc] = trial[rr][cc]
            return
    raise RuntimeError("could not isolate K=3 prevention pattern")


def _best_fork_preventions(board, player, k):
    opponent = 3 - player
    scored = []
    for row in range(len(board)):
        for col in range(len(board)):
            if board[row][col] != EMPTY:
                continue
            board[row][col] = player
            remaining = len(_fork_moves(board, opponent, k))
            board[row][col] = EMPTY
            scored.append((remaining, (row, col)))
    minimum = min(value for value, _ in scored)
    return {move for value, move in scored if value == minimum}


def _make_case(category_index: int, config_index: int, n: int, k: int, player: int):
    category = CATEGORIES[category_index]
    opponent = 3 - player
    direction = DIRECTIONS[(category_index + config_index) % len(DIRECTIONS)]
    line = _line(n, k, direction)
    board = [[EMPTY] * n for _ in range(n)]
    description = category.replace("_", " ")

    if category == "immediate_win":
        _place(board, line[1:k], player)
        allowed = _immediate_moves(board, player, k)

    elif category == "immediate_block":
        _place(board, line[1:k], opponent)
        _place(board, (line[0],), player)
        allowed = _immediate_moves(board, opponent, k)

    elif category == "open_k_minus_2_prevention":
        if k == 3:
            # A central single stone has fork-creation points in all four
            # directions, so no one move can prevent every fork.  On an edge
            # (but not a corner), only the along-edge pair has two distinct
            # completion squares; inward moves have the boundary on one end.
            board[0][n // 2] = opponent
        else:
            _place(board, line[2:k], opponent)
        allowed = _best_fork_preventions(board, player, k)

    elif category == "own_double_threat":
        _place(board, line[2:k], player)
        allowed = _fork_moves(board, player, k)

    elif category == "broken_double_threat":
        gap = 1 + (k - 1) // 2
        _place(board, (line[index] for index in range(1, k) if index != gap), player)
        allowed = _fork_moves(board, player, k)

    elif category == "cross_direction_threat":
        center = (n // 2, n // 2)
        rays = [
            (dr * sign, dc * sign)
            for dr, dc in DIRECTIONS
            for sign in (-1, 1)
        ]
        built = False
        for first, second in combinations(rays, 2):
            trial = [[EMPTY] * n for _ in range(n)]
            coords = [
                (center[0] + step * dr, center[1] + step * dc)
                for dr, dc in (first, second)
                for step in range(1, k - 1)
            ]
            if len(set(coords)) != len(coords) or not all(
                _in_bounds(n, row, col) for row, col in coords
            ):
                continue
            _place(trial, coords, player)
            if _immediate_moves(trial, player, k):
                continue
            if len(_winning_replies(trial, center, player, k)) >= 2:
                board = trial
                built = True
                break
        if not built:
            raise RuntimeError(f"could not build cross threat for N={n}, K={k}")
        allowed = _fork_moves(board, player, k)

    elif category == "attack_defense":
        target = (n // 2, n // 2)
        attack_direction = DIRECTIONS[
            (DIRECTIONS.index(direction) + 1) % len(DIRECTIONS)
        ]
        _place(
            board,
            (
                (target[0] + step * direction[0], target[1] + step * direction[1])
                for step in range(1, k - 1)
            ),
            opponent,
        )
        _place(
            board,
            (
                (
                    target[0] + step * attack_direction[0],
                    target[1] + step * attack_direction[1],
                )
                for step in range(1, k - 1)
            ),
            player,
        )
        allowed = {
            (row, col)
            for row in range(n)
            for col in range(n)
            if board[row][col] == EMPTY
            and len(_winning_replies(board, (row, col), player, k)) >= 1
            and len(_winning_replies(board, (row, col), opponent, k)) >= 2
        }

    elif category == "attack_over_quiet_defense":
        _place(board, line[2:k], player)
        quiet_direction = DIRECTIONS[(DIRECTIONS.index(direction) + 2) % len(DIRECTIONS)]
        quiet_line = _line(n, k, quiet_direction)
        quiet = [coord for coord in quiet_line[2 : max(3, k - 1)] if board[coord[0]][coord[1]] == EMPTY]
        _place(board, quiet[: max(1, k - 3)], opponent)
        allowed = _fork_moves(board, player, k)

    elif category == "boundary_pseudo_threat":
        boundary = [(0, index) for index in range(k + 1)]
        _place(board, boundary[: k - 2], opponent)
        real_direction = DIRECTIONS[(DIRECTIONS.index(direction) + 1) % len(DIRECTIONS)]
        real_line = _line(n, k, real_direction)
        _place(board, real_line[2:k], player)
        allowed = _fork_moves(board, player, k)

    else:
        for row in range(n):
            for col in range(n):
                board[row][col] = BLACK if (row + 2 * col) % 4 < 2 else WHITE
        target = line[k // 2]
        board[target[0]][target[1]] = EMPTY
        allowed = {target}

    if not allowed:
        raise RuntimeError(f"generated case has no allowed move: {category} N={n} K={k}")
    case_id = f"{category}-n{n}-k{k}-p{player}"
    return TacticalCase(
        case_id=case_id,
        category=category,
        board_size=n,
        win_length=k,
        player=player,
        board=tuple(tuple(row) for row in board),
        allowed_moves=tuple(sorted(allowed)),
        is_defense=category in DEFENSE_CATEGORIES,
        description=description,
    )


def build_tactical_cases() -> list[TacticalCase]:
    cases = []
    for category_index, _category in enumerate(CATEGORIES):
        for config_index, (n, k) in enumerate(CONFIGURATIONS):
            for player in (BLACK, WHITE):
                cases.append(_make_case(category_index, config_index, n, k, player))
    return cases


def _percentile95(values):
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[max(0, (95 * len(ordered) + 99) // 100 - 1)]


def _watchdog_move(ai, board, time_limit):
    holder = {}

    def target():
        try:
            holder["move"] = ai.get_move(board, None, time_limit)
        except BaseException as exc:  # benchmark records protocol failures
            holder["error"] = f"{type(exc).__name__}: {exc}"

    start = time.perf_counter()
    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    thread.join(time_limit + max(0.02, time_limit * 0.15))
    elapsed = time.perf_counter() - start
    if thread.is_alive():
        return None, elapsed, "timeout"
    if "error" in holder:
        return None, elapsed, holder["error"]
    return holder.get("move"), elapsed, None


def run_benchmark(time_limit: float = 0.15):
    cases = build_tactical_cases()
    strategies = (
        ("Enhanced Alpha-Beta", EnhancedAlphaBetaAI, (None,)),
        ("Hybrid Threat Search", HybridThreatSearchAI, (None,)),
        ("MCTS + UCT", MCTSAI, MCTS_SEEDS),
    )
    results = {}
    for strategy_name, strategy_class, seeds in strategies:
        elapsed_values = []
        failures = []
        category_counts = {category: [0, 0] for category in CATEGORIES}
        defense_attempts = defense_correct = 0
        correct = attempts = 0
        for case in cases:
            allowed = set(case.allowed_moves)
            for seed in seeds:
                board = case.mutable_board()
                original = [row[:] for row in board]
                ai = strategy_class(case.player, case.board_size, case.win_length)
                if seed is not None:
                    ai._random.seed(seed)
                move, elapsed, error = _watchdog_move(ai, board, time_limit)
                elapsed_values.append(elapsed)
                attempts += 1
                category_counts[case.category][1] += 1
                if case.is_defense:
                    defense_attempts += 1
                legal = (
                    isinstance(move, tuple)
                    and len(move) == 2
                    and all(type(value) is int for value in move)
                    and 0 <= move[0] < case.board_size
                    and 0 <= move[1] < case.board_size
                    and original[move[0]][move[1]] == EMPTY
                )
                passed = error is None and board == original and legal and move in allowed
                if passed:
                    correct += 1
                    category_counts[case.category][0] += 1
                    if case.is_defense:
                        defense_correct += 1
                else:
                    failures.append(
                        {
                            "case_id": case.case_id,
                            "seed": seed,
                            "move": move,
                            "allowed_moves": list(case.allowed_moves),
                            "elapsed_seconds": elapsed,
                            "error": error,
                            "input_modified": board != original,
                            "legal": legal,
                        }
                    )
        results[strategy_name] = {
            "attempts": attempts,
            "correct": correct,
            "accuracy": correct / attempts,
            "defense_attempts": defense_attempts,
            "defense_correct": defense_correct,
            "defense_accuracy": defense_correct / defense_attempts,
            "mean_seconds": statistics.fmean(elapsed_values),
            "p95_seconds": _percentile95(elapsed_values),
            "max_seconds": max(elapsed_values),
            "per_category": {
                category: {
                    "correct": values[0],
                    "attempts": values[1],
                    "accuracy": values[0] / values[1],
                }
                for category, values in category_counts.items()
            },
            "failures": failures,
        }

    ranking = sorted(
        results,
        key=lambda name: (
            -results[name]["accuracy"],
            -results[name]["defense_accuracy"],
            results[name]["p95_seconds"],
            0 if name == "Enhanced Alpha-Beta" else 1,
        ),
    )
    return {
        "generated_on": "2026-10-01",
        "time_limit_seconds": time_limit,
        "configurations": [list(config) for config in CONFIGURATIONS],
        "categories": list(CATEGORIES),
        "case_count": len(cases),
        "mcts_seeds": list(MCTS_SEEDS),
        "basic_tactical_ranking_rule": [
            "overall accuracy descending",
            "defense accuracy descending",
            "P95 elapsed time ascending",
            "Enhanced Alpha-Beta on an exact tie",
        ],
        "basic_tactical_ranking": ranking,
        "hard_gate_rule": "100% overall and defense accuracy with no failures",
        "hard_gate_passed": [
            name
            for name in ranking
            if results[name]["accuracy"] == 1.0
            and results[name]["defense_accuracy"] == 1.0
            and not results[name]["failures"]
        ],
        "final_selection_source": "advanced_match_benchmark.json",
        "strategies": results,
        "cases": [asdict(case) for case in cases],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--time-limit", type=float, default=0.15)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("experiments") / "tactical_benchmark.json",
    )
    args = parser.parse_args()
    result = run_benchmark(args.time_limit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    for name in result["basic_tactical_ranking"]:
        metrics = result["strategies"][name]
        print(
            f"{name}: {metrics['correct']}/{metrics['attempts']} "
            f"({metrics['accuracy']:.1%}), defense {metrics['defense_accuracy']:.1%}, "
            f"P95 {metrics['p95_seconds']:.4f}s"
        )
    print(f"Hard gate passed: {', '.join(result['hard_gate_passed'])}")
    print(f"Wrote: {args.output}")


if __name__ == "__main__":
    main()
