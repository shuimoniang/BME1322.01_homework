"""Multi-ply and false-proof tactical benchmark for the advanced AIs."""

from __future__ import annotations

import argparse
import json
import statistics
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from gomoku_ai import EnhancedAlphaBetaAI, HybridThreatSearchAI, MCTSAI
from tactical_benchmark import _immediate_moves, _winning_replies


CONFIGURATIONS = ((7, 4), (9, 5), (11, 6))
CATEGORIES = (
    "three_ply_forced_win",
    "five_ply_forcing_chain",
    "unique_chain_defense",
    "counter_win_over_defense",
    "shared_reply_false_fork",
    "boundary_false_sequence",
)
MCTS_SEEDS = (7, 19, 41, 73, 101)


@dataclass(frozen=True)
class AdvancedTacticalCase:
    case_id: str
    category: str
    board_size: int
    win_length: int
    player: int
    board: tuple[tuple[int, ...], ...]
    allowed_moves: tuple[tuple[int, int], ...]
    description: str

    def mutable_board(self):
        return [list(row) for row in self.board]


def _forcing_moves(board, player, win_length):
    return [
        (row, col)
        for row in range(len(board))
        for col in range(len(board))
        if board[row][col] == 0
        and _winning_replies(board, (row, col), player, win_length)
    ]


def _threat_proven(board, attacker, player, win_length, plies_left):
    """Small independent VCF oracle used only to label generated cases."""
    if plies_left <= 0:
        return False
    defender = 3 - attacker
    if player == attacker:
        if _immediate_moves(board, attacker, win_length):
            return True
        defender_wins = _immediate_moves(board, defender, win_length)
        if len(defender_wins) >= 2:
            return False
        moves = _forcing_moves(board, attacker, win_length)
        if len(defender_wins) == 1:
            moves = [move for move in moves if move in defender_wins]
        for row, col in moves:
            board[row][col] = attacker
            proven = _threat_proven(
                board, attacker, defender, win_length, plies_left - 1
            )
            board[row][col] = 0
            if proven:
                return True
        return False

    if _immediate_moves(board, defender, win_length):
        return False
    attacker_wins = _immediate_moves(board, attacker, win_length)
    if len(attacker_wins) >= 2:
        return True
    if len(attacker_wins) != 1:
        return False
    row, col = next(iter(attacker_wins))
    board[row][col] = defender
    proven = _threat_proven(
        board, attacker, attacker, win_length, plies_left - 1
    )
    board[row][col] = 0
    return proven


def _proven_root_moves(board, attacker, win_length, plies_left):
    result = []
    for row, col in _forcing_moves(board, attacker, win_length):
        board[row][col] = attacker
        proven = _threat_proven(
            board, attacker, 3 - attacker, win_length, plies_left - 1
        )
        board[row][col] = 0
        if proven:
            result.append((row, col))
    return set(result)


def _three_ply_board(n, k, attacker):
    defender = 3 - attacker
    board = [[0] * n for _ in range(n)]
    middle = n // 2
    for offset in range(1, k - 1):
        board[middle][middle - offset] = attacker
        board[middle - offset][middle] = attacker
    board[middle][middle - (k - 1)] = defender
    board[middle - (k - 1)][middle] = defender
    return board, (middle, middle)


def _five_ply_board(n, k, attacker):
    defender = 3 - attacker
    board = [[0] * n for _ in range(n)]
    middle = n // 2
    attack = (middle, middle - 1)
    center = (middle, middle)

    for offset in range(1, k - 2):
        board[middle][middle - 1 - offset] = attacker
    for offset in range(1, k - 1):
        board[middle - offset][middle] = attacker
        board[middle - offset][middle - 1 + offset] = attacker
    blockers = (
        (middle, middle - (k - 1)),
        (middle - (k - 1), middle),
        (middle - (k - 1), middle + k - 2),
    )
    for row, col in blockers:
        board[row][col] = defender

    # K=4 is compact enough to create unrelated three-ply forks. These two
    # blockers remove them while preserving the intended five-ply chain.
    if k == 4:
        board[1][5] = defender
        board[2][2] = defender
    return board, attack, center


def _make_case(category, n, k, player):
    opponent = 3 - player
    if category == "three_ply_forced_win":
        board, expected = _three_ply_board(n, k, player)
        allowed = _proven_root_moves(board, player, k, 3)
        allowed.add(expected)

    elif category == "five_ply_forcing_chain":
        board, _attack, _center = _five_ply_board(n, k, player)
        allowed = _proven_root_moves(board, player, k, 5)
        allowed -= _proven_root_moves(board, player, k, 3)

    elif category == "unique_chain_defense":
        board, attack, _center = _five_ply_board(n, k, opponent)
        board[attack[0]][attack[1]] = opponent
        allowed = _immediate_moves(board, opponent, k)
        if len(allowed) != 1:
            raise RuntimeError("generated chain must have exactly one current reply")

    elif category == "counter_win_over_defense":
        board, attack, _center = _five_ply_board(n, k, opponent)
        board[attack[0]][attack[1]] = opponent
        row = n - 1
        for col in range(k - 1):
            board[row][col] = player
        allowed = _immediate_moves(board, player, k)

    elif category == "shared_reply_false_fork":
        board = [[0] * n for _ in range(n)]
        row = n // 2
        candidate, shared_reply = (row, k - 2), (row, k - 1)
        for col in range(k + 1):
            if col not in (candidate[1], shared_reply[1]):
                board[row][col] = opponent
        threat_col = n - 1
        for offset in range(k - 1):
            board[offset][threat_col] = opponent
        allowed = _immediate_moves(board, opponent, k)
        replies = _winning_replies(board, candidate, opponent, k)
        if replies != frozenset((shared_reply,)):
            raise RuntimeError("shared-reply pattern was not deduplicated")

    else:
        board = [[0] * n for _ in range(n)]
        for col in range(k - 2):
            board[0][col] = opponent
        threat_col = n - 1
        for row in range(k - 1):
            board[row][threat_col] = opponent
        allowed = _immediate_moves(board, opponent, k)

    if not allowed:
        raise RuntimeError(f"no allowed move generated for {category}, N={n}, K={k}")
    return AdvancedTacticalCase(
        f"{category}-n{n}-k{k}-p{player}",
        category,
        n,
        k,
        player,
        tuple(tuple(row) for row in board),
        tuple(sorted(allowed)),
        category.replace("_", " "),
    )


def build_advanced_cases():
    return [
        _make_case(category, n, k, player)
        for category in CATEGORIES
        for n, k in CONFIGURATIONS
        for player in (1, 2)
    ]


def _watchdog(ai, board, limit):
    holder = {}

    def target():
        try:
            holder["move"] = ai.get_move(board, None, limit)
        except BaseException as exc:
            holder["error"] = f"{type(exc).__name__}: {exc}"

    start = time.perf_counter()
    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    thread.join(limit + max(0.02, limit * 0.15))
    elapsed = time.perf_counter() - start
    if thread.is_alive():
        return None, elapsed, "timeout"
    return holder.get("move"), elapsed, holder.get("error")


def _p95(values):
    ordered = sorted(values)
    return ordered[max(0, (95 * len(ordered) + 99) // 100 - 1)]


def run_benchmark(time_limit=0.3):
    cases = build_advanced_cases()
    strategies = (
        ("Enhanced Alpha-Beta", EnhancedAlphaBetaAI, (None,)),
        ("Hybrid Threat Search", HybridThreatSearchAI, (None,)),
        ("MCTS + UCT", MCTSAI, MCTS_SEEDS),
    )
    results = {}
    for name, cls, seeds in strategies:
        attempts = correct = 0
        elapsed_values = []
        failures = []
        per_category = {category: [0, 0] for category in CATEGORIES}
        for case in cases:
            for seed in seeds:
                board = case.mutable_board()
                original = [row[:] for row in board]
                ai = cls(case.player, case.board_size, case.win_length)
                if seed is not None:
                    ai._random.seed(seed)
                move, elapsed, error = _watchdog(ai, board, time_limit)
                attempts += 1
                elapsed_values.append(elapsed)
                per_category[case.category][1] += 1
                passed = error is None and board == original and move in case.allowed_moves
                if passed:
                    correct += 1
                    per_category[case.category][0] += 1
                else:
                    failures.append(
                        {
                            "case_id": case.case_id,
                            "seed": seed,
                            "move": move,
                            "allowed_moves": case.allowed_moves,
                            "elapsed_seconds": elapsed,
                            "error": error,
                            "input_modified": board != original,
                        }
                    )
        results[name] = {
            "attempts": attempts,
            "correct": correct,
            "accuracy": correct / attempts,
            "mean_seconds": statistics.fmean(elapsed_values),
            "p95_seconds": _p95(elapsed_values),
            "max_seconds": max(elapsed_values),
            "per_category": {
                category: {
                    "correct": counts[0],
                    "attempts": counts[1],
                    "accuracy": counts[0] / counts[1],
                }
                for category, counts in per_category.items()
            },
            "failures": failures,
        }
    return {
        "generated_on": "2026-10-01",
        "time_limit_seconds": time_limit,
        "case_count": len(cases),
        "configurations": CONFIGURATIONS,
        "categories": CATEGORIES,
        "mcts_seeds": MCTS_SEEDS,
        "strategies": results,
        "cases": [asdict(case) for case in cases],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--time-limit", type=float, default=0.3)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("experiments")
        / "advanced_tactical_benchmark.json",
    )
    args = parser.parse_args()
    result = run_benchmark(args.time_limit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    for name, metrics in result["strategies"].items():
        print(
            f"{name}: {metrics['correct']}/{metrics['attempts']} "
            f"({metrics['accuracy']:.1%}), P95 {metrics['p95_seconds']:.4f}s"
        )
    print(f"Wrote: {args.output}")


if __name__ == "__main__":
    main()
