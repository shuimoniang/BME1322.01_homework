"""Run paired short-budget matches from deterministic legal openings."""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from gomoku_ai import MCTSAI
from run_experiments import (
    BLACK,
    EMPTY,
    STRATEGIES,
    WHITE,
    call_get_move,
    check_win,
    percentile,
    stable_seed,
    validate_move,
)


PAIRINGS = (
    ("Enhanced Alpha-Beta", "Hybrid Threat Search"),
    ("Enhanced Alpha-Beta", "MCTS + UCT"),
    ("Hybrid Threat Search", "MCTS + UCT"),
)
CONFIGURATIONS = ((9, 4), (15, 5))
TIME_LIMITS = (0.5, 1.0)
OPENINGS_PER_GROUP = 4
OPENING_PLIES = 8


@dataclass(frozen=True)
class MatchTask:
    key: str
    first_strategy: str
    second_strategy: str
    board_size: int
    win_length: int
    time_limit: float
    opening_index: int
    swap: int
    opening: tuple[tuple[int, int, int], ...]
    black_agent: str
    white_agent: str
    seed: int


def _opening(size: int, win: int, index: int):
    rng = random.Random(stable_seed("advanced-match-opening", size, win, index))
    board = [[EMPTY] * size for _ in range(size)]
    middle = size // 2
    radius = min(3, middle)
    candidates = [
        (row, col)
        for row in range(middle - radius, middle + radius + 1)
        for col in range(middle - radius, middle + radius + 1)
    ]
    rng.shuffle(candidates)
    result = []
    for ply in range(OPENING_PLIES):
        player = BLACK if ply % 2 == 0 else WHITE
        for candidate_index, (row, col) in enumerate(candidates):
            if board[row][col] != EMPTY:
                continue
            board[row][col] = player
            if check_win(board, row, col, player, win):
                board[row][col] = EMPTY
                continue
            result.append((row, col, player))
            candidates.pop(candidate_index)
            break
        else:
            raise RuntimeError(f"could not generate opening N={size}, K={win}")
    return tuple(result)


def build_tasks(time_limits=TIME_LIMITS):
    openings = {
        (size, win, index): _opening(size, win, index)
        for size, win in CONFIGURATIONS
        for index in range(OPENINGS_PER_GROUP)
    }
    tasks = []
    for first, second in PAIRINGS:
        for size, win in CONFIGURATIONS:
            for limit in time_limits:
                for opening_index in range(OPENINGS_PER_GROUP):
                    for swap in (0, 1):
                        black, white = ((first, second) if swap == 0 else (second, first))
                        key = (
                            f"{first}|{second}|N{size}|K{win}|T{limit:g}|"
                            f"O{opening_index}|S{swap}"
                        )
                        tasks.append(
                            MatchTask(
                                key,
                                first,
                                second,
                                size,
                                win,
                                limit,
                                opening_index,
                                swap,
                                openings[(size, win, opening_index)],
                                black,
                                white,
                                stable_seed("advanced-match", key),
                            )
                        )
    return tasks


def _make_ai(name, player, task):
    ai = STRATEGIES[name](player, task.board_size, task.win_length)
    if isinstance(ai, MCTSAI):
        ai._random.seed(stable_seed(task.seed, "mcts", name, player))
    return ai


def play_game(task: MatchTask):
    board = [[EMPTY] * task.board_size for _ in range(task.board_size)]
    last_move = {BLACK: None, WHITE: None}
    for row, col, player in task.opening:
        board[row][col] = player
        last_move[player] = (row, col)

    agents = {
        BLACK: _make_ai(task.black_agent, BLACK, task),
        WHITE: _make_ai(task.white_agent, WHITE, task),
    }
    names = {BLACK: task.black_agent, WHITE: task.white_agent}
    current = BLACK if len(task.opening) % 2 == 0 else WHITE
    winner = EMPTY
    reason = "board full"
    moves = []
    started = time.perf_counter()
    empty_count = task.board_size * task.board_size - len(task.opening)

    for played_ply in range(1, empty_count + 1):
        move, elapsed, error = call_get_move(
            agents[current], board, last_move[3 - current], task.time_limit
        )
        record = {
            "played_ply": played_ply,
            "player": current,
            "agent": names[current],
            "move": list(move) if isinstance(move, (tuple, list)) else None,
            "elapsed_seconds": elapsed,
            "error": error,
        }
        stats = getattr(agents[current], "last_search_stats", None)
        if isinstance(stats, dict):
            record["search_stats"] = dict(stats)
        moves.append(record)

        if error:
            winner = 3 - current
            reason = f"{error}: {names[current]}"
            break
        invalid = validate_move(move, board)
        if invalid:
            winner = 3 - current
            reason = f"illegal move: {names[current]}: {invalid}"
            record["error"] = invalid
            break

        row, col = move
        board[row][col] = current
        last_move[current] = (row, col)
        if check_win(board, row, col, current, task.win_length):
            winner = current
            reason = f"{task.win_length}-in-a-row"
            break
        current = 3 - current

    return {
        **asdict(task),
        "opening": [list(stone) for stone in task.opening],
        "winner_color": winner,
        "winner_agent": names[winner] if winner else None,
        "reason": reason,
        "opening_plies": len(task.opening),
        "played_moves": sum(move["error"] is None for move in moves),
        "duration_seconds": time.perf_counter() - started,
        "moves": moves,
    }


def _load_jsonl(path):
    records, keys = [], set()
    if not path.exists():
        return records, keys
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            key = record.get("key")
            if not isinstance(key, str) or key in keys:
                raise ValueError(f"invalid or duplicate key at {path}:{line_number}")
            records.append(record)
            keys.add(key)
    return records, keys


def _timing(values):
    return {
        "count": len(values),
        "mean_seconds": statistics.fmean(values) if values else None,
        "median_seconds": statistics.median(values) if values else None,
        "p95_seconds": percentile(values, 0.95),
        "max_seconds": max(values) if values else None,
    }


def _selection_record(standings):
    experiments = Path(__file__).with_name("experiments")
    tactical_path = experiments / "tactical_benchmark.json"
    advanced_path = experiments / "advanced_tactical_benchmark.json"
    search_path = experiments / "search_benchmark_after.json"
    if not all(path.exists() for path in (tactical_path, advanced_path, search_path)):
        return None
    tactical = json.loads(tactical_path.read_text(encoding="utf-8"))["strategies"]
    advanced = json.loads(advanced_path.read_text(encoding="utf-8"))["strategies"]
    search = json.loads(search_path.read_text(encoding="utf-8"))["strategies"]
    metrics = {}
    for name in standings:
        basic = tactical[name]
        multi = advanced[name]
        completed_depth = sum(
            record["median_completed_depth"] for record in search.get(name, ())
        )
        metrics[name] = {
            "protocol_and_basic_tactics_pass": (
                basic["accuracy"] == 1.0 and not basic["failures"]
            ),
            "multi_ply_accuracy": multi["accuracy"],
            "paired_match_points": standings[name]["points"],
            "completed_depth_sum": completed_depth,
            "multi_ply_p95_seconds": multi["p95_seconds"],
        }

    current_default = "Hybrid Threat Search"
    ranked = sorted(
        metrics,
        key=lambda name: (
            metrics[name]["protocol_and_basic_tactics_pass"],
            metrics[name]["multi_ply_accuracy"],
            metrics[name]["paired_match_points"],
            metrics[name]["completed_depth_sum"],
            -metrics[name]["multi_ply_p95_seconds"],
            name == current_default,
        ),
        reverse=True,
    )
    return {
        "criteria": [
            "protocol_and_basic_tactics_pass",
            "multi_ply_accuracy",
            "paired_match_points",
            "completed_depth_sum",
            "multi_ply_p95_seconds",
            "keep_current_default_if_fully_tied",
        ],
        "previous_default": current_default,
        "selected": ranked[0],
        "ranking": ranked,
        "metrics": metrics,
    }


def summarize(records, expected_keys):
    strategy_names = sorted({name for pair in PAIRINGS for name in pair})
    standings = {}
    for strategy in strategy_names:
        games = [
            game
            for game in records
            if strategy in (game["first_strategy"], game["second_strategy"])
        ]
        wins = sum(game["winner_agent"] == strategy for game in games)
        draws = sum(game["winner_color"] == EMPTY for game in games)
        losses = len(games) - wins - draws
        times = [
            move["elapsed_seconds"]
            for game in games
            for move in game["moves"]
            if move["agent"] == strategy
        ]
        standings[strategy] = {
            "games": len(games),
            "wins": wins,
            "losses": losses,
            "draws": draws,
            "points": wins + 0.5 * draws,
            "score_rate": (wins + 0.5 * draws) / len(games) if games else None,
            "timing": _timing(times),
        }

    groups = []
    for first, second in PAIRINGS:
        for size, win in CONFIGURATIONS:
            for limit in sorted({game["time_limit"] for game in records}):
                games = [
                    game
                    for game in records
                    if game["first_strategy"] == first
                    and game["second_strategy"] == second
                    and game["board_size"] == size
                    and game["win_length"] == win
                    and game["time_limit"] == limit
                ]
                if not games:
                    continue
                first_wins = sum(game["winner_agent"] == first for game in games)
                second_wins = sum(game["winner_agent"] == second for game in games)
                draws = len(games) - first_wins - second_wins
                groups.append(
                    {
                        "pairing": [first, second],
                        "board_size": size,
                        "win_length": win,
                        "time_limit": limit,
                        "games": len(games),
                        "first_wins": first_wins,
                        "second_wins": second_wins,
                        "draws": draws,
                        "black_wins": sum(
                            game["winner_color"] == BLACK for game in games
                        ),
                        "white_wins": sum(
                            game["winner_color"] == WHITE for game in games
                        ),
                        "mean_total_moves": statistics.fmean(
                            game["opening_plies"] + game["played_moves"]
                            for game in games
                        ),
                    }
                )

    failures = [
        {"key": game["key"], "reason": game["reason"]}
        for game in records
        if game["reason"].startswith(("timeout", "exception", "illegal move"))
    ]
    result = {
        "protocol": {
            "expected_games": len(expected_keys),
            "completed_games": len(records),
            "unique_keys": len({game["key"] for game in records}),
            "complete": {game["key"] for game in records} == expected_keys,
            "failures": failures,
        },
        "configurations": CONFIGURATIONS,
        "time_limits": sorted({game["time_limit"] for game in records}),
        "openings_per_group": OPENINGS_PER_GROUP,
        "opening_plies": OPENING_PLIES,
        "standings": standings,
        "groups": groups,
        "games": records,
    }
    result["default_selection"] = _selection_record(standings)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--time-limits", nargs="+", type=float, default=TIME_LIMITS)
    parser.add_argument(
        "--results",
        type=Path,
        default=Path(__file__).with_name("experiments")
        / "advanced_match_results.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("experiments")
        / "advanced_match_benchmark.json",
    )
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args()

    tasks = build_tasks(tuple(args.time_limits))
    expected_keys = {task.key for task in tasks}
    records, completed = _load_jsonl(args.results)
    unexpected = completed - expected_keys
    if unexpected:
        raise ValueError(f"results contain {len(unexpected)} unexpected task keys")

    if not args.summarize:
        args.results.parent.mkdir(parents=True, exist_ok=True)
        with args.results.open("a", encoding="utf-8") as stream:
            for index, task in enumerate(tasks, 1):
                if task.key in completed:
                    continue
                record = play_game(task)
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
                records.append(record)
                completed.add(task.key)
                print(
                    f"[{index:02d}/{len(tasks)}] {task.key}: "
                    f"{record['winner_agent'] or 'draw'} in "
                    f"{record['opening_plies'] + record['played_moves']} moves",
                    flush=True,
                )

    result = summarize(records, expected_keys)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    for name, metrics in result["standings"].items():
        print(
            f"{name}: {metrics['wins']}-{metrics['losses']}-{metrics['draws']}, "
            f"{metrics['points']:.1f} points"
        )
    print(f"Wrote: {args.output}")


if __name__ == "__main__":
    main()
