"""Reproducible experiment runner for the Assignment 1 report.

The runner mirrors the course arena's wall-clock watchdog and validation rules,
but records per-move timing and flushes one JSON object after every game.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import statistics
import sys
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from .gomoku_ai import (
        BaselineAlphaBetaAI,
        EnhancedAlphaBetaAI,
        HybridThreatSearchAI,
        MCTSAI,
    )
except ImportError:
    from gomoku_ai import BaselineAlphaBetaAI, EnhancedAlphaBetaAI, HybridThreatSearchAI, MCTSAI


EMPTY, BLACK, WHITE = 0, 1, 2
DIRECTIONS = ((0, 1), (1, 0), (1, 1), (1, -1))
MASTER_SEED = 20260928
CONFIGURATIONS = ((9, 4), (15, 5))
TIME_LIMITS = (0.5, 1.0, 5.0)
GAMES_PER_GROUP = 50
STRATEGIES = {
    "Baseline Alpha-Beta": BaselineAlphaBetaAI,
    "Enhanced Alpha-Beta": EnhancedAlphaBetaAI,
    "MCTS + UCT": MCTSAI,
    "Hybrid Threat Search": HybridThreatSearchAI,
}


@dataclass(frozen=True)
class ExperimentTask:
    key: str
    kind: str
    strategy: str
    opponent: str
    board_size: int
    win_length: int
    time_limit: float
    game_index: int
    seed: int
    black_agent: str
    white_agent: str


class SeededRandomAI:
    def __init__(self, player_id: int, board_size: int, win_length: int, seed: int):
        self.player_id = player_id
        self.board_size = board_size
        self.win_length = win_length
        self._random = random.Random(seed)

    def get_move(self, board, last_opponent_move, time_limit):
        legal = [
            (row, col)
            for row in range(self.board_size)
            for col in range(self.board_size)
            if board[row][col] == EMPTY
        ]
        return self._random.choice(legal)


def stable_seed(*parts: object) -> int:
    payload = "|".join(str(part) for part in (MASTER_SEED, *parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def task_key(
    kind: str,
    strategy: str,
    opponent: str,
    board_size: int,
    win_length: int,
    time_limit: float,
    game_index: int,
) -> str:
    return (
        f"{kind}|{strategy}|{opponent}|N{board_size}|K{win_length}|"
        f"T{time_limit:g}|G{game_index:03d}"
    )


def build_tasks(
    games_per_group: int = GAMES_PER_GROUP,
    only: str = "all",
    include_hybrid: bool = False,
) -> list[ExperimentTask]:
    if only == "smoke":
        games_per_group = 2
        include_random = include_selfplay = True
    else:
        include_random = only in ("all", "random")
        include_selfplay = only in ("all", "selfplay")

    tasks: list[ExperimentTask] = []
    for time_limit in TIME_LIMITS:
        for board_size, win_length in CONFIGURATIONS:
            if include_random:
                strategies = (
                    ["Hybrid Threat Search"]
                    if include_hybrid
                    else [name for name in STRATEGIES if name != "Hybrid Threat Search"]
                )
                for strategy in strategies:
                    for game_index in range(games_per_group):
                        ai_is_black = game_index % 2 == 0
                        black = strategy if ai_is_black else "Random AI"
                        white = "Random AI" if ai_is_black else strategy
                        opponent = "Random AI"
                        tasks.append(
                            ExperimentTask(
                                key=task_key(
                                    "vs_random",
                                    strategy,
                                    opponent,
                                    board_size,
                                    win_length,
                                    time_limit,
                                    game_index,
                                ),
                                kind="vs_random",
                                strategy=strategy,
                                opponent=opponent,
                                board_size=board_size,
                                win_length=win_length,
                                time_limit=time_limit,
                                game_index=game_index,
                                seed=stable_seed(
                                    "vs_random",
                                    board_size,
                                    win_length,
                                    time_limit,
                                    game_index,
                                ),
                                black_agent=black,
                                white_agent=white,
                            )
                        )
            if include_selfplay:
                if not include_hybrid:
                    strategy = "Enhanced Alpha-Beta"
                    opponent = "Enhanced Alpha-Beta"
                    for game_index in range(games_per_group):
                        a_is_black = game_index % 2 == 0
                        black = "Enhanced Alpha-Beta A" if a_is_black else "Enhanced Alpha-Beta B"
                        white = "Enhanced Alpha-Beta B" if a_is_black else "Enhanced Alpha-Beta A"
                        tasks.append(
                            ExperimentTask(
                                key=task_key(
                                    "selfplay", strategy, opponent,
                                    board_size, win_length, time_limit, game_index
                                ),
                                kind="selfplay",
                                strategy=strategy,
                                opponent=opponent,
                                board_size=board_size,
                                win_length=win_length,
                                time_limit=time_limit,
                                game_index=game_index,
                                seed=stable_seed(
                                    "selfplay", board_size, win_length,
                                    time_limit, game_index
                                ),
                                black_agent=black,
                                white_agent=white,
                            )
                        )
                if include_hybrid:
                    strategy = "Hybrid Threat Search"
                    opponent = "Enhanced Alpha-Beta"
                    for game_index in range(games_per_group):
                        hybrid_is_black = game_index % 2 == 0
                        black = "Hybrid Threat Search A" if hybrid_is_black else "Enhanced Alpha-Beta B"
                        white = "Enhanced Alpha-Beta B" if hybrid_is_black else "Hybrid Threat Search A"
                        tasks.append(
                            ExperimentTask(
                                key=task_key(
                                    "hybrid_match", strategy, opponent,
                                    board_size, win_length, time_limit, game_index
                                ),
                                kind="hybrid_match",
                                strategy=strategy,
                                opponent=opponent,
                                board_size=board_size,
                                win_length=win_length,
                                time_limit=time_limit,
                                game_index=game_index,
                                seed=stable_seed(
                                    "hybrid_match", board_size, win_length,
                                    time_limit, game_index
                                ),
                                black_agent=black,
                                white_agent=white,
                            )
                        )
    return tasks


def check_win(board: list[list[int]], row: int, col: int, player: int, win_length: int) -> bool:
    size = len(board)
    for dr, dc in DIRECTIONS:
        count = 1
        for sign in (-1, 1):
            rr, cc = row + sign * dr, col + sign * dc
            while 0 <= rr < size and 0 <= cc < size and board[rr][cc] == player:
                count += 1
                rr += sign * dr
                cc += sign * dc
        if count >= win_length:
            return True
    return False


def call_get_move(ai, board, last_opponent_move, time_limit):
    holder: dict[str, Any] = {}

    def target() -> None:
        try:
            holder["move"] = ai.get_move(
                [row[:] for row in board], last_opponent_move, time_limit
            )
        except BaseException as exc:  # Match the course arena's containment policy.
            holder["error"] = repr(exc)

    start = time.perf_counter()
    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    thread.join(time_limit)
    elapsed = time.perf_counter() - start
    if thread.is_alive():
        return None, elapsed, "timeout"
    if "error" in holder:
        return None, elapsed, f"exception: {holder['error']}"
    return holder.get("move"), elapsed, None


def validate_move(move, board) -> str | None:
    if not isinstance(move, (tuple, list)) or len(move) != 2:
        return "not a (row, col) pair"
    row, col = move
    if (
        isinstance(row, bool)
        or isinstance(col, bool)
        or not isinstance(row, int)
        or not isinstance(col, int)
    ):
        return "coordinates must be int"
    size = len(board)
    if not (0 <= row < size and 0 <= col < size):
        return f"out of range: ({row}, {col})"
    if board[row][col] != EMPTY:
        return f"cell occupied: ({row}, {col})"
    return None


def make_ai(name: str, player: int, task: ExperimentTask):
    if name == "Random AI":
        return SeededRandomAI(
            player,
            task.board_size,
            task.win_length,
            stable_seed(task.seed, "random", player),
        )
    base_name = name.removesuffix(" A").removesuffix(" B")
    cls = STRATEGIES[base_name]
    ai = cls(player, task.board_size, task.win_length)
    if isinstance(ai, MCTSAI):
        ai._random.seed(stable_seed(task.seed, "mcts", name, player))
    return ai


def percentile(values: list[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    rank = max(0, math.ceil(probability * len(ordered)) - 1)
    return ordered[rank]


def timing_stats(values: list[float]) -> dict[str, float | int | None]:
    return {
        "count": len(values),
        "mean_seconds": statistics.fmean(values) if values else None,
        "median_seconds": statistics.median(values) if values else None,
        "p95_seconds": percentile(values, 0.95),
        "max_seconds": max(values) if values else None,
    }


def play_game(task: ExperimentTask) -> dict[str, Any]:
    board = [[EMPTY] * task.board_size for _ in range(task.board_size)]
    agents = {
        BLACK: make_ai(task.black_agent, BLACK, task),
        WHITE: make_ai(task.white_agent, WHITE, task),
    }
    names = {BLACK: task.black_agent, WHITE: task.white_agent}
    last_move = {BLACK: None, WHITE: None}
    move_times = {BLACK: [], WHITE: []}
    move_records: list[dict[str, Any]] = []
    placed_moves = 0
    current = BLACK
    winner = EMPTY
    reason = ""
    started = time.perf_counter()

    for ply in range(1, task.board_size * task.board_size + 1):
        move, elapsed, error = call_get_move(
            agents[current], board, last_move[3 - current], task.time_limit
        )
        move_times[current].append(elapsed)
        record = {
            "ply": ply,
            "player": current,
            "agent": names[current],
            "move": list(move) if isinstance(move, (tuple, list)) and len(move) == 2 else None,
            "elapsed_seconds": elapsed,
            "error": error,
        }
        diagnostics = getattr(agents[current], "last_search_stats", None)
        if isinstance(diagnostics, dict):
            record["search_stats"] = dict(diagnostics)
        move_records.append(record)
        if error == "timeout":
            winner = 3 - current
            reason = f"timeout: {names[current]}"
            break
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
        placed_moves += 1
        last_move[current] = (row, col)
        if check_win(board, row, col, current, task.win_length):
            winner = current
            reason = f"{task.win_length}-in-a-row"
            break
        current = 3 - current
    else:
        reason = "board full"

    elapsed_game = time.perf_counter() - started
    record = {
        **asdict(task),
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "winner_color": winner,
        "winner_agent": names[winner] if winner else None,
        "reason": reason,
        "total_moves": placed_moves,
        "move_attempts": len(move_records),
        "duration_seconds": elapsed_game,
        "black_timing": timing_stats(move_times[BLACK]),
        "white_timing": timing_stats(move_times[WHITE]),
        "moves": move_records,
    }
    return record


def load_results(path: Path) -> tuple[list[dict[str, Any]], set[str]]:
    if not path.exists():
        return [], set()
    records: list[dict[str, Any]] = []
    keys: set[str] = set()
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}: {exc}") from exc
            key = record.get("key")
            if not isinstance(key, str):
                raise ValueError(f"missing experiment key at {path}:{line_number}")
            if key in keys:
                raise ValueError(f"duplicate experiment key: {key}")
            keys.add(key)
            records.append(record)
    return records, keys


def append_result(stream, record: dict[str, Any]) -> None:
    stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    stream.flush()
    os.fsync(stream.fileno())


def group_id(record: dict[str, Any]) -> str:
    return (
        f"{record['kind']}|{record['strategy']}|{record['opponent']}|"
        f"N{record['board_size']}|K{record['win_length']}|T{record['time_limit']:g}"
    )


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault(group_id(record), []).append(record)

    groups = []
    for key in sorted(grouped):
        games = sorted(grouped[key], key=lambda item: item["game_index"])
        first = games[0]
        color_wins = {
            "black": sum(game["winner_color"] == BLACK for game in games),
            "white": sum(game["winner_color"] == WHITE for game in games),
            "draw": sum(game["winner_color"] == EMPTY for game in games),
        }
        agent_names = sorted(
            {game["black_agent"] for game in games} | {game["white_agent"] for game in games}
        )
        agent_results: dict[str, Any] = {}
        all_nonrandom_times: list[float] = []
        for agent_name in agent_names:
            wins = losses = draws = black_games = white_games = 0
            by_color = {
                "black": {"wins": 0, "losses": 0, "draws": 0},
                "white": {"wins": 0, "losses": 0, "draws": 0},
            }
            times: list[float] = []
            threat_calls = threat_proofs = threat_nodes = 0
            for game in games:
                if game["black_agent"] == agent_name:
                    color = BLACK
                    black_games += 1
                    timing = game["black_timing"]
                elif game["white_agent"] == agent_name:
                    color = WHITE
                    white_games += 1
                    timing = game["white_timing"]
                else:
                    continue
                times.extend(
                    move["elapsed_seconds"]
                    for move in game["moves"]
                    if move["agent"] == agent_name
                )
                for move in game["moves"]:
                    if move["agent"] != agent_name:
                        continue
                    search_stats = move.get("search_stats")
                    if not isinstance(search_stats, dict):
                        continue
                    nodes = int(search_stats.get("threat_nodes", 0) or 0)
                    if nodes > 0:
                        threat_calls += 1
                        threat_nodes += nodes
                    if search_stats.get("threat_proven"):
                        threat_proofs += 1
                if game["winner_color"] == EMPTY:
                    draws += 1
                    by_color["black" if color == BLACK else "white"]["draws"] += 1
                elif game["winner_color"] == color:
                    wins += 1
                    by_color["black" if color == BLACK else "white"]["wins"] += 1
                else:
                    losses += 1
                    by_color["black" if color == BLACK else "white"]["losses"] += 1
            if agent_name != "Random AI":
                all_nonrandom_times.extend(times)
            total = wins + losses + draws
            agent_results[agent_name] = {
                "games": total,
                "black_games": black_games,
                "white_games": white_games,
                "wins": wins,
                "losses": losses,
                "draws": draws,
                "by_color": by_color,
                "win_rate": wins / total if total else None,
                "score_rate": (wins + 0.5 * draws) / total if total else None,
                "timing": timing_stats(times),
                "threat_search": {
                    "triggered_moves": threat_calls,
                    "proven_moves": threat_proofs,
                    "proof_rate": threat_proofs / threat_calls if threat_calls else None,
                    "nodes": threat_nodes,
                },
            }

        failure_counts = {"timeout": 0, "exception": 0, "illegal_move": 0}
        for game in games:
            reason = game["reason"]
            if reason.startswith("timeout"):
                failure_counts["timeout"] += 1
            elif reason.startswith("exception"):
                failure_counts["exception"] += 1
            elif reason.startswith("illegal move"):
                failure_counts["illegal_move"] += 1
        groups.append(
            {
                "group_id": key,
                "kind": first["kind"],
                "strategy": first["strategy"],
                "opponent": first["opponent"],
                "board_size": first["board_size"],
                "win_length": first["win_length"],
                "time_limit": first["time_limit"],
                "games": len(games),
                "color_results": color_wins,
                "average_game_moves": statistics.fmean(game["total_moves"] for game in games),
                "median_game_moves": statistics.median(game["total_moves"] for game in games),
                "agent_results": agent_results,
                "nonrandom_timing": timing_stats(all_nonrandom_times),
                "protocol_failures": failure_counts,
            }
        )

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "master_seed": MASTER_SEED,
        "environment": {
            "platform": platform.platform(),
            "python": sys.version,
            "processor": platform.processor(),
        },
        "total_games": len(records),
        "unique_keys": len({record["key"] for record in records}),
        "groups": groups,
    }


def write_summary(records: list[dict[str, Any]], path: Path) -> dict[str, Any]:
    result = summarize(records)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)
    return result


def format_duration(seconds: float) -> str:
    seconds = max(0, round(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {seconds:02d}s"
    if minutes:
        return f"{minutes}m {seconds:02d}s"
    return f"{seconds}s"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run reproducible Gomoku report experiments")
    parser.add_argument(
        "--only",
        choices=("all", "random", "selfplay", "smoke"),
        default="all",
        help="experiment subset; smoke runs two games per group",
    )
    parser.add_argument(
        "--resume", action="store_true", help="skip experiment keys already present"
    )
    parser.add_argument(
        "--summarize", action="store_true", help="only rebuild summary.json"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "experiments",
    )
    parser.add_argument(
        "--hybrid", action="store_true",
        help="include the Hybrid Threat Search extension matrix",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    smoke = args.only == "smoke"
    suffix = "_hybrid" if args.hybrid and not smoke else ""
    result_path = args.output_dir / ("smoke_results.jsonl" if smoke else f"results{suffix}.jsonl")
    summary_path = args.output_dir / ("smoke_summary.json" if smoke else f"summary{suffix}.json")
    existing, completed = load_results(result_path)

    if args.summarize:
        summary = write_summary(existing, summary_path)
        print(f"Summarized {summary['total_games']} games -> {summary_path}", flush=True)
        return
    if existing and not args.resume:
        raise SystemExit(f"{result_path} already contains data; use --resume")

    tasks = build_tasks(only=args.only, include_hybrid=args.hybrid)
    pending = [task for task in tasks if task.key not in completed]
    print(
        f"Experiment set: {len(tasks)} games; completed: {len(tasks) - len(pending)}; "
        f"pending: {len(pending)}",
        flush=True,
    )
    started = time.perf_counter()
    mode = "a" if result_path.exists() else "w"
    with result_path.open(mode, encoding="utf-8") as stream:
        for position, task in enumerate(pending, 1):
            game_started = time.perf_counter()
            record = play_game(task)
            append_result(stream, record)
            existing.append(record)
            duration = time.perf_counter() - game_started
            average = (time.perf_counter() - started) / position
            eta = average * (len(pending) - position)
            print(
                f"[{position:04d}/{len(pending):04d}] {task.key} -> "
                f"{record['winner_agent'] or 'Draw'} ({record['reason']}, "
                f"{record['total_moves']} moves, {duration:.2f}s); ETA {format_duration(eta)}",
                flush=True,
            )

    summary = write_summary(existing, summary_path)
    print(
        f"Completed {summary['total_games']} games in "
        f"{format_duration(time.perf_counter() - started)}; summary -> {summary_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
