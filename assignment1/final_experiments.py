"""Run the final, version-bound experiment matrix for the report."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import statistics
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from gomoku_ai import MCTSAI
from run_experiments import (
    BLACK,
    EMPTY,
    MASTER_SEED,
    STRATEGIES,
    WHITE,
    SeededRandomAI,
    check_win,
    percentile,
    stable_seed,
    validate_move,
)


SCHEMA_VERSION = 1
CONFIGURATIONS = ((9, 4), (15, 5))
TIME_LIMITS = (0.5, 1.0, 5.0)
GAMES_PER_GROUP = 50
OPENING_PLIES = 4
FINAL_STRATEGY = "Hybrid Threat Search"
REFERENCE_STRATEGY = "Enhanced Alpha-Beta"
REFERENCE_OPPONENTS = (
    "Baseline Alpha-Beta",
    "MCTS + UCT",
    "Hybrid Threat Search",
)
SEARCH_STAT_KEYS = (
    "completed_depth",
    "nodes",
    "evaluations",
    "analysis_scans",
    "analysis_cache_hits",
    "tt_probes",
    "tt_hits",
    "tt_cutoffs",
    "quiescence_nodes",
    "threat_nodes",
    "threat_disproven",
    "threat_unknown",
    "mcts_iterations",
    "mcts_max_depth",
)


@dataclass(frozen=True)
class FinalExperimentTask:
    key: str
    kind: str
    strategy: str
    opponent: str
    board_size: int
    win_length: int
    time_limit: float
    game_index: int
    opening_index: int | None
    swap: int
    opening: tuple[tuple[int, int, int], ...]
    seed: int
    black_agent: str
    white_agent: str


def source_sha256(path: Path | None = None) -> str:
    target = path or Path(__file__).with_name("gomoku_ai.py")
    return hashlib.sha256(target.read_bytes()).hexdigest()


def _task_key(kind, strategy, opponent, size, win, limit, game_index):
    return (
        f"final-v{SCHEMA_VERSION}|{kind}|{strategy}|{opponent}|"
        f"N{size}|K{win}|T{limit:g}|G{game_index:03d}"
    )


def _opening(size: int, win: int, index: int):
    rng = random.Random(stable_seed("final-opening", size, win, index))
    board = [[EMPTY] * size for _ in range(size)]
    middle = size // 2
    radius = min(3, middle)
    candidates = [
        (row, col)
        for row in range(middle - radius, middle + radius + 1)
        for col in range(middle - radius, middle + radius + 1)
    ]
    rng.shuffle(candidates)
    stones = []
    for ply in range(OPENING_PLIES):
        player = BLACK if ply % 2 == 0 else WHITE
        for candidate_index, (row, col) in enumerate(candidates):
            if board[row][col] != EMPTY:
                continue
            board[row][col] = player
            if check_win(board, row, col, player, win):
                board[row][col] = EMPTY
                continue
            stones.append((row, col, player))
            candidates.pop(candidate_index)
            break
        else:
            raise RuntimeError(f"cannot generate opening N={size}, K={win}")
    return tuple(stones)


def build_tasks(
    games_per_group: int = GAMES_PER_GROUP,
    only: str = "all",
) -> list[FinalExperimentTask]:
    if games_per_group <= 0 or games_per_group % 2:
        raise ValueError("games_per_group must be a positive even integer")
    include_random = only in ("all", "random")
    include_selfplay = only in ("all", "selfplay")
    include_reference = only in ("all", "vs-enhanced")
    opening_count = games_per_group // 2
    openings = {
        (size, win, index): _opening(size, win, index)
        for size, win in CONFIGURATIONS
        for index in range(opening_count)
    }

    tasks = []
    for limit in TIME_LIMITS:
        for size, win in CONFIGURATIONS:
            if include_random:
                for game_index in range(games_per_group):
                    swap = game_index % 2
                    black, white = (
                        (FINAL_STRATEGY, "Random AI")
                        if swap == 0
                        else ("Random AI", FINAL_STRATEGY)
                    )
                    key = _task_key(
                        "vs_random",
                        FINAL_STRATEGY,
                        "Random AI",
                        size,
                        win,
                        limit,
                        game_index,
                    )
                    tasks.append(
                        FinalExperimentTask(
                            key,
                            "vs_random",
                            FINAL_STRATEGY,
                            "Random AI",
                            size,
                            win,
                            limit,
                            game_index,
                            None,
                            swap,
                            (),
                            stable_seed(key),
                            black,
                            white,
                        )
                    )

            if include_selfplay:
                for game_index in range(games_per_group):
                    opening_index = game_index // 2
                    swap = game_index % 2
                    black, white = (
                        (f"{FINAL_STRATEGY} A", f"{FINAL_STRATEGY} B")
                        if swap == 0
                        else (f"{FINAL_STRATEGY} B", f"{FINAL_STRATEGY} A")
                    )
                    key = _task_key(
                        "selfplay",
                        FINAL_STRATEGY,
                        FINAL_STRATEGY,
                        size,
                        win,
                        limit,
                        game_index,
                    )
                    tasks.append(
                        FinalExperimentTask(
                            key,
                            "selfplay",
                            FINAL_STRATEGY,
                            FINAL_STRATEGY,
                            size,
                            win,
                            limit,
                            game_index,
                            opening_index,
                            swap,
                            openings[(size, win, opening_index)],
                            stable_seed(key),
                            black,
                            white,
                        )
                    )

            if include_reference:
                for competitor in REFERENCE_OPPONENTS:
                    for game_index in range(games_per_group):
                        opening_index = game_index // 2
                        swap = game_index % 2
                        black, white = (
                            (competitor, REFERENCE_STRATEGY)
                            if swap == 0
                            else (REFERENCE_STRATEGY, competitor)
                        )
                        key = _task_key(
                            "vs_enhanced",
                            competitor,
                            REFERENCE_STRATEGY,
                            size,
                            win,
                            limit,
                            game_index,
                        )
                        tasks.append(
                            FinalExperimentTask(
                                key,
                                "vs_enhanced",
                                competitor,
                                REFERENCE_STRATEGY,
                                size,
                                win,
                                limit,
                                game_index,
                                opening_index,
                                swap,
                                openings[(size, win, opening_index)],
                                stable_seed(key),
                                black,
                                white,
                            )
                        )
    return tasks


def manifest_payload(games_per_group=GAMES_PER_GROUP):
    tasks = build_tasks(games_per_group)
    return {
        "schema_version": SCHEMA_VERSION,
        "source_sha256": source_sha256(),
        "master_seed": MASTER_SEED,
        "configurations": [list(item) for item in CONFIGURATIONS],
        "time_limits": list(TIME_LIMITS),
        "games_per_group": games_per_group,
        "opening_plies": OPENING_PLIES,
        "final_strategy": FINAL_STRATEGY,
        "reference_strategy": REFERENCE_STRATEGY,
        "reference_opponents": list(REFERENCE_OPPONENTS),
        "expected_games": len(tasks),
        "expected_groups": len({_group_id(asdict(task)) for task in tasks}),
    }


def ensure_manifest(path: Path, games_per_group=GAMES_PER_GROUP):
    expected = manifest_payload(games_per_group)
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        comparable = {key: existing.get(key) for key in expected}
        if comparable != expected:
            raise ValueError(
                "manifest does not match the current source or matrix; "
                "use a new output directory"
            )
        return existing
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        **expected,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return payload


def _base_strategy_name(name: str):
    return name.removesuffix(" A").removesuffix(" B")


def _make_ai(name, player, task):
    base_name = _base_strategy_name(name)
    if base_name == "Random AI":
        return SeededRandomAI(
            player,
            task.board_size,
            task.win_length,
            stable_seed(task.seed, "random", player),
        )
    ai = STRATEGIES[base_name](player, task.board_size, task.win_length)
    if isinstance(ai, MCTSAI):
        ai._random.seed(stable_seed(task.seed, "mcts", name, player))
    return ai


def _call_get_move_checked(ai, board, last_opponent_move, time_limit):
    holder: dict[str, Any] = {}
    call_board = [row[:] for row in board]
    original = [row[:] for row in call_board]

    def target():
        try:
            holder["move"] = ai.get_move(
                call_board, last_opponent_move, time_limit
            )
        except BaseException as exc:
            holder["error"] = f"exception: {type(exc).__name__}: {exc}"

    started = time.perf_counter()
    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    thread.join(time_limit)
    elapsed = time.perf_counter() - started
    if thread.is_alive():
        return None, elapsed, "timeout", None
    modified = call_board != original
    if "error" in holder:
        return None, elapsed, holder["error"], modified
    if modified:
        return holder.get("move"), elapsed, "input board modified", True
    return holder.get("move"), elapsed, None, False


def play_game(task: FinalExperimentTask, source_hash: str):
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
    move_times = {BLACK: [], WHITE: []}
    current = BLACK if len(task.opening) % 2 == 0 else WHITE
    winner = EMPTY
    reason = "board full"
    moves = []
    started = time.perf_counter()
    empty_count = task.board_size * task.board_size - len(task.opening)

    for played_ply in range(1, empty_count + 1):
        move, elapsed, error, input_modified = _call_get_move_checked(
            agents[current], board, last_move[3 - current], task.time_limit
        )
        move_times[current].append(elapsed)
        record = {
            "played_ply": played_ply,
            "player": current,
            "agent": names[current],
            "move": list(move) if isinstance(move, (tuple, list)) else None,
            "elapsed_seconds": elapsed,
            "error": error,
            "input_modified": input_modified,
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

    successful_moves = sum(move["error"] is None for move in moves)
    return {
        **asdict(task),
        "opening": [list(stone) for stone in task.opening],
        "schema_version": SCHEMA_VERSION,
        "source_sha256": source_hash,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "winner_color": winner,
        "winner_agent": names[winner] if winner else None,
        "reason": reason,
        "opening_plies": len(task.opening),
        "played_moves": successful_moves,
        "total_moves": len(task.opening) + successful_moves,
        "duration_seconds": time.perf_counter() - started,
        "black_timing": _timing(move_times[BLACK]),
        "white_timing": _timing(move_times[WHITE]),
        "moves": moves,
    }


def _timing(values):
    return {
        "count": len(values),
        "mean_seconds": statistics.fmean(values) if values else None,
        "median_seconds": statistics.median(values) if values else None,
        "p95_seconds": percentile(values, 0.95),
        "max_seconds": max(values) if values else None,
    }


def load_results(path: Path, expected_hash: str):
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
            if record.get("source_sha256") != expected_hash:
                raise ValueError(f"source hash mismatch at {path}:{line_number}")
            keys.add(key)
            records.append(record)
    return records, keys


def append_result(stream, record):
    stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    stream.flush()
    os.fsync(stream.fileno())


def _group_id(record):
    return (
        f"{record['kind']}|{record['strategy']}|{record['opponent']}|"
        f"N{record['board_size']}|K{record['win_length']}|"
        f"T{record['time_limit']:g}"
    )


def _agent_result(games, agent_name):
    wins = losses = draws = 0
    by_color = {
        "black": {"games": 0, "wins": 0, "losses": 0, "draws": 0},
        "white": {"games": 0, "wins": 0, "losses": 0, "draws": 0},
    }
    times = []
    search_values = {key: [] for key in SEARCH_STAT_KEYS}
    threat_proven = 0
    move_count = 0
    for game in games:
        if game["black_agent"] == agent_name:
            color = BLACK
        elif game["white_agent"] == agent_name:
            color = WHITE
        else:
            continue
        color_name = "black" if color == BLACK else "white"
        by_color[color_name]["games"] += 1
        if game["winner_color"] == EMPTY:
            draws += 1
            by_color[color_name]["draws"] += 1
        elif game["winner_color"] == color:
            wins += 1
            by_color[color_name]["wins"] += 1
        else:
            losses += 1
            by_color[color_name]["losses"] += 1

        for move in game["moves"]:
            if move["agent"] != agent_name:
                continue
            move_count += 1
            times.append(move["elapsed_seconds"])
            stats = move.get("search_stats")
            if not isinstance(stats, dict):
                continue
            threat_proven += bool(stats.get("threat_proven", False))
            for key in SEARCH_STAT_KEYS:
                value = stats.get(key)
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    search_values[key].append(value)
    total = wins + losses + draws
    return {
        "games": total,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "win_rate": wins / total if total else None,
        "score_rate": (wins + 0.5 * draws) / total if total else None,
        "by_color": by_color,
        "moves": move_count,
        "timing": _timing(times),
        "search_stats": {
            key: {
                "count": len(values),
                "mean": statistics.fmean(values) if values else None,
                "p95": percentile(values, 0.95),
                "max": max(values) if values else None,
                "sum": sum(values),
            }
            for key, values in search_values.items()
        },
        "threat_proven_moves": threat_proven,
    }


def _failure_kind(reason):
    if reason.startswith("timeout"):
        return "timeout"
    if reason.startswith("exception"):
        return "exception"
    if reason.startswith("illegal move"):
        return "illegal_move"
    if reason.startswith("input board modified"):
        return "input_modified"
    return None


def summarize(records, expected_tasks):
    expected_keys = {task.key for task in expected_tasks}
    grouped = {}
    for record in records:
        grouped.setdefault(_group_id(record), []).append(record)

    groups = []
    all_failures = []
    for group_key in sorted(grouped):
        games = sorted(grouped[group_key], key=lambda item: item["game_index"])
        first = games[0]
        agent_names = sorted(
            {game["black_agent"] for game in games}
            | {game["white_agent"] for game in games}
        )
        failures = []
        for game in games:
            failure_kind = _failure_kind(game["reason"])
            if failure_kind:
                item = {
                    "key": game["key"],
                    "kind": failure_kind,
                    "reason": game["reason"],
                }
                failures.append(item)
                all_failures.append(item)
        groups.append(
            {
                "group_id": group_key,
                "kind": first["kind"],
                "strategy": first["strategy"],
                "opponent": first["opponent"],
                "board_size": first["board_size"],
                "win_length": first["win_length"],
                "time_limit": first["time_limit"],
                "games": len(games),
                "unique_openings": len(
                    {game["opening_index"] for game in games if game["opening_index"] is not None}
                ),
                "color_results": {
                    "black_wins": sum(game["winner_color"] == BLACK for game in games),
                    "white_wins": sum(game["winner_color"] == WHITE for game in games),
                    "draws": sum(game["winner_color"] == EMPTY for game in games),
                },
                "mean_total_moves": statistics.fmean(
                    game["total_moves"] for game in games
                ),
                "agent_results": {
                    name: _agent_result(games, name) for name in agent_names
                },
                "failures": failures,
            }
        )

    result_keys = {record["key"] for record in records}
    completed_at = max(
        (record.get("completed_at", "") for record in records), default=None
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "source_sha256": source_sha256(),
        "completed_at": completed_at,
        "total_games": len(records),
        "unique_keys": len(result_keys),
        "expected_games": len(expected_keys),
        "expected_groups": len({_group_id(asdict(task)) for task in expected_tasks}),
        "complete": result_keys == expected_keys,
        "missing_keys": sorted(expected_keys - result_keys),
        "unexpected_keys": sorted(result_keys - expected_keys),
        "failure_counts": {
            kind: sum(item["kind"] == kind for item in all_failures)
            for kind in ("timeout", "exception", "illegal_move", "input_modified")
        },
        "failures": all_failures,
        "groups": groups,
    }


def write_summary(records, tasks, path):
    result = summarize(records, tasks)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    temporary.replace(path)
    return result


def _format_duration(seconds):
    seconds = max(0, round(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {seconds:02d}s"
    if minutes:
        return f"{minutes}m {seconds:02d}s"
    return f"{seconds}s"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--only",
        choices=("all", "random", "selfplay", "vs-enhanced"),
        default="all",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--summarize", action="store_true")
    # When omitted, an existing manifest supplies the matrix size.  This keeps
    # --summarize and --resume safe for smoke directories without weakening the
    # normal default for a new final run.
    parser.add_argument("--games-per-group", type=int, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).with_name("experiments") / "final",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "manifest.json"
    games_per_group = args.games_per_group
    if games_per_group is None and manifest_path.exists():
        existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        games_per_group = int(existing_manifest.get("games_per_group", GAMES_PER_GROUP))
    if games_per_group is None:
        games_per_group = GAMES_PER_GROUP
    manifest = ensure_manifest(
        manifest_path, games_per_group
    )
    all_tasks = build_tasks(games_per_group, "all")
    selected_tasks = build_tasks(games_per_group, args.only)
    results_path = args.output_dir / "results.jsonl"
    summary_path = args.output_dir / "summary.json"
    records, completed = load_results(results_path, manifest["source_sha256"])

    if args.summarize:
        summary = write_summary(records, all_tasks, summary_path)
        print(
            f"Summarized {summary['total_games']}/{summary['expected_games']} games "
            f"-> {summary_path}",
            flush=True,
        )
        return
    if records and not args.resume:
        raise SystemExit(f"{results_path} already contains data; use --resume")

    pending = [task for task in selected_tasks if task.key not in completed]
    print(
        f"Experiment set: {len(selected_tasks)} games; "
        f"completed in selected set: {len(selected_tasks) - len(pending)}; "
        f"pending: {len(pending)}",
        flush=True,
    )
    started = time.perf_counter()
    mode = "a" if results_path.exists() else "w"
    with results_path.open(mode, encoding="utf-8") as stream:
        for position, task in enumerate(pending, 1):
            game_started = time.perf_counter()
            record = play_game(task, manifest["source_sha256"])
            append_result(stream, record)
            records.append(record)
            completed.add(task.key)
            duration = time.perf_counter() - game_started
            average = (time.perf_counter() - started) / position
            eta = average * (len(pending) - position)
            print(
                f"[{position:04d}/{len(pending):04d}] {task.key} -> "
                f"{record['winner_agent'] or 'Draw'} ({record['reason']}, "
                f"{record['total_moves']} moves, {duration:.2f}s); "
                f"ETA {_format_duration(eta)}",
                flush=True,
            )

    summary = write_summary(records, all_tasks, summary_path)
    print(
        f"Stored {summary['total_games']}/{summary['expected_games']} games in "
        f"{_format_duration(time.perf_counter() - started)}; summary -> {summary_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
