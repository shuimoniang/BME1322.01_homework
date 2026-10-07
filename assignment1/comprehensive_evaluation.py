"""Run the complete four-strategy evaluation matrix.

This runner deliberately writes to a separate directory from the formal
1500-game matrix.  It reuses the formal runner's watchdog and record schema,
but expands both Random and self-play sections to all four strategies.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from final_experiments import (
    FinalExperimentTask,
    GAMES_PER_GROUP,
    OPENING_PLIES,
    CONFIGURATIONS,
    TIME_LIMITS,
    _opening,
    _format_duration,
    append_result,
    load_results,
    play_game,
    source_sha256,
    stable_seed,
    summarize,
    write_summary,
)


SCHEMA_VERSION = 1
STRATEGIES = (
    "Baseline Alpha-Beta",
    "Enhanced Alpha-Beta",
    "MCTS + UCT",
    "Hybrid Threat Search",
)
DEFAULT_OUTPUT = Path(__file__).with_name("experiments") / "comprehensive_evaluation"


def _task_key(kind, strategy, size, win, limit, game_index):
    return (
        f"comprehensive-v{SCHEMA_VERSION}|{kind}|{strategy}|"
        f"N{size}|K{win}|T{limit:g}|G{game_index:03d}"
    )


def build_tasks(games_per_group=GAMES_PER_GROUP, only="all"):
    if games_per_group <= 0 or games_per_group % 2:
        raise ValueError("games_per_group must be a positive even integer")
    if only not in {"all", "random", "selfplay"}:
        raise ValueError(f"unsupported experiment section: {only}")

    opening_count = games_per_group // 2
    openings = {
        (size, win, index): _opening(size, win, index)
        for size, win in CONFIGURATIONS
        for index in range(opening_count)
    }
    tasks = []
    for limit in TIME_LIMITS:
        for size, win in CONFIGURATIONS:
            for strategy in STRATEGIES:
                if only in {"all", "random"}:
                    for game_index in range(games_per_group):
                        ai_black = game_index % 2 == 0
                        black, white = (
                            (strategy, "Random AI")
                            if ai_black
                            else ("Random AI", strategy)
                        )
                        tasks.append(
                            FinalExperimentTask(
                                key=_task_key(
                                    "vs_random", strategy, size, win, limit, game_index
                                ),
                                kind="vs_random",
                                strategy=strategy,
                                opponent="Random AI",
                                board_size=size,
                                win_length=win,
                                time_limit=limit,
                                game_index=game_index,
                                opening_index=None,
                                swap=0 if ai_black else 1,
                                opening=(),
                                seed=stable_seed(
                                    "comprehensive", "vs_random", strategy,
                                    size, win, limit, game_index
                                ),
                                black_agent=black,
                                white_agent=white,
                            )
                        )

                if only in {"all", "selfplay"}:
                    for game_index in range(games_per_group):
                        opening_index = game_index // 2
                        first_a = game_index % 2 == 0
                        black = f"{strategy} A" if first_a else f"{strategy} B"
                        white = f"{strategy} B" if first_a else f"{strategy} A"
                        tasks.append(
                            FinalExperimentTask(
                                key=_task_key(
                                    "selfplay", strategy, size, win, limit, game_index
                                ),
                                kind="selfplay",
                                strategy=strategy,
                                opponent=strategy,
                                board_size=size,
                                win_length=win,
                                time_limit=limit,
                                game_index=game_index,
                                opening_index=opening_index,
                                swap=0 if first_a else 1,
                                opening=openings[(size, win, opening_index)],
                                seed=stable_seed(
                                    "comprehensive", "selfplay", strategy,
                                    size, win, limit, game_index
                                ),
                                black_agent=black,
                                white_agent=white,
                            )
                        )
    return tasks


def manifest_payload(games_per_group=GAMES_PER_GROUP):
    tasks = build_tasks(games_per_group)
    return {
        "schema_version": SCHEMA_VERSION,
        "source_sha256": source_sha256(),
        "master_seed": 20260928,
        "configurations": [list(item) for item in CONFIGURATIONS],
        "time_limits": list(TIME_LIMITS),
        "games_per_group": games_per_group,
        "opening_plies": OPENING_PLIES,
        "strategies": list(STRATEGIES),
        "expected_games": len(tasks),
        "expected_groups": len({_group_id(asdict(task)) for task in tasks}),
    }


def _group_id(record):
    return (
        f"{record['kind']}|{record['strategy']}|{record['opponent']}|"
        f"N{record['board_size']}|K{record['win_length']}|"
        f"T{record['time_limit']:g}"
    )


def ensure_manifest(path, games_per_group):
    expected = manifest_payload(games_per_group)
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if {key: existing.get(key) for key in expected} != expected:
            raise ValueError(
                "manifest does not match the current source or matrix; "
                "use a new output directory"
            )
        return existing
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {**expected, "created_at": datetime.now(timezone.utc).isoformat()}
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", choices=("all", "random", "selfplay"), default="all")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument("--games-per-group", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "manifest.json"
    games_per_group = args.games_per_group
    if games_per_group is None and manifest_path.exists():
        games_per_group = int(
            json.loads(manifest_path.read_text(encoding="utf-8"))["games_per_group"]
        )
    if games_per_group is None:
        games_per_group = GAMES_PER_GROUP

    manifest = ensure_manifest(manifest_path, games_per_group)
    all_tasks = build_tasks(games_per_group)
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
        f"Experiment set: {len(selected_tasks)} games; pending: {len(pending)}",
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
            elapsed = time.perf_counter() - started
            eta = elapsed / position * (len(pending) - position)
            print(
                f"[{position:04d}/{len(pending):04d}] {task.key} -> "
                f"{record['winner_agent'] or 'Draw'} ({record['reason']}, "
                f"{record['total_moves']} moves, {duration:.2f}s); "
                f"ETA {_format_duration(eta)}",
                flush=True,
            )
    summary = write_summary(records, all_tasks, summary_path)
    print(
        f"Stored {summary['total_games']}/{summary['expected_games']} games "
        f"-> {summary_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
