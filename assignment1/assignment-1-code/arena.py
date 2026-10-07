#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
arena.py — BME1322 Assignment 1 统一裁判程序（Gomoku AI Protocol v1）

用法:
    python arena.py <ai_file_a> <ai_file_b> [--board N] [--win K] [--time T]
                    [--games G] [--quiet]

加载两份符合协议的 gomoku_ai.py，进行 G 局比赛（默认 2 局，交换先后手），
逐步打印棋盘与落子，最后输出比赛结果摘要。

协议要点（与作业讲义一致）:
    - GomokuAI(player_id, board_size, win_length)
    - get_move(board, last_opponent_move, time_limit) -> (row, col)
    - 棋盘: 0=空位, 1=黑棋(先手), 2=白棋; 坐标 (row, col) 均从 0 开始
    - 超时 / 非法落子 / 抛出异常 => 直接判负
"""

import argparse
import importlib.util
import itertools
import sys
import threading
import time
from pathlib import Path

EMPTY, BLACK, WHITE = 0, 1, 2
STONE_CHAR = {EMPTY: ".", BLACK: "X", WHITE: "O"}
DIRECTIONS = ((0, 1), (1, 0), (1, 1), (1, -1))  # 横、竖、两条斜线

_module_seq = itertools.count(1)


# ---------------------------------------------------------------- AI 加载

def load_ai(path: Path):
    """从文件路径加载 GomokuAI 类与显示名 NAME（可缺省）。"""
    if not path.is_file():
        raise FileNotFoundError(f"AI 文件不存在: {path}")
    module_name = f"_gomoku_ai_{next(_module_seq)}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    if not hasattr(module, "GomokuAI"):
        raise AttributeError(f"{path} 中未找到 GomokuAI 类")
    name = getattr(module, "NAME", None)
    if not name:
        name = path.parent.name if path.parent.name not in ("", ".") else path.stem
    return module.GomokuAI, str(name)


# ---------------------------------------------------------------- 规则判定

def check_win(board, r, c, player, k):
    """落子 (r, c) 后，判断 player 是否连成 K 子及以上（含长连）。"""
    n = len(board)
    for dr, dc in DIRECTIONS:
        count = 1
        for sign in (1, -1):
            rr, cc = r + sign * dr, c + sign * dc
            while 0 <= rr < n and 0 <= cc < n and board[rr][cc] == player:
                count += 1
                rr += sign * dr
                cc += sign * dc
        if count >= k:
            return True
    return False


# ---------------------------------------------------------------- 调用 AI

def call_get_move(ai, board, last_opponent_move, time_limit):
    """带超时看门狗地调用 get_move。

    返回 (move, elapsed, error)：
      move   -- 正常返回时的落子坐标；超时/异常时为 None
      elapsed-- 实际耗时（秒）
      error  -- None 或 "timeout" / "exception: ..."
    注意：超时后原线程无法被强制杀死，只能弃置（daemon 线程，
    进程结束时自动回收）；此时立即判负，不再向该 AI 发出后续调用。
    """
    holder = {}

    def target():
        try:
            holder["move"] = ai.get_move([row[:] for row in board],
                                         last_opponent_move, time_limit)
        except BaseException as exc:  # noqa: BLE001 —— 学生代码任何异常都要接住
            holder["error"] = exc

    start = time.perf_counter()
    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    thread.join(time_limit)
    elapsed = time.perf_counter() - start

    if thread.is_alive():
        return None, elapsed, "timeout"
    if "error" in holder:
        return None, elapsed, f"exception: {holder['error']!r}"
    return holder.get("move"), elapsed, None


def validate_move(move, board):
    """校验落子合法性。合法返回 True，否则返回失败原因字符串。"""
    if not isinstance(move, (tuple, list)) or len(move) != 2:
        return "not a (row, col) pair"
    r, c = move
    if isinstance(r, bool) or isinstance(c, bool) \
            or not isinstance(r, int) or not isinstance(c, int):
        return "coordinates must be int"
    n = len(board)
    if not (0 <= r < n and 0 <= c < n):
        return f"out of range: ({r}, {c})"
    if board[r][c] != EMPTY:
        return f"cell occupied: ({r}, {c})"
    return None


# ---------------------------------------------------------------- 对局

def format_board(board):
    n = len(board)
    header = "    " + " ".join(f"{c:>2}" for c in range(n))
    lines = [header]
    for r in range(n):
        cells = " ".join(f"{STONE_CHAR[v]:>2}" for v in board[r])
        lines.append(f"{r:>3} {cells}")
    return "\n".join(lines)


def play_one_game(black, white, args, game_no):
    """进行一局比赛。black/white 为 (实例, 显示名) 二元组。"""
    n, k, t = args.board, args.win, args.time
    board = [[EMPTY] * n for _ in range(n)]
    inst = {BLACK: black[0], WHITE: white[0]}
    name = {BLACK: black[1], WHITE: white[1]}
    last_move = {BLACK: None, WHITE: None}
    player = BLACK
    moves = 0
    winner, reason = 0, ""

    if not args.quiet:
        print(f"\n=== Game {game_no}: {name[BLACK]} (X, 先手) "
              f"vs {name[WHITE]} (O, 后手) | board {n}x{n}, win {k}, time {t}s ===")
        print(format_board(board))

    while moves < n * n:
        cur = inst[player]
        move, elapsed, error = call_get_move(cur, board,
                                             last_move[3 - player], t)
        if error == "timeout":
            winner, reason = 3 - player, f"timeout ({name[player]} > {t:.1f}s)"
            break
        if error:
            winner, reason = 3 - player, f"{name[player]} {error}"
            break
        fail = validate_move(move, board)
        if fail:
            winner, reason = 3 - player, f"{name[player]} illegal move: {fail}"
            break
        r, c = move
        board[r][c] = player
        last_move[player] = (r, c)
        moves += 1
        if not args.quiet:
            print(f"[{moves:>3}] {name[player]}({STONE_CHAR[player]}) "
                  f"-> ({r}, {c})  ({elapsed:.2f}s)")
            print(format_board(board))
        if check_win(board, r, c, player, k):
            winner, reason = player, f"{k}-in-a-row"
            break
        player = 3 - player
    else:
        winner, reason = 0, "board full"

    if not args.quiet:
        print(format_board(board))
    return winner, reason, moves, name


def play_match(args):
    cls_a, name_a = load_ai(Path(args.ai_a))
    cls_b, name_b = load_ai(Path(args.ai_b))
    score = {name_a: 0.0, name_b: 0.0}
    results = []

    for g in range(args.games):
        a_is_black = (g % 2 == 0)
        if a_is_black:
            black = (cls_a(1, args.board, args.win), name_a)
            white = (cls_b(2, args.board, args.win), name_b)
        else:
            black = (cls_b(1, args.board, args.win), name_b)
            white = (cls_a(2, args.board, args.win), name_a)

        winner, reason, moves, name = play_one_game(black, white, args, g + 1)
        if winner == BLACK:
            win_name, lose_name = name[BLACK], name[WHITE]
            score[win_name] += 1.0
        elif winner == WHITE:
            win_name, lose_name = name[WHITE], name[BLACK]
            score[win_name] += 1.0
        else:
            win_name, lose_name = None, None
            score[name[BLACK]] += 0.5
            score[name[WHITE]] += 0.5
        results.append((g + 1, name[BLACK], name[WHITE], win_name, reason, moves))

    print("\n" + "=" * 24 + " 比赛结果 " + "=" * 24)
    for g, bn, wn, winner, reason, moves in results:
        if winner:
            print(f"Game {g}: {bn} (X) vs {wn} (O) -> "
                  f"{winner} wins [{reason}], {moves} moves")
        else:
            print(f"Game {g}: {bn} (X) vs {wn} (O) -> Draw [{reason}], "
                  f"{moves} moves")
    print(f"Match score: {name_a} {score[name_a]:.1f} : "
          f"{score[name_b]:.1f} {name_b}")


# ---------------------------------------------------------------- 入口

def main():
    parser = argparse.ArgumentParser(
        description="BME1322 Assignment 1 五子棋统一裁判 (Gomoku AI Protocol v1)")
    parser.add_argument("ai_a", help="AI 文件 A（gomoku_ai.py 路径）")
    parser.add_argument("ai_b", help="AI 文件 B（gomoku_ai.py 路径）")
    parser.add_argument("--board", type=int, default=15, help="棋盘边长 N（默认 15）")
    parser.add_argument("--win", type=int, default=5, help="连珠数 K（默认 5）")
    parser.add_argument("--time", type=float, default=5.0,
                        help="每步时限（秒，默认 5.0）")
    parser.add_argument("--games", type=int, default=2,
                        help="对局数，双方交换先后手（默认 2）")
    parser.add_argument("--quiet", action="store_true",
                        help="只输出最终结果，不逐步打印棋盘")
    args = parser.parse_args()

    if args.board < 3:
        parser.error("--board 必须 >= 3")
    if not (3 <= args.win <= args.board):
        parser.error("必须满足 3 <= --win <= --board")
    if args.time <= 0:
        parser.error("--time 必须 > 0")
    if args.games < 1:
        parser.error("--games 必须 >= 1")

    play_match(args)


if __name__ == "__main__":
    main()
