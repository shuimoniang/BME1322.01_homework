"""Tkinter interface for generalized Gomoku."""

from __future__ import annotations

import importlib.util
import itertools
import queue
import sys
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

try:
    from .gomoku_ai import BaselineAlphaBetaAI, EnhancedAlphaBetaAI, MCTSAI
    from .gomoku_game import BLACK, EMPTY, WHITE, GomokuGame
except ImportError:
    from gomoku_ai import BaselineAlphaBetaAI, EnhancedAlphaBetaAI, MCTSAI
    from gomoku_game import BLACK, EMPTY, WHITE, GomokuGame


MODE_HUMAN = "人人对战"
MODE_HUMAN_AI = "人机对战"
MODE_AI = "机机对战"
STRATEGIES = {
    "基础 Alpha-Beta": BaselineAlphaBetaAI,
    "增强 Alpha-Beta": EnhancedAlphaBetaAI,
    "MCTS + UCT": MCTSAI,
}
EXTERNAL = "外部 AI"
SIDE_NAME = {BLACK: "黑方", WHITE: "白方"}
STONE_NAME = {BLACK: "黑棋", WHITE: "白棋"}


class GomokuGUI:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("广义五子棋 - 对抗搜索实验平台")
        self.root.geometry("1180x760")
        self.root.minsize(900, 620)

        self.game = GomokuGame()
        self.session_id = 0
        self.task_id = itertools.count(1)
        self.active_task: tuple[int, int] | None = None
        self.active_deadline = 0.0
        self.result_queue: queue.Queue = queue.Queue()
        self.ai_players: dict[int, object | None] = {BLACK: None, WHITE: None}
        self.external_paths: dict[int, Path | None] = {BLACK: None, WHITE: None}
        self.external_names: dict[int, str] = {BLACK: "未选择", WHITE: "未选择"}
        self.paused = False
        self.end_reason = ""

        self._create_variables()
        self._configure_style()
        self._build_layout()
        self._sync_mode_controls()
        self._start_game()
        self.root.after(25, self._poll_ai_results)

    def _create_variables(self) -> None:
        self.mode_var = tk.StringVar(value=MODE_HUMAN_AI)
        self.board_size_var = tk.StringVar(value="15")
        self.win_length_var = tk.StringVar(value="5")
        self.time_limit_var = tk.StringVar(value="5.0")
        self.human_side_var = tk.StringVar(value="黑方")
        self.black_strategy_var = tk.StringVar(value="增强 Alpha-Beta")
        self.white_strategy_var = tk.StringVar(value="增强 Alpha-Beta")
        self.delay_var = tk.IntVar(value=250)
        self.status_var = tk.StringVar(value="准备开始")
        self.detail_var = tk.StringVar(value="")
        self.pause_text_var = tk.StringVar(value="暂停")

    def _configure_style(self) -> None:
        style = ttk.Style(self.root)
        if "vista" in style.theme_names():
            style.theme_use("vista")
        style.configure("Title.TLabel", font=("Microsoft YaHei UI", 16, "bold"))
        style.configure("Status.TLabel", font=("Microsoft YaHei UI", 12, "bold"))
        style.configure("Muted.TLabel", foreground="#5f6b66")
        style.configure("Accent.TButton", font=("Microsoft YaHei UI", 10, "bold"))

    def _build_layout(self) -> None:
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)

        settings = ttk.Frame(self.root, padding=(16, 12))
        settings.grid(row=0, column=0, sticky="ew")
        for col in range(14):
            settings.columnconfigure(col, weight=0)
        settings.columnconfigure(13, weight=1)

        ttk.Label(settings, text="广义五子棋", style="Title.TLabel").grid(
            row=0, column=0, columnspan=2, sticky="w", padx=(0, 20)
        )
        self._labeled_combo(settings, "模式", self.mode_var, [MODE_HUMAN, MODE_HUMAN_AI, MODE_AI], 2)
        self._labeled_entry(settings, "棋盘 N", self.board_size_var, 4, 7)
        self._labeled_entry(settings, "连珠 K", self.win_length_var, 6, 7)
        self._labeled_entry(settings, "每步秒数", self.time_limit_var, 8, 7)
        self.human_side_combo = self._labeled_combo(
            settings, "玩家执子", self.human_side_var, ["黑方", "白方"], 10
        )
        ttk.Button(
            settings, text="新对局", style="Accent.TButton", command=self._start_game
        ).grid(row=0, column=12, padx=(16, 6), sticky="e")
        self.pause_button = ttk.Button(
            settings, textvariable=self.pause_text_var, command=self._toggle_pause
        )
        self.pause_button.grid(row=0, column=13, sticky="w")

        self.mode_combo.bind("<<ComboboxSelected>>", lambda _event: self._sync_mode_controls())
        self.human_side_combo.bind(
            "<<ComboboxSelected>>", lambda _event: self._sync_mode_controls()
        )

        body = ttk.Frame(self.root, padding=(16, 0, 16, 16))
        body.grid(row=1, column=0, sticky="nsew")
        body.columnconfigure(0, weight=1)
        body.columnconfigure(1, weight=0, minsize=300)
        body.rowconfigure(0, weight=1)

        board_frame = ttk.Frame(body)
        board_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 14))
        board_frame.columnconfigure(0, weight=1)
        board_frame.rowconfigure(0, weight=1)
        self.canvas = tk.Canvas(
            board_frame,
            background="#d9aa62",
            highlightthickness=1,
            highlightbackground="#8a714c",
            cursor="hand2",
        )
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.canvas.bind("<Configure>", lambda _event: self._draw_board())
        self.canvas.bind("<Button-1>", self._on_board_click)

        side = ttk.Frame(body, width=300)
        side.grid(row=0, column=1, sticky="ns")
        side.grid_propagate(False)
        side.columnconfigure(0, weight=1)
        side.rowconfigure(5, weight=1)

        ttk.Label(side, textvariable=self.status_var, style="Status.TLabel").grid(
            row=0, column=0, sticky="ew", pady=(2, 2)
        )
        ttk.Label(
            side,
            textvariable=self.detail_var,
            style="Muted.TLabel",
            wraplength=290,
            justify="left",
        ).grid(row=1, column=0, sticky="ew", pady=(0, 12))

        strategy_box = ttk.LabelFrame(side, text="对局参与方", padding=10)
        strategy_box.grid(row=2, column=0, sticky="ew")
        strategy_box.columnconfigure(1, weight=1)
        ttk.Label(strategy_box, text="黑方").grid(row=0, column=0, sticky="w", pady=3)
        self.black_strategy_combo = ttk.Combobox(
            strategy_box,
            textvariable=self.black_strategy_var,
            values=list(STRATEGIES) + [EXTERNAL],
            state="readonly",
            width=18,
        )
        self.black_strategy_combo.grid(row=0, column=1, sticky="ew", padx=6)
        self.black_file_button = ttk.Button(
            strategy_box, text="选择文件", command=lambda: self._choose_external(BLACK)
        )
        self.black_file_button.grid(row=0, column=2)

        ttk.Label(strategy_box, text="白方").grid(row=1, column=0, sticky="w", pady=3)
        self.white_strategy_combo = ttk.Combobox(
            strategy_box,
            textvariable=self.white_strategy_var,
            values=list(STRATEGIES) + [EXTERNAL],
            state="readonly",
            width=18,
        )
        self.white_strategy_combo.grid(row=1, column=1, sticky="ew", padx=6)
        self.white_file_button = ttk.Button(
            strategy_box, text="选择文件", command=lambda: self._choose_external(WHITE)
        )
        self.white_file_button.grid(row=1, column=2)

        speed = ttk.Frame(side)
        speed.grid(row=3, column=0, sticky="ew", pady=(12, 8))
        speed.columnconfigure(1, weight=1)
        ttk.Label(speed, text="机机步间隔").grid(row=0, column=0, sticky="w")
        ttk.Scale(
            speed, from_=0, to=1200, variable=self.delay_var, orient="horizontal"
        ).grid(row=0, column=1, sticky="ew", padx=(8, 0))

        ttk.Separator(side).grid(row=4, column=0, sticky="ew", pady=(2, 8))
        self.move_table = ttk.Treeview(
            side,
            columns=("number", "side", "coordinate", "elapsed"),
            show="headings",
            height=14,
        )
        headings = (("number", "步"), ("side", "棋方"), ("coordinate", "坐标"), ("elapsed", "用时"))
        for key, label in headings:
            self.move_table.heading(key, text=label)
        self.move_table.column("number", width=36, anchor="center", stretch=False)
        self.move_table.column("side", width=50, anchor="center", stretch=False)
        self.move_table.column("coordinate", width=84, anchor="center")
        self.move_table.column("elapsed", width=70, anchor="e", stretch=False)
        self.move_table.grid(row=5, column=0, sticky="nsew")
        scrollbar = ttk.Scrollbar(side, orient="vertical", command=self.move_table.yview)
        scrollbar.grid(row=5, column=1, sticky="ns")
        self.move_table.configure(yscrollcommand=scrollbar.set)

    def _labeled_combo(self, parent, label, variable, values, column):
        ttk.Label(parent, text=label).grid(row=0, column=column, padx=(8, 4), sticky="e")
        combo = ttk.Combobox(
            parent, textvariable=variable, values=values, state="readonly", width=10
        )
        combo.grid(row=0, column=column + 1, sticky="w")
        if variable is self.mode_var:
            self.mode_combo = combo
        return combo

    @staticmethod
    def _labeled_entry(parent, label, variable, column, width):
        ttk.Label(parent, text=label).grid(row=0, column=column, padx=(8, 4), sticky="e")
        entry = ttk.Entry(parent, textvariable=variable, width=width)
        entry.grid(row=0, column=column + 1, sticky="w")
        return entry

    def _sync_mode_controls(self) -> None:
        mode = self.mode_var.get()
        human_state = "readonly" if mode == MODE_HUMAN_AI else "disabled"
        self.human_side_combo.configure(state=human_state)
        self.pause_button.configure(state="normal" if mode == MODE_AI else "disabled")

        if mode == MODE_HUMAN:
            black_state = white_state = "disabled"
        elif mode == MODE_HUMAN_AI:
            human = BLACK if self.human_side_var.get() == "黑方" else WHITE
            black_state = "disabled" if human == BLACK else "readonly"
            white_state = "disabled" if human == WHITE else "readonly"
        else:
            black_state = white_state = "readonly"
        self.black_strategy_combo.configure(state=black_state)
        self.white_strategy_combo.configure(state=white_state)
        self.black_file_button.configure(state="normal" if black_state != "disabled" else "disabled")
        self.white_file_button.configure(state="normal" if white_state != "disabled" else "disabled")

    def _read_configuration(self) -> tuple[int, int, float]:
        try:
            board_size = int(self.board_size_var.get().strip())
            win_length = int(self.win_length_var.get().strip())
            time_limit = float(self.time_limit_var.get().strip())
        except ValueError as exc:
            raise ValueError("N、K 必须是整数，每步秒数必须是数字") from exc
        if board_size < 3:
            raise ValueError("棋盘大小 N 必须至少为 3")
        if not 3 <= win_length <= board_size:
            raise ValueError("必须满足 3 <= K <= N")
        if time_limit <= 0:
            raise ValueError("每步秒数必须大于 0")
        return board_size, win_length, time_limit

    def _start_game(self) -> None:
        try:
            board_size, win_length, time_limit = self._read_configuration()
            game = GomokuGame(board_size, win_length)
            players = self._create_players(board_size, win_length)
        except (ValueError, TypeError, ImportError, AttributeError) as exc:
            messagebox.showerror("无法开始", str(exc), parent=self.root)
            return

        self.session_id += 1
        self.active_task = None
        self.active_deadline = 0.0
        self.game = game
        self.ai_players = players
        self.time_limit = time_limit
        self.paused = False
        self.pause_text_var.set("暂停")
        self.end_reason = ""
        for item in self.move_table.get_children():
            self.move_table.delete(item)
        self._sync_mode_controls()
        self._update_status()
        self._draw_board()
        self.root.after(80, self._maybe_start_ai)

    def _create_players(self, board_size, win_length):
        mode = self.mode_var.get()
        players = {BLACK: None, WHITE: None}
        if mode == MODE_HUMAN:
            return players
        if mode == MODE_HUMAN_AI:
            human = BLACK if self.human_side_var.get() == "黑方" else WHITE
            ai_side = 3 - human
            players[ai_side] = self._new_ai(ai_side, board_size, win_length)
            return players
        players[BLACK] = self._new_ai(BLACK, board_size, win_length)
        players[WHITE] = self._new_ai(WHITE, board_size, win_length)
        return players

    def _new_ai(self, side, board_size, win_length):
        strategy = self.black_strategy_var.get() if side == BLACK else self.white_strategy_var.get()
        if strategy in STRATEGIES:
            return STRATEGIES[strategy](side, board_size, win_length)
        if strategy != EXTERNAL:
            raise ValueError(f"未知 AI 策略：{strategy}")
        path = self.external_paths[side]
        if path is None:
            raise ValueError(f"请先为{SIDE_NAME[side]}选择外部 AI 文件")
        module_name = f"_external_gomoku_{side}_{time.time_ns()}"
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"无法加载文件：{path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        if not hasattr(module, "GomokuAI"):
            raise AttributeError(f"{path.name} 中没有 GomokuAI 类")
        return module.GomokuAI(side, board_size, win_length)

    def _choose_external(self, side) -> None:
        filename = filedialog.askopenfilename(
            parent=self.root,
            title=f"选择{SIDE_NAME[side]} AI 文件",
            filetypes=[("Python 文件", "*.py"), ("所有文件", "*.*")],
        )
        if not filename:
            return
        path = Path(filename)
        self.external_paths[side] = path
        self.external_names[side] = path.stem
        if side == BLACK:
            self.black_strategy_var.set(EXTERNAL)
        else:
            self.white_strategy_var.set(EXTERNAL)
        self.detail_var.set(f"已为{SIDE_NAME[side]}选择：{path.name}")

    def _on_board_click(self, event) -> None:
        if self.game.game_over or self.active_task is not None or self.paused:
            return
        if self.ai_players.get(self.game.current_player) is not None:
            return
        geometry = self._board_geometry()
        if geometry is None:
            return
        left, top, step = geometry
        col = round((event.x - left) / step)
        row = round((event.y - top) / step)
        if not (0 <= row < self.game.board_size and 0 <= col < self.game.board_size):
            return
        if abs(event.x - (left + col * step)) > max(8, step * 0.48):
            return
        if abs(event.y - (top + row * step)) > max(8, step * 0.48):
            return
        error = self.game.validate_move(row, col)
        if error:
            self.detail_var.set(error)
            return
        self._apply_move((row, col), 0.0)

    def _maybe_start_ai(self) -> None:
        if self.game.game_over or self.paused or self.active_task is not None:
            return
        player = self.game.current_player
        ai = self.ai_players.get(player)
        if ai is None:
            return

        task = (self.session_id, next(self.task_id))
        self.active_task = task
        self.active_deadline = time.perf_counter() + self.time_limit
        board = self.game.board_copy()
        last_move = self.game.last_move
        self.status_var.set(f"{SIDE_NAME[player]} AI 思考中…")
        self.detail_var.set(f"本步上限 {self.time_limit:g} 秒")

        thread = threading.Thread(
            target=self._run_ai,
            args=(task, player, ai, board, last_move, self.time_limit),
            daemon=True,
        )
        thread.start()

    def _run_ai(self, task, player, ai, board, last_move, time_limit) -> None:
        start = time.perf_counter()
        try:
            move = ai.get_move(board, last_move, time_limit)
            error = None
        except BaseException as exc:  # Student-loaded code must not stop the GUI.
            move = None
            error = f"运行异常：{exc!r}"
        elapsed = time.perf_counter() - start
        self.result_queue.put((task, player, move, elapsed, error))

    def _poll_ai_results(self) -> None:
        try:
            while True:
                result = self.result_queue.get_nowait()
                self._handle_ai_result(*result)
        except queue.Empty:
            pass

        if (
            self.active_task is not None
            and time.perf_counter() >= self.active_deadline
            and not self.game.game_over
        ):
            loser = self.game.current_player
            self.active_task = None
            self._finish_forfeit(3 - loser, f"{SIDE_NAME[loser]} AI 超时")
        self.root.after(25, self._poll_ai_results)

    def _handle_ai_result(self, task, player, move, elapsed, error) -> None:
        if task != self.active_task or task[0] != self.session_id:
            return
        self.active_task = None
        if self.game.game_over or player != self.game.current_player:
            return
        if elapsed >= self.time_limit:
            self._finish_forfeit(3 - player, f"{SIDE_NAME[player]} AI 超时（{elapsed:.3f}s）")
            return
        if error:
            self._finish_forfeit(3 - player, f"{SIDE_NAME[player]} {error}")
            return
        if (
            not isinstance(move, (tuple, list))
            or len(move) != 2
            or isinstance(move[0], bool)
            or isinstance(move[1], bool)
            or not isinstance(move[0], int)
            or not isinstance(move[1], int)
        ):
            self._finish_forfeit(3 - player, f"{SIDE_NAME[player]} AI 返回了非法坐标")
            return
        error = self.game.validate_move(move[0], move[1])
        if error:
            self._finish_forfeit(3 - player, f"{SIDE_NAME[player]} AI 非法落子：{error}")
            return
        self._apply_move((move[0], move[1]), elapsed)

    def _apply_move(self, move, elapsed) -> None:
        played = self.game.place_move(*move)
        self.move_table.insert(
            "",
            "end",
            values=(
                len(self.game.history),
                "黑" if played.player == BLACK else "白",
                f"({played.row}, {played.col})",
                "-" if elapsed == 0 else f"{elapsed:.3f}s",
            ),
        )
        children = self.move_table.get_children()
        if children:
            self.move_table.see(children[-1])
        self._draw_board()
        self._update_status()
        if not self.game.game_over:
            delay = self.delay_var.get() if self.mode_var.get() == MODE_AI else 80
            self.root.after(max(0, int(delay)), self._maybe_start_ai)

    def _finish_forfeit(self, winner, reason) -> None:
        self.game.winner = winner
        self.game.game_over = True
        self.end_reason = reason
        self._update_status()
        self._draw_board()

    def _toggle_pause(self) -> None:
        if self.mode_var.get() != MODE_AI or self.game.game_over:
            return
        self.paused = not self.paused
        self.pause_text_var.set("继续" if self.paused else "暂停")
        self._update_status()
        if not self.paused:
            self.root.after(20, self._maybe_start_ai)

    def _update_status(self) -> None:
        if self.game.game_over:
            if self.game.is_draw:
                self.status_var.set("和棋")
                self.detail_var.set("棋盘已满，双方均未连珠")
            else:
                self.status_var.set(f"{SIDE_NAME[self.game.winner]}获胜")
                self.detail_var.set(self.end_reason or f"达成 {self.game.win_length} 子连珠")
            return
        if self.paused:
            self.status_var.set("对局已暂停")
            self.detail_var.set(f"当前轮到{SIDE_NAME[self.game.current_player]}")
            return
        self.status_var.set(f"轮到{SIDE_NAME[self.game.current_player]}")
        self.detail_var.set(
            f"{self.game.board_size}×{self.game.board_size} 棋盘，"
            f"{self.game.win_length} 子连珠，已下 {len(self.game.history)} 步"
        )

    def _board_geometry(self):
        width = self.canvas.winfo_width()
        height = self.canvas.winfo_height()
        if width <= 2 or height <= 2:
            return None
        n = self.game.board_size
        margin = 34 if n <= 25 else 18
        span = max(1.0, min(width, height) - margin * 2)
        step = span / max(1, n - 1)
        left = (width - span) / 2
        top = (height - span) / 2
        return left, top, step

    def _draw_board(self) -> None:
        if not hasattr(self, "canvas"):
            return
        self.canvas.delete("all")
        geometry = self._board_geometry()
        if geometry is None:
            return
        left, top, step = geometry
        n = self.game.board_size
        right = left + step * (n - 1)
        bottom = top + step * (n - 1)

        line_color = "#604a2f"
        width = 1 if step < 8 else 1.2
        for index in range(n):
            position = left + index * step
            self.canvas.create_line(position, top, position, bottom, fill=line_color, width=width)
            position = top + index * step
            self.canvas.create_line(left, position, right, position, fill=line_color, width=width)

        if n <= 25 and step >= 18:
            font_size = max(7, min(10, int(step * 0.34)))
            for index in range(n):
                self.canvas.create_text(
                    left + index * step,
                    top - min(16, step * 0.65),
                    text=str(index),
                    fill="#3d3428",
                    font=("Segoe UI", font_size),
                )
                self.canvas.create_text(
                    left - min(18, step * 0.75),
                    top + index * step,
                    text=str(index),
                    fill="#3d3428",
                    font=("Segoe UI", font_size),
                )

        radius = max(1.2, min(16.0, step * 0.43))
        winning = set(self.game.winning_line)
        for row in range(n):
            for col, player in enumerate(self.game.board[row]):
                if player == EMPTY:
                    continue
                x, y = left + col * step, top + row * step
                fill = "#202523" if player == BLACK else "#f7f6f0"
                outline = "#111514" if player == BLACK else "#8a8f8c"
                ring = "#16835d" if (row, col) in winning else outline
                ring_width = max(1, min(4, int(step * 0.12))) if (row, col) in winning else 1
                self.canvas.create_oval(
                    x - radius,
                    y - radius,
                    x + radius,
                    y + radius,
                    fill=fill,
                    outline=ring,
                    width=ring_width,
                )

        if self.game.last_move is not None:
            row, col = self.game.last_move
            x, y = left + col * step, top + row * step
            marker = max(1.5, radius * 0.20)
            self.canvas.create_oval(
                x - marker,
                y - marker,
                x + marker,
                y + marker,
                fill="#d94a3d",
                outline="",
            )


def run() -> None:
    root = tk.Tk()
    GomokuGUI(root)
    root.mainloop()


if __name__ == "__main__":
    run()
