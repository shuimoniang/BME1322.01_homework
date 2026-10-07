"""Arena adapter for the MCTS + UCT strategy."""

import importlib.util
import sys
from pathlib import Path


_CORE_NAME = "_assignment1_gomoku_core"
if _CORE_NAME in sys.modules:
    _core = sys.modules[_CORE_NAME]
else:
    _spec = importlib.util.spec_from_file_location(
        _CORE_NAME, Path(__file__).with_name("gomoku_ai.py")
    )
    if _spec is None or _spec.loader is None:
        raise ImportError("cannot load gomoku_ai.py")
    _core = importlib.util.module_from_spec(_spec)
    sys.modules[_CORE_NAME] = _core
    _spec.loader.exec_module(_core)

MCTSAI = _core.MCTSAI

NAME = "MCTS-UCT"
GomokuAI = MCTSAI
