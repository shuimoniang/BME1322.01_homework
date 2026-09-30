"""Graphical entry point for the generalized Gomoku project."""

try:
    from .gui import run
except ImportError:
    from gui import run


if __name__ == "__main__":
    run()
