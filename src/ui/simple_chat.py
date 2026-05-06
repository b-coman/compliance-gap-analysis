"""Simple chat loop — demo entry point for the simplified architecture.

Mirrors the shape of `notebook_chat.py` but routes queries through the
single-call simplified path in `src/simplified.py` rather than through the
multi-step chain. Intended as the production demo path.

Usage from a Python REPL or notebook:

    from src.ui.simple_chat import chat
    chat()

Type a compliance question; receive a 3-section assessment. Type
'exit', 'quit', or send EOF (Ctrl-D) to leave the loop.
"""
from __future__ import annotations

import sys
from typing import TextIO

from src.simplified import analyse


_EXIT_TOKENS = {"exit", "quit", "q", "bye"}


def chat(*, file: TextIO | None = None) -> None:
    """Interactive chat loop. Reads queries from stdin and writes responses
    to `file` (defaults to stdout, resolved at call-time so pytest's
    `capsys` works correctly).
    """
    out = file if file is not None else sys.stdout

    print("Compliance gap analysis (simplified architecture).", file=out)
    print("Type a question, or 'exit' to leave.\n", file=out)

    while True:
        try:
            query = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.", file=out)
            return

        if not query:
            continue
        if query.lower() in _EXIT_TOKENS:
            print("Goodbye.", file=out)
            return

        try:
            response = analyse(query)
        except Exception as exc:
            print(f"\n[error] {type(exc).__name__}: {exc}\n", file=out)
            continue

        print("\n" + response + "\n", file=out)


if __name__ == "__main__":
    chat()
