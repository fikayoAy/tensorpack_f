"""Module entrypoint to allow `python -m tensorpack` to run the package CLI.

This module performs imports lazily inside the runtime guard to avoid
circular import/timing issues when the package is imported during
installation or by the console script entrypoint.
"""
from __future__ import annotations
import sys


def _run() -> int:
    # Prefer the lightweight CLI shim where available; import inside runtime
    try:
        from .cli import main as _cli_main
        return int(_cli_main())
    except Exception:
        pass

    # Fallback to the package-level main() if present
    try:
        from . import main as _package_main
        return int(_package_main())
    except Exception as e:
        print(f"tensorpack: failed to run package main: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(_run())
