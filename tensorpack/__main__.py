"""Module entrypoint to allow `python -m tensorpack` to run the package CLI."""
from __future__ import annotations
import sys


def _run():
    try:
        # Import the CLI shim; this will import the package and delegate
        from .cli import main
    except Exception as e:
        print(f"Failed to import tensorpack.cli: {e}", file=sys.stderr)
        return 2

    # Delegate to the cli main with original argv
    return main()


if __name__ == '__main__':
    raise SystemExit(_run())
