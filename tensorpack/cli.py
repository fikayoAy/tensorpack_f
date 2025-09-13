"""CLI shim for the tensorpack package.

This module provides a stable entry point that console scripts can target
(`tensorpack.cli:main`) and allows `python -m tensorpack` to work via
`tensorpack.__main__` delegating here.
"""
from __future__ import annotations
import sys
from typing import Optional, Sequence


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the package main() function with optional argv.

    Args:
        argv: Optional list of arguments (if None, use existing sys.argv).

    Returns:
        Exit code (int) returned by package main(), or 0 on success.
    """
    # Prefer the compiled `script` entrypoint (if present).
    # Do not import the top-level package here to avoid import recursion
    # when `tensorpack.main` delegates to this function.
    try:
        from .script import main as _script_main
    except Exception:
        _script_main = None

    # If argv provided, forward it to the script implementation where supported
    if argv is not None and _script_main is not None:
        try:
            return int(_script_main(argv))
        except SystemExit as se:
            return int(se.code) if se.code is not None else 0
        except Exception:
            import traceback
            traceback.print_exc()
            return 1

    if _script_main is not None:
        try:
            return int(_script_main())
        except SystemExit as se:
            return int(se.code) if se.code is not None else 0
        except Exception:
            import traceback
            traceback.print_exc()
            return 1

    # Fallback: try a legacy helper inside a module that doesn't import the package
    try:
        from . import run_cli as _run_cli
        if hasattr(_run_cli, 'main'):
            return int(_run_cli.main(argv) if argv is not None else _run_cli.main())
    except Exception:
        pass

    print("tensorpack package does not expose a main() entry point.", file=sys.stderr)
    return 3


if __name__ == '__main__':
    raise SystemExit(main())
