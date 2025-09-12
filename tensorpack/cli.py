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
    # Defer import to avoid heavy module import at install time
    try:
        import tensorpack as _tp
    except Exception as e:
        print(f"Error importing tensorpack package: {e}", file=sys.stderr)
        return 2

    # If argv provided, set sys.argv accordingly
    if argv is not None:
        sys.argv[:] = list(argv)

    # Call package-level main() if present
    main_fn = getattr(_tp, 'main', None)
    if callable(main_fn):
        try:
            result = main_fn()
            return int(result) if result is not None else 0
        except SystemExit as se:
            # Preserve explicit SystemExit codes
            return int(se.code) if se.code is not None else 0
        except Exception:
            import traceback
            traceback.print_exc()
            return 1

    # Fallback: try legacy CLI helper `run_cli` or offer guidance
    alt = getattr(_tp, 'run_cli', None)
    if callable(alt):
        try:
            alt()
            return 0
        except Exception:
            return 1

    print("tensorpack package does not expose a main() entry point.", file=sys.stderr)
    return 3


if __name__ == '__main__':
    raise SystemExit(main())
