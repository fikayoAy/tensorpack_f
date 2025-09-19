from __future__ import annotations
from .script import main as _script_main


def _run() -> int:
    try:
        return int(_script_main())
    except SystemExit as se:
        return int(se.code) if se.code is not None else 0


if __name__ == "__main__":
    raise SystemExit(_run())
