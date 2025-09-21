from __future__ import annotations
from typing import Optional, Sequence
from .script import main as _script_main


def main(argv: Optional[Sequence[str]] = None) -> int:
    try:
        return int(_script_main(argv) if argv is not None else _script_main())
    except SystemExit as se:
        return int(se.code) if se.code is not None else 0


if __name__ == '__main__':
    raise SystemExit(main())
