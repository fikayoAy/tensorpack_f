"""List relevant entries inside the built wheel under `dist/`.

Usage (PowerShell):
  python tools\list_wheel_contents.py

This script finds the first wheel matching `dist/tensorpack-0.1*.whl`
and prints entries that are either the top-level `tensorpack.py` or
anything under the `tensorpack/` package directory.
"""
from __future__ import annotations
import glob
import zipfile
import sys
from pathlib import Path


def main() -> int:
    dist = Path('dist')
    if not dist.exists():
        print('No dist/ directory found', file=sys.stderr)
        return 2

    files = sorted(dist.glob('tensorpack-0.1*.whl'))
    if not files:
        print('No wheel matching dist/tensorpack-0.1*.whl', file=sys.stderr)
        return 3

    w = files[0]
    print('Inspecting wheel:', w)
    with zipfile.ZipFile(w, 'r') as z:
        names = z.namelist()
        # Filter the interesting entries
        interesting = [n for n in names if n == 'tensorpack.py' or n.startswith('tensorpack/')]
        if not interesting:
            print('No tensorpack entries found in wheel')
            return 4

        print('\nEntries of interest:')
        for n in interesting:
            print(' -', n)

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
