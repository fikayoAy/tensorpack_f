import argparse
import importlib.metadata as m
import sys

from .. import __version__ as _version  # optional if package provides __version__


def _build_parser():
    p = argparse.ArgumentParser(prog='tensorpack', description='Tensorpack CLI')
    p.add_argument('--version', action='store_true', help='Show package version')
    sub = p.add_subparsers(dest='cmd')
    # minimal commands to avoid importing heavy modules at startup
    sub.add_parser('help', help='Show help')
    return p


def main(argv=None):
    argv = argv or sys.argv[1:]
    parser = _build_parser()
    ns = parser.parse_args(argv)
    if ns.version:
        try:
            ver = m.version('tensorpack')
        except Exception:
            ver = getattr(sys.modules.get('tensorpack'), '__version__', '0.0')
        print(ver)
        return 0
    if ns.cmd == 'help' or not ns.cmd:
        parser.print_help()
        return 0
    print(f"Unknown command: {ns.cmd}")
    return 2


if __name__ == '__main__':
    raise SystemExit(main())
