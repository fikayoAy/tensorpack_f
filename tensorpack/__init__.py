"""tensorpack package initializer.

Expose a small `main()` function so the CLI shim can call into the
package. The real implementation may live in `tensorpack.script` (preferred),
or as a top-level `script` module (legacy). This wrapper tries both and
returns appropriate exit codes.
"""
from __future__ import annotations
import sys
from typing import Optional


def _find_main_callable():
	# Try package-local script module first
	try:
		from . import script as _script
		main_fn = getattr(_script, 'main', None)
		if callable(main_fn):
			return main_fn
	except Exception:
		pass

	# Try top-level legacy `script` module
	try:
		import script as _script
		main_fn = getattr(_script, 'main', None)
		if callable(main_fn):
			return main_fn
	except Exception:
		pass

	# No main found
	return None


def main(argv: Optional[list] = None) -> int:
	"""Package-level entry point used by `tensorpack.cli`.

	Args:
		argv: Optional list of args; if provided, replaces `sys.argv`.

	Returns:
		Exit code (int).
	"""
	main_fn = _find_main_callable()
	if main_fn is None:
		print("tensorpack package does not expose a main() entry point.", file=sys.stderr)
		return 3

	if argv is not None:
		sys.argv[:] = list(argv)

	try:
		result = main_fn()
		return int(result) if result is not None else 0
	except SystemExit as se:
		return int(se.code) if se.code is not None else 0
	except Exception:
		import traceback
		traceback.print_exc()
		return 1


__all__ = ["main"]
