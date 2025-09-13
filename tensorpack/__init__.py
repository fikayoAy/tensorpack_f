"""tensorpack package initializer.

Provide a package-level main() that delegates to the CLI shim or the compiled
script extension. This ensures `import tensorpack; tensorpack.main()` and
`python -m tensorpack` work after installation.
"""
from typing import Optional, Sequence


def main(argv: Optional[Sequence[str]] = None) -> int:
	"""Package-level entry point.

	Prefer a compiled `script.main` (extension) if present; otherwise fall
	back to the lightweight `cli.main` shim.
	"""
	# Try compiled extension first (may be a .pyd/.so)
	try:
		from . import script as _script
		main_fn = getattr(_script, 'main', None)
		if callable(main_fn):
			return int(main_fn(argv) if argv is not None else main_fn())
	except Exception:
		pass

	# Fallback to the CLI shim
	try:
		from .cli import main as _cli_main
		return int(_cli_main(argv) if argv is not None else _cli_main())
	except Exception:
		pass

	raise SystemExit("tensorpack package does not expose a main() entry point.")
