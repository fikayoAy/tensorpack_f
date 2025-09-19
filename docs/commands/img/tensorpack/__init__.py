"""TensorPack package.

This package uses lazy imports for heavy submodules so that importing
lightweight entry points (like `tensorpack.cli`) does not pull in
optional or missing dependencies (for example `matrixtransformer`).

Access submodules with `import tensorpack.module` or via attribute
access `tensorpack.module` (PEP 562 lazy loading).
"""

__version__ = "0.1.2"

# Public submodules that can be imported lazily
__all__ = [
    "script",
    "license_manager",
    "matrixtransformer",
    "graph",
]

def __getattr__(name):
    """Lazy-import a submodule on attribute access.

    Example: `from tensorpack import license_manager` will import
    `tensorpack.license_manager` only when requested.
    """
    if name in __all__:
        import importlib
        mod = importlib.import_module(f"{__name__}.{name}")
        globals()[name] = mod
        return mod
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

def __dir__():
    return sorted(list(globals().keys()) + __all__)