from pathlib import Path
from setuptools import setup, Extension
import sys
import os
from typing import Union, Dict, List

try:
    from Cython.Build import cythonize
except Exception:  # pragma: no cover - runtime fallback
    cythonize = None


def discover_extensions(package_dir: Path) -> Dict[str, List[Extension]]:
    """Discover .pyx files and group them.

    Returns a mapping of group name -> list[Extension]. By default we classify
    anything named `license_manager` into the `license` group, and everything
    else into `core`.
    """
    groups: Dict[str, List[Extension]] = {"core": [], "license": []}
    pyx_files = sorted(package_dir.glob("*.pyx"))
    for p in pyx_files:
        # Construct a stable package module name. Use the package folder
        # `tensorpack` as the top-level package so Cython receives a valid
        # module name (avoids cases where package_dir.name could be '.' or
        # otherwise produce a leading dot).
        mod_name = f"tensorpack.{p.stem}"
        ext = Extension(mod_name, [str(p)])
        if p.stem == "license_manager":
            groups.setdefault("license", []).append(ext)
        else:
            groups.setdefault("core", []).append(ext)
    return groups


def add_numpy_include_dirs(ext_list: List[Extension]):
    """If numpy is available, append its include dirs to Extension objects.

    This is executed at import time during CI where numpy gets installed
    before running the build step. If numpy is not importable, this is a no-op.
    """
    try:
        import numpy
        inc = numpy.get_include()
    except Exception:
        return
    for e in ext_list:
        # ensure include_dirs exists and append numpy include
        if getattr(e, 'include_dirs', None) is None:
            e.include_dirs = [inc]
        else:
            e.include_dirs.append(inc)


def parse_groups_arg(argv: List[str]) -> Union[List[str], None]:
    """Look for --groups=core,license or read BUILD_GROUPS env var.

    Returns list of groups to build or None to build all.
    """
    # Check env var first
    env = os.environ.get("BUILD_GROUPS")
    if env:
        return [g.strip() for g in env.split(",") if g.strip()]

    # Look for a --groups=... arg and remove it from argv so setuptools isn't confused
    for i, a in enumerate(list(argv)):
        if a.startswith("--groups="):
            val = a.split("=", 1)[1]
            # remove the custom arg
            del argv[i]
            return [g.strip() for g in val.split(",") if g.strip()]
    return None


HERE = Path(__file__).parent

# Create package directory if it doesn't exist
PKG_DIR = HERE / "tensorpack"
PKG_DIR.mkdir(exist_ok=True)
PKG_INIT = PKG_DIR / "__init__.py"
if not PKG_INIT.exists():
    PKG_INIT.write_text("")

# Source directory is the same as HERE since .pyx files are in root
TP_DIR = HERE

groups_map = discover_extensions(TP_DIR)

# Determine which groups to build: None -> all
requested = parse_groups_arg(sys.argv)
if requested is None:
    selected_groups = list(groups_map.keys())
else:
    # validate groups
    invalid = [g for g in requested if g not in groups_map]
    if invalid:
        print(f"Invalid group(s): {invalid}. Available: {list(groups_map.keys())}")
        sys.exit(1)
    selected_groups = requested

# Flatten selected extensions
extensions = []
for g in selected_groups:
    extensions.extend(groups_map.get(g, []))

# If numpy is installed in the build environment, add its include dirs so
# C extensions can compile against numpy headers.
add_numpy_include_dirs(extensions)

if not extensions:
    print("No extension modules found for the selected groups. Nothing to build.")
    sys.exit(0)

if cythonize is not None:
    ext_modules = cythonize(
        extensions,
        compiler_directives={"language_level": "3", "boundscheck": False, "wraparound": False},
    )
else:
    # If Cython isn't available, setup() will try to compile from .c files if present.
    ext_modules = extensions

setup(
    name="tensorpack",
    version="0.1",
    description="Tensorpack project - compiled extensions",
    author="Fikayomi Ayodele",
    author_email="Ayodeleanjola4@gmail.com",
    url="https://github.com/fikayoAy/tensorpack",
    packages=["tensorpack"],
    py_modules=["tensorpack"],
    ext_modules=ext_modules,
    entry_points={
        'console_scripts': [
            'tensorpack=tensorpack.cli:main',
        ],
    },
    install_requires=[
        "numpy>=1.19.0",
        "pandas>=1.0.0",
        "scipy>=1.6.0",
        "torch>=1.8.0",
        "scikit-learn>=0.24.0",
    ],
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
)
