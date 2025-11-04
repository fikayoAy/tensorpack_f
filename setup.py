from pathlib import Path
from setuptools import setup, find_packages
import sys
import os
from typing import Union, Dict, List

# Enforce Python 3.9
if sys.version_info < (3, 9) or sys.version_info >= (3, 10):
    raise RuntimeError("This package requires Python 3.9.x")

# No Cython imports - using pure Python

def discover_python_modules(package_dir: Path) -> List[str]:
    """Discover .py files.
    
    Returns a list of Python module files to include.
    """
    modules = []
    py_files = sorted(package_dir.glob("*.py"))
    for p in py_files:
        if p.stem != "__init__":  # Skip __init__.py
            modules.append(p.stem)
    return modules


# Remove the numpy include dirs function since we're not using extensions
# Replace with a function to find Python packages

def find_python_packages(package_dir: Path) -> List[str]:
    """Find all Python packages (directories with __init__.py)."""
    packages = []
    for p in package_dir.glob("**/__init__.py"):
        pkg_path = p.parent.relative_to(package_dir.parent)
        packages.append(str(pkg_path).replace('\\', '.').replace('/', '.'))
    return packages


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

# Determine where the runtime modules live.
# Some checkouts place the runtime .py files directly in the project root;
# others place them inside a `tensorpack/` subdirectory. Detect both layouts
# and configure setuptools `packages` and `package_dir` accordingly.
SUBDIR = HERE / "tensorpack"
if SUBDIR.exists() and any(p.suffix == '.py' for p in SUBDIR.glob('*.py')):
    # Standard layout: `tensorpack/` contains the modules
    TP_DIR = SUBDIR
    packages = find_packages(where=str(HERE)) or ["tensorpack"]
    package_dir = None
else:
    # Flat layout: runtime modules are in the project root. Map the package
    # name `tensorpack` to the project root so files are packaged under that
    # package name.
    TP_DIR = HERE
    packages = ["tensorpack"]
    package_dir = {"tensorpack": str(HERE)}

# Get Python modules for debug output
python_modules = discover_python_modules(TP_DIR)
print(f"Found Python modules: {python_modules} (source dir: {TP_DIR})")

# No extensions, pure Python only
setup(
    name="tensorpack-f",
    version="0.1.8",
    description="TensorPack project - pure Python implementation",
    author="Fikayomi Ayodele",
    author_email="Ayodeleanjola4@gmail.com",
    url="https://github.com/fikayoAy/tensorpack",
    packages=packages,
    package_dir=package_dir if package_dir is not None else None,
    include_package_data=True,
    package_data={
        "tensorpack": ["__pycache__/*.pyc"],
    },
    exclude_package_data={
        "tensorpack": ["*.py"],
    },
    entry_points={
        'console_scripts': [
            'tensorpack=tensorpack.script:main',
        ],
    },
    install_requires=[
        "numpy>=1.19.0",
        "pandas>=1.0.0",
        "scipy>=1.6.0",
        "torch>=1.8.0",
        "scikit-learn>=0.24.0",
        "matplotlib>=3.3.0",
        "seaborn>=0.11.0",
        "pillow>=8.0.0",
        "opencv-python-headless>=4.5.0",
        "pyarrow>=5.0.0",
        "h5py>=3.1.0",
        "netCDF4>=1.5.6",
        "tifffile>=2021.4.8",
        "nibabel>=3.2.1",
        "zarr>=2.8.0",
        "requests>=2.25.0",
        "cryptography>=3.4.0",
        
        "numba>=0.53.0",
        "joblib>=1.0.0",
        "networkx>=2.5.0",
        "tqdm>=4.60.0",
        "openpyxl>=3.0.0",
        "xlwt>=1.3.0",
        "transformers>=4.5.0",
        "jsonschema>=3.2.0",
        "sqlalchemy>=1.4.0",
        "psutil>=5.8.0",
        "python-dateutil>=2.8.1",
        "pandas-flavor>=0.2.0",
        "pyyaml>=5.4.0",
        "lxml>=4.6.0",
        "scikit-image>=0.18.0",
        "fastparquet>=0.7.0",
        "regex>=2021.4.4",
        "fonttools>=4.22.0",
        "Flask>=2.0.0",
        "rich>=13.0.0",
        "PyJWT>=2.0.0",
    ],
    python_requires="==3.9.*",
)
