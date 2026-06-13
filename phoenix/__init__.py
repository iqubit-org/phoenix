from . import primitive, utils
from .basics import CNOTEquivCliffordGate, fSwapEquivCliffordGate
from .compiler import compile_hamiltonian_simulation
from .hamiltonian import Hamiltonian
from .utils import post_transpile

# Backwards-compatible alias for the former name (now in ``phoenix.utils``).
optimize_phoenix_circuit_by_qiskit = post_transpile

# Resolved by setuptools-scm at build time (see `pyproject.toml`).
# The generated `_version.py` is installed with the package; in a source
# checkout without a built version, we fall back to installed metadata.
try:
    from ._version import __version__
except ImportError:
    from importlib.metadata import PackageNotFoundError, version

    try:
        __version__ = version("phoenix-quantum")
    except PackageNotFoundError:
        __version__ = "0.0.0+unknown"

__all__ = [
    "__version__",
    "Hamiltonian",
    "compile_hamiltonian_simulation",
    "post_transpile",
    "optimize_phoenix_circuit_by_qiskit",
    "CNOTEquivCliffordGate",
    "fSwapEquivCliffordGate",
]
