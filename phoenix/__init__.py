"""
Phoenix is a Python package for compiling VQA programs whose high-level IR is Pauli strings with coefficients.
"""
from .basic import gates, circuits
from .basic.gates import Gate
from .basic.circuits import Circuit, QASMStringIO
from . import synthesis
from . import models
from . import utils
