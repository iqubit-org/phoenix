"""
Phoenix is a Python package for compiling VQA programs whose high-level IR is Pauli strings with coefficients.
"""
from .basic import gates, circuits
from .basic import Gate, Circuit, QASMStringIO
from . import synthesis
from . import models
from . import utils
