#!/usr/bin/env python
"""
Benchmarking on UCCSD benchmarks with given "compiler" and given "topology".
"""

import sys

sys.path.append('..')
sys.path.append('../..')

import os
import json
import argparse
import pytket.qasm
import qiskit.qasm2
from phoenix import gates
from natsort import natsorted
import bench_utils

from rich.console import Console

console = Console()

INPUT_QASM_DPATH = '../benchmarks/uccsd_qasm'

assert os.path.exists(INPUT_QASM_DPATH)