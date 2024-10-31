"""
All-to-all compilation results to limited connectivity topology (Manhattan, Sycamore)
"""
import sys
import qiskit
import pytket
import pytket.qasm
import pytket.passes
import bench_utils


# TKet, PauliOpt, Phoenix

bench_utils.optimize_with_mapping()