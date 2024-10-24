"""
Test the correctness of some basic circuit properties.
"""
import os

import pytket
import qiskit
import pytket.qasm

import sys

sys.path.append('../..')

from phoenix import Circuit


def test_circ_prop_wrt_tket_cx():
    benchmark_dpath = '../input/cx-basis/'
    qasm_fnames = []
    for dir in os.listdir(benchmark_dpath):
        qasm_fnames.extend(
            [os.path.join(benchmark_dpath, dir, fname) for fname in os.listdir(os.path.join(benchmark_dpath, dir)) if
             fname.endswith('.qasm')])

    for fname in qasm_fnames:
        circ_tket = pytket.qasm.circuit_from_qasm(fname)
        if circ_tket.n_2qb_gates() > 2000:
            continue
        circ = Circuit.from_qasm(fname=fname)
        assert circ.num_gates == circ_tket.n_gates, "num_gates not equal for {}".format(fname)
        assert circ.num_nonlocal_gates == circ_tket.n_2qb_gates(), "num_nonlocal_gates not equal for {}".format(fname)
        assert circ.depth == circ_tket.depth(), "depth not equal for {}".format(fname)
        assert circ.depth_nonlocal == circ_tket.depth_2q(), "depth_nonlocal not equal for {}".format(fname)


def test_circ_prop_wrt_tket_su4():
    benchmark_dpath = '../input/su4-basis/'

    qasm_fnames = []
    for dir in os.listdir(benchmark_dpath):
        qasm_fnames.extend(
            [os.path.join(benchmark_dpath, dir, fname) for fname in os.listdir(os.path.join(benchmark_dpath, dir)) if
             fname.endswith('.qasm')])

    for fname in qasm_fnames:
        circ_tket = pytket.qasm.circuit_from_qasm(fname)
        if circ_tket.n_2qb_gates() > 2000:
            continue
        circ = Circuit.from_qasm(fname=fname)
        assert circ.num_gates == circ_tket.n_gates, "num_gates not equal for {}".format(fname)
        assert circ.num_nonlocal_gates == circ_tket.n_2qb_gates(), "num_nonlocal_gates not equal for {}".format(fname)
        assert circ.depth == circ_tket.depth(), "depth not equal for {}".format(fname)
        assert circ.depth_nonlocal == circ_tket.depth_2q(), "depth_nonlocal not equal for {}".format(fname)


def test_circ_prop_wrt_qiskit_cx():
    circ_files_dpath = '../input/cx-basis/'

    qasm_fnames = []
    for dir in os.listdir(circ_files_dpath):
        qasm_fnames.extend(
            [os.path.join(circ_files_dpath, dir, fname) for fname in os.listdir(os.path.join(circ_files_dpath, dir)) if
             fname.endswith('.qasm')])

    filter_nonlocal = lambda instr: instr.operation.num_qubits > 1

    for fname in qasm_fnames:
        circ_qiskit = qiskit.QuantumCircuit.from_qasm_file(fname)
        if circ_qiskit.depth(filter_nonlocal) > 2000:
            continue
        circ = Circuit.from_qasm(fname=fname)
        assert circ.num_gates == circ_qiskit.size(), "num_gates not equal for {}".format(fname)
        assert circ.num_nonlocal_gates == circ_qiskit.num_nonlocal_gates(), "num_nonlocal_gates not equal for {}".format(
            fname)
        assert circ.depth == circ_qiskit.depth(), "depth not equal for {}".format(fname)
        assert circ.depth_nonlocal == circ_qiskit.depth(filter_nonlocal), "depth_nonlocal not equal for {}".format(
            fname)


def test_circ_prop_wrt_qiskit_su4():
    benchmark_dpath = '../input/su4-basis/'

    qasm_fnames = []
    for dir in os.listdir(benchmark_dpath):
        qasm_fnames.extend(
            [os.path.join(benchmark_dpath, dir, fname) for fname in os.listdir(os.path.join(benchmark_dpath, dir)) if
             fname.endswith('.qasm')])

    filter_nonlocal = lambda instr: instr.operation.num_qubits > 1

    for fname in qasm_fnames:
        circ_qiskit = qiskit.QuantumCircuit.from_qasm_file(fname)
        if circ_qiskit.depth(filter_nonlocal) > 2000:
            continue
        circ = Circuit.from_qasm(fname=fname)
        assert circ.num_gates == circ_qiskit.size(), "num_gates not equal for {}".format(fname)
        assert circ.num_nonlocal_gates == circ_qiskit.num_nonlocal_gates(), "num_nonlocal_gates not equal for {}".format(
            fname)
        assert circ.depth == circ_qiskit.depth(), "depth not equal for {}".format(fname)
        assert circ.depth_nonlocal == circ_qiskit.depth(filter_nonlocal), "depth_nonlocal not equal for {}".format(
            fname)


test_circ_prop_wrt_qiskit_cx()
test_circ_prop_wrt_qiskit_su4()
test_circ_prop_wrt_tket_cx()
test_circ_prop_wrt_tket_su4()
