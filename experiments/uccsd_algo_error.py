import sys

sys.path.append('..')

import argparse

import os
import json
import numpy as np
import cirq
from cirq.contrib.qasm_import import circuit_from_qasm
from phoenix import Circuit, gates
from phoenix import models
from natsort import natsorted
from phoenix.utils.ops import tensor_slots
from phoenix.utils.functions import infidelity

benchmark_dpath = '../benchmarks/uccsd_json'
output_dpath = './output_uccsd/'

parser = argparse.ArgumentParser(prog='Summarize compilation results (gate count and circuit depth statistics)')
parser.add_argument('-c', '--compiler', type=str, help='Compiler name')
args = parser.parse_args()


def ideal_evolution(json_fname):
    X = gates.X.data
    with open(json_fname, 'r') as f:
        data = json.load(f)
    ham = models.HamiltonianModel(data['paulis'], data['coeffs'])
    u = ham.unitary_evolution()
    u = u @ tensor_slots(cirq.kron(*[X for _ in data['front_x_on']]), ham.num_qubits, data['front_x_on'])

    return u


output_dpath = os.path.join(output_dpath, args.compiler, 'all2all_opt')
for fname in natsorted(os.listdir(output_dpath)):


    json_fname = os.path.join(benchmark_dpath, fname.replace('.qasm', '.json'))
    qasm_fname = os.path.join(output_dpath, fname)

    with open(qasm_fname, 'r') as f:
        qasm_str = f.read()
        circ = circuit_from_qasm(qasm_str)

    if cirq.num_qubits(circ) > 10:
        continue

    print(fname)

    print('{} infidelity {:.5f}'.format(fname, infidelity(
        ideal_evolution(json_fname),
        cirq.unitary(circ)
    )))

    # with open(json_fname, 'r') as f:
    #     data = json.load(f)
    # ham = models.HamiltonianModel(data['paulis'], data['coeffs'])
    # ham.unitary_evolution()
