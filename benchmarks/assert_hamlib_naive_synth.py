"""Verify that the Pauli operators and QASM code are equivalent."""

import sys

sys.path.append("../")

import os
import json
from qiskit.quantum_info import Operator
from phoenix import Hamiltonian
from phoenix.utils import infidelity


benchmark_dir = "./hamlib"

categories = [
    "binaryoptimization",
    "discreteoptimization",
    "condensedmatter",
    "chemistry",
]
fnames = []
for dir in categories:
    fnames.extend(
        [
            os.path.join(benchmark_dir, dir, fname)
            for fname in os.listdir(os.path.join(benchmark_dir, dir))
        ]
    )

for fname in fnames:
    if not os.path.exists(fname):
        continue
    print()
    print("verifying {} ...".format(fname))

    with open(fname, "r") as f:
        data = json.load(f)


    if data["num_qubits"] >= 10:
        print("skipping due to large number of qubits")
        continue

    ham = Hamiltonian(data["paulis"], data["coeffs"])
    qc = ham.generate_circuit()
    if qc.size() > 10000:
        print("skipping due to large number of gates")
        continue

    u_ideal = ham.unitary_evolution()
    u_ref = Operator(qc).to_matrix() 
    print("infidelity (ideal v.s. ref):", infidelity(u_ideal, u_ref))
