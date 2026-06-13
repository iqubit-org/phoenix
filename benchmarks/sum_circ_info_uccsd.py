"""
Summarize circuit information (e.g., # qubits, # gates, depth, etc.) of all benchmark programs
"""

import os
import json
import qiskit
import pandas as pd
from phoenix import Hamiltonian
from natsort import natsorted
from rich.console import Console
import warnings

warnings.filterwarnings("ignore")

console = Console()

benchmark_dir = "./uccsd"

fnames = natsorted([os.path.join(benchmark_dir, fname) for fname in os.listdir(benchmark_dir)])

description = pd.DataFrame(
    columns=[
        "program",
        "num_qubits",
        "num_paulis",
        "max_pauli_weight",
        "num_gates",
        "num_2q_gates",
        "depth",
        "depth_2q",
    ]
)

for fname in fnames:
    program_name = os.path.basename(fname).replace(".json", "")
    with open(fname, "r") as f:
        data = json.load(f)

    data['paulis'] = [p[::-1] for p in data['paulis']]  # reverse the order of qubits to be consistent with qiskit
    ham = Hamiltonian(data["paulis"], data["coeffs"])
    qc = ham.generate_circuit()
    qc = qiskit.transpile(qc, basis_gates=["u", "cx"], optimization_level=0)

    description = pd.concat(
        [
            description,
            pd.DataFrame(
                {
                    "program": program_name,
                    "num_qubits": qc.num_qubits,
                    "num_paulis": len(data["paulis"]),
                    "max_pauli_weight": ham.max_weight,
                    "num_gates": qc.size(),
                    "num_2q_gates": qc.num_nonlocal_gates(),
                    "depth": qc.depth(),
                    "depth_2q": qc.depth(lambda instr: instr.operation.num_qubits > 1),
                },
                index=[0],
            ),
        ],
        ignore_index=True,
    )

console.print(description)

description.to_csv("description_uccsd.csv", index=False)
