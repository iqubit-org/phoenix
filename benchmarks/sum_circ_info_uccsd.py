"""
Summarize circuit information (e.g., # qubits, # gates, depth, etc.) of all benchmark programs
"""
import os
import json
import pandas as pd
from qiskit import QuantumCircuit
from natsort import natsorted
from rich.console import Console
import warnings

warnings.filterwarnings('ignore')

console = Console()

benchmark_dpath = './uccsd_qasm/'

qasm_fnames = natsorted(
    [os.path.join(benchmark_dpath, fname) for fname in os.listdir(benchmark_dpath) if fname.endswith('.qasm')])

description = pd.DataFrame(columns=['program', 'num_qubits',
                                    'num_paulis', 'max_pauli_weight',
                                    'num_gates', 'num_2q_gates', 'depth', 'depth_2q'])

for qasm_fname in qasm_fnames:
    json_fname = qasm_fname.replace('qasm', 'json')
    program_name = qasm_fname.split('/')[-1].replace('.qasm', '')
    qc = QuantumCircuit.from_qasm_file(qasm_fname)
    with open(json_fname, 'r') as f:
        data = json.load(f)

    pauli_weights = [len(pauli) - pauli.count('I') for pauli in data['paulis']]

    description = pd.concat([description, pd.DataFrame({
        'program': program_name,
        'num_qubits': qc.num_qubits,
        'num_paulis': len(data['paulis']),
        'max_pauli_weight': max(pauli_weights),
        'num_gates': qc.size(),
        'num_2q_gates': qc.num_nonlocal_gates(),
        'depth': qc.depth(),
        'depth_2q': qc.depth(lambda instr: instr.operation.num_qubits > 1)
    }, index=[0])], ignore_index=True)

console.print(description)

# with open('description_uccsd.txt', 'w') as f:
#     f.write(description.to_string())

# description.to_latex('description_uccsd.tex', index=False)

description.to_csv('description_uccsd.csv', index=False)
