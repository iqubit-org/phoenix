#!/usr/bin/env python
"""
Usage: 
        python get_circ_info.py -f <circuit_file>
    or 
        python get_circ_info.py -d <circuit_dir>
"""
import os
import warnings
import argparse
from qiskit import QuantumCircuit
from rich.console import Console
from rich.table import Table
from collections import Counter
console = Console()

warnings.filterwarnings('ignore')


def get_circ_info(fname):
    """Get information of a quantum circuit from its qasm file."""

    qc = QuantumCircuit.from_qasm_file(fname)
    # dag = circuit_to_dag(qc)
    # layers = list(dag.layers())
    # used_qubits = [q.index for q in qc.qubits if q not in dag.idle_wires()]

    ops = [circ_instr.operation for circ_instr in qc.data]
    gate_names_count = dict(Counter([op.name for op in ops]))
    gate_names_count = dict(sorted(gate_names_count.items()))
    num_gates_count = dict(Counter([op.num_qubits for op in ops]))
    num_gates_count = dict(sorted(num_gates_count.items()))

    # use rich Table
    circ_name = fname.split('/')[-1].split('.')[0]
    table = Table(title=circ_name)
    table.add_column("num_qubits")
    table.add_column("num_gates")
    table.add_column("num_2q_gates")
    table.add_column("depth")
    table.add_column('depth_2q')
    table.add_row(str(qc.num_qubits), 
                  str(len(qc)), 
                  str(qc.num_nonlocal_gates()), 
                  str(qc.depth()),
                  str(qc.depth(lambda instr: instr.operation.num_qubits > 1)))
    console.print(table)

    # print(qc)
    console.print('gate names count: {}'.format(gate_names_count))
    console.print('gate numbers count: {}'.format(num_gates_count))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(prog='script.py')
    parser.add_argument('-f', '--file', type=str, help='circuit file name to get its info')
    parser.add_argument('-d', '--dir', type=str, help='directory of circuit files to get their info')
    args = parser.parse_args()

    if args.file:
        get_circ_info(args.file)
    elif args.dir:
        for fname in os.listdir(args.dir):
            if fname.endswith('.qasm'):
                get_circ_info(os.path.join(args.dir, fname))
                print()
    else:
        raise ValueError('Invalid input, please provide a file or directory.')
