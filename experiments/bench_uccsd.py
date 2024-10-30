"""
Benchmarking on chemistry benchmarks from TKet_Benchmarking.
"""

import sys

sys.path.append('..')

import os
import json
import argparse
import rustworkx as rx
from phoenix.models import HamiltonianModel
from phoenix.utils import arch
from phoenix import transforms
from phoenix import Circuit
from phoenix.utils import  arch

BENCHMARK_DPATH = '../benchmarks/uccsd_json'
OUTPUT_DPATH = './output_chem/phoenix'

parser = argparse.ArgumentParser(description='Benchmarking on UCCSD chemistry benchmarks')
parser.add_argument('-d', '--device', default='all2all', type=str,
                    help='Device topology (default: all2all, hex)')
args = parser.parse_args()


def phoenix_pass(ham: HamiltonianModel, device: rx.rustworkx = None) -> Circuit:
    # topology-agnostic synthesis
    circ = ham.reconfigure_and_generate_circuit()
    if device is None:
        return circ

    # hardware mapping
    circ, _, _ = transforms.circuit_pass.sabre_by_qiskit(circ, device)
    circ = transforms.circuit_pass.phys_circ_opt_by_qiskit(circ)
    return circ


for json_fname in [fname for fname in os.listdir(BENCHMARK_DPATH) if fname.endswith('.json')]:
    with open(os.path.join(BENCHMARK_DPATH, json_fname), 'r') as f:
        data = json.load(f)

    output_fname = json_fname.replace('.json', '.qasm')

    # TODO: delete this line
    if os.path.exists(os.path.join(OUTPUT_DPATH, args.device, output_fname)):
        continue

    if args.device == 'all2all':
        device = None
    elif args.device == 'chain':
        device = arch.gene_chain_1d_graph(data['num_qubits'])
    elif args.device == 'grid':
        device = arch.gene_grid_2d_graph(data['num_qubits'])
    else:
        raise ValueError('Invalid device topology (all2all, chain, grid)')

    ham = HamiltonianModel(data['paulis'], data['coeffs'])
    circ = phoenix_pass(ham, device)
    circ.to_qasm(os.path.join(OUTPUT_DPATH, args.device, output_fname))
    print(f'Output file: {output_fname}')
