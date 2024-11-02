"""
Benchmarking on chemistry benchmarks from TKet_Benchmarking.
"""

import os
import qiskit
import networkx as nx
from qiskit import QuantumCircuit
import warnings

import sys


sys.path.append('..')
from phoenix.utils.functions import infidelity

from phoenix import Circuit
from phoenix.utils import arch


warnings.filterwarnings('ignore')

# qasm_fname = os.path.join('./output_chem/phoenix', 'all2all', 'CH2_cmplt_BK_sto3g.qasm')

chem_name = 'CH2_cmplt_BK_sto3g.qasm'
# chem_name = 'CH2_frz_BK_sto3g.qasm'
# chem_name = 'LiH_frz_P_sto3g.qasm'

qasm_fname = chem_name
qasm_fname_mapped = 'mapped_' + chem_name

manhattan = nx.read_graphml('topo_config/manhattan.graphml')

# relabel str node to int node
mapping = {node: int(node) for node in manhattan.nodes}
manhattan = nx.relabel_nodes(manhattan, mapping)


# manhattan = nx.grid_2d_graph(3, 4)

# manhattan = manhattan.to_directed()

coupling_edges = list(manhattan.edges)
coupling_edges = [list(edge) for edge in coupling_edges]

print(list(coupling_edges))

qc = QuantumCircuit.from_qasm_file(qasm_fname)
qc_mapped = qiskit.transpile(qc, basis_gates=['u1', 'u2', 'u3', 'cx'], optimization_level=3,
                             coupling_map=coupling_edges, layout_method='sabre')

def qc_to_unitary(qc: QuantumCircuit):
    from qiskit.quantum_info import Operator
    return Operator(qc.reverse_bits()).to_matrix()

# print(infidelity(qc_to_unitary(qc), qc_to_unitary(qc_mapped)))

qc_mapped.qasm(filename=qasm_fname_mapped)

print('#gates', qc.size(), qc_mapped.size())
print('#depth', qc.depth(), qc_mapped.depth())
print('#2Q', qc.num_nonlocal_gates(), qc_mapped.num_nonlocal_gates())
print('#2Q-depth', qc.depth(lambda instr: instr.operation.num_qubits > 1),
      qc_mapped.depth(lambda instr: instr.operation.num_qubits > 1))


print(qc_mapped.layout.initial_index_layout())


circ = Circuit.from_qiskit(qc)
circ_mapped = Circuit.from_qiskit(qc_mapped)
init_mapping_inv = {i:j for i,j in zip(qc_mapped.layout.initial_index_layout(), range(qc_mapped.num_qubits))}
final_mapping_inv = {i: j for i, j in zip(qc_mapped.layout.final_index_layout(), range(qc_mapped.num_qubits))}
circ_mapped = circ_mapped.rewire(init_mapping_inv)
init_mapping = {j: i for i, j in init_mapping_inv.items()}
final_mapping = {j: i for i, j in final_mapping_inv.items()}


print(arch.verify_mapped_circuit(circ, circ_mapped, init_mapping, final_mapping))




# print(circ_mapped.num_qubits, circ_mapped.num_nonlocal_gates, circ_mapped.gate_stats())
# print(circ_mapped.qubits)
#
#
# print(qc.num_qubits, qc_mapped.num_qubits)