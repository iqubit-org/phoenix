import os
import cirq
import qiskit
import pytket
import pytket.qasm
import pytket.passes
import qiskit.qasm2
from natsort import natsorted
from qiskit.transpiler import CouplingMap

import bench_utils

input_dpath = './output_uccsd/phoenix/all2all'


def qiskit_post_optimize(circ: qiskit.QuantumCircuit) -> qiskit.QuantumCircuit:
    circ = qiskit.transpile(circ, optimization_level=3,
                            basis_gates=['u1', 'u2', 'u3', 'cx'],
                            coupling_map=CouplingMap(bench_utils.Manhattan_coupling.edge_list()),
                            layout_method='sabre')
    return circ


def tket_post_optimize(circ: pytket.Circuit) -> pytket.Circuit:
    circ = circ.copy()

    pytket.passes.FullPeepholeOptimise().apply(circ)
    pytket.passes.RemoveRedundancies().apply(circ)

    circ = bench_utils.tket_to_qiskit(circ)
    circ = qiskit.transpile(circ, optimization_level=3,
                            basis_gates=['u1', 'u2', 'u3', 'cx'],
                            coupling_map=CouplingMap(bench_utils.Manhattan_coupling.edge_list()),
                            layout_method='sabre')

    # def sabre_by_qiskit(circ: Circuit, device: rx.PyGraph,
    #                     seed: int = None) -> Tuple[Circuit, Dict[int, int], Dict[int, int]]:
    #     """The running efficiency of qiskit.transpiler.passes.SabreLayout is better than what we implemented"""
    #     from qiskit.transpiler import passes, PassManager, CouplingMap
    #
    #     circ = circ.to_qiskit()
    #     pm = PassManager([passes.SabreLayout(CouplingMap(device.edge_list()), seed=seed)])
    #     circ = pm.run(circ)
    #     init_mapping_inv = {i: j for i, j in zip(circ.layout.initial_index_layout(), range(circ.num_qubits))}
    #     final_mapping_inv = {i: j for i, j in zip(circ.layout.final_index_layout(), range(circ.num_qubits))}
    #     init_mapping = {j: i for i, j in init_mapping_inv.items()}
    #     final_mapping = {j: i for i, j in final_mapping_inv.items()}
    #     circ = Circuit.from_qiskit(circ).rewire(init_mapping_inv)
    #     return circ, init_mapping, final_mapping
    #

    circ = bench_utils.qiskit_to_tket(circ)
    pytket.passes.FullPeepholeOptimise(allow_swaps=False).apply(circ)
    pytket.passes.RemoveRedundancies().apply(circ)

    return circ


for fname in natsorted(os.listdir(input_dpath)):
    fname = os.path.join(input_dpath, fname)
    print('Processing', fname)

    circ_qiskit = qiskit.QuantumCircuit.from_qasm_file(fname)
    circ_tket = pytket.qasm.circuit_from_qasm(fname)
    circ_qiskit_opt = qiskit_post_optimize(circ_qiskit)
    circ_tket_opt = tket_post_optimize(circ_tket)

    qiskit.qasm2.dump(circ_qiskit_opt, os.path.join('qiskit_post_opt_manhattan', fname.split('/')[-1]))
    pytket.qasm.circuit_to_qasm(circ_tket_opt, os.path.join('tket_post_opt_manhattan', fname.split('/')[-1]))
