from qiskit import QuantumCircuit

from bench_utils import *

qasm_fname = '../../benchmarks/uccsd_qasm/CH2_cmplt_BK_sto3g.qasm'

# circ = pytket.qasm.circuit_from_qasm(qasm_fname)
# circ_opt = tket_pass(circ)
#
# print(circ.n_2qb_gates(), circ_opt.n_2qb_gates())
# print(circ.depth_2q(), circ_opt.depth_2q())
#
# circ_opt_mapped = optimize_with_mapping(tket_to_qiskit(circ_opt),
#                                         Sycamore)
# circ_opt_mapped = qiskit_to_tket(circ_opt_mapped)
#
# print(circ.n_2qb_gates(), circ_opt.n_2qb_gates(), circ_opt_mapped.n_2qb_gates())
# print(circ.depth_2q(), circ_opt.depth_2q(), circ_opt_mapped.depth_2q())

circ = QuantumCircuit.from_qasm_file(qasm_fname)

circ = qiskit.transpile(circ,
                        basis_gates=['u3', 'cx'],
                        coupling_map=Sycamore,
                        optimization_level=3)
print(circ.num_nonlocal_gates(), circ.depth(
    lambda instr: instr.operation.num_qubits > 1
))
