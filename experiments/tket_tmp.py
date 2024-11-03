import qiskit
import pytket
import pytket.qasm
import pytket.passes
import networkx as nx


def tket_to_qsikit(circ: pytket.Circuit) -> qiskit.QuantumCircuit:
    return qiskit.QuantumCircuit.from_qasm_str(pytket.qasm.circuit_to_qasm_str(circ))


def unroll_u3(circ: pytket.Circuit) -> pytket.Circuit:
    import sys
    sys.path.append('..')

    from phoenix import Circuit
    from phoenix.transforms.circuit_pass import unroll_u3
    return unroll_u3(Circuit.from_tket(circ)).to_tket()


def adaptive_paulisimp(circ: pytket.Circuit) -> pytket.Circuit:
    circ_tmp = circ.copy()
    best_depth_2q = circ.depth_2q()
    best_num_2q_gates = circ.n_2qb_gates()
    while True:
        pytket.passes.PauliSimp().apply(circ_tmp)
        if best_depth_2q > circ_tmp.depth_2q() and best_num_2q_gates > circ_tmp.n_2qb_gates():
            best_depth_2q = circ_tmp.depth_2q()
            best_num_2q_gates = circ_tmp.n_2qb_gates()
            circ = circ_tmp.copy()
        else:
            break
    return circ


def tket_pass(circ: pytket.Circuit) -> pytket.Circuit:
    circ = circ.copy()

    # unroll U3
    circ = unroll_u3(circ)

    # adaptive PauliSimp
    circ = adaptive_paulisimp(circ)

    # full optimization
    pytket.passes.FullPeepholeOptimise().apply(circ)
    pytket.passes.RemoveRedundancies().apply(circ)

    return circ


circ = pytket.qasm.circuit_from_qasm('CH2_cmplt_BK_sto3g.qasm')

circ = unroll_u3(circ)
circ_opt = tket_pass(circ)

# qubit mapping on Manhattan
qc = tket_to_qsikit(circ_opt)
manhattan = nx.read_graphml('manhattan.graphml')
manhattan = nx.relabel_nodes(manhattan, {node: int(node) for node in manhattan.nodes})
coupling_edges = [list(edge) for edge in manhattan.edges]
qc_mapped = qiskit.transpile(qc, basis_gates=['u1', 'u2', 'u3', 'cx'], optimization_level=3,
                             coupling_map=coupling_edges, layout_method='sabre')

print(circ.n_gates, circ_opt.n_gates, qc_mapped.size())
print(circ.n_2qb_gates(), circ_opt.n_2qb_gates(), qc_mapped.num_nonlocal_gates())
print(circ.depth(), circ_opt.depth(), qc_mapped.depth())
print(circ.depth_2q(), circ_opt.depth_2q(), qc_mapped.depth(lambda instr: instr.operation.num_qubits > 1))
