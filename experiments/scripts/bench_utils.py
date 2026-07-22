"""
Benchmarking utilities
"""

import sys

sys.path.append("../..")

import qiskit
import numpy as np
import rustworkx as rx
from typing import Tuple, List
from qiskit.transpiler import CouplingMap

import phoenix

from tetris.utils.hardware import pGraph
from tetris.benchmark.mypauli import pauliString

from rich.console import Console

console = Console()


def naive_pass(
    paulis: List[str],
    coeffs: List[float],
    coupling_map: CouplingMap = None,
) -> qiskit.QuantumCircuit:
    """Naive compilation: the raw Trotter circuit of the input program."""
    paulis = [p[::-1] for p in paulis]
    qc = phoenix.Hamiltonian(paulis, coeffs).generate_circuit()
    if not (coupling_map is None or phoenix.utils.is_all2all_coupling_map(coupling_map)):
        qc = phoenix.utils.optimize_with_mapping(qc, coupling_map)
    # Pure basis translation (NO optimization): every compiled output is
    # counted in the cx/1q basis, so the reference must be too — counting raw
    # rzz/rxx gates as single 2q gates would skew every opt rate by ~2x.
    qc = qiskit.transpile(qc, basis_gates=["u", "cx"], optimization_level=0)
    return qc

def phoenix_pass(
    paulis: List[str],
    coeffs: List[float],
    grouping: str | None = None,
    coupling_map: CouplingMap = None,
    optimize: bool = True
) -> qiskit.QuantumCircuit:
    """Phoenix's high-level optimization.

    ``grouping`` accepts the public modes (None/'holistic'/'support').
    """
    paulis = [
        p[::-1] for p in paulis
    ]  # ! PHOENIX uses little-endian convention for Pauli strings, reverse the input strings here
    ham = phoenix.Hamiltonian(paulis, coeffs)
    qc = phoenix.compile_hamiltonian_simulation(ham, grouping=grouping, optimize=optimize)

    if not (coupling_map is None or phoenix.utils.is_all2all_coupling_map(coupling_map)):
        qc = phoenix.utils.optimize_with_mapping(qc, coupling_map)

    return qc


def paulihedral_pass(
    paulis: List[str], coeffs: List[float], coupling_map: CouplingMap = None
) -> qiskit.QuantumCircuit:
    from tetris.utils.parallel_bl import gate_count_oriented_scheduling
    from tetris.synthesis_SC import block_opt_SC

    if coupling_map is None:
        coupling_map = phoenix.utils.gene_all2all_coupling_map(len(paulis[0]))

    a2 = gate_count_oriented_scheduling(constr_mypauli_blocks(paulis, coeffs))

    qc, total_swaps, total_cx = block_opt_SC(a2, graph=coupling_map_to_pGraph(coupling_map))
    # qc = qiskit.transpile(qc, optimization_level=2, basis_gates=["u1", "u2", "u3", "cx"])

    if coupling_map is None or phoenix.utils.is_all2all_coupling_map(coupling_map):
        qc = phoenix.utils.post_transpile(qc)
    else:
        qc = phoenix.utils.optimize_with_mapping(qc, coupling_map)

    # console.print({
    #     'PH_swap_count': total_swaps,
    #     'PH_cx_count': total_cx,
    #     'CNOT': circ.num_nonlocal_gates(),
    #     'Single': circ.size() - circ.num_nonlocal_gates(),
    #     'Total': circ.size(),
    #     'Depth': circ.depth()})

    return qc


def tetris_pass(
    paulis: List[str], coeffs: List[float], coupling_map: CouplingMap = None
) -> qiskit.QuantumCircuit:
    from tetris.utils.synthesis_lookahead import synthesis_lookahead

    if coupling_map is None:
        coupling_map = phoenix.utils.gene_all2all_coupling_map(len(paulis[0]))

    qc, metrics = synthesis_lookahead(
        constr_mypauli_blocks(paulis, coeffs),
        graph=coupling_map_to_pGraph(coupling_map),
        use_bridge=False,
        swap_coefficient=3,
        k=10,
    )

    if coupling_map is None or phoenix.utils.is_all2all_coupling_map(coupling_map):
        qc = phoenix.utils.post_transpile(qc)
    else:
        qc = phoenix.utils.optimize_with_mapping(qc, coupling_map)

    metrics.update(
        {
            "CNOT": qc.num_nonlocal_gates(),
            "Single": qc.size() - qc.num_nonlocal_gates(),
            "Total": qc.size(),
            "Depth": qc.depth(),
        }
    )
    # console.print(metrics)

    return qc


def quclear_pass(
    paulis: List[str], coeffs: List[float], coupling_map: CouplingMap = None
) -> qiskit.QuantumCircuit:
    from qiskit.quantum_info import Clifford
    from qiskit.synthesis import synth_clifford_greedy
    from quclear.CE_module import CE_recur_tree

    params = np.array(coeffs).real * 2.0
    qc, append_clifford, _ = CE_recur_tree(entanglers=paulis, params=params, barrier=False)
    # append_clifford = append_clifford.decompose("swap")
    append_clifford = synth_clifford_greedy(Clifford(append_clifford))
    qc.compose(append_clifford, inplace=True)
    qc = phoenix.utils.post_transpile(qc)

    if not (coupling_map is None or phoenix.utils.is_all2all_coupling_map(coupling_map)):
        qc = phoenix.utils.optimize_with_mapping(qc, coupling_map)

    return qc


def pauliopt_pass(
    paulis: List[str],
    coeffs: List[float],
    method: str = "steiner_gray",
    coupling_map: CouplingMap = None,
) -> qiskit.QuantumCircuit:
    """PauliOpt synthesis, architecture-aware when ``coupling_map`` is limited.

    Methods (all three take a ``Topology`` natively):
      - ``steiner_gray``       — Steiner-tree based greedy synthesis.
                                 Default; best on limited topologies.
                                 See Goubault de Brugière et al. 2024
                                 (arXiv:2404.03280) and the pauliopt
                                 comparison paper (arXiv:2306.15601).
      - ``annealing``          — Simulated-annealing refinement.
                                 Longer runtime; sensitive to schedule
                                 / nr_iterations (pauliopt default is
                                 only 100 iters — often not enough).
                                 See arXiv:2206.11839.
      - ``divide_and_conquer`` — Recursive block split synthesis.
                                 Newer, experimental.

    Topology handling:
      When ``coupling_map`` encodes a **limited** connectivity, we build
      the pauliopt ``Topology`` directly from it and pass it into the
      synthesizer, so pauliopt does its own architecture-aware routing.
      Only post-optimisation (gate cancellation / 2q resynthesis) is
      then delegated to Qiskit. Previously we forced all-to-all inside
      pauliopt and routed later with Qiskit SABRE, which negated the
      whole point of pauliopt's Steiner-tree machinery.
    """
    from pauliopt.pauli.pauli_polynomial import PauliPolynomial
    from pauliopt.pauli.pauli_gadget import PPhase
    from pauliopt.pauli_strings import I, X, Y, Z
    from pauliopt.topologies import Topology
    from pauliopt.pauli.synthesis.annealing import annealing_synthesis
    from pauliopt.pauli.synthesis.steiner_gray_synthesis import pauli_polynomial_steiner_gray_clifford
    from pauliopt.pauli.synthesis.synthesis_divide_and_conquer import synthesis_divide_and_conquer

    def apply_permutation(qc: qiskit.QuantumCircuit, permutation: list) -> qiskit.QuantumCircuit:
        """Apply a permutation to a qiskit quantum circuit."""
        register = qc.qregs[0]
        qc_out = qiskit.QuantumCircuit(register)
        for instruction in qc:
            op_qubits = [register[permutation[register.index(q)]] for q in instruction.qubits]
            qc_out.append(instruction.operation, op_qubits)
        return qc_out

    n = len(paulis[0])
    all2all = coupling_map is None or phoenix.utils.is_all2all_coupling_map(coupling_map)

    # Build a pauliopt Topology matching the target connectivity.
    if all2all:
        topology = Topology.complete(n)
    else:
        # qiskit CouplingMap is directed; Topology treats couplings as undirected
        topology = Topology(coupling_map.size(), list(coupling_map.get_edges()))

    # Coerce coeffs to real floats: Qiskit's SparsePauliOp exposes complex128
    # coeffs even when imag is 0, but pauliopt -> qiskit's `rz(angle)` rejects
    # complex params.
    coeffs_real = np.asarray(coeffs)
    if np.iscomplexobj(coeffs_real):
        coeffs_real = np.real_if_close(coeffs_real, tol=1e8)
        if np.iscomplexobj(coeffs_real):
            raise ValueError("pauliopt requires real coefficients; got non-trivial complex")
    coeffs_real = coeffs_real.astype(float, copy=False)

    pauli_str_map = {"I": I, "X": X, "Y": Y, "Z": Z}
    pp = PauliPolynomial(num_qubits=n)
    for pauli_str, coeff in zip(paulis, coeffs_real):
        pp >>= PPhase(float(coeff) * 2) @ [pauli_str_map[p] for p in pauli_str]

    if method == "annealing":
        qc = annealing_synthesis(pp.copy(), topology).to_qiskit()
    elif method == "steiner_gray":
        qc, _gadget_perm, perm = pauli_polynomial_steiner_gray_clifford(pp.copy(), topology)
        qc = apply_permutation(qc.to_qiskit(), perm)
    elif method == "divide_and_conquer":
        qc, perm = synthesis_divide_and_conquer(pp.copy(), topology)
        qc = apply_permutation(qc.to_qiskit(), perm)
    else:
        raise ValueError(f"Unknown method: {method}")

    # Post-optimisation. For limited topology the circuit is already
    # architecture-aware (all 2q gates live on coupling-map edges); Qiskit
    # SABRE inside `optimize_with_mapping` should find zero additional
    # SWAPs and only run the cancellation / resynthesis passes.
    if all2all:
        qc = phoenix.utils.post_transpile(qc)
    else:
        qc = phoenix.utils.optimize_with_mapping(qc, coupling_map)

    return qc


def tket_pass(
    paulis: List[str], coeffs: List[float], greedy: bool = True, coupling_map: CouplingMap = None, optimize: bool = True
) -> qiskit.QuantumCircuit:
    qc = phoenix.utils.compile_by_tket(paulis, coeffs, greedy=greedy, little_endian=False, optimize=optimize)
    if not (coupling_map is None or phoenix.utils.is_all2all_coupling_map(coupling_map)):
        qc = phoenix.utils.optimize_with_mapping(qc, coupling_map)
    return qc


def qiskit_pass(
    paulis: List[str], coeffs: List[float], coupling_map: CouplingMap = None, optimize: bool = True
) -> qiskit.QuantumCircuit:
    qc = phoenix.utils.compile_by_qiskit(paulis, coeffs, little_endian=False, optimize=optimize)
    if not (coupling_map is None or phoenix.utils.is_all2all_coupling_map(coupling_map)):
        qc = phoenix.utils.optimize_with_mapping(qc, coupling_map)
    return qc


def paulirl_pass(
    paulis: List[str], coeffs: List[float], coupling_map: CouplingMap = None
) -> qiskit.QuantumCircuit:
    """https://quantum.cloud.ibm.com/docs/en/guides/ai-transpiler-passes"""
    # TODO: perform PauliRL synthesis
    raise NotImplementedError("PauliRL pass not implemented yet cause it is not scalable")


def coupling_map_to_pGraph(coupling_map: CouplingMap) -> pGraph:
    """Used in Paulihedral/Tetris primitive"""
    G = rx.adjacency_matrix(coupling_map.graph)
    C = rx.floyd_warshall_numpy(coupling_map.graph)
    return pGraph(G, C)


def constr_mypauli_blocks(paulis, coeffs) -> List[List[pauliString]]:
    """Used in Paulihedral/Tetris primitive"""
    groups = phoenix.primitive.group_paulis_and_coeffs(paulis, coeffs)

    mypauli_blocks = []
    for paulis, coeffs in groups.values():
        mypauli_blocks.append([])
        for p, c in zip(paulis, coeffs):
            mypauli_blocks[-1].append(pauliString(p, c))
    return mypauli_blocks


def su4_circ_stats(qc: qiskit.QuantumCircuit) -> Tuple[int, int]:
    """Statistic of #2Q and Depth-2Q of SU(4)-based circuit."""
    from canopus import rebase_to_canonical

    qc = rebase_to_canonical(qc)
    num_su4 = qc.count_ops().get("can", 0)
    depth_su4 = qc.depth(lambda instr: instr.operation.name == "can")
    return num_su4, depth_su4
