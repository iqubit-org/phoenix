from typing import List, Union, Tuple

from pauliopt.clifford.tableau import CliffordTableau
from pauliopt.clifford.tableau_synthesis import synthesize_tableau
from pauliopt.circuits import Circuit
from pauliopt.gates import CZ, CY, CX

from pauliopt.pauli.pauli_gadget import PauliGadget
from pauliopt.pauli.pauli_polynomial import PauliPolynomial
from pauliopt.pauli.synthesis.annealing import get_best_gate
from pauliopt.pauli_strings import X, Y, Z, I
from pauliopt.topologies import Topology
import networkx as nx

DESIRED_NEIGHBORS = [(X, X), (Y, X), (Z, Z), (Z, Y)]
UNDESIRABLE_NEIGHBORS = [(X, I), (Y, I), (Z, I)]


def compute_global_permutation(pp: PauliPolynomial, topology: Topology) -> dict:
    """
    Compute a globally optimal permutation.
    :param pp:
    :param topology:
    :return:
    """
    matching_graph = nx.Graph()
    for e1 in range(pp.num_qubits):
        for e2 in range(pp.num_qubits):
            if e1 != e2:
                matching_graph.add_edge(e1, e2, weight=0)
    matching_graph.add_nodes_from(range(pp.num_qubits))
    for gadget in pp.pauli_gadgets:
        assert isinstance(gadget, PauliGadget)
        for e1 in range(pp.num_qubits):
            for e2 in range(pp.num_qubits):
                if e1 != e2:
                    p1 = gadget.paulis[e1]
                    p2 = gadget.paulis[e2]
                    if (p1, p2) in DESIRED_NEIGHBORS or (p2, p1) in DESIRED_NEIGHBORS:
                        matching_graph[e1][e2]["weight"] += topology.dist(e1, e2)
                    elif (p1, p2) in UNDESIRABLE_NEIGHBORS or (
                        p2,
                        p1,
                    ) in UNDESIRABLE_NEIGHBORS:
                        matching_graph[e1][e2]["weight"] -= topology.dist(e1, e2)
    return dict(nx.maximal_matching(matching_graph))


def optimize_pauli_polynomial(
    clifford_left: CliffordTableau,
    pp: PauliPolynomial,
    clifford_right: CliffordTableau,
    topology: Topology,
    gate_set=None,
    leg_cache=None,
) -> Tuple[CliffordTableau, PauliPolynomial, CliffordTableau]:
    """
    Optimize a region of Clifford - PauliPolynomial - Clifford.

    :param clifford_left:
    :param pp:
    :param clifford_right:
    :param topology:
    :param gate_set:
    :param leg_cache:
    :return:
    """
    if gate_set is None:
        gate_set = [CX, CY, CZ]

    for c in range(pp.num_qubits):
        for t in range(pp.num_qubits):
            if c != t:
                gate, effect = get_best_gate(
                    pp, c, t, gate_set, topology, leg_cache=leg_cache
                )
                dist = topology.dist(c, t)
                if effect + 2 * dist <= 0:
                    pp.propagate_inplace(gate)
                    clifford_left.append_gate(gate)
                    clifford_right.prepend_gate(gate)

    return clifford_left, pp, clifford_right


def compare(
    pp: PauliPolynomial, prev_gadget: int, current_gadget: int, next_gadget: int
) -> bool:
    """
    Compare the previous and next gadget to the current gadget.
    :param pp:
    :param prev_gadget:
    :param current_gadget:
    :param next_gadget:
    :return:
    """
    return pp.commutes(current_gadget, next_gadget) and (
        pp.mutual_legs(prev_gadget, current_gadget)
        < pp.mutual_legs(prev_gadget, next_gadget)
    )


def sort_pauli_polynomial(pp: PauliPolynomial) -> PauliPolynomial:
    """
    Insertion sort for the Pauli Polynomial.
    :param pp:
    :return:
    """
    col_idx = 1
    while col_idx < pp.num_gadgets - 1:
        prev_col_idx = col_idx - 1
        col_idx_ = col_idx
        new_col_idx = col_idx
        while new_col_idx < pp.num_gadgets and compare(
            pp, prev_col_idx, col_idx_, new_col_idx
        ):
            pp.swap_gadgets(col_idx_, new_col_idx)
            prev_col_idx = col_idx_
            col_idx_ = new_col_idx
            new_col_idx = col_idx_ + 1
        col_idx += 1
    return pp


def split_pauli_polynomial(
    pp: PauliPolynomial,
) -> Tuple[PauliPolynomial, PauliPolynomial]:
    """
    Split the Pauli Polynomial into two regions.
    :param pp:
    :return:
    """
    pp_left = PauliPolynomial(pp.num_qubits)
    pp_right = PauliPolynomial(pp.num_qubits)

    for gadget in pp.pauli_gadgets[: pp.num_gadgets // 2]:
        pp_left >>= gadget

    for gadget in pp.pauli_gadgets[pp.num_gadgets // 2 :]:
        pp_right >>= gadget

    return pp_left, pp_right


def recursion_synth_divide_and_conquer(
    c_l: CliffordTableau,
    pp: PauliPolynomial,
    c_r: CliffordTableau,
    topology: Topology,
    leg_cache=None,
) -> List[Union[CliffordTableau, PauliPolynomial]]:
    """
    Recursive definition of the algorithm.

    1. optimize the PP
    2. IF the number of gadgets are smaller than two return
    2. ELSE split the PP and optimize the two subregions by testing every combination
    3. Recurse on both subregions
    4. Combine and continue
    :param c_l:
    :param pp:
    :param c_r:
    :param topology:
    :param leg_cache:
    :return:
    """
    c_l, pp, c_r = optimize_pauli_polynomial(
        c_l, pp, c_r, topology, leg_cache=leg_cache
    )
    if pp.num_gadgets <= 2:
        return [c_l, pp, c_r]

    c_center = CliffordTableau(pp.num_qubits)
    pp = sort_pauli_polynomial(pp)
    pp_left, pp_right = split_pauli_polynomial(pp)
    regions_left = recursion_synth_divide_and_conquer(
        c_l, pp_left, c_center, topology, leg_cache=leg_cache
    )
    regions_right = recursion_synth_divide_and_conquer(
        c_center, pp_right, c_r, topology, leg_cache=leg_cache
    )

    return regions_left[:-1] + [c_center] + regions_right[1:]


def synthesis_divide_and_conquer(
    pp: PauliPolynomial, topology: Topology
) -> Tuple[Circuit, List]:
    """
    Divide and conquer synthesis.

    Will return a circuit that is synthesized with an additional global placement of qubits.
    See: TODO reference

    :param pp:
    :param topology:
    :return:
    """
    permutation = compute_global_permutation(pp, topology)
    pp.permute(permutation)

    c_l = CliffordTableau(pp.num_qubits)
    c_r = CliffordTableau(pp.num_qubits)
    legs_cache = {}
    regions = recursion_synth_divide_and_conquer(
        c_l, pp, c_r, topology, leg_cache=legs_cache
    )

    circ_out = Circuit(pp.num_qubits)
    for region in regions:
        if isinstance(region, PauliPolynomial):
            circ_out += region.to_circuit(topology=topology)
        else:
            circ, _ = synthesize_tableau(region, topology, include_swaps=False)
            circ_out += circ

    qubit_placement = list(range(pp.num_qubits))
    for i, j in permutation.items():
        qubit_placement[i], qubit_placement[j] = qubit_placement[j], qubit_placement[i]
    return circ_out, qubit_placement
