"""
Arch/Compiler-related Utils functions
"""
import random
import numpy as np
import rustworkx as rx
from numpy import ndarray
from scipy.stats import unitary_group
from typing import List, Dict
from phoenix.basic import gates
from phoenix.basic.gates import Gate, SWAPGate
from phoenix.basic.circuits import Circuit
from phoenix.utils.functions import calculate_elementary_permutations


def gene_init_mapping(circ: Circuit, device: rx.PyGraph, method: str = 'random') -> Dict[int, int]:
    """
    Generate initial mapping
    
    Args:
        circ: input Circuit instance
        device: coupling graph representing qubit connections
        method: method to generate initial mapping ('random' or 'trivial'), default is 'random'

    Returns:
        A dictionary containing a logical to physical qubits mapping
    """
    n = circ.num_qubits
    logical_qubits = circ.qubits  # logical units
    physical_qubits = sorted(device.node_indices())  # physical units
    # print('-----' * 10)
    # print('logical_qubits:', logical_qubits)
    # print('physical_qubits:', physical_qubits)
    assert len(physical_qubits) >= len(
        logical_qubits), "The number of physical qubits should be greater than the number of logical qubits"
    if method == 'random':
        # random.seed(10)
        random.shuffle(physical_qubits)
        subgraph = device.subgraph(physical_qubits[:n])
        while not rx.is_connected(subgraph):
            random.shuffle(physical_qubits)
            subgraph = device.subgraph(physical_qubits[:n])
        mapping = dict(zip(logical_qubits, physical_qubits[:n]))
    elif method == 'trivial':
        assert rx.is_connected(device.subgraph(physical_qubits[:n])), "The first n physical qubits should be connected"
        mapping = dict(zip(logical_qubits, physical_qubits[:n]))
    else:
        raise ValueError("Invalid method to generate initial mapping")
    return mapping


def is_executable(gate: Gate, mapping: Dict[int, int], dist_mat: ndarray = None, device: rx.PyGraph = None):
    """
    Determine if a gate is executable, i.e., whether the gate acts on qubits which are connected in the coupling graph

    Args:
        gate: input gate
        mapping: logical to physical qubit mapping
        dist_mat: distance matrix of the coupling graph (optional)
        device: coupling graph representing qubit connections (optional)
    
    Returns:
        True if the input gate acts on connected qubits. False otherwise.
    """
    if len(gate.qregs) == 1:  # single qubit gate must be executable
        return True
    assert len(gate.qregs) == 2, "Only 1 or 2 qubit gates are supported"
    assert not (dist_mat is None and device is None), "Either dist_mat or device should be provided"
    if dist_mat is not None:
        return dist_mat[mapping[gate.qregs[0]], mapping[gate.qregs[1]]] == 1
    if device is not None:
        return device.has_edge(mapping[gate.qregs[0]], mapping[gate.qregs[1]])


def update_mapping(mapping: Dict[int, int], swap: SWAPGate) -> Dict[int, int]:
    """Update the logical-to-physical qubit mapping when inserting a SWAP gate"""
    updated_mapping = mapping.copy()
    updated_mapping.update({
        swap.qregs[0]: mapping[swap.qregs[1]],
        swap.qregs[1]: mapping[swap.qregs[0]]
    })
    return updated_mapping


def unify_mapped_circuit(circ_with_swaps: Circuit, mappings: List[Dict[int, int]]) -> Circuit:
    """
    Unify the output circuit, i.e., generate the mapped circuit according to the initial mapping and temporary mappings with SWAP gates inserted

    Args:
        circ_with_swaps: input circuit (with inserted SWAP gates), whose qregs indices are logical qubits
        mappings: initial mapping and all temporary (intermediate) mappings after each SWAP gate

    Returns:
        A new circuit (with SWAP gates decomposed into CNOTs), whose qregs indices are still logical qubits with respect to the initial mapping
    """
    mapping_idx = 0
    mapping = mappings[mapping_idx]
    mapped_circ = Circuit()
    for g in circ_with_swaps:
        if isinstance(g, SWAPGate):
            # NOTE: there must be len(mappings) == len(swaps) + 1
            # mapped_circ.append(
            #     gate.X.on(mapping[g.tqs[0]], mapping[g.tqs[1]]),
            #     gate.X.on(mapping[g.tqs[1]], mapping[g.tqs[0]]),
            #     gate.X.on(mapping[g.tqs[0]], mapping[g.tqs[1]]),
            # )
            mapped_circ.append(gates.SWAP.on([mapping[g.tqs[0]], mapping[g.tqs[1]]]))
            mapping_idx += 1
            mapping = mappings[mapping_idx]
        else:
            mapped_circ.append(g.on(
                [mapping[tq] for tq in g.tqs],
                [mapping[cq] for cq in g.cqs]
            ))
    inverse_mapping = {v: k for k, v in mappings[0].items()}  # inverse of the initial mapping
    mapped_circ = mapped_circ.rewire(inverse_mapping)
    return mapped_circ


def verify_mapped_circuit(circ: Circuit, mapped_circ: Circuit, init_mapping: Dict[int, int],
                          final_mapping: Dict[int, int]) -> bool:
    """
    Verify the correctness of the mapped circuit with respect to the original circuit.
    Fist append necessary SWAP gates on the mapped circuit.
    Then compare the unitary matrices of the two circuits.

    Args:
        circ: original circuit, whose qregs indices are logical qubits
        mapped_circ: mapped circuit, whose qregs indices are logical qubits
        init_mapping: initial mapping from logical to physical qubits
        final_mapping: final mapping from logical to physical qubits

    Returns:
        True if the mapped circuit is correct. False otherwise.
    """
    init_final_mapping = obtain_logic_logic_mapping(init_mapping, final_mapping)
    mat1 = circ.unitary()
    mapped_circ = mapped_circ.deepclone()
    mapped_circ.append(*obtain_appended_swaps(init_final_mapping))
    mat2 = mapped_circ.unitary()

    from phoenix.utils.operations import is_equiv_unitary
    return is_equiv_unitary(mat1, mat2)


def obtain_appended_swaps(mapping) -> List[Gate]:
    """Obtain the corresponding SWAP gates to the mapping relationship, by calculating the series of elementary permutations"""
    swaps = []
    permutations = calculate_elementary_permutations(list(mapping.values()), list(mapping.keys()))
    for perm in permutations:
        swaps.append(gates.SWAP.on([perm[0], perm[1]]))
    return swaps


def obtain_logic_logic_mapping(mapping1: Dict[int, int], mapping2: Dict[int, int]) -> Dict[int, int]:
    """Obtain the logical-logical mapping from two logical-physical mappings"""
    inverse_mapping1 = {v: k for k, v in mapping1.items()}
    inverse_mapping2 = {v: k for k, v in mapping2.items()}
    physical_qubits = sorted(mapping1.values())
    logic_logic_mapping = {inverse_mapping1[q]: inverse_mapping2[q] for q in physical_qubits}
    logic_logic_mapping = dict(sorted(logic_logic_mapping.items(), key=lambda x: x[0]))
    return logic_logic_mapping


def obtain_logical_neighbors(qubit: int, mapping: Dict[int, int], device: rx.PyGraph) -> List[int]:
    """Obtain logical neighbors of a logical qubit according to the logical-physical mapping and device connectivity"""
    # print(qubit, mapping[qubit])
    physical_neighbors = device.neighbors(mapping[qubit])
    inverse_mapping = {v: k for k, v in mapping.items()}
    # NOTE: physical neighbors may not exist in the mapping because num_device_qubits >=  num_logical_qubits
    logical_neighbors = [inverse_mapping[q] for q in physical_neighbors if q in inverse_mapping]
    return logical_neighbors


def read_device_topology(fname: str) -> rx.PyGraph:
    """
    Read device topology from a file (e.g., .graphml file) into a rustworkx.PyGraph instance
    """
    g = rx.read_graphml(fname)[0]
    device = rx.PyGraph()
    device.add_nodes_from(g.node_indices())
    edges = list(g.edge_list())
    device.add_edges_from([(src, dst, {}) for src, dst in edges])
    return device


def gene_grid_2d_graph(total_number: int) -> rx.PyGraph:
    """Generate a 2D grid graph that is most nearly square"""
    # 26: 5x5 + 1
    # 17: 4x4 + 1
    n = int(np.sqrt(total_number))
    m = int(np.ceil(total_number / n))  # m >= n
    # remainder = total_number % n
    g = rx.generators.grid_graph(n, m)
    return g.subgraph(range(total_number))


def gene_chain_1d_graph(total_number: int) -> rx.PyGraph:
    return rx.generators.path_graph(total_number)


def gene_quantum_volume_circuit(num_qubits: int, depth: int, seed=None) -> Circuit:
    """
    Generate a quantum-volume circuit.

    Args:
        num_qubits (int): number of quantum wires
        depth (int): layers of operations (i.e. critical path length)
        seed (int): sets random seed (optional)

    Returns:
        Circuit: constructed circuit

    References:
        Cross, A. W., Bishop, L. S., Sheldon, S., Nation, P. D., & Gambetta, J. M. (2019).
            Validating quantum computers using randomized model circuits.
            Physical Review A, 100(3), 032328.
    """
    ...
    circ = Circuit()
    rng = np.random.default_rng(seed)
    width = int(num_qubits // 2)
    unitary_seeds = rng.integers(low=1, high=1000, size=[depth, width])

    perm_0 = list(range(num_qubits))
    for d in range(depth):
        perm = rng.permutation(perm_0)
        for w in range(width):
            seed_u = unitary_seeds[d][w]
            su4 = unitary_group.rvs(4, random_state=np.random.default_rng(seed_u))
            circ.append(gates.UnivGate(su4).on([int(perm[2 * w]), int(perm[2 * w + 1])]))

    return circ


def gene_random_circuit(num_qubits: int, depth: int, seed=None) -> Circuit:
    """
    Generate random circuit of arbitrary size and form.

    This function leverages traits from qiskit.circuit.random.random_circuit function.


    Args:
        num_qubits (int): number of quantum wires
        depth (int): layers of operations (i.e. critical path length)
        seed (int): sets random seed (optional)

    Returns:
        Circuit: constructed circuit
    """
    from qiskit.circuit.library.standard_gates import (XGate, YGate, ZGate, HGate, SGate, SdgGate, TGate, TdgGate,
                                                       RXGate, RYGate, RZGate, CXGate, CYGate, CZGate, CHGate)
    from qiskit import QuantumCircuit, QuantumRegister
    one_q_ops = [
        XGate,
        YGate,
        ZGate,
        HGate,
        SGate,
        SdgGate,
        TGate,
        TdgGate,
        RXGate,
        RYGate,
        RZGate,
    ]

    max_operands = 2
    one_param = [RXGate, RYGate, RZGate]
    two_param = []
    three_param = []
    two_q_ops = [CXGate, CYGate, CZGate, CHGate]
    three_q_ops = []

    qr = QuantumRegister(num_qubits, "q")
    qc = QuantumCircuit(num_qubits)

    if seed is None:
        seed = np.random.randint(0, np.iinfo(np.int32).max)
    rng = np.random.default_rng(seed)

    # apply arbitrary random operations at every depth
    for _ in range(depth):
        # choose either 1, 2, or 3 qubits for the operation
        remaining_qubits = list(range(num_qubits))
        rng.shuffle(remaining_qubits)
        while remaining_qubits:
            max_possible_operands = min(len(remaining_qubits), max_operands)
            num_operands = rng.choice(range(max_possible_operands)) + 1
            operands = [remaining_qubits.pop() for _ in range(num_operands)]
            if num_operands == 1:
                operation = rng.choice(one_q_ops)
            elif num_operands == 2:
                operation = rng.choice(two_q_ops)
            elif num_operands == 3:
                operation = rng.choice(three_q_ops)
            if operation in one_param:
                num_angles = 1
            elif operation in two_param:
                num_angles = 2
            elif operation in three_param:
                num_angles = 3
            else:
                num_angles = 0
            angles = [rng.uniform(0, 2 * np.pi) for x in range(num_angles)]
            register_operands = [qr[i] for i in operands]
            op = operation(*angles)

            # with some low probability, condition on classical bit values
            qc.append(op, register_operands)

    return Circuit.from_qasm(qc.qasm())


def gene_random_clifford_circuit(num_qubits: int, depth: int, seed=None) -> Circuit:
    """
    Generate random Clifford circuit of arbitrary size and form, including only
        H, S, Sdg, X, Y, Z, CNOT gates.

    This function leverages traits from qiskit.circuit.random.random_circuit function.

    Args:
        num_qubits (int): number of quantum wires
        depth (int): layers of operations (i.e. critical path length)
        seed (int): sets random seed (optional)

    Returns:
        Circuit: constructed Clifford circuit
    """
    from qiskit.circuit.library.standard_gates import (XGate, YGate, ZGate, HGate, SGate, SdgGate, CXGate)
    from qiskit import QuantumCircuit, QuantumRegister
    one_q_ops = [
        XGate,
        YGate,
        ZGate,
        HGate,
        SGate,
        SdgGate,
    ]

    max_operands = 2
    one_param = []
    two_param = []
    three_param = []
    two_q_ops = [CXGate]
    three_q_ops = []

    qr = QuantumRegister(num_qubits, "q")
    qc = QuantumCircuit(num_qubits)

    if seed is None:
        seed = np.random.randint(0, np.iinfo(np.int32).max)
    rng = np.random.default_rng(seed)

    # apply arbitrary random operations at every depth
    for _ in range(depth):
        # choose either 1, 2, or 3 qubits for the operation
        remaining_qubits = list(range(num_qubits))
        rng.shuffle(remaining_qubits)
        while remaining_qubits:
            max_possible_operands = min(len(remaining_qubits), max_operands)
            num_operands = rng.choice(range(max_possible_operands)) + 1
            operands = [remaining_qubits.pop() for _ in range(num_operands)]
            if num_operands == 1:
                operation = rng.choice(one_q_ops)
            elif num_operands == 2:
                operation = rng.choice(two_q_ops)
            elif num_operands == 3:
                operation = rng.choice(three_q_ops)
            if operation in one_param:
                num_angles = 1
            elif operation in two_param:
                num_angles = 2
            elif operation in three_param:
                num_angles = 3
            else:
                num_angles = 0
            angles = [rng.uniform(0, 2 * np.pi) for x in range(num_angles)]
            register_operands = [qr[i] for i in operands]
            op = operation(*angles)

            # with some low probability, condition on classical bit values
            qc.append(op, register_operands)

    return Circuit.from_qasm(qc.qasm())
