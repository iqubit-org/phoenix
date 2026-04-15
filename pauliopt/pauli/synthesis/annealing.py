from typing import Tuple, List

import numpy as np

from pauliopt.circuits import Circuit
from pauliopt.clifford.tableau import CliffordTableau
from pauliopt.clifford.tableau_synthesis import synthesize_tableau
from pauliopt.gates import CZ, CY, CX, CliffordGate

from pauliopt.pauli.pauli_polynomial import PauliPolynomial
from pauliopt.phase.optimized_circuits import _validate_temp_schedule
from pauliopt.topologies import Topology


def pick_random_control_and_target(num_qubits: int) -> Tuple[int, int]:
    """
    Given the number of qubits pick a random, but distinct control and target qubit.
    :param num_qubits:
    :return:
    """
    control = np.random.choice([q for q in range(num_qubits)])
    target = np.random.choice([q for q in range(num_qubits) if q != control])
    return control, target


def compute_effect(
    pp: PauliPolynomial, gate: CliffordGate, topology: Topology, leg_cache: dict = None
) -> int:
    """
    Compute the effect on the PauliPolynomial as: #CX_prev - #CX_new
    :param pp:
    :param gate:
    :param topology:
    :param leg_cache:
    :return:
    """
    pp_ = pp.copy()
    pp_ = pp_.propagate(gate)
    return pp_.two_qubit_count(topology, leg_cache=leg_cache) - pp.two_qubit_count(
        topology, leg_cache=leg_cache
    )


def get_best_gate(
    pp: PauliPolynomial,
    control: int,
    target: int,
    gate_set: List[CliffordGate],
    topology: Topology,
    leg_cache: dict = None,
):
    """
    Get the gate with the best effect from the gate set.
    :param pp:
    :param control:
    :param target:
    :param gate_set:
    :param topology:
    :param leg_cache:
    :return:
    """
    gate_scores = []
    for gate in gate_set:
        gate = gate(control, target)
        effect = compute_effect(pp, gate, topology, leg_cache=leg_cache)
        gate_scores.append((gate, effect))
    return min(gate_scores, key=lambda x: x[1])


def count_legs(pp: PauliPolynomial, gate: CliffordGate) -> int:
    """
    Count the legs on the pauli-polynomial.
    :param pp:
    :param gate:
    :return:
    """
    pp_ = pp.copy()
    pp_.propagate(gate)
    return pp_.num_legs() - pp.num_legs()


def annealing_synthesis(
    pp: PauliPolynomial,
    topology: Topology,
    schedule: tuple = ("geometric", 1.0, 0.1),
    nr_iterations: int = 100,
    gate_set: List[CliffordGate] = None,
) -> Circuit:
    """
    Simple annealing based synthesis of a pauli-polynomial.

    See: TODO reference

    :param pp:
    :param topology:
    :param schedule:
    :param nr_iterations:
    :param gate_set:
    :return:
    """
    if gate_set is None:
        gate_set = [CX, CY, CZ]

    leg_cache = {}
    clifford_tableau = CliffordTableau(n_qubits=pp.num_qubits)

    schedule = _validate_temp_schedule(schedule)
    random_nrs = np.random.uniform(0.0, 1.0, size=(nr_iterations,))
    for it in range(nr_iterations):
        t = schedule(it, nr_iterations)
        ctrl, trg = pick_random_control_and_target(pp.num_qubits)
        gate, effect = get_best_gate(
            pp, ctrl, trg, gate_set, topology, leg_cache=leg_cache
        )
        accept_step = effect < 0 or random_nrs[it] < np.exp(-np.log(2) * effect / t)
        if accept_step:
            clifford_tableau.append_gate(gate)
            pp.propagate_inplace(gate)

    clifford_circ, _ = synthesize_tableau(
        clifford_tableau, topology, include_swaps=False
    )
    pp_circuit = pp.to_circuit(topology)

    qc = Circuit(pp.num_qubits)
    qc += clifford_circ
    qc += pp_circuit
    qc += clifford_circ.inverse()
    return qc
