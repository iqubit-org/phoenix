"""Generate several 6/8-qubit UCCSD Hamiltonian examples and store them as JSON.

This script mirrors ``qiskit_nature_uccsd_hamiltonian.py`` but uses active-space reductions to
keep the qubit count small. For standard second-quantized electronic-structure problems, the
Jordan-Wigner qubit count is twice the number of spatial orbitals, so practical UCCSD examples in
the 5-8 qubit window land on 6 or 8 qubits.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from qiskit import QuantumCircuit
from qiskit.circuit.library import PauliEvolutionGate
from qiskit_nature.second_q.circuit.library import HartreeFock, UCCSD
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.second_q.transformers import ActiveSpaceTransformer
from qiskit_nature.units import DistanceUnit

from qiskit_nature_uccsd_hamiltonian import dump_qubit_hamiltonian_json


@dataclass(frozen=True)
class ActiveSpaceExample:
    name: str
    atom: str
    active_electrons: int | tuple[int, int]
    active_spatial_orbitals: int
    output_json: str


EXAMPLES: tuple[ActiveSpaceExample, ...] = (
    ActiveSpaceExample(
        name="LiH active space (2e, 3o)",
        atom="Li 0.0 0.0 0.0; H 0.0 0.0 1.6",
        active_electrons=2,
        active_spatial_orbitals=3,
        output_json="LiH_as_2e_3o_JW_sto3g.json",
    ),
    ActiveSpaceExample(
        name="LiH active space (2e, 4o)",
        atom="Li 0.0 0.0 0.0; H 0.0 0.0 1.6",
        active_electrons=2,
        active_spatial_orbitals=4,
        output_json="LiH_as_2e_4o_JW_sto3g.json",
    ),
    ActiveSpaceExample(
        name="BeH2 active space (2e, 3o)",
        atom="Be 0.0 0.0 0.0; H 0.0 0.0 -1.3; H 0.0 0.0 1.3",
        active_electrons=2,
        active_spatial_orbitals=3,
        output_json="BeH2_as_2e_3o_JW_sto3g.json",
    ),
    ActiveSpaceExample(
        name="BeH2 active space (4e, 4o)",
        atom="Be 0.0 0.0 0.0; H 0.0 0.0 -1.3; H 0.0 0.0 1.3",
        active_electrons=4,
        active_spatial_orbitals=4,
        output_json="BeH2_as_4e_4o_JW_sto3g.json",
    ),
)


def build_example(example: ActiveSpaceExample) -> None:
    driver = PySCFDriver(
        atom=example.atom,
        basis="sto3g",
        charge=0,
        spin=0,
        unit=DistanceUnit.ANGSTROM,
    )
    problem = driver.run()

    transformer = ActiveSpaceTransformer(
        num_electrons=example.active_electrons,
        num_spatial_orbitals=example.active_spatial_orbitals,
    )
    active_problem = transformer.transform(problem)

    mapper = JordanWignerMapper()
    qubit_hamiltonian = mapper.map(active_problem.hamiltonian.second_q_op())
    evolution_time = 0.25
    evolution_gate = PauliEvolutionGate(qubit_hamiltonian, time=evolution_time)

    evolution_circuit = QuantumCircuit(qubit_hamiltonian.num_qubits, name=example.name)
    evolution_circuit.append(evolution_gate, range(qubit_hamiltonian.num_qubits))

    hartree_fock = HartreeFock(
        active_problem.num_spatial_orbitals,
        active_problem.num_particles,
        mapper,
    )
    uccsd = UCCSD(
        active_problem.num_spatial_orbitals,
        active_problem.num_particles,
        mapper,
        initial_state=hartree_fock,
    )

    assert evolution_gate.num_qubits == qubit_hamiltonian.num_qubits
    assert uccsd.num_qubits == qubit_hamiltonian.num_qubits
    assert 5 <= qubit_hamiltonian.num_qubits <= 8

    output_path = Path(__file__).resolve().parent / "hams" / example.output_json
    dump_qubit_hamiltonian_json(qubit_hamiltonian, output_path)

    print(f"Example: {example.name}")
    print(f"  Active electrons: {active_problem.num_particles}")
    print(f"  Active spatial orbitals: {active_problem.num_spatial_orbitals}")
    print(f"  Qubits: {qubit_hamiltonian.num_qubits}")
    print(f"  Hamiltonian terms: {len(qubit_hamiltonian.paulis)}")
    print(f"  UCCSD parameter count: {uccsd.num_parameters}")
    print(f"  Evolution time: {evolution_time}")
    print(f"  JSON: {output_path}")
    print(f"  Circuit depth (undecomposed): {evolution_circuit.depth()}")
    print()


def main() -> None:
    for example in EXAMPLES:
        build_example(example)


if __name__ == "__main__":
    main()
