from __future__ import annotations

from dataclasses import dataclass

from qiskit import QuantumCircuit
from qiskit.circuit.library import PauliEvolutionGate

from ..basics import CNOTEquivCliffordGate, fSwapEquivCliffordGate
from ..hamiltonian import Hamiltonian

# _EACH_GROUP_SYNTHESIS_BASIS_GATES = ["cx", "rz", "sx", "x"]
_EACH_GROUP_SYNTHESIS_BASIS_GATES = ["cx", "u"]
_SUCCESSIVE_2Q_PAULI_ROTATION_GATES = {"rxx", "rxy", "rxz", "ryx", "ryy", "ryz", "rzx", "rzy", "rzz"}

@dataclass
class SimplificationStep:
    clifford: CNOTEquivCliffordGate | fSwapEquivCliffordGate
    local_hamiltonian: Hamiltonian
    qubits: tuple[int, int]


def constr_circuit_from_simp_steps(ham: Hamiltonian, steps: list[SimplificationStep], optimize : bool = True) -> QuantumCircuit:
    qc_pre = QuantumCircuit(ham.num_qubits)
    qc_post = QuantumCircuit(ham.num_qubits)

    for step in steps:
        qc_post.append(PauliEvolutionGate(step.local_hamiltonian), range(ham.num_qubits))
        qc_post.append(step.clifford, step.qubits)

        qc_pre.append(step.clifford, step.qubits)

    qc_post = qc_post.reverse_ops()
    qc_pre.append(PauliEvolutionGate(ham), range(ham.num_qubits))

    qc = qc_pre.compose(qc_post).decompose("PauliEvolution")

    if optimize:
        qc = _optimize_phoenix_circuit_by_qiskit_each_group(qc)

    return qc


def _synthesize_successive_2q_pauli_rotation_block(block: QuantumCircuit) -> QuantumCircuit:
    from qiskit.transpiler import PassManager, passes

    pm = PassManager()
    pm.append(passes.Collect2qBlocks())
    pm.append(passes.ConsolidateBlocks(basis_gates=["cx"]))
    pm.append(passes.UnitarySynthesis(basis_gates=_EACH_GROUP_SYNTHESIS_BASIS_GATES))
    pm.append(passes.Optimize1qGatesDecomposition(basis=_EACH_GROUP_SYNTHESIS_BASIS_GATES[1:]))
    pm.append(passes.CommutativeCancellation())
    return pm.run(block)


def _optimize_phoenix_circuit_by_qiskit_each_group(qc: QuantumCircuit) -> QuantumCircuit:
    """Resynthesize only successive 2Q Pauli-rotation runs.

    This is a narrow version of :func:`~phoenix.utils.post_transpile`:
    it scans the circuit, finds maximal consecutive runs of 2-qubit Pauli
    rotations acting on the same qubit pair, and applies 2-qubit unitary
    synthesis to each run independently.  Other instructions are preserved
    verbatim.
    """

    optimized = QuantumCircuit(*qc.qregs, *qc.cregs, name=qc.name)
    run_instrs = []
    run_pair_qubits = None

    def flush_run():
        nonlocal run_instrs, run_pair_qubits, optimized
        if not run_instrs:
            return

        local_block = QuantumCircuit(2)
        local_qubits = tuple(local_block.qubits)
        for instr in run_instrs:
            mapped_qargs = [local_qubits[run_pair_qubits.index(qubit)] for qubit in instr.qubits]
            local_block.append(instr.operation, mapped_qargs, instr.clbits)

        synthesized_block = _synthesize_successive_2q_pauli_rotation_block(local_block)
        optimized.compose(synthesized_block, qubits=list(run_pair_qubits), inplace=True)

        run_instrs = []
        run_pair_qubits = None

    for instr in qc.data:
        if _is_successive_2q_pauli_rotation(instr):
            pair = tuple(instr.qubits)
            if not run_instrs:
                run_instrs = [instr]
                run_pair_qubits = pair
                continue
            if set(pair) == set(run_pair_qubits):
                run_instrs.append(instr)
                continue

        flush_run()
        optimized.append(instr.operation, instr.qubits, instr.clbits)

    flush_run()
    return optimized


def _is_successive_2q_pauli_rotation(instr) -> bool:
    return (
        instr.operation.num_qubits == 2
        and instr.operation.name in _SUCCESSIVE_2Q_PAULI_ROTATION_GATES
        and len(instr.qubits) == 2
        and instr.qubits[0] != instr.qubits[1]
    )
