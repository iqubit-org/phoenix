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


def cnot_equiv_commute(g1: CNOTEquivCliffordGate, q1: tuple,
                       g2: CNOTEquivCliffordGate, q2: tuple) -> bool:
    """Exact O(1) commutation test: axes must match on every shared qubit."""
    def axis(g, qs, q):
        return g.pauli_0 if q == qs[0] else g.pauli_1

    return all(axis(g1, q1, q) == axis(g2, q2, q) for q in set(q1) & set(q2))


def asap_order(items: list[dict], n: int) -> list[int]:
    """Greedy ASAP list scheduling over a commutation partial order.

    Each item is a dict with ``qubits`` (occupied for one unit layer) and
    ``deps`` (ids of items it must follow); an item starts at the max of its
    dependencies' finish layers and its qubits' free layers. A lazy min-heap
    on (start, id) yields a deterministic minimal-start order; start
    estimates only ever grow, so stale entries are re-pushed on pop.
    """
    import heapq
    from collections import defaultdict

    nitems = len(items)
    finish = [0] * nitems
    placed = [False] * nitems
    ndeps = [len(it["deps"]) for it in items]
    dependents = defaultdict(list)
    for i, it in enumerate(items):
        for d in it["deps"]:
            dependents[d].append(i)
    qubit_free = [0] * n

    def start_of(i: int) -> int:
        it = items[i]
        s = 0
        for d in it["deps"]:
            s = max(s, finish[d])
        for q in it["qubits"]:
            s = max(s, qubit_free[q])
        return s

    heap = [(start_of(i), i) for i in range(nitems) if ndeps[i] == 0]
    heapq.heapify(heap)
    order: list[int] = []
    while heap:
        s, i = heapq.heappop(heap)
        if placed[i]:
            continue
        cur = start_of(i)
        if cur > s and heap and heap[0][0] < cur:
            heapq.heappush(heap, (cur, i))  # stale estimate: requeue
            continue
        placed[i] = True
        finish[i] = cur + 1
        for q in items[i]["qubits"]:
            qubit_free[q] = cur + 1
        order.append(i)
        for j in dependents[i]:
            ndeps[j] -= 1
            if ndeps[j] == 0:
                heapq.heappush(heap, (start_of(j), j))
    assert len(order) == nitems, "scheduling must place every item (deps form a DAG)"
    return order


def schedule_cnot_equiv_clifford(circ: QuantumCircuit, cancel: bool = True) -> QuantumCircuit:
    """Depth-optimize a circuit made exclusively of ``CNOTEquivCliffordGate``s.

    Cancels commutation-reachable self-inverse pairs (qiskit
    ``CommutativeInverseCancellation``), then reorders by ASAP list scheduling
    over the exact commutation DAG. Operator-exact (including global phase).

    Raises ``ValueError`` if any operation is not a ``CNOTEquivCliffordGate``.
    """
    for inst in circ.data:
        if not isinstance(inst.operation, CNOTEquivCliffordGate):
            raise ValueError(
                f"schedule_cnot_equiv_clifford: unsupported gate {inst.operation.name!r}"
            )

    if cancel and circ.size() > 1:
        from qiskit.transpiler.passes import CommutativeInverseCancellation

        circ = CommutativeInverseCancellation(matrix_based=True)(circ)

    items: list[dict] = []
    on_q: list[list[int]] = [[] for _ in range(circ.num_qubits)]
    for inst in circ.data:
        qs = tuple(circ.find_bit(q).index for q in inst.qubits)
        deps: set[int] = set()
        for q in qs:
            for j in reversed(on_q[q]):
                if not cnot_equiv_commute(items[j]["gate"], items[j]["qubits"],
                                          inst.operation, qs):
                    deps.add(j)
        i = len(items)
        items.append({"gate": inst.operation, "qubits": qs, "deps": deps})
        for q in qs:
            on_q[q].append(i)

    out = circ.copy_empty_like()
    for i in asap_order(items, circ.num_qubits):
        out.append(items[i]["gate"], items[i]["qubits"])
    return out
