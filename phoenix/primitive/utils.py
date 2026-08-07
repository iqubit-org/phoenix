from __future__ import annotations

from dataclasses import dataclass

from qiskit import QuantumCircuit
from qiskit.circuit.library import CXGate, PauliEvolutionGate, RXXGate, RYYGate, RZXGate, RZZGate
from qiskit.transpiler import AnalysisPass, PassManager
from qiskit.transpiler.passes import ConsolidateBlocks, UnitarySynthesis

from ..basics import CNOTEquivCliffordGate, fSwapEquivCliffordGate
from ..hamiltonian import Hamiltonian

_PAULI_EXP_GATES = (RXXGate, RYYGate, RZZGate, RZXGate)

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
        qc = pauli_exp_consolidation.run(qc)

    return qc





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


def _is_twoq_pauli_exponential(node) -> bool:
    """Whether a DAG node is one of the 2Q Pauli-rotation gates to fuse."""
    return (
        node.op.num_qubits == 2
        and isinstance(node.op, _PAULI_EXP_GATES)
    )


class CollectSuccessive2QPauliExponentials(AnalysisPass):
    """Collect maximal runs of >=2 target Pauli exponentials on one qubit pair.

    Any other operation—including CNOTEquivCliffordGate—is a hard boundary
    and is never included in a collected block.
    """

    def __init__(self, min_block_size: int = 1):
        super().__init__()
        self.min_block_size = min_block_size

    def run(self, dag):
        blocks = []

        # This recognises adjacency on the relevant wires; unrelated operations
        # on other qubits do not unnecessarily break a candidate block.
        for run in dag.collect_2q_runs():
            segment = []
            segment_qargs = None

            for node in run:
                qargs = tuple(node.qargs)
                eligible = _is_twoq_pauli_exponential(node)

                if eligible and (
                    segment_qargs is None or qargs == segment_qargs
                ):
                    segment.append(node)
                    segment_qargs = qargs
                    continue

                # A non-target operation, e.g. cxy/cxz/czz, ends the segment.
                if len(segment) >= self.min_block_size:
                    blocks.append(tuple(segment))

                segment = [node] if eligible else []
                segment_qargs = qargs if eligible else None

            if len(segment) >= self.min_block_size:
                blocks.append(tuple(segment))

        # ConsolidateBlocks consumes precisely this list.
        self.property_set["block_list"] = blocks
        return dag

pauli_exp_consolidation = PassManager([
    CollectSuccessive2QPauliExponentials(),
    ConsolidateBlocks(
        kak_basis_gate=CXGate(),
        force_consolidate=True,
    ),
    UnitarySynthesis(
        basis_gates=["u", "cx"],
        min_qubits=2,
        approximation_degree=1.0,
    ),
])
