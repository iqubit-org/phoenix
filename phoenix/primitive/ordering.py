from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import CircuitInstruction
from qiskit.converters import circuit_to_dag
from ..basics import CNOTEquivCliffordGate, fSwapEquivCliffordGate



class CircuitTetris:
    """A simplified IR group represented as a circuit with Tetris structure and metadata."""

    def __init__(
        self,
        circuit: QuantumCircuit,
        left_end: np.ndarray,
        right_end: np.ndarray,
        head_cliffs: list[CircuitInstruction],
        tail_cliffs: list[CircuitInstruction],
    ):
        self.circuit = circuit
        self.left_end = left_end
        self.right_end = right_end
        self.head_cliffs = head_cliffs
        self.tail_cliffs = tail_cliffs
        self.head_depth2q = _compute_cliff_depth2q(head_cliffs)
        self.tail_depth2q = _compute_cliff_depth2q(tail_cliffs)

    @classmethod
    def from_circuit(cls, qc: QuantumCircuit) -> CircuitTetris:
        qc_copy = qc.copy()
        left_end, right_end = compute_endian_vectors(qc_copy)
        head_cliffs = extract_head_cliffs(qc_copy)
        tail_cliffs = extract_tail_cliffs(qc_copy)
        return cls(qc_copy, left_end, right_end, head_cliffs, tail_cliffs)




def compute_endian_vectors(circuit: QuantumCircuit) -> tuple[np.ndarray, np.ndarray]:
    """Compute left_end and right_end vectors with a single DAG conversion.

    left_end[q]: how many 2Q layers from the left until qubit q is first touched.
    right_end[q]: how many 2Q layers from the right until qubit q is last touched.

    Example:
        q0: ─────────────●──     left_end[0] = 2,  right_end[0] = 0
        q1: ──●──────────┼──     left_end[1] = 0,  right_end[1] = 2
        q2: ──X────●─────X──     left_end[2] = 0,  right_end[2] = 0
        q3: ───────X────────     left_end[3] = 1,  right_end[3] = 1
    """
    num_qubits = circuit.num_qubits
    if num_qubits == 0:
        empty = np.zeros(num_qubits, dtype=int)
        return empty, empty.copy()

    dag = circuit_to_dag(circuit)
    layers = list(dag.layers())

    def _scan(layer_iter):
        end = np.full(num_qubits, -1, dtype=int)
        layer_idx = 0
        for layer in layer_iter:
            has_2q = False
            for node in layer["graph"].op_nodes():
                if node.op.num_qubits >= 2:
                    has_2q = True
                    for qubit in node.qargs:
                        q_idx = circuit.find_bit(qubit).index
                        if q_idx < num_qubits and end[q_idx] < 0:
                            end[q_idx] = layer_idx
            if has_2q:
                layer_idx += 1
            if np.all(end >= 0):
                break
        max_layer = end.max() + 1 if layer_idx > 0 else 0
        end[end < 0] = max_layer
        return end

    left_end = _scan(iter(layers))
    right_end = _scan(reversed(layers))
    return left_end, right_end


def extract_head_cliffs(qc: QuantumCircuit) -> list[CircuitInstruction]:
    """
    Extract consecutive 2Q Clifford blocks from the circuit head, skipping single-qubit rotation gates.
    Single-qubit rotation gates block their acting qubits, preventing earlier 2Q gates from being collected.
    
    Returns:
        list of CircuitInstruction
    """
    cliffs = []
    blocked_qubits = set()
    
    for instr in qc.data:
        gate = instr.operation
        qubits = tuple(qc.find_bit(q).index for q in instr.qubits)
        
        if isinstance(gate, (CNOTEquivCliffordGate, fSwapEquivCliffordGate)):
            if not (qubits[0] in blocked_qubits or qubits[1] in blocked_qubits):
                cliffs.append(instr)
            else:
                blocked_qubits.update(qubits)
        
        elif gate.num_qubits == 1:
            blocked_qubits.add(qubits[0])
        
        else:
            blocked_qubits.update(qubits)
        
        if len(blocked_qubits) >= qc.num_qubits:
            break
    
    return cliffs


def extract_tail_cliffs(qc: QuantumCircuit) -> list[CircuitInstruction]:
    """
    Extract consecutive 2Q Clifford blocks from the circuit tail, skipping single-qubit rotation gates.
    Single-qubit rotation gates block their acting qubits, preventing earlier 2Q gates from being collected.
    ! NOTE: This scans backwards from the circuit tail, so the returned list is in reverse gate order.
    
    Returns:
        list of CircuitInstruction
    """
    cliffs = []
    blocked_qubits = set()
    
    for instr in reversed(qc.data):
        gate = instr.operation
        qubits = tuple(qc.find_bit(q).index for q in instr.qubits)
        
        if isinstance(gate, (CNOTEquivCliffordGate, fSwapEquivCliffordGate)):
            # 2Q Clifford: only collect if both qubits are not blocked
            if not (qubits[0] in blocked_qubits or qubits[1] in blocked_qubits):
                cliffs.append(instr)
            else:
                # This gate is blocked, mark its qubits as blocked (to prevent earlier gates)
                blocked_qubits.update(qubits)
        
        elif gate.num_qubits == 1:
            # Single-qubit gate: block this qubit
            blocked_qubits.add(qubits[0])
        
        else:
            # Other multi-qubit gates: block all involved qubits
            blocked_qubits.update(qubits)
        
        # Early exit when all qubits are blocked
        if len(blocked_qubits) >= qc.num_qubits:
            break
    
    return cliffs



def _compute_cliff_depth2q(cliffs: list[CircuitInstruction]) -> int:
    """Compute 2Q depth of Clifford instructions via qubit-occupation tracking (no QuantumCircuit construction)."""
    if not cliffs:
        return 0
    qubit_depth: dict = {}
    for instr in cliffs:
        q0, q1 = instr.qubits[0], instr.qubits[1]
        d = max(qubit_depth.get(q0, 0), qubit_depth.get(q1, 0)) + 1
        qubit_depth[q0] = d
        qubit_depth[q1] = d
    return max(qubit_depth.values())


def assembling_cost(lhs: CircuitTetris, rhs: CircuitTetris) -> float:
    cost = depth_cost(lhs.right_end, rhs.left_end)
    bonus, lhs_tail_simplified, rhs_head_simplified = cancellation_bonus(lhs.tail_cliffs, rhs.head_cliffs, return_simplified_blocks=True)

    if bonus > 0:
        lhs_depth_reduced = lhs.tail_depth2q - _compute_cliff_depth2q(lhs_tail_simplified)
        rhs_depth_reduced = rhs.head_depth2q - _compute_cliff_depth2q(rhs_head_simplified)
        bonus += lhs_depth_reduced * lhs.circuit.num_qubits + rhs_depth_reduced * rhs.circuit.num_qubits

    cost -= bonus

    return cost


def depth_cost(left_end: np.ndarray, right_end: np.ndarray) -> float:
    """
    Calculate depth overhead when concatenating two circuits.
    
    Based on the Tetris-like stacking of endian vectors.
    """    
    if np.all(right_end[left_end == 0] > 0) and np.all(left_end[right_end == 0] > 0):
        cost = (left_end + right_end - 1).sum()
    else:
        cost = (left_end + right_end).sum()
    
    return max(0, cost)


def cancellation_bonus(lhs_tail: list[CircuitInstruction],
                       rhs_head: list[CircuitInstruction],
                       return_simplified_blocks: bool = False) -> float | tuple[float, list[CircuitInstruction], list[CircuitInstruction]]:
    """
    Calculate cancellation bonus between two Clifford blocks, considering gate reordering.
    
    Gates can "pass through" other gates with disjoint qubits to reach the boundary.

    Args:
        lhs_tail: List of Clifford gates at the tail of the left circuit
        rhs_head: List of Clifford gates at the head of the right circuit
        return_simplified_blocks: Whether to return lhs/rhs Clifford gate lists with cancellable gates removed

    """
    bonus = 0.0
    used_tail = set()
    used_head = set()

    def _is_reachable(cliffs: list[CircuitInstruction], idx: int, used: set) -> bool:
        """
        Check if cliffs[idx] can move to the boundary (pass through all preceding unused gates).
        
        Condition: All unused gates before idx must have qubits disjoint from cliffs[idx]'s qubits.
        """
        target_qubits = {cliffs[idx].qubits[0], cliffs[idx].qubits[1]}
        
        for k in range(idx):
            if k not in used:
                other_qubits = {cliffs[k].qubits[0], cliffs[k].qubits[1]}
                if not target_qubits.isdisjoint(other_qubits):
                    return False  # Blocked, cannot pass through
        
        return True
    
    for i, t in enumerate(lhs_tail):
        for j, h in enumerate(rhs_head):
            if j in used_head:
                continue

            if t == h:
                # Check if both sides can move to the boundary
                if (_is_reachable(lhs_tail, i, used_tail) and 
                    _is_reachable(rhs_head, j, used_head)):
                    bonus += 2.0
                    used_tail.add(i)
                    used_head.add(j)
                    break  # t matched, continue to next
    
    if return_simplified_blocks:
        lhs_tail_simplified = [t for i, t in enumerate(lhs_tail) if i not in used_tail]
        rhs_head_simplified = [h for j, h in enumerate(rhs_head) if j not in used_head]
        return bonus, lhs_tail_simplified, rhs_head_simplified
    
    return bonus



def order_circuits(circuits: list[QuantumCircuit], method: str = 'trivial', **kwargs) -> QuantumCircuit:
    """
    Order and assemble a list of circuits to minimize total cost.
    
    The assembling cost is a heuristic that estimates depth overhead and gate cancellation
    potential between adjacent circuit blocks. The actual gate cancellation is realized
    by optimize_phoenix_circuit_by_qiskit() in the compiler.
    
    Args:
        circuits: List of QuantumCircuits to order and assemble
        method: Ordering algorithm
            - 'trivial': Simple concatenation in original order (fastest, no optimization)
            - 'greedy': Greedy selection with lookahead (recommended for most cases)
            - 'tsp': TSP-based ordering (DP for n<=20, 2-opt for larger; good quality)
        **kwargs: Additional arguments passed to the ordering algorithm
            For 'greedy': lookahead (default 40)
            For 'tsp': dp_threshold (default 18), max_2opt_iterations (default 1000)

    Returns:
        Assembled QuantumCircuit
    """
    if method == 'trivial':
        qc_final = _order_circuit_trivial(circuits)
    elif method == 'greedy':
        qc_final = _order_circuit_greedy(circuits, **kwargs)
    elif method == 'tsp':
        qc_final = _order_circuit_tsp(circuits, **kwargs)
    else:
        raise ValueError(f"Unknown ordering method: {method}")
    return qc_final


def _order_circuit_trivial(circuits: list[QuantumCircuit]) -> QuantumCircuit:
    qc_final = QuantumCircuit(circuits[0].num_qubits)
    for qc in circuits:
        qc_final.compose(qc, inplace=True)
    return qc_final

def _cancel_matching_cliffs(qc: QuantumCircuit,
                            lhs_tail: list[CircuitInstruction],
                            rhs_head: list[CircuitInstruction]) -> bool:
    """Remove matching Clifford pairs at the junction of two composed circuits (in-place).

    Returns True if any gates were removed.
    """
    _, lhs_remaining, rhs_remaining = cancellation_bonus(
        lhs_tail, rhs_head, return_simplified_blocks=True)

    remaining_ids = set(id(t) for t in lhs_remaining) | set(id(h) for h in rhs_remaining)
    all_ids = set(id(t) for t in lhs_tail) | set(id(h) for h in rhs_head)
    to_remove = all_ids - remaining_ids

    if to_remove:
        qc.data = [instr for instr in qc.data if id(instr) not in to_remove]
        return True
    return False


def _order_circuit_greedy(circuits: list[QuantumCircuit], lookahead: int = 40, **kwargs) -> QuantumCircuit:
    """Order circuits using a greedy algorithm with lookahead."""
    circuits = circuits.copy()
    tetris = CircuitTetris.from_circuit(circuits.pop(0))
    tetris_list = [CircuitTetris.from_circuit(circ) for circ in circuits]

    while tetris_list:
        costs = {i: assembling_cost(tetris, tts) for i, tts in enumerate(tetris_list[:lookahead])}
        i = min(costs, key=costs.get)
        next_tetris = tetris_list.pop(i)

        old_tail = tetris.tail_cliffs
        tetris.circuit.compose(next_tetris.circuit, inplace=True)

        cancelled = _cancel_matching_cliffs(tetris.circuit, old_tail, next_tetris.head_cliffs)
        if cancelled:
            tetris.tail_cliffs = extract_tail_cliffs(tetris.circuit)
            tetris.tail_depth2q = _compute_cliff_depth2q(tetris.tail_cliffs)
        else:
            tetris.tail_cliffs = next_tetris.tail_cliffs
            tetris.tail_depth2q = next_tetris.tail_depth2q
        tetris.right_end = next_tetris.right_end.copy()

    return tetris.circuit



# =============================================================================
# TSP (Traveling Salesman Problem) based Ordering Algorithm
# =============================================================================

def _compute_cost_matrix(tetris_list: list[CircuitTetris]) -> np.ndarray:
    """
    Precompute the cost matrix for all pairs of circuit blocks.
    
    cost_matrix[i, j] = assembling_cost(tetris_list[i], tetris_list[j])
    
    This represents the cost of placing block j immediately after block i.
    """
    n = len(tetris_list)
    cost_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                cost_matrix[i, j] = assembling_cost(tetris_list[i], tetris_list[j])
            else:
                cost_matrix[i, j] = float('inf')  # Can't go from a node to itself
    return cost_matrix


def _tsp_dp_solve(cost_matrix: np.ndarray, start_idx: int = 0) -> tuple[list[int], float]:
    """
    Solve TSP exactly using Held-Karp dynamic programming algorithm.
    
    Time complexity: O(n² × 2^n)
    Space complexity: O(n × 2^n)
    
    Practical for n <= 20 (2^20 ≈ 1M states).
    
    Args:
        cost_matrix: n×n matrix where cost_matrix[i,j] is cost from i to j
        start_idx: Fixed starting node index
    
    Returns:
        Tuple of (optimal ordering as list of indices, total cost)
    """
    n = cost_matrix.shape[0]
    
    if n <= 1:
        return [start_idx], 0.0
    
    # dp[mask][i] = minimum cost to visit all nodes in 'mask' ending at node i
    # mask is a bitmask representing which nodes have been visited
    INF = float('inf')
    dp = np.full((1 << n, n), INF)
    parent = np.full((1 << n, n), -1, dtype=int)
    
    # Base case: start at start_idx
    start_mask = 1 << start_idx
    dp[start_mask][start_idx] = 0
    
    # Fill DP table
    for mask in range(1, 1 << n):
        # Must include start_idx
        if not (mask & start_mask):
            continue
            
        for last in range(n):
            if not (mask & (1 << last)):
                continue
            if dp[mask][last] == INF:
                continue
                
            # Try extending to a new node
            for next_node in range(n):
                if mask & (1 << next_node):
                    continue  # Already visited
                    
                new_mask = mask | (1 << next_node)
                new_cost = dp[mask][last] + cost_matrix[last][next_node]
                
                if new_cost < dp[new_mask][next_node]:
                    dp[new_mask][next_node] = new_cost
                    parent[new_mask][next_node] = last
    
    # Find the optimal ending node (all nodes visited)
    full_mask = (1 << n) - 1
    best_last = -1
    best_cost = INF
    
    for last in range(n):
        if dp[full_mask][last] < best_cost:
            best_cost = dp[full_mask][last]
            best_last = last
    
    # Reconstruct path
    path = []
    mask = full_mask
    current = best_last
    
    while current != -1:
        path.append(current)
        prev = parent[mask][current]
        mask ^= (1 << current)
        current = prev
    
    path.reverse()
    
    return path, best_cost


def _tsp_2opt_improve(ordering: list[int], cost_matrix: np.ndarray, 
                       max_iterations: int = 1000) -> tuple[list[int], float]:
    """
    Improve an ordering using 2-opt local search.
    
    2-opt repeatedly reverses segments of the path to reduce total cost.
    This is the classic TSP local search that works well in practice.
    
    Time complexity per iteration: O(n²)
    
    Args:
        ordering: Initial ordering (list of node indices)
        cost_matrix: n×n cost matrix
        max_iterations: Maximum number of improvement iterations
    
    Returns:
        Tuple of (improved ordering, total cost)
    """
    n = len(ordering)
    if n <= 2:
        cost = sum(cost_matrix[ordering[i]][ordering[i+1]] for i in range(n-1)) if n > 1 else 0
        return ordering, cost
    
    ordering = list(ordering)  # Make a copy

    current_cost = sum(cost_matrix[ordering[k]][ordering[k+1]] for k in range(n - 1))

    for _ in range(max_iterations):
        improved = False

        for i in range(1, n - 1):
            for j in range(i + 1, n):
                # Compute cost delta for reversing segment [i, j]: O(j-i) instead of O(n)
                delta = (cost_matrix[ordering[i-1]][ordering[j]]
                         - cost_matrix[ordering[i-1]][ordering[i]])
                if j < n - 1:
                    delta += (cost_matrix[ordering[i]][ordering[j+1]]
                              - cost_matrix[ordering[j]][ordering[j+1]])
                for k in range(i, j):
                    delta += (cost_matrix[ordering[k+1]][ordering[k]]
                              - cost_matrix[ordering[k]][ordering[k+1]])

                if delta < -1e-10:
                    ordering[i:j+1] = ordering[i:j+1][::-1]
                    current_cost += delta
                    improved = True
                    break

            if improved:
                break

        if not improved:
            break

    return ordering, current_cost


def _tsp_greedy_initial(cost_matrix: np.ndarray, start_idx: int = 0) -> list[int]:
    """
    Generate an initial ordering using greedy nearest neighbor heuristic.
    
    Args:
        cost_matrix: n×n cost matrix
        start_idx: Starting node
    
    Returns:
        Greedy ordering starting from start_idx
    """
    n = cost_matrix.shape[0]
    visited = {start_idx}
    ordering = [start_idx]
    current = start_idx
    
    while len(ordering) < n:
        # Find the nearest unvisited node
        best_next = -1
        best_cost = float('inf')
        
        for j in range(n):
            if j not in visited and cost_matrix[current][j] < best_cost:
                best_cost = cost_matrix[current][j]
                best_next = j
        
        if best_next == -1:
            break
            
        ordering.append(best_next)
        visited.add(best_next)
        current = best_next
    
    return ordering


def _order_circuit_tsp(
    circuits: list[QuantumCircuit],
    dp_threshold: int = 18,
    max_2opt_iterations: int = 1000,
    **kwargs
) -> QuantumCircuit:
    """
    Order circuits by solving the problem as a Traveling Salesman Problem (TSP).
    
    The IR ordering problem is naturally a TSP: each circuit block is a "city",
    and assembling_cost(block_i, block_j) is the "distance" from i to j.
    
    For small n (≤ dp_threshold): Uses exact DP solution (Held-Karp algorithm)
    For large n: Uses greedy + 2-opt local search
    
    Args:
        circuits: List of QuantumCircuits to order
        dp_threshold: Use DP for n ≤ this value (default 18)
        max_2opt_iterations: Max iterations for 2-opt improvement
    
    Returns:
        Assembled QuantumCircuit in optimized order
    """
    if len(circuits) <= 1:
        return _order_circuit_trivial(circuits)
    
    # Convert to CircuitTetris and compute cost matrix
    tetris_list = [CircuitTetris.from_circuit(circ) for circ in circuits]
    n = len(tetris_list)
    cost_matrix = _compute_cost_matrix(tetris_list)
    
    # Fix start index as 0 (first circuit)
    start_idx = 0
    
    if n <= dp_threshold:
        # Use exact DP solution
        ordering, total_cost = _tsp_dp_solve(cost_matrix, start_idx)
    else:
        # Use greedy + 2-opt
        ordering = _tsp_greedy_initial(cost_matrix, start_idx)
        ordering, total_cost = _tsp_2opt_improve(ordering, cost_matrix, max_2opt_iterations)
    
    # Assemble circuits in the optimized order with gate cancellation
    qc_final = QuantumCircuit(circuits[0].num_qubits)
    tail_cliffs: list[CircuitInstruction] = []
    for idx in ordering:
        t = tetris_list[idx]
        qc_final.compose(t.circuit, inplace=True)
        cancelled = _cancel_matching_cliffs(qc_final, tail_cliffs, t.head_cliffs)
        if cancelled:
            tail_cliffs = extract_tail_cliffs(qc_final)
        else:
            tail_cliffs = t.tail_cliffs

    return qc_final


def repr_circuit_instr(instr: CircuitInstruction) -> str:
    """A simple string representation of a CircuitInstruction."""
    gate = instr.operation
    qubits = tuple([q._index for q in instr.qubits])
    return f"{gate.name}@{qubits}"
