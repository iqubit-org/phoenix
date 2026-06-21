"""Topology-aware peel-forward compilation (co-design, design doc §13).

ISOLATED experimental module — duplicates engine logic from ``peel.py`` on
purpose (to be merged once finalized). It only READS shared building blocks
(``CLIFFORD_OPTIONS`` and the precomputed 16-code tables ``_NEWCODE16`` /
``_DELTA16`` / ``_SIGN16``) and delegates the all-to-all case to the untouched
``peel_compile``, so the existing all-to-all engine is provably unchanged.

Mechanism (fully topology-valid by construction, NO post-hoc SABRE routing):

- Initial mapping: logical qubits placed along a Hamiltonian path of the
  coupling graph (so JW-style chains land locally), as a free relabeling.
- Per episode: a target row is reduced to weight <= 2 ON A COUPLING EDGE using
  only physical-edge moves —
    * REDUCTION: a weight-reducing 2q Clifford (one of 9) on a physical edge
      whose both endpoints are in the target's support; chosen by whole-table
      benefit (preserves cross-row sharing, like all-to-all peel);
    * ROUTING: when no support pair is adjacent, a SWAP along the shortest path
      moves the nearest support pair one hop closer (every SWAP on a physical
      edge).
- Emission rule: a row is emitted iff weight <= 1, OR weight == 2 and its
  support is a coupling edge. Rows at weight-2-on-non-edge stay active.
- Terminal: the forward frame replayed in reverse — every move is self-inverse
  and on a physical edge, so the terminal is nearest-neighbor by construction.

Termination is structural: each reduction strictly drops the target weight
(<= n-2 reductions/episode), each routing strictly drops the nearest-support
distance (<= diameter SWAPs between reductions); bounded, zero hyperparameters.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from itertools import combinations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.transpiler import CouplingMap

from ..hamiltonian import Hamiltonian
from .peel import _DELTA16, _NEWCODE16, _SIGN16, peel_compile
from .simplification import CLIFFORD_OPTIONS

# ---------------------------------------------------------------------------
# Coupling-graph helpers
# ---------------------------------------------------------------------------


def _adjacency(coupling_map: CouplingMap, n: int) -> list[set[int]]:
    adj: list[set[int]] = [set() for _ in range(n)]
    for a, b in coupling_map.get_edges():
        if a != b:
            adj[a].add(b)
            adj[b].add(a)
    return adj


def _bfs_trees(adj: list[set[int]], n: int):
    """All-pairs BFS: dist[a][b] and parent[a][b] (= the neighbour of b on a
    shortest path from b toward a; i.e. one hop from b toward root a)."""
    INF = np.iinfo(np.int32).max
    dist = np.full((n, n), INF, dtype=np.int64)
    parent = np.full((n, n), -1, dtype=np.int64)
    for a in range(n):
        dist[a, a] = 0
        q = deque([a])
        while q:
            u = q.popleft()
            for v in adj[u]:
                if dist[a, v] == INF:
                    dist[a, v] = dist[a, u] + 1
                    parent[a, v] = u  # one step from v toward a
                    q.append(v)
    return dist, parent


def _hamiltonian_path(adj: list[set[int]], n: int) -> list[int]:
    """A true Hamiltonian path (every consecutive pair physically adjacent),
    found by Warnsdorff-guided DFS with backtracking — tractable for the
    benchmark sizes (n <= ~30). Falls back to the best partial greedy path if
    the graph is not traceable (never happens for line/grid/heavy-hex)."""
    import sys

    sys.setrecursionlimit(10000)
    best_partial = []

    def dfs(u, visited, path):
        nonlocal best_partial
        if len(path) > len(best_partial):
            best_partial = list(path)
        if len(path) == n:
            return True
        # Warnsdorff: try neighbours with fewest onward options first
        nbrs = sorted(
            (v for v in adj[u] if not visited[v]),
            key=lambda v: sum(1 for w in adj[v] if not visited[w]),
        )
        for v in nbrs:
            visited[v] = True
            path.append(v)
            if dfs(v, visited, path):
                return True
            path.pop()
            visited[v] = False
        return False

    starts = sorted(range(n), key=lambda i: len(adj[i]))  # low-degree first
    result = None
    for s in starts[: min(len(starts), 4)]:  # a few low-degree seeds suffice
        visited = [False] * n
        visited[s] = True
        best_partial = []
        if dfs(s, visited, [s]):
            result = best_partial
            break
    if result is None:
        result = best_partial  # best effort
    # ensure a full permutation: append any qubits the path missed
    seen = set(result)
    result = list(result) + [q for q in range(n) if q not in seen]
    return result


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


@dataclass
class TopoPeelResult:
    moves: list  # ('R', ci, a, b) reduction | ('S', a, b) swap
    emissions: list  # (t, labels, coeffs)
    num_qubits: int
    init_perm: list  # init_perm[logical] = physical qubit it was mapped to


def _is_edge(edge_set: set, a: int, b: int) -> bool:
    return (a, b) in edge_set


def peel_forward_topo(
    ham: Hamiltonian, coupling_map: CouplingMap, verbose: bool = False
) -> TopoPeelResult:
    n = ham.num_qubits
    adj = _adjacency(coupling_map, n)
    dist, parent = _bfs_trees(adj, n)
    edge_set = {(a, b) for a in range(n) for b in adj[a]}

    # ---- initial mapping: logical i -> physical path[i] (free relabeling) ----
    path = _hamiltonian_path(adj, n)
    x0 = np.asarray(ham.paulis.x, dtype=np.uint8)
    z0 = np.asarray(ham.paulis.z, dtype=np.uint8)
    x = np.zeros_like(x0)
    z = np.zeros_like(z0)
    x[:, path] = x0  # column of logical i moves to physical path[i]
    z[:, path] = z0
    coeffs = np.real(np.asarray(ham.coeffs)).astype(np.float64).copy()
    m = x.shape[0]

    active = np.ones(m, dtype=bool)
    w = (x | z).sum(axis=1).astype(np.int64)
    moves: list = []
    emissions: list = []
    target = -1

    def _supp(r):
        return np.where((x[r] | z[r]) != 0)[0]

    def _emittable(r) -> bool:
        if w[r] <= 1:
            return True
        if w[r] == 2:
            s = _supp(r)
            return _is_edge(edge_set, int(s[0]), int(s[1]))
        return False

    def _emit():
        nonlocal target
        rows = [int(r) for r in np.where(active)[0] if _emittable(int(r))]
        if not rows:
            return
        from qiskit.quantum_info import Pauli

        labels = [Pauli((z[r].astype(bool), x[r].astype(bool))).to_label() for r in rows]
        emissions.append((len(moves), labels, [float(coeffs[r]) for r in rows]))
        active[np.array(rows)] = False
        if target in rows:
            target = -1
        if verbose:
            print(f"  t={len(moves)}: emit {len(rows)} rows {labels if n<=12 else ''}")

    def _apply_reduction(ci: int, a: int, b: int):
        ai = np.where(active)[0]
        cq = (x[ai, a] | (z[ai, a] << 1) | (x[ai, b] << 2) | (z[ai, b] << 3)).astype(np.int64)
        nc = _NEWCODE16[ci][cq]
        x[ai, a] = (nc & 1).astype(np.uint8)
        z[ai, a] = ((nc >> 1) & 1).astype(np.uint8)
        x[ai, b] = ((nc >> 2) & 1).astype(np.uint8)
        z[ai, b] = ((nc >> 3) & 1).astype(np.uint8)
        w[ai] += _DELTA16[ci][cq]
        coeffs[ai] *= _SIGN16[ci][cq]
        moves.append(("R", ci, a, b))
        if verbose:
            print(f"  move {len(moves)} [R]: {CLIFFORD_OPTIONS[ci].name}@({a},{b})")
        _emit()

    def _apply_swap(a: int, b: int):
        # SWAP conjugation: P_a <-> P_b on every row (column swap, no phase).
        x[:, [a, b]] = x[:, [b, a]]
        z[:, [a, b]] = z[:, [b, a]]
        moves.append(("S", a, b))
        if verbose:
            print(f"  move {len(moves)} [S]: swap@({a},{b})")
        _emit()

    def _spanning_tree(s_list):
        """BFS spanning tree of the support's induced subgraph (edges within S).
        Returns (tree_parent dict, reached set). Disconnected if reached < S."""
        s_set = set(s_list)
        root = s_list[0]
        tp = {root: -1}
        dq = deque([root])
        while dq:
            u = dq.popleft()
            for v in adj[u]:
                if v in s_set and v not in tp:
                    tp[v] = u
                    dq.append(v)
        return tp, set(tp)

    _emit()
    while active.any():
        if target < 0:
            wa = np.where(active, w, np.iinfo(np.int64).max)
            target = int(np.argmin(wa))
            if verbose:
                print(f"  [target] row {target}, w={w[target]}")

        s = _supp(target).tolist()
        tp, reached = _spanning_tree(s)

        if len(reached) < len(s):
            # support disconnected on the graph: route nearest cross-component
            # pair one hop together (one-time, until connected)
            rest = [v for v in s if v not in reached]
            u, v = min(((u, v) for u in reached for v in rest), key=lambda p: dist[p[0], p[1]])
            hop = int(parent[u, v])  # neighbour of v toward u
            _apply_swap(v, hop)
            continue

        # connected: eliminate a LEAF of the tree (survivor stays on the parent,
        # keeping the support connected — no gap, no spurious routing). Choose
        # among leaf × Clifford by whole-table benefit (cross-row sharing).
        children: dict[int, int] = {}
        for v, p in tp.items():
            if p >= 0:
                children[p] = children.get(p, 0) + 1
        leaves = [v for v in s if tp[v] >= 0 and children.get(v, 0) == 0]
        if not leaves:  # |S| == 1 shouldn't reach here (emitted); defensive
            leaves = [v for v in s if v != s[0]]

        ai = np.where(active)[0]
        best = None  # (dW_table, ci, leaf, parent)
        for leaf in leaves:
            p = tp[leaf]
            tcode = int(x[target, leaf] | (z[target, leaf] << 1)
                        | (x[target, p] << 2) | (z[target, p] << 3))
            code = (x[ai, leaf] | (z[ai, leaf] << 1)
                    | (x[ai, p] << 2) | (z[ai, p] << 3)).astype(np.int64)
            for ci in range(9):
                nc = int(_NEWCODE16[ci][tcode])
                if _DELTA16[ci][tcode] == -1 and (nc & 0b0011) == 0:  # leaf -> I
                    dW = int(_DELTA16[ci][code].sum())
                    key = (dW, ci, leaf, p)
                    if best is None or key < best:
                        best = key
        if best is None:
            # no leaf-eliminating Clifford (rare): fall back to any reducing
            # Clifford on a leaf edge (may create a gap -> routing next iter)
            for leaf in leaves:
                p = tp[leaf]
                tcode = int(x[target, leaf] | (z[target, leaf] << 1)
                            | (x[target, p] << 2) | (z[target, p] << 3))
                for ci in range(9):
                    if _DELTA16[ci][tcode] == -1:
                        best = (0, ci, leaf, p)
                        break
                if best:
                    break
        assert best is not None, "leaf edge must admit a weight-reducing Clifford"
        _apply_reduction(best[1], best[2], best[3])

    return TopoPeelResult(moves=moves, emissions=emissions, num_qubits=n, init_perm=path)


# ---------------------------------------------------------------------------
# Circuit construction (sequential; ASAP can be added later)
# ---------------------------------------------------------------------------


def topo_peel_circuit(result: TopoPeelResult, terminal: str = "replay") -> QuantumCircuit:
    from qiskit.circuit.library import SwapGate

    n = result.num_qubits
    by_t: dict[int, tuple[list, list]] = {}
    for t, labels, cs in result.emissions:
        bucket = by_t.setdefault(t, ([], []))
        bucket[0].extend(labels)
        bucket[1].extend(cs)

    qc = QuantumCircuit(n)

    def _emit_blocks(t):
        if t not in by_t:
            return
        labels, cs = by_t[t]
        # group by support so same-edge rotations consolidate into one block
        from qiskit.quantum_info import Pauli

        buckets: dict[tuple, tuple[list, list]] = {}
        for lbl, c in zip(labels, cs):
            p = Pauli(lbl)
            key = tuple(int(q) for q in np.where(np.asarray(p.x) | np.asarray(p.z))[0])
            bk = buckets.setdefault(key, ([], []))
            bk[0].append(lbl)
            bk[1].append(c)
        for key, (lbls, ccs) in buckets.items():
            qc.append(PauliEvolutionGate(Hamiltonian(lbls, ccs)), range(n))

    def _apply_move(mv):
        if mv[0] == "R":
            _, ci, a, b = mv
            qc.append(CLIFFORD_OPTIONS[ci], (a, b))
        else:
            _, a, b = mv
            qc.append(SwapGate(), (a, b))

    for t in range(len(result.moves) + 1):
        _emit_blocks(t)
        if t < len(result.moves):
            _apply_move(result.moves[t])

    if terminal == "replay":
        for mv in reversed(result.moves):
            _apply_move(mv)
    elif terminal != "absorb":
        raise ValueError(f"Unknown terminal: {terminal!r}")

    return qc.decompose("PauliEvolution")


def peel_topo_compile(
    ham: Hamiltonian, coupling_map: CouplingMap | None = None,
    terminal: str = "replay", verbose: bool = False,
) -> QuantumCircuit:
    """Topology-aware peel. ``coupling_map=None`` (or all-to-all) delegates to
    the untouched all-to-all ``peel_compile`` so that path is unchanged."""
    from phoenix.utils import is_all2all_coupling_map

    if coupling_map is None or is_all2all_coupling_map(coupling_map):
        return peel_compile(ham)
    res = peel_forward_topo(ham, coupling_map, verbose=verbose)
    return topo_peel_circuit(res, terminal=terminal)
