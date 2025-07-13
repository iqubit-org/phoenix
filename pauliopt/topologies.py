"""
    This module contains utility code to deal with qubit topologies.
"""

import re
from typing import (
    Collection,
    Dict,
    Final,
    FrozenSet,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    TypedDict,
    Union,
)
import numpy as np
import numpy.typing as npt


class Coupling(FrozenSet[int]):
    """
    Type for couplings in a qubit topology, i.e. unordered
    pairs of adjacent qubits.
    """

    def __new__(cls, fst: int, snd: int) -> "Coupling":
        if not isinstance(fst, int):
            raise TypeError(f"Expected integer, found {fst}")
        if not isinstance(snd, int):
            raise TypeError(f"Expected integer, found {snd}")
        if len({fst, snd}) != 2:
            raise ValueError("Expected a pair of distinct qubits.")
        # see https://github.com/python/mypy/issues/6061
        return super(Coupling, cls).__new__(cls, {fst, snd})  # type: ignore

    @property
    def as_pair(self) -> Tuple[int, int]:
        """
        Returns the coupling as a (increasingly) ordered pair.
        """
        return (min(*self), max(*self))  # pylint: disable = not-an-iterable

    def __str__(self) -> str:
        fst, snd = sorted(self)
        return f"{{{fst}, {snd}}}"

    def __repr__(self) -> str:
        fst, snd = sorted(self)
        return f"Coupling({fst}, {snd})"


CouplingLike = Collection[int]
"""
    Type alias for things that could be used to specify couplings,
    namely any collection of `int` (subject to additional restrictions).

    In an ideal world, this should be "int collections of len 2",
    but static typing does not yet allow for such a constraint.
"""


def _validate_coupling(num_qubits: int, coupling_like: CouplingLike) -> Coupling:
    # Assumes `qubits` was already validated.
    if not isinstance(coupling_like, Collection) or len(coupling_like) != 2:
        raise TypeError(f"Expected a pair, found {coupling_like}.")
    fst, snd = coupling_like
    if not 0 <= fst < num_qubits:
        raise TypeError(f"Invalid qubit {fst}.")
    if not 0 <= snd < num_qubits:
        raise TypeError(f"Invalid qubit {snd}.")
    return Coupling(fst, snd)


class TopologyDict(TypedDict, total=True):
    """
    The type of the dictionary returned by `Topology.as_dict`,
    suitable for JSON serialization.
    """

    num_qubits: int
    """
        Property exposing the number of qubits in the topology.
    """

    couplings: List[List[int]]
    """
        Property exposing the couplings between qubits in the topology.
    """


Layouts: Final[Tuple[str, ...]] = (
    "circular",
    "kamada_kawai",
    "random",
    "shell",
    "spring",
    "spectral",
    "spiral",
)
"""
    Possible layout values for `Topology.draw`
"""


def _floyd_warshall(topology: "Topology"):
    next = np.ones((topology.num_qubits, topology.num_qubits), dtype=int)
    dist = np.inf * np.ones((topology.num_qubits, topology.num_qubits))
    G = topology.to_nx
    for u, v in G.edges():
        dist[u, v] = 1.0
        dist[v, u] = 1.0
        next[u, v] = v
        next[v, u] = u
    for v in G.nodes:
        dist[v, v] = 0
        next[v, v] = v
    for k in G.nodes:
        for i in G.nodes:
            for j in G.nodes:
                if dist[i][j] > dist[i][k] + dist[k][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]
                    next[i][j] = next[i][k]
    return dist, next


class Topology:
    """
    Container class for a qubit topology.
    """

    _num_qubits: int
    _couplings: FrozenSet[Coupling]
    _adjacent: Tuple[FrozenSet[int], ...]
    _dist: np.ndarray
    _named: Optional[str] = None

    def __init__(self, num_qubits: int, couplings: Collection[CouplingLike]):
        if not isinstance(num_qubits, int) or num_qubits <= 0:
            raise TypeError("Number of qubits must be a positive integer.")
        if not isinstance(couplings, Collection):
            raise TypeError("Couplings must be a collection of pairs of qubits.")
        self._num_qubits = num_qubits
        self._couplings = frozenset(
            _validate_coupling(num_qubits, c) for c in couplings
        )
        _adjacent: Tuple[Set[int], ...] = tuple(set() for _ in range(num_qubits))
        for fst, snd in self.couplings:
            _adjacent[fst].add(snd)
            _adjacent[snd].add(fst)
        self._adjacent = tuple(frozenset(n) for n in _adjacent)
        self._dist, self._next = _floyd_warshall(self)

    @property
    def num_qubits(self) -> int:
        """
        Readonly property returning the number of qubits in this topology.
        """
        return self._num_qubits

    @property
    def qubits(self) -> range:
        """
        Readonly property returning the range of qubits in this topology.
        """
        return range(self._num_qubits)

    @property
    def couplings(self) -> FrozenSet[Coupling]:
        """
        Readonly property exposing the couplings between qubits in this topology.
        """
        return self._couplings

    @property
    def as_dict(self) -> Union[str, TopologyDict]:
        """
        Readonly property returning this topology as
        a dictionary, for serialization purposes.
        """
        if self._named is not None:
            return self._named
        return {
            "num_qubits": self.num_qubits,
            "couplings": sorted(list(c.as_pair) for c in self.couplings),
        }

    @property
    def is_planar(self) -> bool:
        """
        Whether this qubit topology is a planar graph.
        """
        try:
            # pylint: disable = import-outside-toplevel
            import networkx as nx  # type: ignore
        except ModuleNotFoundError as e:  # pylint: disable = unused-variable
            raise ModuleNotFoundError("You must install the 'networkx' library.")
        G = self.to_nx
        is_planar, _ = nx.check_planarity(G)
        return is_planar

    @property
    def available_nx_layouts(self) -> Tuple[str, ...]:
        """
        Readonly property returning the available layouts for this qubit topology.
        """
        if self.is_planar:
            return Layouts + ("planar",)
        return Layouts

    @property
    def to_nx(self):
        """
        Readonly property returning a NetworkX graph version of this topology.
        Requires the 'networkx' library to work.
        """
        try:
            # pylint: disable = import-outside-toplevel
            import networkx as nx  # type: ignore
        except ModuleNotFoundError as _:
            raise ModuleNotFoundError("You must install the 'networkx' library.")
        g = nx.Graph()
        g.add_nodes_from(sorted(self.qubits))
        g.add_edges_from(sorted(self.couplings, key=lambda c: c.as_pair))
        return g

    def draw(
        self,
        layout: str = "kamada_kawai",
        *,
        figsize: Optional[Tuple[int, int]] = None,
        filename: Optional[str] = None,
        **kwargs,
    ):
        """
        Draws this qubit topology using NetworkX and Matplotlib.

        The `layout` keyword argument can be used to select a NetworkX layout
        from the available ones (exposed by `Topology.available_nx_layouts`).
        The `figsize` keyword argument is passed to `matplotlib.pyplot.figure`:
        if specified, it determines the width and height of the figure being drawn.
        Keyword arguments `kwargs` are those of `networkx.draw_networkx`.
        If the keyword argument `filename` is set, the figure is also saved.
        """
        try:
            # pylint: disable = import-outside-toplevel
            import networkx as nx  # type: ignore
        except ModuleNotFoundError as _:
            raise ModuleNotFoundError("You must install the 'networkx' library.")
        try:
            # pylint: disable = import-outside-toplevel
            import matplotlib.pyplot as plt  # type: ignore
        except ModuleNotFoundError as _:
            raise ModuleNotFoundError("You must install the 'matplotlib' library.")
        G = self.to_nx
        kwargs = {**kwargs}
        layouts = self.available_nx_layouts
        if "pos" not in kwargs:
            if layout not in layouts:
                raise ValueError(
                    f"Invalid layout found: {layout}. "
                    f"Valid layouts: {', '.join(repr(l) for l in layouts)}"
                )
            kwargs["pos"] = getattr(nx, layout + "_layout")(G)
        if "node_color" not in kwargs:
            kwargs["node_color"] = "#dddddd"
        plt.figure(figsize=figsize)
        nx.draw_networkx(G, **kwargs)
        if filename is not None:
            plt.savefig(filename)
        plt.show()

    def adjacent(self, qubit: int) -> FrozenSet[int]:
        """
        Readonly property exposing the (frozen) set of qubits adjacent
        to (i.e. couple with) the given qubit.
        """
        if not isinstance(qubit, int):
            raise TypeError("Qubit should be an integer.")
        if qubit not in self:
            raise ValueError(f"Invalid qubit {qubit}.")
        return self._adjacent[qubit]

    def incident(self, qubit: int) -> Iterator[Coupling]:
        """
        Readonly property returning an iterator running over all couplings
        incident onto the given qubit.

        This is returned as an iterator, rather than a collection,
        because the couplings are generated on the fly (i.e. this is not
        merely exposing some internal collection).
        """
        adjacent = self.adjacent(qubit)
        return (Coupling(qubit, q) for q in adjacent)

    def dist(self, fro: int, to: int) -> int:
        """
        Returns the distance between two given qubits in the topology.
        """
        if not isinstance(fro, int) or fro not in self:
            raise TypeError(f"Expected a valid qubit, found {fro}.")
        if not isinstance(to, int) or to not in self:
            raise TypeError(f"Expected a valid qubit, found {to}.")
        return self._dist[fro, to]

    def mapped_fwd(self, mapping: Union[Sequence[int], Dict[int, int]]) -> "Topology":
        """
        Returns a topology with the same couplings, but remapping the qubits using
        the given mapping.
        """
        if isinstance(mapping, Sequence):
            if len(mapping) < self.num_qubits:
                raise ValueError(
                    f"Expected mapping keys [0,...,{self._num_qubits}], "
                    f"found {sorted(mapping)} instead."
                )
            _mapping = list(mapping)
        elif isinstance(mapping, Mapping):
            _mapping = []
            for i in range(self._num_qubits):
                if i not in mapping:
                    raise ValueError(
                        f"Expected mapping keys [0,...,{self._num_qubits}], "
                        f"found {sorted(mapping.keys())} instead."
                    )
                _mapping.append(mapping[i])
        else:
            raise TypeError(
                f"Expected Sequence[int] or Mapping[int, int], "
                f"found {type(mapping)} instead."
            )
        if set(_mapping) != set(range(self._num_qubits)):
            raise ValueError(
                f"Expected mapping values [0,...,{self._num_qubits}], "
                f"found {sorted(_mapping)} instead."
            )
        mapped_couplings = [
            {_mapping[x] for x in coupling} for coupling in self._couplings
        ]
        return Topology(self.num_qubits, mapped_couplings)

    def shortest_path(self, fro: int, to: int):
        """
        Computes the shortest path using the next lookup table from the Floyd–Warshall algorithm
        """
        if self._next[fro, to] is None:
            raise Exception("Unconnected Architecture")
        else:
            path = [fro]
            while fro != to:
                fro = self._next[fro, to]
                path.append(fro)
            return path

    def steiner_tree(self, terminals: list[int], subgraph: list[int] = []):
        """
        Computes the Steiner tree over the topology with given terminals.
        `subgraph` can be used to find the Steiner tree of a subgraph of the topology.
        Requires networkx to be installed to work.
        Returns a networkx Graph.
        """
        try:
            # pylint: disable = import-outside-toplevel
            from networkx.algorithms.approximation.steinertree import steiner_tree
        except ModuleNotFoundError as _:
            raise ModuleNotFoundError("You must install the 'networkx' library.")
        if subgraph:
            G = self.to_nx.subgraph(subgraph)
        else:
            G = self.to_nx
        return steiner_tree(G, terminals)

    def non_cutting_qubits(self, subgraph: List[int] = []) -> List[int]:
        try:
            # pylint: disable = import-outside-toplevel
            from networkx import articulation_points
        except ModuleNotFoundError as _:
            raise ModuleNotFoundError("You must install the 'networkx' library.")
        if subgraph:
            G = self.to_nx.subgraph(subgraph)
        else:
            G = self.to_nx
            subgraph = self.qubits
        cut_vertices = list(articulation_points(G))
        return [q for q in subgraph if q not in cut_vertices]

    def mapped_bwd(self, mapping: Union[Sequence[int], Dict[int, int]]) -> "Topology":
        """
        Returns a topology with the same couplings, but remapping the qubits using
        the inverse of the given mapping.
        """
        if isinstance(mapping, Sequence):
            if len(mapping) < self.num_qubits:
                raise ValueError(
                    f"Expected mapping keys [0,...,{self._num_qubits}], "
                    f"found {sorted(mapping)} instead."
                )
            _rev_mapping = {mapping[i]: i for i in mapping}
        elif isinstance(mapping, Mapping):
            _rev_mapping = {}
            for i in range(self._num_qubits):
                if i not in mapping:
                    raise ValueError(
                        f"Expected mapping keys [0,...,{self._num_qubits}], "
                        f"found {sorted(mapping.keys())} instead."
                    )
                _rev_mapping[mapping[i]] = i
        else:
            raise TypeError(
                f"Expected Sequence[int] or Mapping[int, int], "
                f"found {type(mapping)} instead."
            )
        if set(_rev_mapping.keys()) != set(range(self._num_qubits)):
            raise ValueError(
                f"Expected mapping values [0,...,{self._num_qubits}], "
                f"found {sorted(_rev_mapping.keys())} instead."
            )
        mapped_couplings = [
            {_rev_mapping[x] for x in coupling} for coupling in self._couplings
        ]
        return Topology(self.num_qubits, mapped_couplings)

    def __contains__(self, x: Union[int, Coupling, Tuple[int, int]]) -> bool:
        if isinstance(x, int):
            return 0 <= x < self._num_qubits
        if isinstance(x, Coupling):
            return x in self._couplings
        if isinstance(x, tuple) and len(x) == 2:
            fst, snd = x
            if isinstance(fst, int) and isinstance(snd, int):
                return Coupling(fst, snd) in self
        raise TypeError(f"Expected qubit or coupling, found {x}")

    def __repr__(self) -> str:
        if self.couplings:
            return (
                f"Topology({self.num_qubits}, "
                f"[{', '.join(str(c) for c in self.couplings)}])"
            )
        return f"Topology({set(self.qubits)})"

    def __hash__(self) -> int:
        return hash((self.qubits, self.couplings))

    def __eq__(self, other) -> bool:
        if self is other:
            return True
        if not isinstance(other, Topology):
            return NotImplemented
        if self.qubits != other.qubits:
            return False
        if self.couplings != other.couplings:
            return False
        return True

    @staticmethod
    def from_dict(topology: Union[TopologyDict, str]) -> "Topology":
        """
        Creates a `Topology` instance from a dictionary in the
        format obtained from `Topology.as_dict`,
        for de-serialization purposes.
        """
        if isinstance(topology, str):
            line_pattern = re.compile(r"line\(([0-9]+)\)")
            if match := line_pattern.match(topology):
                return Topology.line(int(match[1]))
            cycle_pattern = re.compile(r"cycle\(([0-9]+)\)")
            if match := cycle_pattern.match(topology):
                return Topology.cycle(int(match[1]))
            complete_pattern = re.compile(r"complete\(([0-9]+)\)")
            if match := complete_pattern.match(topology):
                return Topology.complete(int(match[1]))
            grid_pattern = re.compile(r"grid\(([0-9]+),([0-9]+)\)")
            if match := grid_pattern.match(topology):
                return Topology.grid(int(match[1]), int(match[2]))
            periodic_grid_pattern = re.compile(r"periodic_grid\(([0-9]+),([0-9]+)\)")
            if match := periodic_grid_pattern.match(topology):
                return Topology.periodic_grid(int(match[1]), int(match[2]))
            raise ValueError(f"Unexpected special topology {topology}")
        if "num_qubits" not in topology:
            raise TypeError("Expected key 'qubits'.")
        if "couplings" not in topology:
            raise TypeError("Expected key 'couplings'.")
        return Topology(topology["num_qubits"], topology["couplings"])

    @staticmethod
    def line(num_qubits: int) -> "Topology":
        """
        Creates a line topology on the given number of qubits.
        """
        if not isinstance(num_qubits, int) or num_qubits <= 0:
            raise TypeError("Number of qubits must be positive integer.")
        couplings = [[i, i + 1] for i in range(num_qubits - 1)]
        top = Topology(num_qubits, couplings)
        top._named = f"line({num_qubits})"  # pylint: disable = protected-access
        return top

    @staticmethod
    def cycle(num_qubits: int) -> "Topology":
        """
        Creates a cycle topology on the given number of qubits.
        """
        if not isinstance(num_qubits, int) or num_qubits <= 0:
            raise TypeError("Number of qubits must be positive integer.")
        couplings = [[i, (i + 1) % num_qubits] for i in range(num_qubits)]
        top = Topology(num_qubits, couplings)
        top._named = f"cycle({num_qubits})"  # pylint: disable = protected-access
        return top

    @staticmethod
    def complete(num_qubits: int) -> "Topology":
        """
        Creates a complete topology on the given number of qubits.
        """
        if not isinstance(num_qubits, int) or num_qubits <= 0:
            raise TypeError("Number of qubits must be positive integer.")
        couplings = [
            [i, j] for i in range(num_qubits) for j in range(i + 1, num_qubits)
        ]
        top = Topology(num_qubits, couplings)
        top._named = f"complete({num_qubits})"  # pylint: disable = protected-access
        return top

    @staticmethod
    def grid(num_rows: int, num_cols: int) -> "Topology":
        """
        Creates a grid topology with the given number of rows and cols.
        Qubits are indexed by rows.
        """
        if not isinstance(num_rows, int) or num_rows <= 0:
            raise TypeError("Number of rows must be positive integer.")
        if not isinstance(num_cols, int) or num_cols <= 0:
            raise TypeError("Number of cols must be positive integer.")
        num_qubits = num_rows * num_cols

        def qubit(r, c):
            return num_cols * r + c

        couplings: List[List[int]] = []
        for r in range(num_rows):
            for c in range(num_cols):
                if r < num_rows - 1:
                    couplings.append([qubit(r, c), qubit(r + 1, c)])
                if c < num_cols - 1:
                    couplings.append([qubit(r, c), qubit(r, c + 1)])
        top = Topology(num_qubits, couplings)
        top._named = (
            f"grid({num_rows},{num_cols})"  # pylint: disable = protected-access
        )
        return top

    @staticmethod
    def periodic_grid(num_rows: int, num_cols: int) -> "Topology":
        """
        Creates a periodic grid topology with the given number of rows and cols.
        Qubits are indexed by rows.
        """
        if not isinstance(num_rows, int) or num_rows <= 0:
            raise TypeError("Number of rows must be positive integer.")
        if not isinstance(num_cols, int) or num_cols <= 0:
            raise TypeError("Number of cols must be positive integer.")
        num_qubits = num_rows * num_cols

        def qubit(r, c):
            return num_cols * r + c

        couplings: List[List[int]] = []
        for r in range(num_rows):
            for c in range(num_cols):
                couplings.append([qubit(r, c), qubit((r + 1) % num_rows, c)])
                couplings.append([qubit(r, c), qubit(r, (c + 1) % num_cols)])
        top = Topology(num_qubits, couplings)
        top._named = f"periodic_grid({num_rows},{num_cols})"  # pylint: disable = protected-access
        return top

    @staticmethod
    def from_qiskit_config(config) -> "Topology":
        """
        Static method to construct the topology from a
        Qiskit backend configuration.

        This method relies on the `qiskit` library being available.
        Specifically, the `config` argument must be of type
        `qiskit.providers.models.QasmBackendConfiguration`.
        """
        try:
            # pylint: disable = import-outside-toplevel
            from qiskit.providers.models import QasmBackendConfiguration  # type: ignore
        except ModuleNotFoundError as _:
            raise ModuleNotFoundError("You must install the 'qiskit' library.")
        if not isinstance(config, QasmBackendConfiguration):
            raise TypeError(
                "Argument backend must be of type "
                "`qiskit.providers.models.QasmBackendConfiguration`, "
                f"found {type(config)}."
            )
        config_dict = config.to_dict()
        num_qubits: int = config_dict["n_qubits"]
        coupling_map: List[List[int]] = config_dict["coupling_map"]
        return Topology(num_qubits, coupling_map)

    @staticmethod
    def from_qiskit_backend(backend) -> "Topology":
        """
        Static method to construct the topology from a Qiskit backend.

        This method relies on the `qiskit` library being available.
        Specifically, the `backend` argument must be of type
        `qiskit.providers.Backend`.
        """
        try:
            # pylint: disable = import-outside-toplevel, unused-import
            from qiskit.providers import Backend  # type: ignore
        except ModuleNotFoundError as _:
            raise ModuleNotFoundError("You must install the 'qiskit' library.")
        if not isinstance(backend, Backend):
            raise TypeError(
                "Argument backend must be of type " "`qiskit.providers.Backend`."
            )
        return Topology.from_qiskit_config(backend.configuration())


class Matching:
    """
    Mutable container class for a matching on a qubit topology.
    """

    _topology: Topology
    _matched_couplings: Set[Coupling]
    _matched_qubits: Set[int]
    _incident_coupling: List[Optional[Coupling]]

    def __init__(self, topology: Topology):
        if not isinstance(topology, Topology):
            raise TypeError(f"Expected Topology, found {topology}.")
        self._topology = topology
        self._matched_couplings = set()
        self._matched_qubits = set()
        self._incident_coupling = [None for _ in topology.qubits]

    @property
    def topology(self) -> Topology:
        """
        Readonly property exposing the qubit topology
        underlying this matching.
        """
        return self._topology

    @property
    def matched_couplings(self) -> FrozenSet[Coupling]:
        """
        Readonly property returning the collection of couplings
        currently in this matching.

        This collection is freshly generated at every call.
        """
        return frozenset(self._matched_couplings)

    @property
    def matched_qubits(self) -> FrozenSet[int]:
        """
        Readonly property returning the collection of qubits
        currently matched in this matching.

        This collection is freshly generated at every call.
        """
        return frozenset(self._matched_qubits)

    @property
    def flippable_couplings(self) -> FrozenSet[Coupling]:
        """
        Readonly property returning the collection of couplings
        that can be currently flipped in this matching, namely:

        - all couplings currently in the matching (will be removed by flip);
        - all couplings with both qubits currently not matched by the matching
          (will be added by flip).

        This collection is freshly generated at every call.
        """
        return frozenset(self._iter_flippable_couplings())

    def incident(self, qubit: int) -> Optional[Coupling]:
        """
        Returns the coupling incident to the given qubit in this matching,
        or `None` if the qubit is not matched.
        """
        return self._incident_coupling[qubit]

    def is_flippable(self, coupling: CouplingLike) -> bool:
        """
        Checks whether the coupling can be flipped:

        - always true if the coupling is already present in the matching;
        - otherwise true only if neither qubit in the coupling is currently matched.
        """
        coupling = _validate_coupling(self.topology.num_qubits, coupling)
        if coupling not in self.topology.couplings:
            raise ValueError(f"Invalid coupling {coupling} for the given topology.")
        return self._is_flippable(coupling)

    def flip(self, coupling: CouplingLike) -> "Matching":
        """
        Flips the given coupling in the matching (removes it if it is already present,
        adds it if it is not yed present and can be added).
        Raises `ValueError` if the coupling is not flippable.

        The matching is modified in-place and then returned, as per the
        [fluent API pattern](https://en.wikipedia.org/wiki/Fluent_interface).
        """
        coupling = _validate_coupling(self.topology.num_qubits, coupling)
        return self._flip(coupling)

    def _is_flippable(self, coupling: Coupling) -> bool:
        if coupling in self._matched_couplings:
            return True
        fst, snd = coupling
        if fst in self._matched_qubits:
            return False
        if snd in self._matched_qubits:
            return False
        return True

    def _flip(self, coupling: Coupling) -> "Matching":
        if not self._is_flippable(coupling):
            raise ValueError(
                f"Cannot add coupling {coupling} to matching:"
                f"one of the qubits is already matched. "
            )
        fst, snd = coupling
        if coupling in self._matched_couplings:
            self._matched_couplings.remove(coupling)
            self._matched_qubits.remove(fst)
            self._matched_qubits.remove(snd)
            self._incident_coupling[fst] = None
            self._incident_coupling[snd] = None
        else:
            self._matched_couplings.add(coupling)
            self._matched_qubits.add(fst)
            self._matched_qubits.add(snd)
            self._incident_coupling[fst] = coupling
            self._incident_coupling[snd] = coupling
        return self

    def _iter_flippable_couplings(self) -> Iterator[Coupling]:
        for coupling in self._topology.couplings:
            fst, snd = coupling
            if coupling in self._matched_couplings:
                yield coupling
            elif fst not in self._matched_qubits and snd not in self._matched_qubits:
                yield coupling
            else:
                # coupling is not flippable
                pass
