"""
    This module contains code to optimize circuits of mixed ZX phase gadgets
    using topologically-aware circuits of CNOTs.
"""

from collections import deque
from math import ceil, log10
from typing import (
    Callable,
    Deque,
    Dict,
    List,
    Optional,
    Protocol,
    Literal,
    Sequence,
    runtime_checkable,
    Set,
    Tuple,
    TypedDict,
    Union,
)
import numpy as np  # type: ignore
from pauliopt.phase.phase_circuits import PhaseCircuit, PhaseCircuitView
from pauliopt.phase.cx_circuits import CXCircuitLayer, CXCircuit, CXCircuitView
from pauliopt.topologies import Topology
from pauliopt.utils import (
    AngleVar,
    TempSchedule,
    StandardTempSchedule,
    StandardTempSchedules,
    SVGBuilder,
)


@runtime_checkable
class AnnealingCostLogger(Protocol):
    """
    Protocol for logger of initial/final cost in annealing.
    """

    def __call__(self, cx_count: int, num_iters: int): ...


@runtime_checkable
class AnnealingIterLogger(Protocol):
    """
    Protocol for logging of iteration info in annealing.
    """

    def __call__(
        self,
        it: int,
        prev_cx_count: int,
        new_cx_count: int,
        accepted: bool,
        flip: Tuple[int, Tuple[int, int]],
        t: float,
        num_iters: int,
    ):
        # pylint: disable = too-many-arguments
        ...


class AnnealingLoggers(TypedDict, total=False):
    """
    Typed dictionary of loggers for annealing.
    """

    log_start: AnnealingCostLogger
    log_iter: AnnealingIterLogger
    log_end: AnnealingCostLogger


def _validate_temp_schedule(
    schedule: Union[StandardTempSchedule, TempSchedule]
) -> TempSchedule:
    if not isinstance(schedule, TempSchedule):
        if not isinstance(schedule, tuple) or len(schedule) != 3:
            raise TypeError(
                f"Expected triple (schedule_name, t_init, t_final), "
                f"found {schedule}"
            )
        schedule_name, t_init, t_final = schedule
        if schedule_name not in StandardTempSchedules:
            raise TypeError(
                f"Invalid standard temperature schedule name {schedule_name}, "
                f"allowed names are: {list(StandardTempSchedules.keys())}"
            )
        if not isinstance(t_init, (int, float)) or not isinstance(
            t_final, (int, float)
        ):
            raise TypeError("Expected t_init and t_final to be int or float.")
        schedule = StandardTempSchedules[schedule_name](t_init, t_final)
    return schedule


def _validate_loggers(
    loggers: AnnealingLoggers,
) -> Tuple[
    Optional[AnnealingCostLogger],
    Optional[AnnealingIterLogger],
    Optional[AnnealingCostLogger],
]:
    log_start = loggers.get("log_start", None)
    log_iter = loggers.get("log_iter", None)
    log_end = loggers.get("log_end", None)
    if log_start is not None and not isinstance(log_start, AnnealingCostLogger):
        raise TypeError(f"Expected AnnealingCostLogger, found {type(log_start)}")
    if log_iter is not None and not isinstance(log_iter, AnnealingIterLogger):
        raise TypeError(f"Expected AnnealingCostLogger, found {type(log_iter)}")
    if log_end is not None and not isinstance(log_end, AnnealingCostLogger):
        raise TypeError(f"Expected AnnealingCostLogger, found {type(log_end)}")
    return log_start, log_iter, log_end


class OptimizedPhaseCircuit:
    # pylint: disable = too-many-instance-attributes
    """
    Container for a phase circuit to be progressively optimized.
    The original phase circuit is passed to the constructor, together
    with a qubit topology and a fixed number of layers constraining the
    CX circuit to be used for simplification.

    To understand the structure of the optimized phase circuit,
    consider the following code snippet:

    ```py
        opt_circ = PhaseCircuitCXBlockOptimizer(orig_circ, topology, num_cx_layers)
        # perform optimization using the methods of `opt_circ`
        phase_block = opt_circ.phase_block
        cx_block = opt_circ.cx_block
    ```

    The optimized circuit is obtained by composing three blocks:

    1. a first block of CX gates, given by `cx_block.dag()`
       (the same CX gates of `cx_block`, but in reverse order);
    2. a central block of phase gadgets, given by `phase_block`;
    3. a final block of CX gates, given by `cx_block`.

    An optional keyword argument `circuit_rep` (default: 1) can be passed to the
    constructor to indicate that the original circuit is to be repeated a certain
    number of times (default: 1). In the optimized circuit, this is achieved by
    repeating the `phase_block` part (at point 2. above) a number of times
    given by the `circuit_rep` argument.
    The first and last CX blocks are left unaltered (because the intermediate CX
    blocks would cancel each other out in pairs when repeating the optimized circuit).
    """

    _topology: Topology
    _circuit_rep: int
    _phase_block: PhaseCircuit
    _phase_block_view: PhaseCircuitView
    _cx_block: CXCircuit
    _cx_block_view: CXCircuitView
    _init_cx_count: int
    _cx_count: int
    _init_cx_blocks_count: int
    _cx_blocks_count: int
    _gadget_cx_count_cache: Dict[int, Dict[Tuple[int, ...], int]]
    _rng_seed: Optional[int]
    _rng: np.random.Generator
    _fresh_angle_vars: Union[None, Callable[[int], AngleVar]]
    _phase_method: Literal["naive", "paritysynth", "steiner-graysynth"]
    _cx_method: Literal["permrowcol", "naive"]
    _reallocate: bool
    _output_mapping: Sequence[int]
    _qubit_mapping: Sequence[int]

    def __init__(
        self,
        phase_block: Union[PhaseCircuit, PhaseCircuitView],
        topology: Topology,
        cx_block: Union[int, CXCircuit, CXCircuitView],
        *,
        circuit_rep: int = 1,
        rng_seed: Optional[int] = None,
        fresh_angle_vars: Union[None, str, Callable[[int], AngleVar]] = None,
        qubit_mapping: Sequence[int] = None,
        phase_method: Literal["naive", "paritysynth", "steiner-graysynth"] = "naive",
        cx_method: Literal["permrowcol", "naive"] = "naive",
        reallocate: bool = False,
    ):
        if not isinstance(phase_block, PhaseCircuit):
            raise TypeError(f"Expected PhaseCircuit, found {type(phase_block)}.")
        if not isinstance(topology, Topology):
            raise TypeError(f"Expected Topology, found {type(topology)}.")
        if not isinstance(circuit_rep, int) or circuit_rep <= 0:
            raise TypeError(f"Expected positive integer, found {circuit_rep}.")
        if not isinstance(cx_block, (int, CXCircuit, CXCircuitView)):
            raise TypeError(
                f"Expected int, CXCircuit or CXCircuitView, found {type(cx_block)}."
            )
        if isinstance(cx_block, int) and cx_block <= 0:
            raise TypeError(
                f"Expected positive integer number of CX layers, found {cx_block}."
            )
        if rng_seed is not None and not isinstance(rng_seed, int):
            raise TypeError("RNG seed must be integer or None.")
        self._qubit_mapping = (
            qubit_mapping if qubit_mapping else range(topology.num_qubits)
        )
        self._topology = topology.mapped_fwd(self._qubit_mapping)
        self._circuit_rep = circuit_rep
        self._phase_block = phase_block.cloned()
        if isinstance(cx_block, int):
            self._cx_block = CXCircuit(
                self._topology,
                [CXCircuitLayer(self._topology) for _ in range(cx_block)],
            )
        else:
            self._cx_block = cx_block.clone()
        self._rng_seed = rng_seed
        self._rng = np.random.default_rng(seed=rng_seed)
        self._phase_method = phase_method
        self._cx_method = cx_method
        self._reallocate = reallocate
        self._phase_block_view = PhaseCircuitView(self._phase_block)
        self._cx_block_view = CXCircuitView(self._cx_block)
        self._gadget_cx_count_cache = {}
        self._init_cx_count, self._init_cx_blocks_count = self._compute_cx_count()
        self._cx_count = self._init_cx_count
        self._cx_blocks_count = self._init_cx_blocks_count
        if isinstance(fresh_angle_vars, str):
            self._fresh_angle_vars = lambda i: AngleVar(
                f"{fresh_angle_vars}[{i}]", f"{fresh_angle_vars}_{i}"
            )
        else:
            self._fresh_angle_vars = fresh_angle_vars

    @property
    def topology(self) -> Topology:
        """
        Readonly property exposing the topology constraining the circuit optimization.
        """
        return self._topology

    @property
    def num_qubits(self) -> int:
        """
        Readonly property exposing the number of qubits spanned by the circuit to be optimized.
        """
        return self._phase_block.num_qubits

    @property
    def circuit_rep(self) -> int:
        """
        Readonly property exposing the number of times that the original circuit is
        to be repeated, for use when computing CX counts.
        """
        return self._circuit_rep

    @property
    def phase_block(self) -> PhaseCircuitView:
        """
        Readonly property exposing a readonly view on the phase block of the optimized circuit.
        """
        return self._phase_block_view

    @property
    def cx_block(self) -> CXCircuitView:
        """
        Readonly property exposing a readonly view on the CX block of the optimized circuit.
        """
        return self._cx_block_view

    @property
    def init_cx_count(self) -> int:
        """
        Readonly property exposing the CX count for the original circuit.
        """
        return self._init_cx_count

    @property
    def cx_count(self) -> int:
        """
        Readonly property exposing the current CX count for the optimized circuit.
        """
        return self._cx_count

    @property
    def init_cx_blocks_count(self) -> int:
        """
        Readonly property exposing the overall CX count for the two conjugating
        CX blocks at the time the circuit was instantiated.
        """
        return self._init_cx_blocks_count

    @property
    def cx_blocks_count(self) -> int:
        """
        Readonly property exposing the overall CX count for the conjugating
        CX blocks in the currently optimized circuit.
        """
        return self._cx_blocks_count

    @property
    def init_phase_block_cx_count(self) -> int:
        """
        Readonly property exposing the overall CX count for a single
        phase block at the time the circuit was instantiated.
        """
        return (self.init_cx_count - self.init_cx_blocks_count) // self.circuit_rep

    @property
    def phase_block_cx_count(self) -> int:
        """
        Readonly property exposing the overall CX count for a single
        phase block in the currently optimized circuit.
        """
        return (self.cx_count - self.cx_blocks_count) // self.circuit_rep

    def clone(self, rng_seed: Optional[int] = None) -> "OptimizedPhaseCircuit":
        """
        Returns a copy of this optimized phase circuit.
        """
        return OptimizedPhaseCircuit(
            self.phase_block,
            self.topology,
            self.cx_block,
            circuit_rep=self.circuit_rep,
            rng_seed=rng_seed,
        )

    def to_qiskit(self, simplified: bool = True):
        """
        Returns the optimized circuit as a Qiskit circuit.

        This method relies on the `qiskit` library being available.
        Specifically, the `circuit` argument must be of type
        `qiskit.providers.BaseBackend`.
        """
        try:
            # pylint: disable = import-outside-toplevel
            from qiskit.circuit import QuantumCircuit  # type: ignore
        except ModuleNotFoundError as _:
            raise ModuleNotFoundError("You must install the 'qiskit' library.")
        if simplified:
            self.simplify()
        circuit = QuantumCircuit(self.num_qubits)
        front_cx_block = CXCircuit(self.topology, self._cx_block)
        synthesized_cx_block1 = front_cx_block.to_qiskit(
            self._cx_method, reallocate=self._reallocate
        )
        synthesized_phase_block, cxs = self._phase_block.to_qiskit(
            self.topology,
            simplified=simplified,
            method=self._phase_method,
            return_cx=True,
        )
        for layer in self._cx_block:
            cxs >>= layer
        synthesized_cx_block2 = cxs.to_qiskit(
            self._cx_method, reallocate=self._reallocate
        )
        circuit.compose(synthesized_cx_block1.inverse(), inplace=True)
        for _ in range(self._circuit_rep):
            circuit.compose(synthesized_phase_block, inplace=True)
        circuit.compose(synthesized_cx_block2, inplace=True)
        if self._reallocate:
            undo_mapping = [
                self._qubit_mapping.index(i) for i in range(self.num_qubits)
            ]
            initial_layout = [
                undo_mapping[i] for i in synthesized_cx_block1.metadata["final_layout"]
            ]
            final_layout = [
                undo_mapping[i] for i in synthesized_cx_block2.metadata["final_layout"]
            ]
            circuit.metadata = {
                "initial_layout": initial_layout,
                "final_layout": final_layout,
            }
        return circuit

    def simplify(self):
        """
        Simplifies the phase block according to the commutation and fusion
        rules for phase gadgets.
        """
        new_phase_block = self._phase_block.cloned()
        for _ in range(1, self._circuit_rep):
            new_phase_block >>= self._phase_block
        self._phase_block = new_phase_block.simplified()
        self._phase_block_view = PhaseCircuitView(self._phase_block)
        self._circuit_rep = 1
        self._gadget_cx_count_cache = {}
        self._cx_count, self._cx_blocks_count = self._compute_cx_count()

    def anneal(
        self,
        num_iters: int,
        *,
        schedule: Union[StandardTempSchedule, TempSchedule] = ("linear", 1.0, 0.1),
        loggers: AnnealingLoggers = {},
    ):
        # pylint: disable = dangerous-default-value
        # pylint: disable = too-many-locals
        """
        Performs a cycle of simulated annealing optimization,
        using the given number of iterations, temperature schedule,
        initial/final temperatures.

        The circuit is modified in-place and then returned, as per the
        [fluent API pattern](https://en.wikipedia.org/wiki/Fluent_interface).
        """
        # Validate arguments:
        if not isinstance(num_iters, int) or num_iters <= 0:
            raise TypeError(f"Expected a positive integer, found {num_iters}.")
        schedule = _validate_temp_schedule(schedule)
        log_start, log_iter, log_end = _validate_loggers(loggers)
        # Log start:
        if log_start is not None:
            log_start(self._cx_count, num_iters)
        # Pre-sample random numbers to use in iterations:
        rand = self._rng.uniform(size=num_iters)
        # Run iterations:
        for it in range(num_iters):
            old_output_mapping = [i for i in self._output_mapping]
            t = schedule(it, num_iters=num_iters)
            layer_idx, (ctrl, trgt) = self.random_flip_cx()
            new_cx_count, new_cx_blocks_count = self._compute_cx_count()
            cx_count_diff = new_cx_count - self._cx_count
            accept_step = cx_count_diff < 0 or rand[it] < np.exp(
                -np.log(2) * cx_count_diff / t
            )
            if log_iter is not None:
                log_iter(
                    it,
                    self._cx_count,
                    new_cx_count,
                    accept_step,
                    (layer_idx, (ctrl, trgt)),
                    t,
                    num_iters,
                )
            if accept_step:
                # Accept changes:
                self._cx_count = new_cx_count
                self._cx_blocks_count = new_cx_blocks_count
            else:
                # Undo changes:
                self._flip_cx(layer_idx, ctrl, trgt)
                if self._reallocate:
                    self._output_mapping = old_output_mapping
        # Log end:
        if log_end is not None:
            log_end(self._cx_count, num_iters)

    def random_flip_cx(self) -> Tuple[int, Tuple[int, int]]:
        """
        Randomly flips a CX gate in the CX circuit used for the optimization,
        updating both the CX circuit and the circuit being optimized.

        Returns the layer index and gate (pair of control and target) that were
        flipped (e.g. in case the flip needs to be subsequently undone).
        """
        while True:
            layer_idx = int(self._rng.integers(len(self._cx_block)))
            ctrl, trgt = self._cx_block[layer_idx].random_flip_cx(self._rng)
            if layer_idx < len(self._cx_block) - 1 and self._cx_block[
                layer_idx + 1
            ].has_cx(ctrl, trgt):
                # Try again if CX gate already present in layer above (to avoid redundancy)
                continue
            if layer_idx > 0 and self._cx_block[layer_idx - 1].has_cx(ctrl, trgt):
                # Try again if CX gate already present in layer below (to avoid redundancy)
                continue
            self._flip_cx(layer_idx, ctrl, trgt)
            return layer_idx, (ctrl, trgt)

    def is_cx_flippable(self, layer_idx: int, ctrl: int, trgt: int) -> bool:
        """
        Checks whether the given CX gate can be flipped in the given layer.
        """
        if not isinstance(layer_idx, int) or not 0 <= layer_idx < len(self._cx_block):
            raise TypeError(f"Invalid layer index {layer_idx} for CX circuit.")
        layer = self._cx_block[layer_idx]
        return layer.is_cx_flippable(ctrl, trgt)

    def flip_cx(self, layer_idx: int, ctrl: int, trgt: int):
        """
        Performs the actions needed to flip the given CX gate in the given layer
        of the CX circuit used for the optimization:

        - undoes all gates in layers subsequent to the given layer which are
          causally following the given gate, starting from the last layer and
          working backwards towards the gate;
        - applies the desired gate;
        - redoes all gate undone, in reverse order (starting from the gate and
          working forwards towards the last layer).
        """
        if not self.is_cx_flippable(layer_idx, ctrl, trgt):
            raise ValueError(
                f"Gate {(ctrl, trgt)} cannot be flipped in layer number {layer_idx}."
            )
        self._flip_cx(layer_idx, ctrl, trgt)

    def _flip_cx(self, layer_idx: int, ctrl: int, trgt: int) -> None:
        conj_by: Deque[Tuple[int, int]] = deque([(ctrl, trgt)])
        qubits_spanned: Set[int] = set([ctrl, trgt])
        for layer in reversed(self._cx_block[:layer_idx]):
            new_qubits_spanned: Set[int] = set()
            incident_gates: Set[Tuple[int, int]] = set()
            for q in qubits_spanned:
                incident_gate = layer.incident(q)
                if incident_gate is not None:
                    incident_gates.add(incident_gate)
                    new_qubits_spanned.update({incident_gate[0], incident_gate[1]})
            for incident_gate in incident_gates:
                conj_by.appendleft(incident_gate)  # will first undo the gate ...
                # ... then do all gates already in conj_by ...
                conj_by.append(incident_gate)  # ... then finally redo the gate
            qubits_spanned.update(new_qubits_spanned)
        # Flip the gate in the CX circuit:
        self._cx_block[layer_idx].flip_cx(ctrl, trgt)
        # Conjugate the optimized phase gadget circuit by all necessary gates:
        for cx in conj_by:
            self._phase_block.conj_by_cx(*cx)

    def _compute_cx_count(self) -> Tuple[int, int]:
        # pylint: disable = protected-access
        self._output_mapping = list(range(self.num_qubits))
        if self._cx_method == "naive":
            front_cx_count = self._cx_block.num_gates
        else:
            front_cx_count = CXCircuit.from_parity_matrix(
                self._cx_block.parity_matrix(),
                self.topology,
                reallocate=self._reallocate,
                method=self._cx_method,
            ).num_gates
        phase_qiskit, cxs = self._phase_block.to_qiskit(
            self.topology,
            simplified=False,
            method=self._phase_method,
            cx_synth="naive",
            reallocate=self._reallocate,
            return_cx=True,
        )
        counts = phase_qiskit.count_ops()
        if "cx" in counts:
            phase_cx_count = counts["cx"]
        else:
            phase_cx_count = 0
        cxs >>= self._cx_block
        if self._cx_method == "naive":
            new_cxs = cxs
        else:
            new_cxs = CXCircuit.from_parity_matrix(
                cxs.parity_matrix(),
                self.topology,
                reallocate=self._reallocate,
                method=self._cx_method,
            )
        back_cx_count = new_cxs.num_gates
        cx_count = front_cx_count + phase_cx_count + back_cx_count
        if self._reallocate:
            self._output_mapping = new_cxs._output_mapping
        return cx_count, 2 * front_cx_count

    def to_svg(
        self,
        *,
        zcolor: str = "#CCFFCC",
        xcolor: str = "#FF8888",
        hscale: float = 1.0,
        vscale: float = 1.0,
        scale: float = 1.0,
        svg_code_only: bool = False,
    ):
        """
        Returns an SVG representation of this optimized circuit, using
        the ZX calculus to express phase gadgets and CX gates.

        The keyword arguments `zcolor` and `xcolor` can be used to
        specify a colour for the Z and X basis spiders in the circuit.
        The keyword arguments `hscale` and `vscale` can be used to
        scale the circuit representation horizontally and vertically.
        The keyword argument `scale` can be used to scale the circuit
        representation isotropically.
        The keyword argument `svg_code_only` (default `False`) can be used
        to specify that the SVG code itself be returned, rather than the
        IPython `SVG` object.
        """
        if not isinstance(zcolor, str):
            raise TypeError("Keyword argument 'zcolor' must be string.")
        if not isinstance(xcolor, str):
            raise TypeError("Keyword argument 'xcolor' must be string.")
        if not isinstance(hscale, (int, float)) or hscale <= 0.0:
            raise TypeError("Keyword argument 'hscale' must be positive float.")
        if not isinstance(vscale, (int, float)) or vscale <= 0.0:
            raise TypeError("Keyword argument 'vscale' must be positive float.")
        if not isinstance(scale, (int, float)) or scale <= 0.0:
            raise TypeError("Keyword argument 'scale' must be positive float.")
        return self._to_svg(
            zcolor=zcolor,
            xcolor=xcolor,
            hscale=hscale,
            vscale=vscale,
            scale=scale,
            svg_code_only=svg_code_only,
        )

    def __eq__(self, other) -> bool:
        if self is other:
            return True
        if not isinstance(other, OptimizedPhaseCircuit):
            return NotImplemented
        if self.circuit_rep != other.circuit_rep:
            return False
        if self.phase_block != other.phase_block:
            return False
        if self.cx_block != other.cx_block:
            return False
        return True

    def _to_svg(
        self,
        *,
        zcolor: str = "#CCFFCC",
        xcolor: str = "#FF8888",
        hscale: float = 1.0,
        vscale: float = 1.0,
        scale: float = 1.0,
        svg_code_only: bool = False,
    ):
        # pylint: disable = too-many-locals, too-many-statements, too-many-branches
        # TODO: reuse from phase circuit once cleaned up and restructured
        num_qubits = self.num_qubits
        vscale *= scale
        hscale *= scale
        cx_block = self._cx_block
        phase_block = self._phase_block
        pre_cx_gates = [gate for layer in reversed(cx_block) for gate in layer.gates]
        gadgets = list(phase_block.gadgets) * self._circuit_rep
        post_cx_gates = list(reversed(pre_cx_gates))
        _layers: List[int] = [0 for _ in range(num_qubits)]
        pre_cx_gates_depths: List[int] = []
        max_cx_gates_depth: int = 0
        for gate in pre_cx_gates:
            m = min(gate)
            M = max(gate)
            d = max(_layers[q] for q in range(m, M + 1))
            max_cx_gates_depth = max(max_cx_gates_depth, d + 1)
            pre_cx_gates_depths.append(d)
            for q in range(m, M + 1):
                _layers[q] = d + 1
        num_digits = int(ceil(log10(num_qubits)))
        line_height = int(ceil(30 * vscale))
        row_width = int(ceil(120 * hscale))
        cx_row_width = int(ceil(40 * hscale))
        pad_x = int(ceil(10 * hscale))
        margin_x = int(ceil(40 * hscale))
        pad_y = int(ceil(20 * vscale))
        r = pad_y // 2 - 2
        font_size = 2 * r
        pad_x += font_size * (num_digits + 1)
        delta_fst = row_width // 4
        delta_snd = 2 * row_width // 4
        width = (
            2 * pad_x
            + 2 * margin_x
            + row_width * len(gadgets)
            + 2 * max_cx_gates_depth * cx_row_width
        )
        height = pad_y + line_height * (num_qubits + 1)
        builder = SVGBuilder(width, height)
        levels: List[int] = [0 for _ in range(num_qubits)]
        max_lvl = 0
        base_x = pad_x + margin_x
        for ctrl, trgt in pre_cx_gates:
            qubit_span = range(min(ctrl, trgt), max(ctrl, trgt) + 1)
            lvl = max(levels[q] for q in qubit_span)
            max_lvl = max(max_lvl, lvl)
            x = base_x + lvl * row_width // 3
            for q in qubit_span:
                levels[q] = lvl + 1
            y_ctrl = pad_y + (ctrl + 1) * line_height
            y_trgt = pad_y + (trgt + 1) * line_height
            builder.line((x, y_ctrl), (x, y_trgt))
            builder.circle((x, y_ctrl), r, zcolor)
            builder.circle((x, y_trgt), r, xcolor)
        base_x = base_x + (max_lvl + 1) * row_width // 3
        levels = [0 for _ in range(num_qubits)]
        max_lvl = 0
        for i, gadget in enumerate(gadgets):
            angle = gadget.angle
            if isinstance(angle, AngleVar) and self._fresh_angle_vars is not None:
                angle = self._fresh_angle_vars(i)
            fill = zcolor if gadget.basis == "Z" else xcolor
            other_fill = xcolor if gadget.basis == "Z" else zcolor
            qubit_span = range(min(gadget.qubits), max(gadget.qubits) + 1)
            lvl = max(levels[q] for q in qubit_span)
            max_lvl = max(max_lvl, lvl)
            x = base_x + lvl * row_width
            for q in qubit_span:
                levels[q] = lvl + 1
            if len(gadget.qubits) > 1:
                text_y = pad_y + min(gadget.qubits) * line_height + line_height // 2
                for q in gadget.qubits:
                    y = pad_y + (q + 1) * line_height
                    builder.line((x, y), (x + delta_fst, text_y))
                for q in gadget.qubits:
                    y = pad_y + (q + 1) * line_height
                    builder.circle((x, y), r, fill)
                builder.line((x + delta_fst, text_y), (x + delta_snd, text_y))
                builder.circle((x + delta_fst, text_y), r, other_fill)
                builder.circle((x + delta_snd, text_y), r, fill)
                builder.text(
                    (x + delta_snd + 2 * r, text_y), str(angle), font_size=font_size
                )
            else:
                for q in gadget.qubits:
                    y = pad_y + (q + 1) * line_height
                    builder.circle((x, y), r, fill)
                builder.text(
                    (x + r, y - line_height // 3), str(angle), font_size=font_size
                )
        base_x = base_x + (max_lvl + 1) * row_width
        levels = [0 for _ in range(num_qubits)]
        max_lvl = 0
        for ctrl, trgt in post_cx_gates:
            qubit_span = range(min(ctrl, trgt), max(ctrl, trgt) + 1)
            lvl = max(levels[q] for q in qubit_span)
            max_lvl = max(max_lvl, lvl)
            x = base_x + lvl * row_width // 3
            for q in qubit_span:
                levels[q] = lvl + 1
            y_ctrl = pad_y + (ctrl + 1) * line_height
            y_trgt = pad_y + (trgt + 1) * line_height
            builder.line((x, y_ctrl), (x, y_trgt))
            builder.circle((x, y_ctrl), r, zcolor)
            builder.circle((x, y_trgt), r, xcolor)
        width = base_x + max_lvl * row_width // 3 + pad_x + margin_x
        _builder = SVGBuilder(width, height)
        for q in range(num_qubits):
            y = pad_y + (q + 1) * line_height
            _builder.line((pad_x, y), (width - pad_x, y))
            _builder.text((0, y), f"{str(q):>{num_digits}}", font_size=font_size)
            _builder.text(
                (width - pad_x + r, y), f"{str(q):>{num_digits}}", font_size=font_size
            )
        _builder >>= builder
        svg_code = repr(_builder)
        if svg_code_only:
            return svg_code
        try:
            # pylint: disable = import-outside-toplevel
            from IPython.core.display import SVG  # type: ignore
        except ModuleNotFoundError as _:
            raise ModuleNotFoundError("You must install the 'IPython' library.")
        return SVG(svg_code)

    def _repr_svg_(self):
        """
        Magic method for IPython/Jupyter pretty-printing.
        See https://ipython.readthedocs.io/en/stable/api/generated/IPython.display.html
        """
        return self._to_svg(svg_code_only=True)


def iter_anneal(
    circuit: PhaseCircuit,
    topology: Topology,
    cx_blocks,
    num_iters: int,
    num_anneal_iters: int,
    opt_kwargs: Dict = {},
    anneal_kwargs: Dict = {},
) -> OptimizedPhaseCircuit:
    topology = Topology(topology.num_qubits, topology.couplings)
    best_circuit, best_cx_count = None, None
    for _ in range(num_iters):
        opt = OptimizedPhaseCircuit(circuit.copy(), topology, cx_blocks, **opt_kwargs)
        opt.anneal(num_anneal_iters, **anneal_kwargs)

        if best_cx_count is None or best_cx_count > opt.cx_count:
            best_circuit = opt
            best_cx_count = opt.cx_count
    return best_circuit


def reverse_traversal_anneal(
    circuit: PhaseCircuit,
    topology: Topology,
    cx_blocks,
    num_iters: int,
    num_anneal_iters: int,
    opt_kwargs: Dict = {},
    anneal_kwargs: Dict = {},
) -> OptimizedPhaseCircuit:
    topology = Topology(topology.num_qubits, topology.couplings)
    best_circuit, best_cx_count, best_reversed = None, None, None
    mapping = None
    reversed_circuit = False
    reversed_phase = PhaseCircuit(circuit.num_qubits, list(reversed(circuit)))
    if isinstance(cx_blocks, int):
        reversed_cx_blocks = cx_blocks
    else:
        reversed_cx_blocks = list(reversed(cx_blocks))
    for it in range(num_iters):
        if reversed_circuit:
            opt = OptimizedPhaseCircuit(
                reversed_phase.copy(),
                topology,
                reversed_cx_blocks,
                qubit_mapping=mapping,
                **opt_kwargs,
            )
        else:
            opt = OptimizedPhaseCircuit(
                circuit.copy(), topology, cx_blocks, qubit_mapping=mapping, **opt_kwargs
            )
        if cx_blocks and num_anneal_iters:
            opt.anneal(num_anneal_iters, **anneal_kwargs)

        if not best_cx_count or best_cx_count > opt.cx_count:
            best_circuit = opt
            best_cx_count = opt.cx_count
            best_reversed = reversed_circuit

        mapping = opt._output_mapping
        reversed_circuit = not reversed_circuit
    if best_reversed:
        reversed_phase = PhaseCircuit(
            best_circuit.num_qubits, list(reversed(best_circuit._phase_block))
        )
        reversed_cx_blocks = CXCircuit(
            best_circuit.topology, list(reversed(best_circuit._cx_block))
        )
        mapping = best_circuit._qubit_mapping
        best_circuit = OptimizedPhaseCircuit(
            reversed_phase, best_circuit.topology, reversed_cx_blocks, **opt_kwargs
        )
        best_circuit._qubit_mapping = mapping
    return best_circuit


def reverse_traversal(
    circuit: PhaseCircuit,
    topology: Topology,
    cx_blocks,
    num_iters: int,
    num_anneal_iters=0,
    opt_kwargs: Dict = {},
    anneal_kwargs: Dict = {},
) -> OptimizedPhaseCircuit:
    opt = reverse_traversal_anneal(
        circuit, topology, cx_blocks, num_iters, 0, opt_kwargs, anneal_kwargs
    )
    if num_anneal_iters > 0:
        opt.anneal(num_anneal_iters, **anneal_kwargs)
    return opt
