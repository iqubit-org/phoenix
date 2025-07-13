from abc import ABC, abstractmethod
from itertools import combinations
from math import ceil
from typing import List, Union

from pauliopt.pauli_strings import Pauli
from pauliopt.phase import X as XHead
from pauliopt.phase import Z as ZHead
from pauliopt.phase import pi
from pauliopt.phase.phase_circuits import PhaseGadget
from pauliopt.utils import Angle


class Gate(ABC):
    """Base class for quantum gates."""

    def __init__(self, *qubits):
        self.qubits = qubits
        self.n_qubits = len(qubits)
        self.name = self.__class__.__name__
        if len(qubits) != self.n_qubits:
            name, n_qubits = self.name, self.n_qubits
            raise ValueError(f"{name} takes {n_qubits} qubits, not {len(qubits)}.")

    def __repr__(self) -> str:
        args = map(repr, self.qubits)
        return f"{self.name}({', '.join(args)})"

    @property
    def width(self):
        """Width of gate used for drawing."""
        if getattr(self, "draw_as_zx", False):
            n_columns = max(self._gen_columns())
            return n_columns * 20 + 40

        vscale = 1.0
        line_height = int(ceil(30 * vscale))
        min_qubit = min(self.qubits)
        max_qubit = max(self.qubits)
        text = repr(self)
        box_height = max(30, (max_qubit - min_qubit) * line_height + 10)
        box_width = max(8 * len(text), box_height * 1.5)
        return int(box_width)

    def _gen_columns(self):
        n_columns = 0
        gate_cls = None
        # shift by 3 columns if a multi-legged gadget was drawn
        big_gadget_drawn = False
        for g in self.decomp:
            if gate_cls != (type(g), getattr(g, "basis", None)):
                gate_cls = (type(g), getattr(g, "basis", None))
                n_columns += 3 if big_gadget_drawn else 1
                big_gadget_drawn = False
            if isinstance(g, PhaseGadget) and len(g.qubits) > 1:
                big_gadget_drawn = True
            yield n_columns
        n_columns += 3 if big_gadget_drawn else 1
        big_gadget_drawn = False
        yield n_columns

    def draw_zx(self, builder, base, row_width, params):
        n_columns = max(self._gen_columns())
        step = row_width / n_columns

        busy = set()
        for g, column_idx in zip(self.decomp, self._gen_columns()):
            x, y = base[0] + column_idx * step, base[1]

            if len(g.qubits) > 1:
                # gadget body blocks text
                busy.add(min(g.qubits) + 1)
            text_above = True
            if len(g.qubits) == 1 and tuple(g.qubits)[0] in busy:
                text_above = False
            if isinstance(g, PhaseGadget):
                draw_gadget(builder, (x, y), row_width, params, g, text_above)
            else:
                g.draw(builder, base, row_width, params)

    def draw_box(self, builder, base, row_width, params):
        line_height = params["line_height"]
        font_size = params["font_size"]

        min_qubit = min(self.qubits)
        max_qubit = max(self.qubits)
        x = base[0] + row_width / 2
        y = base[1] + ((min_qubit + max_qubit) / 2 + 1) * line_height
        text = repr(self)
        box_height = max(30, (max_qubit - min_qubit) * line_height + 10)
        box_width = max(8 * len(text), box_height * 1.5)

        builder.rect((x, y), box_width, box_height, "#FFFFFF")
        builder.text((x, y), text, font_size=font_size, center=True)

    def draw(self, builder, base, row_width, params):
        # builder.line((base[0], 0), (base[0], 100))
        # builder.line((base[0] + row_width, 0), (base[0] + row_width, 100))
        if getattr(self, "draw_as_zx", False):
            self.draw_zx(builder, base, row_width, params)
        else:
            self.draw_box(builder, base, row_width, params)

    @property
    def gadgets(self):
        """List of gadgets used to implement this gate."""
        if not hasattr(self, "decomp"):
            raise NotImplementedError

        gadgets = [
            g if isinstance(g, (ZHead, XHead)) else g.gadgets for g in self.decomp
        ]
        return gadgets

    @abstractmethod
    def to_qiskit(self):
        pass

    @abstractmethod
    def inverse(self):
        pass

    def apply_permutation(self, permutation: list) -> None:
        register = list(range(len(permutation)))
        self.qubits = tuple(
            [permutation[register.index(qubit)] for qubit in self.qubits]
        )

    def copy(self):
        return self.__class__(*self.qubits)


class PhaseGate(Gate, ABC):
    def __init__(self, phase: Angle, *qubits):
        super().__init__(*qubits)
        self.phase = phase

    def __repr__(self) -> str:
        args = map(repr, self.qubits)
        return f"{self.name}({self.phase}, {', '.join(args)})"

    def get_phase_as_float(self):
        return self.phase if not isinstance(self.phase, Angle) else float(self.phase)

    def get_phase_as_angle(self):
        return self.phase if isinstance(self.phase, Angle) else Angle(self.phase)

    def inverse(self):
        return self.__class__(-self.phase, *self.qubits)

    def copy(self):
        return self.__class__(self.phase, self.qubits)


PROPAGATION_H = {
    "X": (Pauli.Z, 1),
    "Y": (Pauli.Y, -1),
    "Z": (Pauli.X, 1),
    "I": (Pauli.I, 1),
}

PROPAGATION_S = {
    "X": (Pauli.Y, -1),
    "Y": (Pauli.X, 1),
    "Z": (Pauli.Z, 1),
    "I": (Pauli.I, 1),
}

PROPAGATION_CX = {
    "XX": (Pauli.X, Pauli.I, 1),
    "XY": (Pauli.Y, Pauli.Z, 1),
    "XZ": (Pauli.Y, Pauli.Y, -1),
    "XI": (Pauli.X, Pauli.X, 1),
    "YX": (Pauli.Y, Pauli.I, 1),
    "YY": (Pauli.X, Pauli.Z, -1),
    "YZ": (Pauli.X, Pauli.Y, 1),
    "YI": (Pauli.Y, Pauli.X, 1),
    "ZX": (Pauli.Z, Pauli.X, 1),
    "ZY": (Pauli.I, Pauli.Y, 1),
    "ZZ": (Pauli.I, Pauli.Z, 1),
    "ZI": (Pauli.Z, Pauli.I, 1),
    "IX": (Pauli.I, Pauli.X, 1),
    "IY": (Pauli.Z, Pauli.Y, 1),
    "IZ": (Pauli.Z, Pauli.Z, 1),
    "II": (Pauli.I, Pauli.I, 1),
}


class CliffordGate(Gate, ABC):
    def __init__(self, *qubits):
        super().__init__(*qubits)

    @abstractmethod
    def get_h_s_cx_decomposition(self) -> List[Union["H", "S", "CX"]]:
        """
        Every clifford must be decomposable into a list of H, S and CX gates.
        Returns:

        """
        pass

    def propagate_pauli(self, gadget: "pauliopt.pauli.pauli_gadget.PauliGadget"):
        """
        Propagate a pauli gate through a gadget using the H, S, CX decomposition rules.

        One can define for H, S and CX propagation rules, which are defined in the dictionaries above.
        Args:
            gadget:

        Returns:

        """
        h_s_cx_decomposition = self.get_h_s_cx_decomposition()
        for gate in reversed(h_s_cx_decomposition):
            if gate.name == "H":
                assert isinstance(gate, SingleQubitClifford)
                p_string = gadget.paulis[gate.qubit].value
                new_p, phase_change = PROPAGATION_H[p_string]
                gadget.paulis[gate.qubit] = new_p
                if phase_change == -1:
                    gadget.angle *= phase_change
            elif gate.name == "S":
                assert isinstance(gate, SingleQubitClifford)
                p_string = gadget.paulis[gate.qubit].value
                new_p, phase_change = PROPAGATION_S[p_string]
                gadget.paulis[gate.qubit] = new_p
                if phase_change == -1:
                    gadget.angle *= phase_change
            elif gate.name == "CX":
                assert isinstance(gate, TwoQubitClifford)
                p_string = (
                    gadget.paulis[gate.control].value + gadget.paulis[gate.target].value
                )
                p_c, p_t, phase_change = PROPAGATION_CX[p_string]
                gadget.paulis[gate.control] = p_c
                gadget.paulis[gate.target] = p_t
                if phase_change == -1:
                    gadget.angle *= phase_change

        return gadget


class SingleQubitClifford(CliffordGate, ABC):

    def __init__(self, qubit: int):
        super().__init__((qubit))

    @property
    def qubit(self):
        return self.qubits[0]


class TwoQubitClifford(CliffordGate, ABC):
    def __init__(self, control, target):
        qubits = (control, target)
        super().__init__(*qubits)

    @property
    def control(self):
        return self.qubits[0]

    @property
    def target(self):
        return self.qubits[1]

    def copy(self):
        return self.__class__(self.control, self.target)


class H(SingleQubitClifford):
    n_qubits = 1
    width = 40

    @property
    def decomp(self):
        p = pi / 2
        (q,) = self.qubits
        return [ZHead(p) @ {q}, XHead(p) @ {q}, ZHead(p) @ {q}]

    def draw(self, builder, base, row_width, params):
        line_height = params["line_height"]
        r = params["r"]
        hcolor = params["hcolor"]
        qubit = self.qubits[0]
        x = base[0] + row_width / 2
        y = base[1] + (qubit + 1) * line_height
        builder.rect((x, y), 1.5 * r, 1.5 * r, hcolor)

    def to_qiskit(self):
        try:
            from qiskit.circuit.library import HGate
        except ImportError:
            raise ImportError("Please install qiskit to use this feature.")

        return HGate(), self.qubits

    def inverse(self):
        return H(*self.qubits)

    def get_h_s_cx_decomposition(self) -> List[Union["H", "S", "CX"]]:
        return [H(*self.qubits)]


class X(SingleQubitClifford):
    n_qubits = 1
    draw_as_zx = True

    @property
    def decomp(self):
        return [XHead(pi) @ {self.qubits[0]}]

    def to_qiskit(self):
        try:
            from qiskit.circuit.library import XGate
        except ImportError:
            raise ImportError("Please install qiskit to use this feature.")

        return XGate(), self.qubits

    def inverse(self):
        return X(*self.qubits)

    def get_h_s_cx_decomposition(self):
        return [H(*self.qubits), S(*self.qubits), S(*self.qubits), H(*self.qubits)]


class Z(SingleQubitClifford):
    n_qubits = 1
    draw_as_zx = True

    @property
    def decomp(self):
        return [ZHead(pi) @ {self.qubits[0]}]

    def to_qiskit(self):
        try:
            from qiskit.circuit.library import ZGate
        except ImportError:
            raise ImportError("Please install qiskit to use this feature.")

        return ZGate(), self.qubits

    def inverse(self):
        return Z(*self.qubits)

    def get_h_s_cx_decomposition(self) -> List[Union["H", "S", "CX"]]:
        return [S(*self.qubits), S(*self.qubits)]


class Y(SingleQubitClifford):
    n_qubits = 1
    draw_as_zx = True

    @property
    def decomp(self):
        # TODO check this
        return [ZHead(pi) @ {self.qubits[0]}, XHead(pi) @ {self.qubits[0]}]

    def to_qiskit(self):
        try:
            from qiskit.circuit.library import YGate
        except ImportError:
            raise ImportError("Please install qiskit to use this feature.")

        return YGate(), self.qubits

    def inverse(self):
        return Y(*self.qubits)

    def get_h_s_cx_decomposition(self) -> List[Union["H", "S", "CX"]]:
        return (
            Sdg(*self.qubits).get_h_s_cx_decomposition()
            + X(*self.qubits).get_h_s_cx_decomposition()
            + S(*self.qubits).get_h_s_cx_decomposition()
        )


class S(SingleQubitClifford):
    n_qubits = 1
    draw_as_zx = True

    @property
    def decomp(self):
        return [ZHead(pi / 2) @ {self.qubits[0]}]

    def to_qiskit(self):
        try:
            from qiskit.circuit.library import SGate
        except ImportError:
            raise ImportError("Please install qiskit to use this feature.")

        return SGate(), self.qubits

    def inverse(self):
        return Sdg(*self.qubits)

    def get_h_s_cx_decomposition(self) -> List[Union["H", "S", "CX"]]:
        return [S(*self.qubits)]


class Sdg(SingleQubitClifford):
    n_qubits = 1
    draw_as_zx = True

    @property
    def decomp(self):
        return [ZHead(-pi / 2) @ {self.qubits[0]}]

    def to_qiskit(self):
        try:
            from qiskit.circuit.library import SdgGate
        except ImportError:
            raise ImportError("Please install qiskit to use this feature.")

        return SdgGate(), self.qubits

    def inverse(self):
        return S(*self.qubits)

    def get_h_s_cx_decomposition(self) -> List[Union["H", "S", "CX"]]:
        return [S(*self.qubits), S(*self.qubits), S(*self.qubits)]


class V(SingleQubitClifford):
    n_qubits = 1
    draw_as_zx = True

    @property
    def decomp(self):
        return [XHead(pi / 2) @ {self.qubits[0]}]

    def to_qiskit(self):
        try:
            from qiskit.circuit.library import SXGate
        except ImportError:
            raise ImportError("Please install qiskit to use this feature.")

        return SXGate(), self.qubits

    def inverse(self):
        return Vdg(*self.qubits)

    def get_h_s_cx_decomposition(self) -> List[Union["H", "S", "CX"]]:
        return [H(*self.qubits), S(*self.qubits), H(*self.qubits)]


class Vdg(SingleQubitClifford):
    n_qubits = 1
    draw_as_zx = True

    @property
    def decomp(self):
        return [XHead(-pi / 2) @ {self.qubits[0]}]

    def to_qiskit(self):
        try:
            from qiskit.circuit.library import SXdgGate
        except ImportError:
            raise ImportError("Please install qiskit to use this feature.")

        return SXdgGate(), self.qubits

    def inverse(self):
        return V(*self.qubits)

    def get_h_s_cx_decomposition(self) -> List[Union["H", "S", "CX"]]:
        return (
            [H(*self.qubits)]
            + Sdg(*self.qubits).get_h_s_cx_decomposition()
            + [H(*self.qubits)]
        )


class T(Gate):
    n_qubits = 1
    draw_as_zx = True

    @property
    def decomp(self):
        return [ZHead(pi / 4) @ {self.qubits[0]}]

    def to_qiskit(self):
        try:
            from qiskit.circuit.library import TGate
        except ImportError:
            raise ImportError("Please install qiskit to use this feature.")

        return TGate(), self.qubits

    def inverse(self):
        return Tdg(*self.qubits)


class Tdg(Gate):
    n_qubits = 1
    draw_as_zx = True

    @property
    def decomp(self):
        return [ZHead(-pi / 4) @ {self.qubits[0]}]

    def to_qiskit(self):
        try:
            from qiskit.circuit.library import TdgGate
        except ImportError:
            raise ImportError("Please install qiskit to use this feature.")

        return TdgGate(), self.qubits

    def inverse(self):
        return T(*self.qubits)


class SWAP(CliffordGate):
    n_qubits = 2
    draw_as_zx = True

    @property
    def decomp(self):
        q0, q1 = self.qubits
        return [CX(q0, q1), CX(q1, q0), CX(q0, q1)]

    def to_qiskit(self):
        try:
            from qiskit.circuit.library import SwapGate
        except ImportError:
            raise ImportError("Please install qiskit to use this feature.")

        return SwapGate(), self.qubits

    def inverse(self):
        return SWAP(*self.qubits)

    def get_h_s_cx_decomposition(self) -> List[Union["H", "S", "CX"]]:
        q0, q1 = self.qubits
        return [CX(q0, q1), CX(q1, q0), CX(q0, q1)]


class CX(TwoQubitClifford):
    n_qubits = 2
    draw_as_zx = True
    width = 40

    @property
    def decomp(self):
        q0, q1 = self.qubits
        return [H(q1), CZ(q0, q1), H(q1)]

    def draw(self, builder, base, row_width, params):
        r = params["r"]
        line_height = params["line_height"]
        zcolor = params["zcolor"]
        xcolor = params["xcolor"]

        ctrl, trgt = self.qubits
        x = base[0] + row_width / 2
        y_ctrl = base[1] + (ctrl + 1) * line_height
        y_trgt = base[1] + (trgt + 1) * line_height
        builder.line((x, y_ctrl), (x, y_trgt))
        builder.circle((x, y_ctrl), r, zcolor)
        builder.circle((x, y_trgt), r, xcolor)

    def get_h_s_cx_decomposition(self) -> List[Union["H", "S", "CX"]]:
        q0, q1 = self.qubits
        return [CX(q0, q1)]

    def to_qiskit(self):
        try:
            from qiskit.circuit.library import CXGate
        except ImportError:
            raise ImportError("Please install qiskit to use this feature.")

        return CXGate(), self.qubits

    def inverse(self):
        return CX(*self.qubits)


class CY(TwoQubitClifford):
    n_qubits = 2
    draw_as_zx = True

    @property
    def decomp(self):
        q0, q1 = self.qubits
        return [XHead(pi / 2) @ {q1}, CZ(q0, q1), XHead(-pi / 2) @ {q1}]

    def to_qiskit(self):
        try:
            from qiskit.circuit.library import CYGate
        except ImportError:
            raise ImportError("Please install qiskit to use this feature.")

        return CYGate(), self.qubits

    def inverse(self):
        return CY(*self.qubits)

    def get_h_s_cx_decomposition(self) -> List[Union["H", "S", "CX"]]:
        q0, q1 = self.qubits
        return Sdg(q1).get_h_s_cx_decomposition() + [CX(q0, q1)] + [S(q1)]


class CZ(TwoQubitClifford):
    n_qubits = 2
    draw_as_zx = True

    @property
    def decomp(self):
        q0, q1 = self.qubits
        return [ZHead(pi / 2) @ {q0, q1}]

    def draw(self, builder, base, row_width, params):
        r = params["r"]
        line_height = params["line_height"]
        zcolor = params["zcolor"]
        hcolor = params["hcolor"]

        ctrl, trgt = self.qubits
        x = base[0] + row_width / 2
        y_ctrl = base[1] + (ctrl + 1) * line_height
        y_trgt = base[1] + (trgt + 1) * line_height
        y_mid = (y_ctrl + y_trgt) / 2
        builder.line((x, y_ctrl), (x, y_trgt))
        builder.circle((x, y_ctrl), r, zcolor)
        builder.circle((x, y_trgt), r, zcolor)
        builder.rect((x, y_mid), r, r, hcolor)

    def to_qiskit(self):
        try:
            from qiskit.circuit.library import CZGate
        except ImportError:
            raise ImportError("Please install qiskit to use this feature.")

        return CZGate(), self.qubits

    def inverse(self):
        return CZ(*self.qubits)

    def get_h_s_cx_decomposition(self) -> List[Union["H", "S", "CX"]]:
        q0, q1 = self.qubits
        return [H(q1), CX(q0, q1), H(q1)]


class CCX(Gate):
    n_qubits = 3
    draw_as_zx = True

    @property
    def decomp(self):
        q0, q1, q2 = self.qubits
        return [H(q2), CCZ(q0, q1, q2), H(q2)]

    def to_qiskit(self):
        try:
            from qiskit.circuit.library import CCXGate
        except ImportError:
            raise ImportError("Please install qiskit to use this feature.")

        return CCXGate(), self.qubits

    def inverse(self):
        return CCX(*self.qubits)


class CCZ(Gate):
    n_qubits = 3
    draw_as_zx = True

    @property
    def decomp(self):
        gs = []

        for i in range(1, self.n_qubits + 1):
            for qs in combinations(self.qubits, i):
                gs.append(ZHead(-(1 ** len(qs)) * pi / 2) @ set(qs))
        return gs

    def to_qiskit(self):
        try:
            from qiskit.circuit.library import CCZGate
        except ImportError:
            raise ImportError("Please install qiskit to use this feature.")

        return CCZGate(), self.qubits

    def inverse(self):
        return CCZ(*self.qubits)


class Rx(PhaseGate):
    n_qubits = 1
    draw_as_zx = True

    @property
    def decomp(self):
        (q,) = self.qubits
        return [XHead(self.get_phase_as_angle()) @ {q}]

    def to_qiskit(self):
        try:
            from qiskit.circuit.library import RXGate
        except ImportError:
            raise ImportError("Please install qiskit to use this feature.")
        return RXGate(self.get_phase_as_float()), self.qubits


class Ry(PhaseGate):
    n_qubits = 1
    draw_as_zx = True

    @property
    def decomp(self):
        (q,) = self.qubits
        return [
            XHead(pi / 2) @ {q},
            ZHead(self.get_phase_as_angle()) @ {q},
            XHead(-pi / 2) @ {q},
        ]

    def to_qiskit(self):
        try:
            from qiskit.circuit.library import RYGate
        except ImportError:
            raise ImportError("Please install qiskit to use this feature.")

        return RYGate(self.get_phase_as_float()), self.qubits


class Rz(PhaseGate):
    n_qubits = 1
    draw_as_zx = True

    @property
    def decomp(self):
        (q,) = self.qubits
        return [ZHead(self.get_phase_as_angle()) @ {q}]

    def to_qiskit(self):
        try:
            from qiskit.circuit.library import RZGate
        except ImportError:
            raise ImportError("Please install qiskit to use this feature.")
        return RZGate(self.get_phase_as_float()), self.qubits


class CRx(PhaseGate):
    n_qubits = 2
    draw_gadget = True

    @property
    def decomp(self):
        p = self.phase
        q0, q1 = self.qubits
        return [H(q1), CRz(p, q0, q1), H(q1)]

    def to_qiskit(self):
        try:
            from qiskit.circuit.library import CRXGate
        except ImportError:
            raise ImportError("Please install qiskit to use this feature.")
        return CRXGate(self.get_phase_as_float()), self.qubits


class CRy(PhaseGate):
    n_qubits = 2
    draw_gadget = True

    @property
    def decomp(self):
        p = self.phase
        q0, q1 = self.qubits
        return [XHead(pi / 2, q1), CRz(p, q0, q1), XHead(-pi / 2, q1)]

    def to_qiskit(self):
        try:
            from qiskit.circuit.library import CRYGate
        except ImportError:
            raise ImportError("Please install qiskit to use this feature.")

        return CRYGate(self.get_phase_as_float()), self.qubits


class CRz(PhaseGate):
    n_qubits = 2
    draw_as_zx = True

    @property
    def decomp(self):
        p = self.phase
        q0, q1 = self.qubits
        return [ZHead(-p / 2) @ {q0, q1}, ZHead(p / 2) @ {q1}]

    def to_qiskit(self):
        try:
            from qiskit.circuit.library import CRZGate
        except ImportError:
            raise ImportError("Please install qiskit to use this feature.")

        return CRZGate(self.get_phase_as_float()), self.qubits


CNOT = CX


def draw_gadget(builder, base, row_width, params, gadget, text_above=True):
    line_height = params["line_height"]
    text_off_x, text_off_y = params["text_off"]
    font_size = params["font_size"]
    r = params["r"]

    if gadget.basis == "Z":
        color1, color2 = params["zcolor"], params["xcolor"]
    elif gadget.basis == "X":
        color1, color2 = params["xcolor"], params["zcolor"]
    else:
        raise ValueError

    step = row_width / 4
    body = min(gadget.qubits) + 0.5
    x = base[0]
    x_body = x + step
    y_body = base[1] + (body + 1) * line_height

    if len(gadget.qubits) > 1:
        for q in gadget.qubits:
            y = base[1] + (q + 1) * line_height
            builder.line((x, y), (x_body, y_body))
            builder.circle((x, y), r, color1)
        builder.line((x_body, y_body), (x_body + step, y_body))
        builder.circle((x_body, y_body), r, color2)
        builder.circle((x_body + step, y_body), r, color1)

        text_x = x_body + step + text_off_x
        text_y = y_body
    else:
        assert len(gadget.qubits) == 1
        q = tuple(gadget.qubits)[0]
        y = base[1] + (q + 1) * line_height
        builder.circle((x, y), r, color1)

        text_x = x + text_off_x
        if text_above:
            text_y = y + text_off_y
        else:
            text_y = y - text_off_y

    builder.text((text_x, text_y), str(gadget.angle), font_size=font_size)
