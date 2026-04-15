import math
from typing import List

import numpy as np

from pauliopt.circuits import Circuit
from pauliopt.gates import CliffordGate
from pauliopt.pauli.pauli_gadget import PauliGadget
from pauliopt.topologies import Topology
from pauliopt.utils import SVGBuilder
from pauliopt.pauli_strings import X, Y, Z, I

LATEX_HEADER = """\documentclass[preview]{standalone}

\\usepackage{tikz}
\\usetikzlibrary{zx-calculus}
\\usetikzlibrary{quantikz}
\\usepackage{graphicx}

\\tikzset{
diagonal fill/.style 2 args={fill=#2, path picture={
\\fill[#1, sharp corners] (path picture bounding box.south west) -|
                         (path picture bounding box.north east) -- cycle;}},
reversed diagonal fill/.style 2 args={fill=#2, path picture={
\\fill[#1, sharp corners] (path picture bounding box.north west) |- 
                         (path picture bounding box.south east) -- cycle;}}
}

\\tikzset{
diagonal fill/.style 2 args={fill=#2, path picture={
\\fill[#1, sharp corners] (path picture bounding box.south west) -|
                         (path picture bounding box.north east) -- cycle;}}
}

\\tikzset{
pauliY/.style={
zxAllNodes,
zxSpiders,
inner sep=0mm,
minimum size=2mm,
shape=rectangle,
%fill=colorZxX
diagonal fill={colorZxX}{colorZxZ}
}
}

\\tikzset{
pauliX/.style={
zxAllNodes,
zxSpiders,
inner sep=0mm,
minimum size=2mm,
shape=rectangle,
fill=colorZxX
}
}

\\tikzset{
pauliZ/.style={
zxAllNodes,
zxSpiders,
inner sep=0mm,
minimum size=2mm,
shape=rectangle,
fill=colorZxZ
}
}

\\tikzset{
pauliPhase/.style={
zxAllNodes,
zxSpiders,
inner sep=0.5mm,
minimum size=2mm,
shape=rectangle,
fill=white
}
}
"""


class PauliPolynomial:
    def __init__(self, num_qubits: int):
        self.num_qubits: int = num_qubits
        self.pauli_gadgets: List[PauliGadget] = []
        self.global_phase: float = 0.0

    def __irshift__(self, gadget: PauliGadget):
        if not len(gadget) == self.num_qubits:
            raise Exception(
                f"Pauli Polynomial has {self.num_qubits} qubits, but Pauli gadget has: "
                f"{len(gadget)} qubits"
            )
        self.pauli_gadgets.append(gadget)
        return self

    def __rshift__(self, pauli_polynomial: "PauliPolynomial"):
        for gadget in pauli_polynomial.pauli_gadgets:
            self.pauli_gadgets.append(gadget)
        return self

    def __repr__(self):
        if len(self.pauli_gadgets) > 0:
            pad_length = max([len(str(gadget.angle)) for gadget in self.pauli_gadgets])
        else:
            pad_length = 0
        return "\n".join(
            [self[i].to_string(pad_length=pad_length) for i in range(self.num_gadgets)]
        )

    def __len__(self):
        return len(self.pauli_gadgets)

    def __getitem__(self, index):
        if isinstance(index, tuple):
            index = list(index)
            pp_ = PauliPolynomial(self.num_qubits)
            pp_.pauli_gadgets = [self[i] for i in index]
            return pp_
        elif isinstance(index, list):
            pp_ = PauliPolynomial(self.num_qubits)
            pp_.pauli_gadgets = [self[i] for i in index]
            return pp_
        else:
            return self.pauli_gadgets[index]

    @property
    def num_gadgets(self):
        return len(self.pauli_gadgets)

    def num_legs(self):
        legs = 0
        for gadget in self.pauli_gadgets:
            legs += gadget.num_legs()
        return legs

    def assign_time(self, time: float):
        for gadget in self.pauli_gadgets:
            assert isinstance(gadget, PauliGadget)
            gadget.assign_time(time)

    def set_random_angles(self, allowed_angles: list):
        for gadget in self.pauli_gadgets:
            angle = np.random.choice(allowed_angles)
            gadget.set_angle(angle)

    def to_qiskit(self, topology: Topology = None, time: float = 1):
        num_qubits = self.num_qubits
        if topology is None:
            topology = Topology.complete(num_qubits)
        try:
            from qiskit import QuantumCircuit
        except:
            raise Exception("Please install qiskit to export Clifford Regions")

        qc = QuantumCircuit(num_qubits)
        for gadget in self.pauli_gadgets:
            qc.compose(gadget.to_qiskit(topology=topology, time=time), inplace=True)
        qc.global_phase += self.global_phase
        return qc

    def to_circuit(self, topology=None):
        num_qubits = self.num_qubits
        if topology is None:
            topology = Topology.complete(num_qubits)
        qc = Circuit(num_qubits)

        for gadget in self.pauli_gadgets:
            qc += gadget.to_circuit(topology)
        return qc

    def propagate(self, gate: CliffordGate, sub_columns=None):
        if sub_columns is None:
            sub_columns = list(range(self.num_gadgets))

        pp_ = PauliPolynomial(self.num_qubits)
        for idx, gadget in enumerate(self.pauli_gadgets):
            if idx in sub_columns:
                pp_ >>= gate.propagate_pauli(gadget)
            else:
                pp_ >>= gadget
        return pp_

    def propagate_inplace(self, gate: CliffordGate, sub_columns=None):
        if sub_columns is None:
            sub_columns = list(range(self.num_gadgets))

        for col in sub_columns:
            gate.propagate_pauli(self[col])

    def copy(self):
        pp_ = PauliPolynomial(self.num_qubits)
        for gadget in self.pauli_gadgets:
            pp_ >>= gadget.copy()
        return pp_

    def two_qubit_count(self, topology, leg_cache=None):
        if leg_cache is None:
            leg_cache = {}
        count = 0
        for gadget in self.pauli_gadgets:
            count += gadget.two_qubit_count(topology, leg_cache=leg_cache)
        return count

    def commutes(self, col1, col2):
        gadget1 = self.pauli_gadgets[col1]
        gadget2 = self.pauli_gadgets[col2]
        return gadget1.commutes(gadget2)

    def mutual_legs(self, col1: int, col2: int):
        gadget1 = self.pauli_gadgets[col1]
        gadget2 = self.pauli_gadgets[col2]
        return gadget1.mutual_legs(gadget2)

    def to_svg(
        self,
        hscale: float = 1.0,
        vscale: float = 1.0,
        scale: float = 1.0,
        svg_code_only=False,
    ):
        vscale *= scale
        hscale *= scale

        x_color = "#CCFFCC"
        z_color = "#FF8888"
        y_color = "ycolor"

        num_qubits = self.num_qubits
        num_gadgets = self.num_gadgets

        # general width and height of a square
        square_width = int(math.ceil(20 * vscale))
        square_height = int(math.ceil(20 * vscale))

        # width of the text of the phases # TODO round floats (!!)
        text_width = int(math.ceil(50 * vscale))

        bend_degree = 10

        # margins between the angle and the legs
        margin_angle_x = int(math.ceil(20 * hscale))
        margin_angle_y = int(math.ceil(20 * hscale))

        # margins between each element
        margin_x = int(math.ceil(10 * hscale))
        margin_y = int(math.ceil(10 * hscale))

        width = (
            num_gadgets * (square_width + margin_x + margin_angle_x + text_width)
            + margin_x
        )
        height = (num_qubits) * (square_height + margin_y) + (
            square_height + margin_y + margin_angle_y
        )

        builder = SVGBuilder(width, height)
        builder = builder.add_diagonal_fill(x_color, z_color, y_color)

        prev_x = {qubit: 0 for qubit in range(num_qubits)}

        x = margin_x

        for gadget in self.pauli_gadgets:
            paulis = gadget.paulis
            y = margin_y
            text_coords = (square_width + margin_x + margin_angle_x + x, y)
            text_left_lower_corder = (text_coords[0], text_coords[1] + square_height)
            for qubit in range(num_qubits):
                if qubit == 0:
                    y += square_height + margin_y + margin_angle_y
                else:
                    y += square_height + margin_y
                center_coords = (x + square_width, y)
                if paulis[qubit] == I:
                    continue

                builder.line(
                    (prev_x[qubit], y + square_height // 2), (x, y + square_height // 2)
                )
                prev_x[qubit] = x + square_width
                builder.line_bend(
                    text_left_lower_corder, center_coords, degree=qubit * bend_degree
                )
                if paulis[qubit] == X:
                    builder.square((x, y), square_width, square_height, x_color)
                elif paulis[qubit] == Y:
                    builder.square((x, y), square_width, square_height, y_color)
                elif paulis[qubit] == Z:
                    builder.square((x, y), square_width, square_height, z_color)

            builder = builder.text_with_square(
                text_coords, text_width, square_height, str(gadget.angle)
            )
            x += square_width + margin_x + text_width + margin_angle_x
        y = margin_y
        for qubit in range(num_qubits):
            if qubit == 0:
                y += square_height + margin_y + margin_angle_y
            else:
                y += square_height + margin_y
            builder.line(
                (prev_x[qubit], y + square_height // 2), (width, y + square_height // 2)
            )
        svg_code = repr(builder)

        if svg_code_only:
            return svg_code
        try:
            # pylint: disable = import-outside-toplevel
            from IPython.core.display import SVG  # type: ignore
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError("You must install the 'IPython' library.") from e

        return SVG(svg_code)

    def _repr_svg_(self):
        """
        Magic method for IPython/Jupyter pretty-printing.
        See https://ipython.readthedocs.io/en/stable/api/generated/IPython.display.html
        """
        return self.to_svg(svg_code_only=True)

    def to_latex(self, file_name=None):
        out_str = LATEX_HEADER
        out_str += "\\begin{document}\n"
        out_str += "\\begin{ZX}\n"

        angle_line = "\zxNone{} \t\t&"

        angle_pad_max = max(
            [len(str(gadget.angle.repr_latex)) for gadget in self.pauli_gadgets]
        )
        lines = {q: "\\zxNone{} \\rar \t&" for q in range(self.num_qubits)}
        for gadget in self.pauli_gadgets:
            assert isinstance(gadget, PauliGadget)
            pad_ = "".join([" " for _ in range(self.num_qubits + 26)])
            pad_angle = "".join(
                [" " for _ in range(angle_pad_max - len(str(gadget.angle.repr_latex)))]
            )
            angle_line += (
                f" \\zxNone{{}}  {pad_}&"
                f" |[pauliPhase]| {gadget.angle.repr_latex} {pad_angle}&"
                f" \\zxNone{{}}      &"
            )
            paulis = gadget.paulis
            for q in range(self.num_qubits):
                us = "".join(["u" for _ in range(q)])

                pad_angle = "".join([" " for _ in range(angle_pad_max)])
                if paulis[q] != I:
                    pad_ = "".join([" " for _ in range(self.num_qubits - q)])
                    lines[q] += (
                        f" |[pauli{paulis[q].value}]| "
                        f"\\ar[ruu{us}, bend right] \\rar {pad_}&"
                        f" \\zxNone{{}} \\rar {pad_angle} &"
                        f" \\zxNone{{}} \\rar &"
                    )
                else:
                    pad_ = "".join([" " for _ in range(self.num_qubits + 22)])
                    lines[q] += (
                        f" \\zxNone{{}} \\rar {pad_}& "
                        f"\\zxNone{{}} \\rar {pad_angle} & "
                        f"\\zxNone{{}} \\rar &"
                    )
        out_str += angle_line + "\\\\ \n"
        out_str += "\\\\ \n"
        for q in range(self.num_qubits):
            out_str += lines[q] + "\\\\ \n"
        out_str += "\\end{ZX} \n"
        out_str += "\\end{document}\n"
        if file_name is not None:
            with open(f"{file_name}.tex", "w") as f:
                f.write(out_str)
        return out_str

    def swap_gadgets(self, col1, col2):
        self.pauli_gadgets[col1], self.pauli_gadgets[col2] = (
            self.pauli_gadgets[col2],
            self.pauli_gadgets[col1],
        )

    def swap_rows(self, row1, row2):
        for l in range(self.num_gadgets):
            self.pauli_gadgets[l].swap_rows(row1, row2)

    def permute(self, permutation: dict):
        for gadget in self.pauli_gadgets:
            assert isinstance(gadget, PauliGadget)
            gadget.permute(permutation)
