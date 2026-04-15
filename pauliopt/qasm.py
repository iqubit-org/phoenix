"""
    QASM file parsing
"""

from fractions import Fraction
import re
from typing import (
    cast,
    Dict,
    Iterator,
    List,
    Optional,
    overload,
    Sequence,
    Tuple,
    Union,
)
from pauliopt.utils import Angle


def assert_same_size_targets(*trgts: "QASM.RegTarget"):
    """
    Asserts that all register targets have the same size.
    """
    size: Optional[int] = None
    for trgt in trgts:
        if not isinstance(trgt, QASM.RegTarget):
            raise TypeError(f"Expected register target, found {trgt}")
        if size is None:
            size = trgt.size
        elif size != trgt.size:
            raise ValueError(
                f"Expected all targets of size {size}, "
                f"found a target of size {trgt.size} instead."
            )


class QASM(Sequence["QASM.Statement"]):
    """
    QASM program.
    Based on the QASM spec from [arXiv: 1707.03429](https://arxiv.org/abs/1707.03429)
    """

    _statements: Tuple["QASM.Statement", ...]
    _registers: Dict[str, Union["QASM.QReg", "QASM.CReg"]]

    def __init__(self, *statements: "QASM.Statement"):
        for s in statements:
            if not isinstance(s, QASM.Statement):
                raise ValueError(f"Invalid statement: {s}")
        self._statements = statements
        self._registers = {}
        version_declared = False
        for s in statements:
            if isinstance(s, QASM.Version):
                if version_declared:
                    raise ValueError(f"Repeated OPENQASM version statements: {s}")
                version_declared = True
                continue
            if not version_declared and not isinstance(s, QASM.Comment):
                raise ValueError(
                    f"The first non-comment statement must be the OPENQASM version"
                    f"declaration. Instead, found: '{s}'"
                )
            if isinstance(s, (QASM.QReg, QASM.CReg)):
                if s.name in self._registers:
                    raise ValueError(f"Repeated register declaration: {s.name}")
                self._registers[s.name] = s
                continue
            for r in s.registers:
                if r.name not in self._registers:
                    raise ValueError(f"Unknown register {r.name} for statement: {s}")
                if r != self._registers[r.name]:
                    raise ValueError(f"Register clash for name {r.name}: ")

    @property
    def num_qubits(self) -> int:
        """Number of qubits in this circuit."""
        return sum(
            reg.size for reg in self._registers.values() if isinstance(reg, QASM.QReg)
        )

    @property
    def num_bits(self) -> int:
        """Number of bits in this circuit."""
        return sum(
            reg.size for reg in self._registers.values() if isinstance(reg, QASM.CReg)
        )

    @property
    def registers(self) -> Iterator[Union["QASM.QReg", "QASM.CReg"]]:
        """Iterator over the registers of this QASM program."""
        return (reg for _, reg in self._registers.items())

    def __iter__(self) -> Iterator["QASM.Statement"]:
        return iter(self._statements)

    def __len__(self) -> int:
        return len(self._statements)

    @overload
    def __getitem__(self, idx: int) -> "QASM.Statement": ...

    @overload
    def __getitem__(self, idx: slice) -> Sequence["QASM.Statement"]: ...

    def __getitem__(self, idx: Union[int, slice]):
        return self._statements[idx]

    def __str__(self) -> str:
        return "\n".join(str(statement) for statement in self) + "\n"

    class Statement:
        """QASM statement."""

        @property
        def registers(self) -> Sequence["QASM.Reg"]:
            """List of registers involved in this statement."""
            return []

        @staticmethod
        def parse(line: str) -> "QASM.Statement":
            """
            Attempts to parse a QASM statement from a line of code.
            """
            ...

    class Version(Statement):
        """QASM version statement."""

        _version: str

        def __init__(self, version: str):
            if version != "2.0":
                raise TypeError(f"Invalid version {version}")
            self._version = version

        @property
        def version(self) -> str:
            """QASM version."""
            return self._version

        def __str__(self):
            return f"OPENQASM {self.version};"

    class Reg(Statement):
        """Register."""

        _name: str
        _size: int

        def __init__(self, name: str, size: int):
            if not isinstance(name, str) or not name:
                raise TypeError(f"Invalid register name {name}")
            if not isinstance(size, int) or size <= 0:
                raise TypeError(f"Invalid register size {size}")
            self._name = name
            self._size = size

        @property
        def registers(self) -> Sequence["QASM.Reg"]:
            """List of registers involved in this statement."""
            return [self]

        @property
        def name(self):
            """Register name."""
            return self._name

        @property
        def size(self):
            """Register size."""
            return self._size

    class QReg(Reg):
        """Quantum register."""

        def __init__(self, name: str, size: int):
            super().__init__(name, size)

        def __str__(self):
            return f"qreg {self.name}[{self.size}];"

        def __eq__(self, other):
            if not isinstance(other, QASM.QReg):
                return NotImplemented
            return self.name == other.name and self.size == other.size

    class CReg(Reg):
        """Classical register."""

        def __init__(self, name: str, size: int):
            super().__init__(name, size)

        def __str__(self):
            return f"creg {self.name}[{self.size}];"

        def __eq__(self, other):
            if not isinstance(other, QASM.CReg):
                return NotImplemented
            return self.name == other.name and self.size == other.size

    class Include(Statement):
        """QASM include statement."""

        _filename: str

        def __init__(self, filename: str):
            if not isinstance(filename, str) or not filename:
                raise TypeError(f"Invalid filename {filename}")
            self._filename = filename

        @property
        def filename(self) -> str:
            """Filename."""
            return self._filename

        def __str__(self):
            return f'include "{self.filename}";'

    class Comment(Statement):
        """QASM comment statement."""

        _text: str

        def __init__(self, text: str):
            if not isinstance(text, str):
                raise TypeError(f"Invalid comment text {text}")
            self._text = text

        @property
        def text(self) -> str:
            """Comment text."""
            return self._text

        def __str__(self):
            return f"// {self.text}"

    class RegTarget:
        """A qreg or creg target."""

        _register: Union["QASM.QReg", "QASM.CReg"]
        _pos: Optional[int]

        def __init__(
            self, register: Union["QASM.QReg", "QASM.CReg"], pos: Optional[int] = None
        ):
            if not isinstance(register, (QASM.QReg, QASM.CReg)):
                raise TypeError(f"Invalid register {register}")
            if pos is not None and (not isinstance(pos, int) or pos < 0):
                raise TypeError(f"Invalid register position {pos}")
            self._register = register
            self._pos = pos

        @property
        def registers(self) -> Sequence["QASM.Reg"]:
            """List of registers involved in this statement."""
            return [self.register]

        @property
        def register(self) -> Union["QASM.QReg", "QASM.CReg"]:
            """Register."""
            return self._register

        @property
        def pos(self) -> Optional[int]:
            """Optional register position."""
            return self._pos

        @property
        def size(self):
            """Size of this target"""
            return 1 if self.pos is not None else self.register.size

        def __str__(self):
            if self.pos is not None:
                return f"{self.register.name}[{self.pos}]"
            return f"{self.register.name}"

    class QRegTarget(RegTarget):
        """A qreg target."""

        def __init__(self, register: "QASM.QReg", pos: Optional[int] = None):
            if not isinstance(register, QASM.QReg):
                raise TypeError(f"Invalid quantum register {register}")
            super().__init__(register, pos)

        @property
        def register(self) -> "QASM.QReg":
            """Register."""
            return cast(QASM.QReg, self._register)

    class CRegTarget(RegTarget):
        """A creg target."""

        def __init__(self, register: "QASM.CReg", pos: Optional[int] = None):
            if not isinstance(register, QASM.CReg):
                raise TypeError(f"Invalid classical register {register}")
            super().__init__(register, pos)

        @property
        def register(self) -> "QASM.CReg":
            """Register."""
            return cast(QASM.CReg, self._register)

    class UGate(Statement):
        """Statement for a U3 gate."""

        _theta: Angle
        _phi: Angle
        _lam: Angle
        _qubit: "QASM.QRegTarget"

        def __init__(
            self, theta: Angle, phi: Angle, lam: Angle, qubit: "QASM.QRegTarget"
        ):
            if not isinstance(theta, Angle):
                raise TypeError(f"Invalid angle expression {theta}")
            if not isinstance(phi, Angle):
                raise TypeError(f"Invalid angle expression {phi}")
            if not isinstance(lam, Angle):
                raise TypeError(f"Invalid angle expression {lam}")
            if not isinstance(qubit, QASM.QRegTarget):
                raise TypeError(f"Invalid qubit/qreg {qubit}")
            self._theta = theta
            self._phi = phi
            self._lam = lam
            self._qubit = qubit

        @property
        def registers(self) -> Sequence["QASM.Reg"]:
            """List of registers involved in this statement."""
            return [self.qubit.register]

        @property
        def theta(self) -> Angle:
            """Theta angle for the U3 gate."""
            return self._theta

        @property
        def phi(self) -> Angle:
            """Phi angle for the U3 gate."""
            return self._phi

        @property
        def lam(self) -> Angle:
            """Lambda angle for the U3 gate."""
            return self._lam

        @property
        def qubit(self) -> "QASM.QRegTarget":
            """Qubit/qreg for the U3 gate."""
            return self._qubit

        def __str__(self):
            return (
                f"U({QASM._angle_str(self.theta)}, "
                f"{QASM._angle_str(self.phi)}, "
                f"{QASM._angle_str(self.lam)})"
                f" {self.qubit};"
            )

    class CXGate(Statement):
        """Statement for a CX gate."""

        _control: "QASM.QRegTarget"
        _target: "QASM.QRegTarget"

        def __init__(self, control: "QASM.QRegTarget", target: "QASM.QRegTarget"):
            if not isinstance(control, QASM.QRegTarget):
                raise TypeError(f"Invalid qubit/qreg {control}")
            if not isinstance(target, QASM.QRegTarget):
                raise TypeError(f"Invalid qubit/qreg {target}")
            self._control = control
            self._target = target
            if (
                control.pos is None
                and target.pos is None
                and control.size != target.size
            ):
                raise ValueError(
                    f"Mismatched register sizes: {control.size} vs {target.size}."
                )

        @property
        def registers(self) -> Sequence["QASM.Reg"]:
            """List of registers involved in this statement."""
            return [self.control.register, self.target.register]

        @property
        def control(self) -> "QASM.QRegTarget":
            """Control qubit/qreg for the CX gate."""
            return self._control

        @property
        def target(self) -> "QASM.QRegTarget":
            """Target qubit/qreg for the CX gate."""
            return self._target

        def __str__(self):
            return f"CX {self.control} {self.target};"

    class Measure(Statement):
        """Statement for a measurement."""

        _qubit: "QASM.QRegTarget"
        _bit: "QASM.CRegTarget"

        def __init__(self, qubit: "QASM.QRegTarget", bit: "QASM.CRegTarget"):
            if not isinstance(qubit, QASM.QRegTarget):
                raise TypeError(f"Invalid qubit/qreg {qubit}")
            if not isinstance(bit, QASM.CRegTarget):
                raise TypeError(f"Invalid bit/creg {bit}")
            assert_same_size_targets(qubit, bit)
            self._qubit = qubit
            self._bit = bit

        @property
        def registers(self) -> Sequence["QASM.Reg"]:
            """List of registers involved in this statement."""
            return [self.qubit.register, self.bit.register]

        @property
        def qubit(self) -> "QASM.QRegTarget":
            """Qubit/qreg to be measured."""
            return self._qubit

        @property
        def bit(self) -> "QASM.CRegTarget":
            """Bit/creg to store measurement outcome."""
            return self._bit

        def __str__(self):
            return f"measure {self.qubit} -> {self.bit};"

    class Reset(Statement):
        """Statement for a reset."""

        _qubit: "QASM.QRegTarget"

        def __init__(self, qubit: "QASM.QRegTarget"):
            if not isinstance(qubit, QASM.QRegTarget):
                raise TypeError(f"Invalid qubit/qreg {qubit}")
            self._qubit = qubit

        @property
        def registers(self) -> Sequence["QASM.Reg"]:
            """List of registers involved in this statement."""
            return [self.qubit.register]

        @property
        def qubit(self) -> "QASM.QRegTarget":
            """Qubit/qreg to be measured."""
            return self._qubit

        def __str__(self):
            return f"reset {self.qubit};"

    class Gate(Statement):
        """Statement for a named gate."""

        _name: str
        _params: Tuple[Angle, ...]
        _targets: Tuple["QASM.RegTarget", ...]

        def __init__(
            self,
            name: str,
            targets: Sequence["QASM.RegTarget"],
            params: Sequence[Angle] = tuple(),
        ):
            if not isinstance(name, str) or not name:
                raise TypeError(f"Invalid gate name {name}")
            if not isinstance(targets, Sequence) or not all(
                isinstance(p, QASM.RegTarget) for p in targets
            ):
                raise TypeError(f"Invalid sequence of register targets: {targets}")
            if not targets:
                raise TypeError("Empty sequence of register targets.")
            if not isinstance(params, Sequence) or not all(
                isinstance(p, Angle) for p in params
            ):
                raise TypeError(f"Invalid sequence of angle parameters: {params}")
            self._name = name
            self._params = tuple(params)
            self._targets = tuple(targets)
            assert_same_size_targets(*self._targets)

        @property
        def registers(self) -> Sequence["QASM.Reg"]:
            """List of registers involved in this statement."""
            return [t.register for t in self.targets]

        @property
        def name(self) -> str:
            """Name for this gate."""
            return self._name

        @property
        def params(self) -> Tuple[Angle, ...]:
            """Tuple of angle parameters for this gate."""
            return self._params

        @property
        def targets(self) -> Tuple["QASM.RegTarget", ...]:
            """Tuple of register targets for this gate."""
            return self._targets

        def __str__(self):
            if self.params:
                return (
                    f"{self.name}({', '.join(QASM._angle_str(p) for p in self.params)}) "
                    f"{', '.join(str(t) for t in self.targets)};"
                )
            return f"{self.name} " f"{', '.join(str(t) for t in self.targets)};"

    class Barrier(Statement):
        """Statement for a barrier."""

        _targets: Tuple["QASM.QRegTarget", ...]

        def __init__(self, targets: Sequence["QASM.QRegTarget"]):
            if not isinstance(targets, Sequence) or not all(
                isinstance(p, QASM.QRegTarget) for p in targets
            ):
                raise TypeError(f"Invalid sequence of register targets: {targets}")
            self._targets = tuple(targets)

        @property
        def registers(self) -> Sequence["QASM.Reg"]:
            """List of registers involved in this statement."""
            return [t.register for t in self.targets]

        @property
        def targets(self) -> Tuple["QASM.RegTarget", ...]:
            """Tuple of register targets for this barrier."""
            return self._targets

        def __str__(self):
            return f"barrier {', '.join(str(t) for t in self.targets)};"

    class Conditional(Statement):
        """Statement for a conditional statement."""

        _register: "QASM.CReg"
        _value: int
        _statement: "QASM.Statement"

        def __init__(
            self, register: "QASM.CReg", value: int, statement: "QASM.Statement"
        ):
            if not isinstance(register, QASM.CReg):
                raise TypeError(f"Invalid classical register {register}")
            if not isinstance(value, int) or not 0 <= value < 2**register.size:
                raise TypeError(f"Invalid value {value} for register {register}.")
            if not isinstance(statement, QASM.Statement):
                raise TypeError(f"Invalid statement: {statement}")
            self._register = register
            self._value = value
            self._statement = statement

        @property
        def registers(self) -> Sequence["QASM.Reg"]:
            """List of registers involved in this statement."""
            return list(self.statement.registers) + [self.register]

        @property
        def register(self) -> "QASM.CReg":
            """The register being tested in this conditional statement."""
            return self._register

        @property
        def value(self) -> int:
            """The register value being tested in this conditional statement."""
            return self._value

        @property
        def statement(self) -> "QASM.Statement":
            """The statement to execute if the given creg has the given value."""
            return self._statement

        def __str__(self):
            return f"if({self.register.name}=={self.value}) {self.statement}"

    @staticmethod
    def _angle_str(angle: Angle) -> str:
        num = angle.value.numerator
        den = angle.value.denominator
        if num == 0:
            return "0"
        if num == 1:
            if den == 1:
                return "pi"
            return "pi/%d" % den
        return "%d*pi/%d" % (num, den)

    @staticmethod
    def _parse_angle(angle_str: str) -> Angle:
        angle_str = angle_str.strip()
        angle_str = re.sub(r" *\* *", "*", angle_str)
        angle_str = re.sub(r" */ *", "/", angle_str)
        angle_str = re.sub(r" *pi *", "pi", angle_str)
        pattern = re.compile(r"pi")
        match = pattern.fullmatch(angle_str)
        if match:
            return Angle(Fraction(1, 1))
        pattern = re.compile(r"0")
        match = pattern.fullmatch(angle_str)
        if match:
            return Angle(Fraction(0, 1))
        pattern = re.compile(r"pi/(.+)")
        match = pattern.fullmatch(angle_str)
        if match:
            return Angle(Fraction(1, int(match[1])))
        pattern = re.compile(r"(.+)\*pi")
        match = pattern.fullmatch(angle_str)
        if match:
            return Angle(Fraction(match[1]))
        pattern = re.compile(r"(.+)\*pi/(.+)")
        match = pattern.fullmatch(angle_str)
        if match:
            return Angle(Fraction(int(match[1]), int(match[2])))
        # TODO: implement angle approximation
        raise ValueError(
            f"Only fractional multiples of pi are accepted as angles."
            f"Instead, found '{angle_str}'"
        )

    @staticmethod
    def _parse_comment(line: str) -> "QASM.Comment":
        return QASM.Comment(line[2:])

    @staticmethod
    def _parse_version(tokens: List[str]) -> "QASM.Version":
        return QASM.Version(tokens[1])

    @staticmethod
    def _parse_include(tokens: List[str]) -> "QASM.Include":
        if len(tokens) != 2:
            raise ValueError(
                f"Expected exactly 2 statement tokens for include statement. "
                f"Instead, found {tokens}"
            )
        pattern = re.compile(r"\"(.+)\"")
        match = pattern.fullmatch(tokens[1])
        if not match:
            raise ValueError(f"Invalid include filename: '{tokens[1]}'")
        return QASM.Include(match[1])

    @staticmethod
    def _parse_qreg(tokens: List[str]) -> "QASM.QReg":
        if len(tokens) != 2:
            raise ValueError(
                f"Expected exactly 2 statement tokens for qreg statement. "
                f"Instead, found {tokens}"
            )
        pattern = re.compile(r"([a-zA-Z0-9]+)\[(.+)\]")
        match = pattern.fullmatch(tokens[1])
        if not match:
            raise ValueError(f"Invalid register declaration: '{tokens[1]}'")
        return QASM.QReg(match[1], int(match[2]))

    @staticmethod
    def _parse_creg(tokens: List[str]) -> "QASM.CReg":
        if len(tokens) != 2:
            raise ValueError(
                f"Expected exactly 2 statement tokens for creg statement. "
                f"Instead, found {tokens}"
            )
        pattern = re.compile(r"([a-zA-Z0-9]+)\[(.+)\]")
        match = pattern.fullmatch(tokens[1])
        if not match:
            raise ValueError(f"Invalid register declaration: '{tokens[1]}'")
        return QASM.CReg(match[1], int(match[2]))

    @staticmethod
    def _parse_qreg_target(
        token: str, qregs: Dict[str, "QASM.QReg"]
    ) -> "QASM.QRegTarget":
        if token in qregs:
            return QASM.QRegTarget(qregs[token])
        pattern = re.compile(r"([a-zA-Z0-9]+)\[(.+)\]")
        match = pattern.fullmatch(token)
        if not match or match[1] not in qregs:
            raise ValueError(f"Invalid register target or unknown register: '{token}'")
        return QASM.QRegTarget(qregs[match[1]], int(match[2]))

    @staticmethod
    def _parse_creg_target(
        token: str, cregs: Dict[str, "QASM.CReg"]
    ) -> "QASM.CRegTarget":
        if token in cregs:
            return QASM.CRegTarget(cregs[token])
        pattern = re.compile(r"([a-zA-Z0-9]+)\[(.+)\]")
        match = pattern.fullmatch(token)
        if not match or match[1] not in cregs:
            raise ValueError(f"Invalid register target or unknown register: '{token}'")
        return QASM.CRegTarget(cregs[match[1]], int(match[2]))

    @staticmethod
    def _parse_reg_target(
        token: str, qregs: Dict[str, "QASM.QReg"], cregs: Dict[str, "QASM.CReg"]
    ) -> "QASM.RegTarget":
        if token in qregs:
            return QASM.QRegTarget(qregs[token])
        if token in cregs:
            return QASM.CRegTarget(cregs[token])
        pattern = re.compile(r"([a-zA-Z0-9]+)\[(.+)\]")
        match = pattern.fullmatch(token)
        if not match or (match[1] not in qregs and match[1] not in cregs):
            raise ValueError(f"Invalid register target or unknown register: '{token}'")
        if match[1] in qregs:
            return QASM.QRegTarget(qregs[match[1]], int(match[2]))
        return QASM.CRegTarget(cregs[match[1]], int(match[2]))

    @staticmethod
    def _parse_u(tokens: List[str], qregs: Dict[str, "QASM.QReg"]) -> "QASM.UGate":
        if len(tokens) != 2:
            raise ValueError(
                f"Expected exactly 2 statement tokens for U gate statement. "
                f"Instead, found {tokens}"
            )
        pattern = re.compile(r"U\((.+)\)")
        match = pattern.fullmatch(tokens[0])
        if not match:
            raise ValueError(f"Invalid first token for U gate statement: '{tokens[0]}'")
        params = match[1].split(",")
        if len(params) != 3:
            raise ValueError(
                f"Expected exactly 3 parameter tokens for U gate statement. "
                f"Instead, found {params}"
            )
        angles = [QASM._parse_angle(p) for p in params]
        qubit = QASM._parse_qreg_target(tokens[1], qregs)
        return QASM.UGate(angles[0], angles[1], angles[2], qubit)

    @staticmethod
    def _parse_cx(tokens: List[str], qregs: Dict[str, "QASM.QReg"]) -> "QASM.CXGate":
        if len(tokens) != 2:
            raise ValueError(
                f"Expected exactly 2 statement tokens for CX gate statement. "
                f"Instead, found {tokens}"
            )
        qreg_targets = tokens[1].split(",")
        if len(qreg_targets) != 2:
            raise ValueError(
                f"Expected exactly 2 qreg target tokens for CX gate statement. "
                f"Instead, found {qreg_targets}"
            )
        ctrl = QASM._parse_qreg_target(qreg_targets[0], qregs)
        trgt = QASM._parse_qreg_target(qreg_targets[1], qregs)
        return QASM.CXGate(ctrl, trgt)

    @staticmethod
    def _parse_measure(
        tokens: List[str], qregs: Dict[str, "QASM.QReg"], cregs: Dict[str, "QASM.CReg"]
    ) -> "QASM.Measure":
        if len(tokens) != 2:
            raise ValueError(
                f"Expected exactly 2 statement tokens for measure statement. "
                f"Instead, found {tokens}"
            )
        targets = tokens[1].split("->")
        if len(targets) != 2:
            raise ValueError(
                f"Expected exactly 2 target tokens for measure statement. "
                f"Instead, found {targets}"
            )
        qubit = QASM._parse_qreg_target(targets[0], qregs)
        bit = QASM._parse_creg_target(targets[1], cregs)
        return QASM.Measure(qubit, bit)

    @staticmethod
    def _parse_reset(tokens: List[str], qregs: Dict[str, "QASM.QReg"]) -> "QASM.Reset":
        if len(tokens) != 2:
            raise ValueError(
                f"Expected exactly 2 statement tokens for reset statement. "
                f"Instead, found {tokens}"
            )
        qubit = QASM._parse_qreg_target(tokens[1], qregs)
        return QASM.Reset(qubit)

    @staticmethod
    def _parse_barrier(
        tokens: List[str], qregs: Dict[str, "QASM.QReg"]
    ) -> "QASM.Barrier":
        if len(tokens) != 2:
            raise ValueError(
                f"Expected exactly 2 statement tokens for barrier statement. "
                f"Instead, found {tokens}"
            )
        qreg_targets = tokens[1].split(",")
        if not qreg_targets:
            raise ValueError(
                f"Expected at least one qreg target token for barrier statement. "
                f"Instead, found {qreg_targets}"
            )
        qubits = [QASM._parse_qreg_target(t, qregs) for t in qreg_targets]
        return QASM.Barrier(qubits)

    @staticmethod
    def _parse_conditional(
        tokens: List[str], qregs: Dict[str, "QASM.QReg"], cregs: Dict[str, "QASM.CReg"]
    ) -> "QASM.Conditional":
        if len(tokens) < 2:
            raise ValueError(
                f"Expected at least 2 statement tokens for conditional statement. "
                f"Instead, found {tokens}"
            )
        pattern = re.compile(r"if\((.+)==(.+)\)")
        match = pattern.fullmatch(tokens[0])
        if not match:
            raise ValueError(
                f"Invalid first token for conditional statement: '{tokens[0]}'"
            )
        register = cregs[match[1]]
        value = int(match[2])
        return QASM.Conditional(
            register, value, QASM._parse_statement(tokens[1:], qregs, cregs)
        )

    @staticmethod
    def _parse_gate(
        tokens: List[str], qregs: Dict[str, "QASM.QReg"], cregs: Dict[str, "QASM.CReg"]
    ) -> "QASM.Gate":
        if len(tokens) != 2:
            raise ValueError(
                f"Expected exactly 2 statement tokens for named gate statement. "
                f"Instead, found {tokens}"
            )
        pattern = re.compile(r"([a-zA-Z0-9]+)\((.+)\)")
        match = pattern.fullmatch(tokens[0])
        if match:
            name = match[1]
            params = match[2].split(",")
        else:
            name = tokens[0]
            params = []
        angles = [QASM._parse_angle(p) for p in params]
        target_tokens = tokens[1].split(",")
        if len(target_tokens) == 0:
            raise ValueError(
                f"Expected at least one target token for named gate statement. "
                f"Instead, found {target_tokens}"
            )
        targets = [QASM._parse_reg_target(t, qregs, cregs) for t in target_tokens]
        return QASM.Gate(name, targets, angles)

    @staticmethod
    def _parse_statement(
        tokens: List[str], qregs: Dict[str, "QASM.QReg"], cregs: Dict[str, "QASM.CReg"]
    ) -> "QASM.Statement":
        # pylint: disable = too-many-return-statements
        if tokens[0] == "OPENQASM":
            return QASM._parse_version(tokens)
        if tokens[0] == "qreg":
            qreg: QASM.QReg = QASM._parse_qreg(tokens)
            if qreg.name in qregs or qreg.name in cregs:
                raise ValueError(f"Repeated register declaration: {qreg.name}")
            qregs[qreg.name] = qreg
            return qreg
        if tokens[0] == "creg":
            creg: QASM.CReg = QASM._parse_creg(tokens)
            if creg.name in qregs or creg.name in cregs:
                raise ValueError(f"Repeated register declaration: {creg.name}")
            cregs[creg.name] = creg
            return creg
        if tokens[0] == "include":
            return QASM._parse_include(tokens)
        if tokens[0].startswith("U"):
            return QASM._parse_u(tokens, qregs)
        if tokens[0] == "CX":
            return QASM._parse_cx(tokens, qregs)
        if tokens[0] == "measure":
            return QASM._parse_measure(tokens, qregs, cregs)
        if tokens[0] == "reset":
            return QASM._parse_reset(tokens, qregs)
        if tokens[0] == "barrier":
            return QASM._parse_barrier(tokens, qregs)
        if tokens[0].startswith("if"):
            return QASM._parse_conditional(tokens, qregs, cregs)
        return QASM._parse_gate(tokens, qregs, cregs)

    @staticmethod
    def parse(program: str):
        """Parses a QASM program into a `QASM` object."""
        # pylint: disable = too-many-branches, too-many-statements
        if not isinstance(program, str):
            raise TypeError("Expected a string.")
        lines = program.split("\n")
        statements: List[QASM.Statement] = []
        cregs: Dict[str, QASM.CReg] = {}
        qregs: Dict[str, QASM.QReg] = {}
        for line in lines:
            line = line.strip()
            line = re.sub(r" *, *", ",", line)
            line = re.sub(r" *== *", "==", line)
            line = re.sub(r" *-> *", "->", line)
            line = re.sub(r"\( *", "(", line)
            line = re.sub(r" *\)", ")", line)
            line = re.sub(r"\[ *", "[", line)
            line = re.sub(r" *\]", "]", line)
            line = re.sub(r"  *", " ", line)
            if line.startswith("//"):
                statements.append(QASM._parse_comment(line))
                continue
            if not line:
                continue
            if line.startswith("gate"):
                # TODO: implement gate definition statements
                continue
            if line.startswith("gate"):
                # TODO: implement opaque gate definition statements
                continue
            if not line.endswith(";"):
                raise ValueError(
                    f"Statements should be terminated by semicolon. "
                    f"Instead, found: {line}"
                )
            line = line[:-1]  # remove final semicolon
            tokens: List[str] = line.split(" ")
            tokens = [t for t in tokens if len(t) != 0]
            try:
                statements.append(QASM._parse_statement(tokens, qregs, cregs))
            except ValueError as _:
                raise ValueError(f"An error arose while parsing line: {line}")
        return QASM(*statements)


# # Some testing:
# code = """
# // quantum teleportation example
# OPENQASM 2.0;
# include "qelib1.inc";
# qreg q[3];
# creg c0[1];
# creg c1[1];
# creg c2[1];
# // optional post-rotation for state tomography
# gate post q { }
# u3(pi/2,pi/4,3*pi/4) q[0];
# h q[1];
# cx q[1],q[2];
# barrier q;
# cx q[0],q[1];
# h q[0];
# measure q[0] -> c0[0];
# measure q[1] -> c1[0];
# if(c0==1) z q[2];
# if(c1==1) x q[2];
# post q[2];
# measure q[2] -> c2[0];
# """
# print(QASM.parse(code))
