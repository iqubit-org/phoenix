"""
    Utility classes and functions for the `pauliopt` library.
"""

from abc import ABC, abstractmethod
import math
from decimal import Decimal
from fractions import Fraction
from types import MappingProxyType
from typing import (
    Any,
    Callable,
    ClassVar,
    Dict,
    Final,
    List,
    Literal,
    Mapping,
    Optional,
    overload,
    Protocol,
    runtime_checkable,
    Sequence,
    Tuple,
    Union,
)
import numpy as np
import networkx as nx


def is_cutting(vertex, g):
    """
    Check if the given vertex is a cutting vertex in the given graph.

    :param vertex: The vertex to check
    :param g: The graph to check
    """
    return vertex in nx.articulation_points(g)


def calculate_orthogonal_point(a, b, d, left):
    direction_vector = b - a
    magnitude = np.linalg.norm(direction_vector)
    normalized_direction_vector = direction_vector / magnitude
    if left:
        orthogonal_vector = np.array(
            [-normalized_direction_vector[1], normalized_direction_vector[0]]
        )
    else:
        orthogonal_vector = np.array(
            [normalized_direction_vector[1], -normalized_direction_vector[0]]
        )
    midpoint = (a + b) / 2
    orthogonal_point = midpoint + d * orthogonal_vector
    return int(orthogonal_point[0]), int(orthogonal_point[1])


AngleInitT = Union[int, Fraction, str, Decimal, float]


class AngleExpr(ABC):
    """
    A container class for angle expressions.
    """

    def __pos__(self) -> "AngleExpr":
        return self

    def __neg__(self) -> "AngleExpr":
        return SumprodAngleExpr(self, coeffs=-1)

    def __add__(self, other: "AngleExpr") -> "AngleExpr":
        if isinstance(other, AngleExpr):
            return SumprodAngleExpr(self, other)
        return NotImplemented

    def __sub__(self, other: "AngleExpr") -> "AngleExpr":
        if isinstance(other, AngleExpr):
            return SumprodAngleExpr(self, -other)
        return NotImplemented

    def __mod__(self, other: "AngleExpr") -> "AngleExpr":
        if isinstance(other, AngleExpr):
            return ModAngleExpr(self, other)
        return NotImplemented

    def __mul__(self, other: int) -> "AngleExpr":
        if isinstance(other, int):
            return SumprodAngleExpr(self, coeffs=other)
        return NotImplemented

    def __rmul__(self, other: int) -> "AngleExpr":
        if isinstance(other, int):
            return SumprodAngleExpr(self, coeffs=other)
        return NotImplemented

    def __truediv__(self, other: int) -> "AngleExpr":
        if isinstance(other, int):
            return SumprodAngleExpr(self, coeffs=Fraction(1, other))
        return NotImplemented

    @abstractmethod
    def __hash__(self) -> int: ...

    @abstractmethod
    def __str__(self) -> str: ...

    @abstractmethod
    def __repr__(self) -> str: ...

    @property
    @abstractmethod
    def repr_latex(self) -> str:
        """
        LaTeX math mode representation of this number.
        """
        ...

    @property
    @abstractmethod
    def to_qiskit(self) -> Any: ...

    @property
    def is_zero(self) -> bool:
        return False

    @property
    def is_pi(self) -> bool:
        return False

    @property
    def is_zero_or_pi(self) -> bool:
        return self.is_zero or self.is_pi

    def _repr_latex_(self) -> str:
        """
        Magic method for IPython/Jupyter pretty-printing.
        See https://ipython.readthedocs.io/en/stable/api/generated/IPython.display.html
        """
        return "$%s$" % self.repr_latex

    @abstractmethod
    def __eq__(self, other: Any) -> bool: ...


class Angle(AngleExpr):
    """
    A container class for angles,
    as rational multiples of PI modulo 2PI.
    """

    _value: Fraction

    def __init__(self, theta: Union["Angle", AngleInitT]):
        if isinstance(theta, Angle):
            self._value = theta.value
        else:
            self._value = Fraction(theta)

    @property
    def value(self) -> Fraction:
        """
        The value of this angle as a fraction of PI.
        """
        return self._value % 2

    @property
    def as_root_of_unity(self) -> Tuple[int, int]:
        """
        Returns `(a,n)` where `n` is the smallest such
        that this angle is an $n$-th root of unity
        and `0 <= a < n` such that this is $e^{i 2\\pi \\frac{a}{n}}$
        """
        num = self.value.numerator
        den = self.value.denominator
        a: int = num // 2 if num % 2 == 0 else num
        order: int = den if num % 2 == 0 else 2 * den
        return (a, order)

    @property
    def order(self) -> int:
        """
        The order of this angle as a root of unity.
        """
        num = self.value.numerator
        den = self.value.denominator
        return den if num % 2 == 0 else 2 * den

    @property
    def is_zero_or_pi(self) -> bool:
        """
        Whether this angle is a multiple of pi.
        """
        num = self.value.numerator
        den = self.value.denominator
        return num % den == 0

    @property
    def is_zero(self) -> bool:
        """
        Whether this angle is a multiple of 2pi.
        """
        num = self.value.numerator
        den = self.value.denominator
        return num % (2 * den) == 0

    @property
    def is_pi(self) -> bool:
        """
        Whether this angle is an odd multiple of pi.
        """
        num = self.value.numerator
        den = self.value.denominator
        return num % den == 0 and not num % (2 * den) == 0

    @property
    def to_qiskit(self) -> float:
        return float(self)

    def __pos__(self) -> "Angle":
        return self

    def __neg__(self) -> "Angle":
        return Angle(-self._value)

    @overload
    def __add__(self, other: "Angle") -> "Angle": ...

    @overload
    def __add__(self, other: "AngleExpr") -> "AngleExpr": ...

    def __add__(self, other: "AngleExpr") -> "AngleExpr":
        if isinstance(other, Angle):
            return Angle(self._value + other._value)
        return super().__add__(other)

    @overload
    def __sub__(self, other: "Angle") -> "Angle": ...

    @overload
    def __sub__(self, other: "AngleExpr") -> "AngleExpr": ...

    def __sub__(self, other: "AngleExpr") -> "AngleExpr":
        if isinstance(other, Angle):
            return Angle(self._value - other._value)
        return super().__sub__(other)

    @overload
    def __mod__(self, other: "Angle") -> "Angle": ...

    @overload
    def __mod__(self, other: "AngleExpr") -> "AngleExpr": ...

    def __mod__(self, other: "AngleExpr") -> "AngleExpr":
        if isinstance(other, Angle):
            return Angle(self._value % other._value)
        return super().__mod__(other)

    def _mul(self, other: Union[int, Fraction]) -> "Angle":
        return Angle(self._value * other)

    def __mul__(self, other: int) -> "Angle":
        if isinstance(other, int):
            return self._mul(other)
        return NotImplemented

    def __rmul__(self, other: int) -> "Angle":
        if isinstance(other, int):
            return self._mul(other)
        return NotImplemented

    def __truediv__(self, other: int) -> "Angle":
        if isinstance(other, int):
            return self._mul(Fraction(1, other))
        return NotImplemented

    def __hash__(self) -> int:
        return hash(repr(self))

    def __str__(self) -> str:
        num = self.value.numerator
        den = self.value.denominator
        if num == 0:
            return "0"
        if num == 1:
            if den == 1:
                return "π"
            return "π/%d" % den
        return "%dπ/%d" % (num, den)

    def __repr__(self) -> str:
        num = self.value.numerator
        den = self.value.denominator
        if num == 0:
            return "0"
        if num == 1:
            if den == 1:
                return "pi"
            return "pi/%d" % den
        return "%d*pi/%d" % (num, den)

    @property
    def repr_latex(self) -> str:
        """
        LaTeX math mode representation of this number.
        """
        num = self.value.numerator
        den = self.value.denominator
        if num == 0:
            return "0"
        if num == 1:
            if den == 1:
                return "\\pi"
            return "\\frac{\\pi}{%d}" % den
        return "\\frac{%d\\pi}{%d}" % (num, den)

    def _repr_latex_(self) -> str:
        """
        Magic method for IPython/Jupyter pretty-printing.
        See https://ipython.readthedocs.io/en/stable/api/generated/IPython.display.html
        """
        return "$%s$" % self.repr_latex

    def __eq__(self, other: Any) -> bool:
        if other == 0:
            return self.value == 0
        if not isinstance(other, Angle):
            return NotImplemented
        return self.value == other.value

    def __float__(self) -> float:
        return float(self.value) * math.pi

    @overload
    @staticmethod
    def random(
        subdivision: int = 4,
        *,
        size: Literal[1] = 1,
        rng_seed: Optional[int] = None,
        nonzero: bool = False,
    ) -> "Angle": ...

    @overload
    @staticmethod
    def random(
        subdivision: int = 4,
        *,
        size: int = 1,
        rng_seed: Optional[int] = None,
        nonzero: bool = False,
    ) -> Union["Angle", Tuple["Angle", ...]]: ...

    @staticmethod
    def random(
        subdivision: int = 4,
        *,
        size: int = 1,
        rng_seed: Optional[int] = None,
        nonzero: bool = False,
    ) -> Union["Angle", Tuple["Angle", ...]]:
        """
        Generates a random angle with the given `subdivision`:
        `r * pi/subdivision` for random `r in range(2*subdivision)`.
        """
        if not isinstance(subdivision, int):
            raise TypeError()
        if not isinstance(size, int):
            raise TypeError()
        if not isinstance(nonzero, bool):
            raise TypeError()
        if subdivision <= 0:
            raise ValueError("Subdivision must be positive.")
        if size <= 0:
            raise ValueError("Size must be positive.")
        if rng_seed is not None and not isinstance(rng_seed, int):
            raise TypeError("RNG seed must be integer or 'None'.")
        rng = np.random.default_rng(seed=rng_seed)
        if nonzero:
            rs = 1 + rng.integers(
                2 * subdivision - 1, size=size
            )  # type: ignore[attr-defined]
        else:
            rs = rng.integers(2 * subdivision, size=size)  # type: ignore[attr-defined]
        if size == 1:
            return Angle(Fraction(int(rs[0]), subdivision))
        return tuple(Angle(Fraction(int(r), subdivision)) for r in rs)

    zero: Final["Angle"]  # type: ignore
    """ A constant for the angle 0. """

    pi: Final["Angle"]  # type: ignore
    """ A constant for the angle pi. """


# Set static constants for Angle:
Angle.zero = Angle(0)  # type: ignore
Angle.pi = Angle(1)  # type: ignore

pi: Final[Angle] = Angle.pi
""" Constant for `Angle.pi`. """

π: Final[Angle] = Angle.pi  # pylint: disable=non-ascii-name
""" Constant for `Angle.pi`. """


def SumprodAngleExpr(
    *exprs: AngleExpr, coeffs: Union[int, Fraction, Sequence[Union[int, Fraction]]] = 1
) -> AngleExpr:
    if not isinstance(coeffs, Sequence):
        coeffs = (coeffs,)
    if len(coeffs) != len(exprs):
        raise ValueError(
            f"Expected a sequence of {len(exprs)} coefficients, "
            f"found {len(coeffs)}."
        )
    _coeffs: Dict[AngleExpr, Fraction] = {}
    _const: Angle = Angle.zero
    for e, c in zip(exprs, coeffs):
        if isinstance(e, Angle):
            _const += e._mul(c)  # pylint: disable = protected-access
        elif isinstance(e, _SumprodAngleExpr):
            for sub_e, sub_c in e.coeffs.items():
                new_c = c * sub_c
                if sub_e in _coeffs:
                    new_c += _coeffs[sub_e]
                _coeffs[sub_e] = new_c
        elif c != 0:
            c = Fraction(c)
            if e in _coeffs:
                c += _coeffs[e]
            _coeffs[e] = c
    _coeffs = {
        e: c
        for e, c in sorted(_coeffs.items(), key=lambda i: -i[1])
        if c != 0 and not e.is_zero
    }
    if not _coeffs:
        return _const
    return _SumprodAngleExpr(_coeffs, _const)


class _SumprodAngleExpr(AngleExpr):
    _coeffs: Mapping[AngleExpr, Fraction]
    _const: "Angle"

    def __init__(
        self, coeffs: Mapping[AngleExpr, Fraction], const: "Angle" = Angle.zero
    ):
        if any(isinstance(e, Angle) for e in coeffs):
            raise ValueError(
                "Keys of 'coeff' argument of _SumprodAngleExpr"
                "constructor cannot be Angle."
            )
        self._coeffs = MappingProxyType(
            {e: c for e, c in coeffs.items() if c != 0 and not e.is_zero}
        )
        self._const = const

    @property
    def coeffs(self) -> Mapping[AngleExpr, Fraction]:
        return self._coeffs

    @property
    def const(self) -> "Angle":
        return self._const

    @property
    def is_zero(self) -> bool:
        return not self.coeffs and self.const.is_zero

    @property
    def is_pi(self) -> bool:
        return not self.coeffs and self.const.is_pi

    @property
    def to_qiskit(self) -> Any:
        return sum(
            (c * e.to_qiskit for e, c in self.coeffs.items()), self.const.to_qiskit
        )

    def __hash__(self) -> int:
        return hash((_SumprodAngleExpr, tuple(self.coeffs.items()), self.const))

    def _str_repr(self, f: Callable[[Union[Angle, AngleExpr]], str]) -> str:
        if not self.coeffs:
            return f(self.const)
        pos_sub_e = {e: c for e, c in self.coeffs.items() if c > 0}
        neg_sub_e = {e: c for e, c in self.coeffs.items() if c < 0}
        s = "+".join(("" if c == 1 else str(c)) + f(e) for e, c in pos_sub_e.items())
        if neg_sub_e:
            s += "-" + "".join(
                ("-" if c == -1 else str(c)) + f(e) for e, c in pos_sub_e.items()
            )
        if self.const != 0:
            s += f(self.const)
        return s

    def __str__(self) -> str:
        return self._str_repr(str)

    def __repr__(self) -> str:
        return self._str_repr(repr)

    @property
    def repr_latex(self) -> str:
        """
        LaTeX math mode representation of this number.
        """
        return self._str_repr(lambda e: e.repr_latex)

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, _SumprodAngleExpr):
            return self.coeffs == other.coeffs and self.const == other.const
        if isinstance(other, AngleExpr):
            return False
        return NotImplemented


def ModAngleExpr(expr: AngleExpr, mod: AngleExpr) -> AngleExpr:
    if isinstance(expr, Angle) and isinstance(mod, Angle):
        return expr % mod
    return _ModAngleExpr(expr, mod)


class _ModAngleExpr(AngleExpr):
    _expr: AngleExpr
    _mod: AngleExpr

    def __init__(self, expr: AngleExpr, mod: AngleExpr):
        if isinstance(expr, Angle) and isinstance(mod, Angle):
            raise ValueError(
                "Arguments to _ModAngleExpr constructor cannot both be Angle."
            )
        self._expr = expr
        self._mod = mod

    @property
    def expr(self) -> AngleExpr:
        return self._expr

    @property
    def mod(self) -> AngleExpr:
        return self._mod

    @property
    def is_zero(self) -> bool:
        return self.expr.is_zero or self.expr == self.mod

    @property
    def to_qiskit(self) -> Any:
        return self.expr.to_qiskit % self.mod.to_qiskit

    def __hash__(self) -> int:
        return hash((_ModAngleExpr, self.expr, self.mod))

    def __str__(self) -> str:
        return f"{str(self.expr)}%{str(self.mod)}"

    def __repr__(self) -> str:
        return f"{repr(self.expr)}%{repr(self.mod)}"

    @property
    def repr_latex(self) -> str:
        """
        LaTeX math mode representation of this number.
        """
        return rf"{self.expr.repr_latex} mod\left({self.mod.repr_latex}\right)"

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, _ModAngleExpr):
            return self._expr == other._expr and self._mod == other._mod
        if isinstance(other, AngleExpr):
            return False
        return NotImplemented


class AngleVar(AngleExpr):
    _global_id: ClassVar[int] = 0
    _qiskit_bindings: ClassVar[Dict[int, Any]]
    _id: int
    _label: str
    _latex_label: str

    def __init__(self, label: str, latex_label: Optional[str] = None):
        self._label = label
        if latex_label is None:
            latex_label = label
        self._latex_label = latex_label
        self._id = AngleVar._global_id
        AngleVar._global_id += 1

    @property
    def to_qiskit(self) -> Any:
        if self._id in AngleVar._qiskit_bindings:
            return AngleVar._qiskit_bindings[self._id]
        try:
            # pylint: disable = import-outside-toplevel
            from qiskit.circuit import Parameter  # type: ignore
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError("You must install the 'qiskit' library.") from e
        p = Parameter(self._repr_latex_)
        AngleVar._qiskit_bindings[self._id] = p
        return p

    def __hash__(self) -> int:
        return hash((AngleVar, self._id))

    def __str__(self) -> str:
        return self._label

    def __repr__(self) -> str:
        return self._label

    @property
    def repr_latex(self) -> str:
        """
        LaTeX math mode representation of this number.
        """
        return self._latex_label

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, AngleVar):
            return self is other
        if isinstance(other, AngleExpr):
            return False
        return NotImplemented


def _validate_vec2(vec2: Tuple[int, int]) -> None:
    if not isinstance(vec2, tuple) or len(vec2) != 2:
        raise TypeError("Expected pair.")
    if not all(isinstance(x, (int, float)) for x in vec2):
        raise TypeError("Expected pair of integers or floats.")


class SVGBuilder:
    """
    Utility class for building certain SVG images.
    Follows the [Fluent interface pattern](https://en.wikipedia.org/wiki/Fluent_interface).
    """

    _width: int
    _height: int
    _tags: List[str]

    def __init__(self, width: int, height: int):
        if not isinstance(width, int) or width <= 0:
            raise TypeError("Width should be positive integer.")
        if not isinstance(height, int) or height <= 0:
            raise TypeError("Height should be positive integer.")
        self._width = width
        self._height = height
        self._def_object_ids = []
        self._tags = []

    @property
    def width(self) -> int:
        """
        The figure width.
        """
        return self._width

    @width.setter
    def width(self, new_width: int) -> None:
        """
        Set the figure width.
        """
        if not isinstance(new_width, int) or new_width <= 0:
            raise TypeError("Width should be positive integer.")
        self._width = new_width

    @property
    def height(self) -> int:
        """
        The figure height.
        """
        return self._height

    @height.setter
    def height(self, new_height: int) -> None:
        """
        Set the figure height.
        """
        if not isinstance(new_height, int) or new_height <= 0:
            raise TypeError("Height should be positive integer.")
        self._height = new_height

    @property
    def tags(self) -> Sequence[str]:
        """
        The current sequence of tags.
        """
        return self._tags

    def line(self, fro: Tuple[int, int], to: Tuple[int, int]) -> "SVGBuilder":
        """
        Draws a line from given coordinates to given coordinates.
        """
        _validate_vec2(fro)
        _validate_vec2(to)
        fx, fy = fro
        tx, ty = to
        tag = f'<path fill="none" stroke="black"' f' d="M {fx}, {fy} L {tx}, {ty}"/>'
        self._tags.append(tag)
        return self

    def line_bend(
        self, fro: Tuple[int, int], to: Tuple[int, int], left=False, degree=5
    ):
        _validate_vec2(fro)
        _validate_vec2(to)

        fx, fy = fro
        tx, ty = to
        bx, by = calculate_orthogonal_point(
            np.asarray(fro), np.asarray(to), d=degree, left=left
        )

        tag = (
            f'<path d="M {fx} {fy} Q {bx} {by} {tx} {ty}" fill="none" stroke="black"/>'
        )
        self._tags.append(tag)
        return self

    def add_diagonal_fill(self, color_1: str, color_2: str, id: str) -> "SVGBuilder":
        tag = (
            f'<linearGradient id="{id}" x1="0%" x2="100%" y1="100%" y2="0%">'
            f'<stop offset="0%"  stop-color="{color_1}"/>'
            f'<stop offset="50%"  stop-color="{color_1}"/>'
            f'<stop offset="50%"  stop-color="{color_2}"/>'
            f'<stop offset="100%"  stop-color="{color_2}"/>'
            f"</linearGradient>"
        )

        self._def_object_ids.append(id)
        self._tags.append(tag)
        return self

    def square(
        self, centre: Tuple[int, int], width: int, height: int, fill
    ) -> "SVGBuilder":
        _validate_vec2(centre)
        x, y = centre
        if fill in self._def_object_ids:
            tag = (
                f'<rect fill="url(#{fill})" stroke="black" x="{x}" y="{y}" '
                f'width="{width}" height="{height}"/>'
            )
            self._tags.append(tag)
        elif isinstance(fill, str):
            tag = (
                f'<rect fill="{fill}" stroke="black" x="{x}" y="{y}" '
                f'width="{width}" height="{height}"/>'
            )
            self._tags.append(tag)
        else:
            raise TypeError(f"Fill must be string or a defined Tag. Got: {fill} ")
        return self

    def text_with_square(
        self, centre: Tuple[int, int], width: int, height: int, text: str
    ) -> "SVGBuilder":
        _validate_vec2(centre)
        x, y = centre
        tag = (
            f'<svg x="{x}" y="{y}" height="{height}" width="{width}">'
            f'<rect x="0" y="0" width="100%" height="100%" '
            f'stroke="black" fill="white" stroke-width="5 %"/>'
            f'<text x="50%" y="50%" width="100%" height="100%" font-size="100%" '
            f'dominant-baseline="middle" text-anchor="middle" >{text}</text>'
            f"</svg>"
        )
        self._tags.append(tag)
        return self

    def circle(self, centre: Tuple[int, int], r: int, fill: str) -> "SVGBuilder":
        """
        Draws a circle with given centre and radius.
        """
        _validate_vec2(centre)
        if not isinstance(fill, str):
            raise TypeError("Fill must be string.")
        x, y = centre
        tag = f'<circle fill="{fill}" stroke="black"' f' cx="{x}" cy="{y}" r="{r}"/>'
        self._tags.append(tag)
        return self

    def rect(
        self, centre: Tuple[int, int], width: int, height: int, fill: str
    ) -> "SVGBuilder":
        """
        Draws a rectangle with given centre, width and height.
        """
        _validate_vec2(centre)
        if not isinstance(fill, str):
            raise TypeError("Fill must be string.")
        x, y = centre
        tag = (
            f'<rect fill="{fill}" stroke="black"'
            f' x="{x - width // 2}" y="{y - height // 2}"'
            f' width="{width}" height="{height}"/>'
        )
        self._tags.append(tag)
        return self

    def text(
        self, pos: Tuple[int, int], text: str, *, font_size: int = 10, center=False
    ) -> "SVGBuilder":
        """
        Draws text at the given position (stroke/fill not used).
        """
        _validate_vec2(pos)
        if not isinstance(text, str):
            raise TypeError("Text must be string.")
        if not isinstance(font_size, int) or font_size <= 0:
            raise TypeError("Font size must be positive integer.")
        x, y = pos
        attrs = f'x="{x}" y="{y + font_size // 4}" font-size="{font_size}"'
        attrs += f' text-anchor="middle"' if center else ""
        tag = f"<text {attrs}>{text}</text>"
        self._tags.append(tag)
        return self

    def __repr__(self) -> str:
        # pylint: disable = line-too-long
        body = "\n".join(self._tags)
        headers = "\n".join(
            [
                '<?xml version="1.0" encoding="utf-8"?>',
                '<!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 1.1//EN" "http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd">',
            ]
        )
        return f'{headers}\n<svg version="1.1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="{self.width}" height="{self.height}">{body}\n</svg>'

    def __irshift__(self, other: "SVGBuilder") -> "SVGBuilder":
        if not isinstance(other, SVGBuilder):
            raise TypeError(f"Expected SVGBuilder, found {type(other)}")
        for tag in other._tags:
            self._tags.append(tag)
        return self


@runtime_checkable
class TempSchedule(Protocol):
    """
    Protocol for a temperature schedule.
    The temperature is a number (int or float) computed from the iteration
    number `it` (starting from 0) and the total number of iterations `num_iter`
    (passed as a keyword argument).
    """

    def __call__(self, it: int, num_iters: int) -> float: ...


@runtime_checkable
class TempScheduleProvider(Protocol):
    """
    Protocol for a function constructing a temperature schedule
    from an initial and final temperatures.
    """

    def __call__(
        self, t_init: Union[int, float], t_final: Union[int, float]
    ) -> TempSchedule: ...


def linear_temp_schedule(
    t_init: Union[int, float], t_final: Union[int, float]
) -> TempSchedule:
    """
    Returns a straight/linear temperature schedule for given initial and final temperatures,
    from https://link.springer.com/article/10.1007/BF00143921
    """
    if not isinstance(t_init, (int, float)):
        raise TypeError(f"Expected int or float, found {type(t_init)}.")
    if not isinstance(t_final, (int, float)):
        raise TypeError(f"Expected int or float, found {type(t_final)}.")

    def temp_schedule(it: int, num_iters: int) -> float:
        return t_init + (t_final - t_init) * it / (num_iters - 1)

    return temp_schedule


def geometric_temp_schedule(
    t_init: Union[int, float], t_final: Union[int, float]
) -> TempSchedule:
    """
    Returns a geometric temperature schedule for given initial and final temperatures,
    from https://link.springer.com/article/10.1007/BF00143921
    """
    if not isinstance(t_init, (int, float)):
        raise TypeError(f"Expected int or float, found {type(t_init)}.")
    if not isinstance(t_final, (int, float)):
        raise TypeError(f"Expected int or float, found {type(t_final)}.")

    def temp_schedule(it: int, num_iters: int) -> float:
        return t_init * ((t_final / t_init) ** (it / (num_iters - 1.0)))  # type: ignore[no-any-return]

    return temp_schedule


def reciprocal_temp_schedule(
    t_init: Union[int, float], t_final: Union[int, float]
) -> TempSchedule:
    """
    Returns a reciprocal temperature schedule for given initial and final temperatures,
    from https://link.springer.com/article/10.1007/BF00143921
    """
    if not isinstance(t_init, (int, float)):
        raise TypeError(f"Expected int or float, found {type(t_init)}.")
    if not isinstance(t_final, (int, float)):
        raise TypeError(f"Expected int or float, found {type(t_final)}.")

    def temp_schedule(it: int, num_iters: int) -> float:
        num = t_init * t_final * (num_iters - 1)
        denom = (t_final * num_iters - t_init) + (t_init - t_final) * (it + 1)
        return num / denom

    return temp_schedule


def log_temp_schedule(
    t_init: Union[int, float], t_final: Union[int, float]
) -> TempSchedule:
    """
    Returns a logarithmic temperature schedule for given initial and final temperatures,
    from https://link.springer.com/article/10.1007/BF00143921
    """
    if not isinstance(t_init, (int, float)):
        raise TypeError(f"Expected int or float, found {type(t_init)}.")
    if not isinstance(t_final, (int, float)):
        raise TypeError(f"Expected int or float, found {type(t_final)}.")

    def temp_schedule(it: int, num_iters: int) -> float:
        num = t_init * t_final * (math.log(num_iters + 1) - math.log(2))
        denom = (t_final * math.log(num_iters + 1) - t_init * math.log(2)) + (
            t_init - t_final
        ) * math.log(it + 2)
        return num / denom

    return temp_schedule


StandardTempScheduleName = Literal["linear", "geometric", "reciprocal", "log"]
"""
    Names of the standard temperature schedules.
"""

StandardTempSchedule = Tuple[
    StandardTempScheduleName, Union[int, float], Union[int, float]
]
"""
    Type for standard temperature schedules.
"""

StandardTempSchedules: Final[
    Mapping[StandardTempScheduleName, TempScheduleProvider]
] = {
    "linear": linear_temp_schedule,
    "geometric": geometric_temp_schedule,
    "reciprocal": reciprocal_temp_schedule,
    "log": log_temp_schedule,
}
"""
    Dictionary of standard temperature schedule providers.
"""
