"""
Quantum Circuit
"""
import io
import re
import cirq
import qiskit
import numpy as np
import networkx as nx
import rustworkx as rx
from math import pi
from scipy import linalg
from itertools import chain
from typing import List, Tuple, Dict, Union
from copy import deepcopy, copy
from collections import Counter
from qiskit import QuantumCircuit
from phoenix.basic import gates
from phoenix.basic.gates import Gate
from phoenix.utils.ops import replace_close_to_zero_with_zero
from phoenix.utils.functions import limit_angle
from phoenix.utils.graphs import draw_circ_dag_mpl, draw_circ_dag_graphviz, rx_to_nx_graph, node_index


class Circuit(list):
    def __init__(self, gates: List[Gate] = None):
        if gates is not None:
            assert all([g.qregs for g in gates]), "Each gate should act on specific qubit(s)"
        super().__init__(gates if gates else [])

    def __hash__(self):
        return hash(id(self))

    def append(self, *gates):
        assert all([g.qregs for g in gates]), "Each gate should act on specific qubit(s)"
        for g in gates:
            super().append(g)

    def prepend(self, *gates):
        assert all([g.qregs for g in gates]), "Each gate should act on specific qubit(s)"
        for g in reversed(gates):
            super().insert(0, g)

    def deepclone(self):
        """Deep duplicate"""
        return Circuit(deepcopy(self))

    def clone(self):
        """Shadow duplicate"""
        return Circuit(copy(self))

    def __add__(self, other):
        return Circuit(deepcopy(self.gates) + deepcopy(other.gates))

    def __repr__(self):
        # return 'Circuit(num_gates: {}, num_qubits: {}, with_measure: {})'.format(self.num_gates, self.num_qubits,
        #                                                                          self.with_measure)
        return 'Circuit(num_gates: {}, num_qubits: {})'.format(self.num_gates, self.num_qubits)

    def compose(self, other):
        self.extend(other)

    @property
    def nonlocal_structure(self):
        """Obtain its circuit structure only with nonlocal gates"""
        return Circuit([g for g in self if g.num_qregs > 1])

    def to_qiskit(self):
        """Convert to qiskit.QuantumCircuit instance"""
        return QuantumCircuit.from_qasm_str(self.to_qasm())

    @classmethod
    def from_qiskit(cls, qc: QuantumCircuit):
        """Convert from qiskit.QuantumCircuit instance"""
        return _from_qiskit(qc)

    def to_cirq(self):
        """Convert to cirq.Circuit instance"""
        return _to_cirq(self)

    @classmethod
    def from_cirq(cls, circ: cirq.Circuit):
        """Convert from cirq.Circuit instance"""
        return cls.from_qasm(cirq.qasm(circ))

    def to_tket(self):
        """Convert to pytket.circuit.Circuit instance"""
        from pytket.qasm import circuit_from_qasm_str
        return circuit_from_qasm_str(self.to_qasm())
    
    @classmethod
    def from_tket(cls, circ):
        """Convert from pytket.circuit.Circuit instance"""
        from pytket.qasm import circuit_to_qasm_str
        return cls.from_qasm(circuit_to_qasm_str(circ))

    def to_bqskit(self):
        """Convert to bqskit.Circuit instance"""
        return _to_bqskit(self)

    @classmethod
    def from_bqskit(cls, circ):
        """Convert from bqskit.Circuit instance"""
        return _from_bqskit(circ)

    def to_qasm(self, fname: str = None):
        """Convert self to QSAM string"""
        if not self:
            return ''

        circuit = deepcopy(self)
        output = QASMStringIO()
        output.write_header()
        n = self.num_qubits_with_dummy

        # write customized gate OpenQASM definitions
        for qasm_def in np.unique([g.qasm_def for g in self if g.qasm_def is not None]):
            output.write(qasm_def)
            output.write_line_gap(2)

        # no measurement, just computing gates
        output.write_comment('Qubits: {}'.format(list(range(n))))
        output.write_qregs(n)
        output.write_line_gap()

        tuples_parsed = parse_to_qasm_tuples(circuit)

        output.write_comment('Quantum gate operations')
        for opr, idx in tuples_parsed:
            output.write_operation(opr, 'q', *idx)

        qasm_str = output.getvalue()
        output.close()
        if fname is not None:
            with open(fname, 'w') as f:
                f.write(qasm_str)
        return qasm_str

    @classmethod
    def from_qasm(cls, qasm_str: str = None, fname: str = None):
        """
        Convert QASM string to Circuit instance

        First parse each line as a list of strings, e.g.,

        'cx q[2], q[1];',                     -->  ['cx', 'q[2]', 'q[1]']
        'h q[2];',                            -->  ['h', 'q[2]']
        'rx(0.50) q[0];',                     -->  ['rx', '0.50', 'q[0]']
        'ry(0.50) q[1];',                     -->  ['ry', '0.50', 'q[1]']
        'u3(0.30, 0.12, 1.12) q[2];',         -->  ['u3', '0.30', '0.12', '1.12', 'q[2]']
        'cu3(0.30, 0.12, 1.12) q[0], q[1];',  -->  ['cu3', '0.30', '0.12', '1.12', 'q[0]', 'q[1]']

        Then construct the corresponding Gate instance according to the parsed list.
        """
        if qasm_str is None and fname is None:
            raise ValueError("Either qasm_str or fname should be given")
        
        # For performance and robustness, we use the qiskit parser to parse OpenQASM
        if qasm_str is None:
            return cls.from_qiskit(QuantumCircuit.from_qasm_file(fname))
        return cls.from_qiskit(QuantumCircuit.from_qasm_str(qasm_str))

    def rewire(self, mapping: Dict[int, int]):
        """
        Rewire the circuit according to a given mapping
        """
        mapped_circ = Circuit()
        for g in self:
            mapped_circ.append(g.on(
                [mapping[tq] for tq in g.tqs],
                [mapping[cq] for cq in g.cqs]
            ))
        return mapped_circ

    def inverse(self):
        """Inverse of the original circuit by reversing the order of gates' hermitian conjugates"""
        circ = Circuit()
        for g in reversed(self):
            if isinstance(g, gates.Can):
                circ.append(gates.Z.on(g.tqs[0]))
                circ.append(gates.Can(g.angles[0], g.angles[1], -g.angles[2]).on(g.tqs, g.cqs))
                circ.append(gates.Z.on(g.tqs[0]))
            else:
                circ.append(g.hermitian())
        return circ

    def unitary(self, msb: bool = False, with_dummy: bool = False) -> np.ndarray:
        """
        Convert a quantum circuit to a unitary matrix.

        Args:
            msb (bool): if True, means the most significant bit (MSB) is on the left, i.e., little-endian representation
            with_dummy (bool): if True, means taking dummy qubits into account when calculating the unitary matrix

        Returns:
            Matrix, Equivalent unitary matrix representation.
        """
        from phoenix.utils.ops import tensor_slots, controlled_unitary_matrix, circuit_to_unitary

        if self.num_qubits > 12:
            raise ValueError('Circuit to compute unitary matrix has too many qubits')
        if self.num_qubits > 7:
            return circuit_to_unitary(self, 'cirq')

        ops = []
        if with_dummy:
            n = self.num_qubits_with_dummy
            circ = self
        else:
            n = self.num_qubits
            circ = self.rewire({p: q for p, q in zip(self.qubits, range(n))})
        for g in circ:
            if g.n_qubits > int(np.log2(g.data.shape[0])) == 1:
                # identical tensor-product gate expanded from single-qubit gate
                data = cirq.kron(*([g.data] * g.n_qubit))
            else:
                data = g.data

            if not g.cqs:
                mat = tensor_slots(data, n, g.tqs)
            else:
                mat = controlled_unitary_matrix(data, len(g.cqs))
                mat = tensor_slots(mat, n, g.cqs + g.tqs)

            ops.append(mat)

        unitary = cirq.dot(*reversed(ops))
        if msb:
            unitary = tensor_slots(unitary, n, list(range(n - 1, -1, -1)))
        return unitary

    def layer(self, on_body='dag') -> List[List[Gate]]:
        """
        Divide a circuit into different layers
        ---
        For large-size circuit, it is recommended to use on_body='dag' for efficiency.
        """
        if on_body == 'circuit':
            from phoenix.utils.passes import obtain_front_layer_from_circuit
            layers = []
            circ = self.clone()
            while circ:
                front_layer = obtain_front_layer_from_circuit(circ)
                layers.append(front_layer)
                for g in front_layer:
                    circ.remove(g)
        elif on_body == 'dag':
            from phoenix.utils.passes import dag_to_layers
            layers = dag_to_layers(self.to_dag())
            layers = list(map(_sort_gates_on_qreg, layers))
        else:
            raise ValueError('Unsupported argument on_body (options: "circuit" or "dag")')
        return layers

    def to_dag(self, backend='rustworkx') -> Union[rx.PyDiGraph, nx.DiGraph]:
        """
        Convert a circuit into a Directed Acyclic Graph (DAG) according to dependency of each gate's qubits.

        Args:
            backend: 'networkx' or 'rustworkx'
        """
        all_gates = self.gates
        dag = rx.PyDiGraph(multigraph=False)
        dag.add_nodes_from(all_gates)
        while all_gates:
            g = all_gates.pop(0)
            qregs = set(g.qregs)
            for g_opt in all_gates:  # traverse the subsequent optional gates
                qregs_opt = set(g_opt.qregs)
                if dependent_qubits := qregs_opt & qregs:
                    dag.add_edge(node_index(dag, g), node_index(dag, g_opt), {'qubits': list(dependent_qubits)})
                    qregs -= qregs_opt
                if not qregs:
                    break
        if backend == 'networkx':
            return rx_to_nx_graph(dag)
        return dag

    def draw_circ_dag_mpl(self, fname: str = None, figsize=None, fix_layout=False):
        return draw_circ_dag_mpl(self.to_dag(), fname, figsize, fix_layout)

    def draw_circ_dag_graphviz(self, fname: str = None):
        return draw_circ_dag_graphviz(self.to_dag(), fname)

    def gate_stats(self) -> Dict[str, int]:
        """Statistics of gate names and occurring number in the circuit"""
        return dict(sorted(Counter([g.name for g in self.gates]).items()))

    def gate_count(self, gate_type: Union[str, type]) -> int:
        """Count the number of specific gate type"""
        if isinstance(gate_type, str):
            gate_type = gate_type.lower()
            return sum([g.name.lower() == gate_type for g in self.gates])
        return sum([isinstance(g, gate_type) for g in self.gates])

    def canonical_stats(self) -> Dict[Tuple[float, float, float], int]:
        """Statistic of Canonical coordinates within the chamber 1/2 ≥ x ≥ y ≥ |z|"""
        canonicals = [g for g in self.gates if isinstance(g, gates.Can)]
        counts = Counter([tuple(np.round(can.kak_coefficients / (pi / 2), 5)) for can in canonicals])
        counts = dict(sorted(counts.items(), key=lambda x: x[1], reverse=True))
        return counts

    def qubit_dependency(self, mapping: Dict[int, int] = None) -> nx.Graph:
        """
        Return the qubit dependency graph.
        If mapping is given, returned is physical qubit dependency graph; Otherwise, logical qubit dependency graph.
        """
        dependency = nx.Graph()
        dependency.add_nodes_from(self.qubits)
        for g in self:
            if g.num_qregs > 2:
                raise ValueError('Only support 1Q or 2Q gates with designated qubits')
            if g.num_qregs > 1:
                if mapping:
                    qregs = [mapping[q] for q in g.qregs]
                else:
                    qregs = g.qregs
                dependency.add_edge(*qregs)
        return dependency

    def draw_qubit_dependency(self, mapping: Dict[int, int] = None):
        """
        Return a visual representation of the qubit connectivity graph as a graphviz
        object.

        :returns:   Representation of the qubit connectivity graph of the circuit
        :rtype:     graphviz.Graph
        """
        import graphviz as gv
        G = gv.Graph(
            "Qubit connectivity",
            node_attr={
                "shape": "circle",
                "color": "blue",
                "fontname": "Courier",
                "fontsize": "8",
            },
            engine="neato"
        )
        G.edges(
            (f'q{src}', f'q{tgt}') for src, tgt in self.qubit_dependency().edges()
        )
        return G

    @property
    def instr_seq(self) -> List[Tuple[str, Tuple[float], Tuple[int], Tuple[int]]]:
        instr_seq = []
        for g in self:
            if g.angle:
                params = (g.angle,)
            elif g.angles:
                params = g.angles
            elif g.exponent:
                params = (g.exponent,)
            elif g.params:
                params = g.params
            else:
                params = ()
            instr_seq.append((g.name, params, tuple(sorted(g.cqs)), tuple(sorted(g.tqs))))
        return instr_seq

    @property
    def gates(self, with_measure: bool = True) -> List[Gate]:
        if with_measure:
            list(self)
        return list(filter(lambda g: not isinstance(g, gates.Measurement), self))

    @property
    def num_gates(self):
        return len(self)

    @property
    def num_nonlocal_gates(self):
        return len([g for g in self if g.num_qregs > 1])

    @property
    def depth(self):
        """number of circuit layers (i.e., length of critical path)"""
        wire_lengths = [0] * self.num_qubits_with_dummy
        for g in self:
            qubits = g.qregs
            current_depth = max(wire_lengths[q] for q in qubits) + 1
            for q in qubits:
                wire_lengths[q] = current_depth
        return max(wire_lengths)


    @property
    def depth_nonlocal(self):
        """number of circuit layers including only nonlocal gates"""
        wire_lengths = [0] * self.num_qubits_with_dummy
        for g in self:
            if g.num_qregs == 1:
                continue
            qubits = g.qregs
            current_depth = max(wire_lengths[q] for q in qubits) + 1
            for q in qubits:
                wire_lengths[q] = current_depth
        return max(wire_lengths)

    @property
    def qubits(self) -> List[int]:
        """qubits indices in the quantum circuit"""
        return sorted(list(set(chain.from_iterable([g.qregs for g in self]))))

    @property
    def num_qubits(self):
        """number of qubits in the quantum circuit (actually used)"""
        return len(self.qubits)

    @property
    def num_qubits_with_dummy(self):
        """number of qubits in the quantum circuit (including dummy qubits)"""
        return max(self.qubits) + 1 if self.gates else 0

    @property
    def max_gate_weight(self):
        """maximum gate weight"""
        return max([g.num_qregs for g in self])

    @property
    def with_measure(self):
        """Whether the circuit contains measurement gates"""
        for g in self:
            if isinstance(g, gates.Measurement):
                return True
        return False


class QASMStringIO(io.StringIO):
    """
    Specific StringIO extension class for QASM string generation
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def write_qregs(self, n: int) -> int:
        """
        Write quantum register declarations into the string stream.

        Args:
            n: number of qubits
        """
        n = super().write('qreg q[{}];\n'.format(n))
        return n

    def write_operation(self, opr: str, qreg_name: str, *args) -> int:
        """
        Write computational gate operation into the string stream.

        Args:
            opr: e.g. 'cx'
            qreg_name: e.g. 'q'
            args: e.g. 0, 1
        """
        if len(args) == 0:
            line = opr + ' ' + qreg_name + ';\n'
        else:
            line_list_qubits = []
            for idx in args:
                line_list_qubits.append(qreg_name + '[{}]'.format(idx))
            line = opr + ' ' + ', '.join(line_list_qubits) + ';\n'
        n = super().write(line)
        return n

    def write_line_gap(self, n: int = 1) -> int:
        n = super().write('\n' * n)
        return n

    def write_comment(self, comment: str) -> int:
        n = super().write('// ' + comment + '\n')
        return n

    def write_header(self) -> int:
        """
        Write the QASM text file header
        """
        n1 = super().write('OPENQASM 2.0;\n')
        n2 = super().write('include "qelib1.inc";\n')
        n3 = self.write_line_gap()
        return n1 + n2 + n3


def parse_to_qasm_tuples(circuit: Circuit) -> List[Tuple[str, List[int]]]:
    """
    Parse each Gate instance into a tuple consisting gate name and quantum register indices of a list

    Args:
        circuit (Circuit): input Circuit instance

    Returns:
        List of tuples representing designated quantum operation, e.g. [('u3', [0]), ..., ('cu3', [0, 1])]
    """
    parsed_list = []
    for g in circuit:
        if not ((g.n_qubits == 1 and len(g.cqs) <= 1) or
                (len(g.tqs) == 2 and len(g.cqs) <= 1 and isinstance(g, (gates.SWAPGate, gates.ISWAPGate))) or
                (len(g.tqs) == 2 and len(g.cqs) == 0 and isinstance(g, (gates.RXX, gates.RYY, gates.RZZ, gates.Can))) or
                (len(g.tqs) == 1 and len(g.cqs) == 2 and isinstance(g, (gates.XGate, gates.ZGate))) or
                (len(g.tqs) == 1 and len(g.cqs) >= 3 and isinstance(g, gates.XGate)) or
                (len(g.tqs) == 2 and len(g.cqs) == 0 and isinstance(g, gates.Clifford2QGate))):
            raise ValueError('Only support 1Q or 2Q gates with designated qubits except for CCX, CCZ and MCX')

        if isinstance(g, gates.IGate):
            gname = 'id'
        elif isinstance(g, gates.VGate):
            gname = 'sx'
        else:
            gname = g.name.lower()

        if g.cqs:
            if gname not in gates.CONTROLLABLE_GATES:
                raise ValueError(f'{g} is not supported for transformation')
            if gname == 'x' and len(g.cqs) >= 3:
                opr = 'mcx'
            elif gname in gates.FIXED_GATES:
                if len(g.cqs) == 2:
                    opr = 'cc{}'.format(gname)
                else:
                    opr = 'c{}'.format(gname)
            elif gname == 'u3':
                angles = list(map(limit_angle, g.angles))
                factors = list(map(lambda x: x / pi, angles))
                # opr = 'cu3({:.4f}, {:.4f}, {:.4f})'.format(*angles)
                opr = 'cu3({}*pi, {}*pi, {}*pi)'.format(*factors)
            elif gname == 'u2':
                angles = list(map(limit_angle, g.angles))
                factors = list(map(lambda x: x / pi, angles))
                opr = 'cu2({}*pi, {}*pi)'.format(*factors)
            else:  # CR(X/Y/Z) and U1 gate
                angle = limit_angle(g.angle)
                factor = angle / pi
                opr = 'c{}({}*pi)'.format(gname, factor)
        else:
            if gname in gates.FIXED_GATES:
                opr = gname
            elif gname in ['u3', 'can']:
                angles = list(map(limit_angle, g.angles))
                factors = list(map(lambda x: x / pi, angles))
                opr = '{}({}*pi, {}*pi, {}*pi)'.format(gname, *factors)
            elif gname == 'u2':
                angles = list(map(limit_angle, g.angles))
                factors = list(map(lambda x: x / pi, angles))
                opr = '{}({}*pi, {}*pi)'.format(gname, *factors)
            elif match := re.match(r"c\((x|y|z), (x|y|z)\)", gname):
                opr = 'c{}{}'.format(*match.groups())
            else:  # R(X/Y/Z), R(XX/YY/ZZ), U1
                angle = limit_angle(g.angle)
                factor = angle / pi
                opr = '{}({}*pi)'.format(gname, factor)
        parsed_list.append((opr, g.qregs))
    return parsed_list


def _sort_gates_on_qreg(circuit: List[Gate], descend=False) -> List[Gate]:
    if descend:
        return sorted(circuit, key=lambda g: max(g.qregs))
    else:
        return sorted(circuit, key=lambda g: min(g.qregs))


def _bqskit_operation_to_phoenix_gate(op, params=None) -> Gate:
    import bqskit.ir.gates as bqskit_gates

    if params is None:
        params = op.params

    if isinstance(op.gate, bqskit_gates.ConstantUnitaryGate):
        return gates.UnivGate(op.get_unitary().numpy).on(list(op.location))
    elif isinstance(op.gate, bqskit_gates.CCXGate):
        return gates.X.on(op.location[2], [op.location[0], op.location[1]])
    elif isinstance(op.gate, bqskit_gates.XGate):
        return gates.X.on(op.location[0])
    elif isinstance(op.gate, bqskit_gates.YGate):
        return gates.Y.on(op.location[0])
    elif isinstance(op.gate, bqskit_gates.ZGate):
        return gates.Z.on(op.location[0])
    elif isinstance(op.gate, bqskit_gates.HGate):
        return gates.H.on(op.location[0])
    elif isinstance(op.gate, bqskit_gates.SGate):
        return gates.S.on(op.location[0])
    elif isinstance(op.gate, bqskit_gates.SdgGate):
        return gates.SDG.on(op.location[0])
    elif isinstance(op.gate, bqskit_gates.TGate):
        return gates.T.on(op.location[0])
    elif isinstance(op.gate, bqskit_gates.TdgGate):
        return gates.TDG.on(op.location[0])
    elif isinstance(op.gate, bqskit_gates.SwapGate):
        return gates.SWAP.on(list(op.location))
    elif isinstance(op.gate, bqskit_gates.ISwapGate):
        return gates.ISWAP.on(list(op.location))
    elif isinstance(op.gate, bqskit_gates.SqrtISwapGate):
        return gates.SQiSW.on(list(op.location))
    elif isinstance(op.gate, bqskit_gates.CXGate):
        return gates.X.on(op.location[1], op.location[0])
    elif isinstance(op.gate, bqskit_gates.CYGate):
        return gates.Y.on(op.location[1], op.location[0])
    elif isinstance(op.gate, bqskit_gates.CZGate):
        return gates.Z.on(op.location[1], op.location[0])
    elif isinstance(op.gate, bqskit_gates.RXGate):
        return gates.RX(params[0]).on(op.location[0])
    elif isinstance(op.gate, bqskit_gates.RYGate):
        return gates.RY(params[0]).on(op.location[0])
    elif isinstance(op.gate, bqskit_gates.RZGate):
        return gates.RZ(params[0]).on(op.location[0])
    elif isinstance(op.gate, bqskit_gates.U1Gate):
        return gates.U1(params[0]).on(op.location[0])
    elif isinstance(op.gate, bqskit_gates.U2Gate):
        return gates.U2(*params).on(op.location[0])
    elif isinstance(op.gate, bqskit_gates.U3Gate):
        return gates.U3(*params).on(op.location[0])
    elif isinstance(op.gate, bqskit_gates.CRXGate):
        return gates.RX(params[0]).on(op.location[1], op.location[0])
    elif isinstance(op.gate, bqskit_gates.CRYGate):
        return gates.RY(params[0]).on(op.location[1], op.location[0])
    elif isinstance(op.gate, bqskit_gates.CRZGate):
        return gates.RZ(params[0]).on(op.location[1], op.location[0])
    elif isinstance(op.gate, bqskit_gates.RXXGate):
        return gates.RXX(params[0]).on(list(op.location))
    elif isinstance(op.gate, bqskit_gates.RYYGate):
        return gates.RYY(params[0]).on(list(op.location))
    elif isinstance(op.gate, bqskit_gates.RZZGate):
        return gates.RZZ(params[0]).on(list(op.location))
    elif isinstance(op.gate, bqskit_gates.CanonicalGate):
        return gates.Canonical(*params).on(list(op.location))
    else:
        raise ValueError(f'Unsupported gate {op.gate}')


def _from_bqskit(circ) -> Circuit:
    try:
        import bqskit
    except ImportError:
        raise ImportError('bqskit is not installed')
    assert isinstance(circ, bqskit.Circuit), 'Input should be an instance of bqskit.Circuit'
    return Circuit(list(map(_bqskit_operation_to_phoenix_gate, circ.operations())))


def _to_bqskit(circ: Circuit):
    try:
        from bqskit.ir.circuit import Circuit as bqskit_Circuit
        import bqskit.ir.gates as bqskit_gates
    except ImportError:
        raise ImportError('bqskit is not installed')
    
    assert circ.max_gate_weight <= 3, 'Only support 1Q, 2Q or 3Q (CCX) gates with designated qubits'

    n = circ.num_qubits_with_dummy
    c = bqskit_Circuit(n)

    for g in circ:
        if g.num_qregs == 3:
            assert isinstance(g, gates.XGate) and len(g.cqs) == 2, 'The only supported 3Q gate is CCX gate'
            c.append_gate(bqskit_gates.CCXGate(), g.qregs)
        elif isinstance(g, gates.UnivGate):
            c.append_gate(bqskit_gates.ConstantUnitaryGate(g.data), g.tqs)
        elif isinstance(g, gates.XGate):
            if g.cqs:
                c.append_gate(bqskit_gates.CXGate(), g.qregs)
            else:
                c.append_gate(bqskit_gates.XGate(), g.tq)
        elif isinstance(g, gates.YGate):
            if g.cqs:
                c.append_gate(bqskit_gates.CYGate(), g.qregs)
            else:
                c.append_gate(bqskit_gates.YGate(), g.tq)
        elif isinstance(g, gates.ZGate):
            if g.cqs:
                c.append_gate(bqskit_gates.CZGate(), g.qregs)
            else:
                c.append_gate(bqskit_gates.ZGate(), g.tq)
        elif isinstance(g, gates.HGate):
            if g.cqs:
                c.append_gate(bqskit_gates.CHGate(), g.qregs)
            else:
                c.append_gate(bqskit_gates.HGate(), g.tq)
        elif isinstance(g, gates.SGate):
            if g.cqs:
                c.append_gate(bqskit_gates.CSGate(), g.qregs)
            else:
                c.append_gate(bqskit_gates.SGate(), g.tq)
        elif isinstance(g, gates.SDGGate):
            c.append_gate(bqskit_gates.SdgGate(), g.tq)
        elif isinstance(g, gates.TGate):
            c.append_gate(bqskit_gates.TGate(), g.tq)
        elif isinstance(g, gates.TDGGate):
            c.append_gate(bqskit_gates.TdgGate(), g.tq)
        elif isinstance(g, gates.SWAPGate):
            c.append_gate(bqskit_gates.SwapGate(), g.tqs)
        elif isinstance(g, gates.RX):
            if g.cqs:
                c.append_gate(bqskit_gates.CRXGate(), g.qregs, [g.angle])
            else:
                c.append_gate(bqskit_gates.RXGate(), g.tq, [g.angle])
        elif isinstance(g, gates.RY):
            if g.cqs:
                c.append_gate(bqskit_gates.CRYGate(), g.qregs, [g.angle])
            else:
                c.append_gate(bqskit_gates.RYGate(), g.tq, [g.angle])
        elif isinstance(g, gates.RZ):
            if g.cqs:
                c.append_gate(bqskit_gates.CRZGate(), g.qregs, [g.angle])
            else:
                c.append_gate(bqskit_gates.RZGate(), g.tq, [g.angle])
        elif isinstance(g, gates.U1):
            c.append_gate(bqskit_gates.U1Gate(), g.tq, [g.angle])
        elif isinstance(g, gates.U2):
            c.append_gate(bqskit_gates.U2Gate(), g.tq, g.angles)
        elif isinstance(g, gates.U3):
            c.append_gate(bqskit_gates.U3Gate(), g.tq, g.angles)
        elif isinstance(g, gates.RXX):
            c.append_gate(bqskit_gates.RXXGate(), g.tqs, [g.angle])
        elif isinstance(g, gates.RYY):
            c.append_gate(bqskit_gates.RYYGate(), g.tqs, [g.angle])
        elif isinstance(g, gates.RZZ):
            c.append_gate(bqskit_gates.RZZGate(), g.tqs, [g.angle])
        elif isinstance(g, gates.ISWAPGate):
            c.append_gate(bqskit_gates.ISwapGate(), g.tqs)
        elif isinstance(g, gates.SQiSWGate):
            c.append_gate(bqskit_gates.SqrtISwapGate(), g.tqs)
        elif isinstance(g, gates.Canonical):
            c.append_gate(bqskit_gates.CanonicalGate(), g.tqs, g.angles)
        elif isinstance(g, gates.Clifford2QGate):
            c.append_gate(bqskit_gates.Clifford2QGate(g.pauli_0, g.pauli_1), g.qregs)
        else:
            raise ValueError(f'Unsupported gate {g}')

    return c


def _to_cirq(circ: Circuit) -> cirq.Circuit:
    class CirqClifford2QGate(cirq.Gate):
        """Use universal controlled gates to represent generic 2Q Clifford gates"""

        def __init__(self, pauli_0: str, pauli_1: str):
            super(CirqClifford2QGate, self)
            assert pauli_0 in ['X', 'Y', 'Z'] and pauli_1 in ['X', 'Y', 'Z']
            self.pauli_0, self.pauli_1 = pauli_0, pauli_1

        def _num_qubits_(self):
            return 2

        def _unitary_(self):
            import qiskit.quantum_info as qi
            I = qi.Pauli('I')
            P0, P1 = qi.Pauli(self.pauli_0), qi.Pauli(self.pauli_1)
            return qi.SparsePauliOp([I ^ I, P0 ^ I, I ^ P1, P0 ^ P1],
                                    [1 / 2, 1 / 2, 1 / 2, -1 / 2]).to_matrix()

        def _circuit_diagram_info_(self, args):
            return [f"C({self.pauli_0}, {self.pauli_1})"] * self.num_qubits()

    class CirqCanonical(cirq.Gate):
        r"""
        Canonical gate with respect to Weyl chamber

        .. math::
            \mathrm{Can}(x, y, z) = e^{- i \frac{\pi}{2}(x XX + y YY + z ZZ)}
        """

        def __init__(self, x, y, z):
            super(CirqCanonical, self)
            self.x, self.y, self.z = x, y, z

        def _num_qubits_(self):
            return 2

        def _unitary_(self):
            return linalg.expm(-1j * np.pi / 2 * (self.x * np.kron(cirq.unitary(cirq.X), cirq.unitary(cirq.X)) +
                                                  self.y * np.kron(cirq.unitary(cirq.Y), cirq.unitary(cirq.Y)) +
                                                  self.z * np.kron(cirq.unitary(cirq.Z), cirq.unitary(cirq.Z))))

        def _circuit_diagram_info_(self, args):
            x, y, z = replace_close_to_zero_with_zero(np.round([self.x, self.y, self.z], 3))
            return [f"Can({x}, {y}, {z})"] * self.num_qubits()

    c = cirq.Circuit()
    qubits = cirq.LineQubit.range(circ.num_qubits_with_dummy)
    for g in circ:
        if isinstance(g, (gates.XGate, gates.YGate, gates.ZGate,
                          gates.HGate, gates.SWAPGate)):
            acted = [qubits[cq] for cq in g.cqs] + [qubits[tq] for tq in g.tqs]
            c.append(getattr(cirq, g.name).controlled(len(g.cqs)).on(*acted))
        elif isinstance(g, gates.VGate):
            c.append(cirq.XPowGate(exponent=0.5).on(qubits[g.tq]))
        elif isinstance(g, gates.TGate):
            c.append(cirq.T.on(qubits[g.tq]))
        elif isinstance(g, gates.TDGGate):
            c.append((cirq.T ** -1).on(qubits[g.tq]))
        elif isinstance(g, gates.SGate):
            c.append(cirq.S.on(qubits[g.tq]))
        elif isinstance(g, gates.SDGGate):
            c.append((cirq.S ** -1).on(qubits[g.tq]))
        elif isinstance(g, gates.RX) and not np.allclose(g.angle, 0):
            acted = [qubits[cq] for cq in g.cqs] + [qubits[g.tq]]
            c.append(cirq.Rx(rads=g.angle).controlled(len(g.cqs)).on(*acted))
        elif isinstance(g, gates.RY) and not np.allclose(g.angle, 0):
            acted = [qubits[cq] for cq in g.cqs] + [qubits[g.tq]]
            c.append(cirq.Ry(rads=g.angle).controlled(len(g.cqs)).on(*acted))
        elif isinstance(g, gates.RZ) and not np.allclose(g.angle, 0):
            acted = [qubits[cq] for cq in g.cqs] + [qubits[g.tq]]
            c.append(cirq.Rz(rads=g.angle).controlled(len(g.cqs)).on(*acted))
        elif isinstance(g, (gates.U1, gates.PhaseShift)) and not np.allclose(g.angle, 0):
            c.append(cirq.ZPowGate(exponent=g.angle / pi).on(qubits[g.tq]))
        elif isinstance(g, gates.U2):
            if not np.allclose(g.angles[1], 0):
                c.append(cirq.ZPowGate(exponent=g.angles[1] / pi, global_shift=-0.5).on(qubits[g.tq]))
            c.append(cirq.YPowGate(exponent=0.5, global_shift=-0.5).on(qubits[g.tq]))
            if not np.allclose(g.angles[0], 0):
                c.append(cirq.ZPowGate(exponent=g.angles[0] / pi, global_shift=-0.5).on(qubits[g.tq]))
        elif isinstance(g, gates.U3):
            if not np.allclose(g.angles[2], 0):
                c.append(cirq.ZPowGate(exponent=g.angles[2] / pi, global_shift=-0.5).on(qubits[g.tq]))
            if not np.allclose(g.angles[0], 0):
                c.append(cirq.YPowGate(exponent=g.angles[0] / pi, global_shift=-0.5).on(qubits[g.tq]))
            if not np.allclose(g.angles[1], 0):
                c.append(cirq.ZPowGate(exponent=g.angles[1] / pi, global_shift=-0.5).on(qubits[g.tq]))
        elif isinstance(g, gates.RXX) and not np.allclose(g.angle, 0):
            c.append(cirq.XXPowGate(exponent=g.angle / pi, global_shift=-0.5).on(qubits[g.tqs[0]], qubits[g.tqs[1]]))
        elif isinstance(g, gates.RYY) and not np.allclose(g.angle, 0):
            c.append(cirq.YYPowGate(exponent=g.angle / pi, global_shift=-0.5).on(qubits[g.tqs[0]], qubits[g.tqs[1]]))
        elif isinstance(g, gates.RZZ) and not np.allclose(g.angle, 0):
            c.append(cirq.ZZPowGate(exponent=g.angle / pi, global_shift=-0.5).on(qubits[g.tqs[0]], qubits[g.tqs[1]]))
        elif isinstance(g, gates.ISWAPGate):
            c.append(cirq.ISWAP.on(qubits[g.tqs[0]], qubits[g.tqs[1]]))
        elif isinstance(g, gates.SQiSWGate):
            c.append(cirq.SQRT_ISWAP.on(qubits[g.tqs[0]], qubits[g.tqs[1]]))
        elif isinstance(g, gates.Clifford2QGate):
            c.append(CirqClifford2QGate(g.pauli_0, g.pauli_1).on(qubits[g.tqs[0]], qubits[g.tqs[1]]))
        elif isinstance(g, gates.Canonical):
            c.append(CirqCanonical(
                g.angles[0] / pi, g.angles[1] / pi, g.angles[2] / pi).on(qubits[g.tqs[0]], qubits[g.tqs[1]]))
        else:
            raise ValueError(f'Unsupported gate {g}')

    return c


def _from_qiskit(qc: qiskit.QuantumCircuit) -> Circuit:
    """Convert from qiskit.QuantumCircuit instance to Circuit instance"""
    # herein we do not use "qasm-based" conversion, to avoid precision loss
    circ = Circuit()
    for instr in qc.data:
        opr = instr.operation
        qubits = [q._index for q in list(instr.qubits)]
        params = opr.params
        if opr.name == 'id':
            circ.append(gates.I.on(qubits))
        elif opr.name == 'x':
            circ.append(gates.X.on(qubits))
        elif opr.name == 'y':
            circ.append(gates.Y.on(qubits))
        elif opr.name == 'z':
            circ.append(gates.Z.on(qubits))
        elif opr.name == 'sx':
            circ.append(gates.V.on(qubits))
        elif opr.name == 'h':
            circ.append(gates.H.on(qubits))
        elif opr.name == 't':
            circ.append(gates.T.on(qubits))
        elif opr.name == 'tdg':
            circ.append(gates.TDG.on(qubits))
        elif opr.name == 's':
            circ.append(gates.S.on(qubits))
        elif opr.name == 'sdg':
            circ.append(gates.SDG.on(qubits))
        elif opr.name == 'rx':
            circ.append(gates.RX(*params).on(qubits))
        elif opr.name == 'ry':
            circ.append(gates.RY(*params).on(qubits))
        elif opr.name == 'rz':
            circ.append(gates.RZ(*params).on(qubits))
        elif opr.name == 'u1':
            circ.append(gates.U1(*params).on(qubits))
        elif opr.name == 'u2':
            circ.append(gates.U2(*params).on(qubits))
        elif opr.name == 'u3' or opr.name == 'u':
            circ.append(gates.U3(*params).on(qubits))
        elif opr.name == 'p':
            circ.append(gates.P(*params).on(qubits))
        elif opr.name == 'rxx':
            circ.append(gates.RXX(*params).on(qubits))
        elif opr.name == 'ryy':
            circ.append(gates.RYY(*params).on(qubits))
        elif opr.name == 'rzz':
            circ.append(gates.RZZ(*params).on(qubits))
        elif opr.name == 'swap':
            circ.append(gates.SWAP.on(qubits))
        elif opr.name == 'iswap':
            circ.append(gates.ISWAP.on(qubits))
        elif opr.name == 'can':
            circ.append(gates.Can(*params).on(qubits))
        elif opr.name == 'cx':
            circ.append(gates.X.on(qubits[1], qubits[0]))
        elif opr.name == 'cy':
            circ.append(gates.Y.on(qubits[1], qubits[0]))
        elif opr.name == 'cz':
            circ.append(gates.Z.on(qubits[1], qubits[0]))
        elif opr.name == 'ch':
            circ.append(gates.H.on(qubits[1], qubits[0]))
        elif opr.name == 'cs':
            circ.append(gates.S.on(qubits[1], qubits[0]))
        elif opr.name == 'cp':
            circ.append(gates.P(*params).on(qubits[1], qubits[0]))
        elif opr.name == 'cu3' or opr.name == 'cu':
            circ.append(gates.U3(*params).on(qubits[1], qubits[0]))
        elif opr.name == 'cswap':
            circ.append(gates.SWAP.on(qubits[1:], qubits[0]))
        elif opr.name == 'crx':
            circ.append(gates.RX(*params).on(qubits[1], qubits[0]))
        elif opr.name == 'cry':
            circ.append(gates.RY(*params).on(qubits[1], qubits[0]))
        elif opr.name == 'crz':
            circ.append(gates.RZ(*params).on(qubits[1], qubits[0]))
        elif opr.name == 'ccx':
            circ.append(gates.X.on(qubits[2], qubits[:2]))
        elif opr.name == 'ccz':
            circ.append(gates.Z.on(qubits[2], qubits[:2]))
        elif match := re.match(r"c(x|y|z)(x|y|z)", opr.name):
            circ.append(gates.Clifford2QGate(match.groups()[0].upper(), match.groups()[1].upper()).on(qubits))
        else:
            raise ValueError(f'Unsupported gate {opr.name}')

    return circ
