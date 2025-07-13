from rich.console import Console
from rich.table import Table
from typing import Union
import pytket
import qiskit
from phoenix.basic.circuits import Circuit

console = Console()


def print_circ_info(circ: Union[Circuit, pytket.Circuit, qiskit.QuantumCircuit], title=None):
    """Get information of a quantum circuit from its qasm file."""

    if isinstance(circ, Circuit):
        num_qubits = circ.num_qubits
        num_gates = circ.num_gates
        num_nonlocal_gates = circ.num_nonlocal_gates
        depth = circ.depth
        depth_nonlocal = circ.depth_nonlocal
    elif isinstance(circ, pytket.Circuit):
        num_qubits = circ.n_qubits
        num_gates = circ.n_gates
        num_nonlocal_gates = circ.n_2qb_gates()
        depth = circ.depth()
        depth_nonlocal = circ.depth_2q()
    elif isinstance(circ, qiskit.QuantumCircuit):
        num_qubits = circ.num_qubits
        num_gates = circ.size()
        num_nonlocal_gates = circ.num_nonlocal_gates()
        depth = circ.depth()
        depth_nonlocal = circ.depth(lambda instr: instr.operation.num_qubits > 1)
    else:
        raise ValueError('Unsupported circuit type {}'.format(type(circ)))

    # use rich Table
    table = Table(title=title)
    table.add_column("num_qubits")
    table.add_column("num_gates")
    table.add_column("num_2q_gates")
    table.add_column("depth")
    table.add_column("depth_2q")
    table.add_row(str(num_qubits),
                  str(num_gates), str(num_nonlocal_gates),
                  str(depth), str(depth_nonlocal))
    console.print(table)
