from qiskit import QuantumCircuit
import pennylane
import subprocess
from collections import Counter
import numpy as np
from qiskit.synthesis import gridsynth_unitary, gridsynth_rz

def decompose_rz(angle, epsilon=1e-10) -> QuantumCircuit:
    """Decomposes an RZ rotation into the Clifford+T basis using the Ross-Selinger decomposition (GridSynth).

    Args:
        angle (float): The rotation angle for the RZ gate.
        epsilon (float): The maximum permissible operator norm error per rotation gate. Defaults to ``1e-4``.

    Returns:
        QuantumCircuit: A Qiskit QuantumCircuit representing the decomposed RZ rotation in the Clifford+T basis.
    """
    op = pennylane.RZ(angle, wires=0)
    operations = pennylane.ops.decompositions.rs_decomposition(op, epsilon=epsilon)
    operations = [op for op in operations if not isinstance(op, pennylane.GlobalPhase)] 
    tape = pennylane.tape.QuantumScript(operations)
    qc = QuantumCircuit.from_qasm_str(pennylane.to_openqasm(tape))
    qc.remove_final_measurements(inplace=True)
    return qc


def gridsynth_call(theta, epsilon=1e-10):
    path = "/Users/anan/.local/bin/gridsynth"
    cmd = [path, str(theta), "--epsilon", str(epsilon)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.stdout.strip()

res = gridsynth_call("pi/128")
print(res)
print(Counter(res))


qc = decompose_rz(np.pi/128, epsilon=1e-10)
print(qc.count_ops())


qc = gridsynth_rz(np.pi/128, epsilon=1e-10)
print(qc.count_ops())
