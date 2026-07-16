import sys
import json
from pathlib import Path

# Allow importing the vendored `pauliopt/` at the repo root.
_DIR = Path(__file__).resolve().parent
sys.path.append(str(_DIR.parent))

import qiskit
import phoenix
import numpy as np
from qiskit.quantum_info import Operator
from rich.console import Console

from phoenix.primitive.grouping import group_paulis_and_coeffs

console = Console()


def main():
    # with open('../benchmarks/uccsd/NH_frz_JW_sto3g.json', 'r') as f:
    with open(_DIR / "hams/BeH2_as_4e_4o_JW_sto3g.json", "r") as f:
        data = json.load(f)


    ham = phoenix.Hamiltonian(data['paulis'], data['coeffs'])
    # ham = phoenix.Hamiltonian(['XXXZIYZI', 'YXXZIYYI', 'ZXXZIYZI'], [-0.0125, -0.0125, -0.0125])
    # print(ham.paulis.to_labels())
    # print(ham.coeffs)

    u = ham.unitary_evolution()

    qc_trivial = ham.generate_circuit()
    console.rule("Trivial synthesis")
    phoenix.utils.print_circ_info(qc_trivial, title="Trivial synthesized circuit")
    print("Infidelity", phoenix.utils.infidelity(u, Operator(qc_trivial).to_matrix()))

    console.rule("Phoenix synthesis")
    # qc_phoenix = phoenix.compile_hamiltonian_simulation(ham, method='smt', smt_min_depth=smt_min_depth, smt_max_depth=50)
    qc_phoenix = phoenix.compile_hamiltonian_simulation(ham)
    # print(qc_phoenix)
    phoenix.utils.print_circ_info(qc_phoenix, title="Phoenix synthesized circuit")
    print("Infidelity", phoenix.utils.infidelity(u, Operator(qc_phoenix).to_matrix()))

    console.rule("Qiskit synthesis")
    qc_qiskit = phoenix.utils.compile_by_qiskit(ham.paulis.to_labels(), ham.coeffs)
    phoenix.utils.print_circ_info(qc_qiskit, title="Qiskit synthesized circuit")
    print("Infidelity", phoenix.utils.infidelity(u, Operator(qc_qiskit).to_matrix()))

    console.rule("TKet synthesis")
    qc_tket = phoenix.utils.compile_by_tket(ham.paulis.to_labels(), ham.coeffs, little_endian=True)
    phoenix.utils.print_circ_info(qc_tket, title="TKet synthesized circuit")
    print("Infidelity", phoenix.utils.infidelity(u, Operator(qc_tket).to_matrix()))


if __name__ == "__main__":
    main()
