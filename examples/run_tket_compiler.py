import qiskit
import phoenix
import json
from qiskit.quantum_info import Operator

from rich.console import Console

console = Console()


def main():
    with open('./hams/BeH2_as_4e_4o_JW_sto3g.json', 'r') as f:
        data = json.load(f)

    M =30
    data['paulis'] = data['paulis'][:M]
    data['coeffs'] = data['coeffs'][:M]
    ham = phoenix.Hamiltonian(data['paulis'], data['coeffs'])
    u = ham.unitary_evolution()

    qc_trivial = ham.generate_circuit()
    console.rule("Trivial synthesis")
    phoenix.utils.print_circ_info(qc_trivial, title='Trivial synthesized circuit')
    print('Infidelity', phoenix.utils.infidelity(u, Operator(qc_trivial).to_matrix()))

    console.rule("Qiskit synthesis")
    qc_qiskit = phoenix.utils.qiskit_pass(ham.paulis.to_labels(), ham.coeffs)
    # print(phoenix.utils.remove_1q_fixed_gates(qc_qiskit))
    phoenix.utils.print_circ_info(qc_qiskit, title='Qiskit synthesized circuit')
    print('Infidelity', phoenix.utils.infidelity(u, Operator(qc_qiskit).to_matrix()))


    console.rule("TKet synthesis (greedy=True)")
    qc_tket = phoenix.utils.tket_pass(ham.paulis.to_labels(), ham.coeffs, little_endian=True, greedy=True)
    phoenix.utils.print_circ_info(qc_tket, title='TKet synthesized circuit')
    print('Infidelity', phoenix.utils.infidelity(u, Operator(qc_tket).to_matrix()))
    print(qc_tket)

    console.rule("TKet synthesis (greedy=False)")
    qc_tket = phoenix.utils.tket_pass(ham.paulis.to_labels(), ham.coeffs, little_endian=True, greedy=False)
    phoenix.utils.print_circ_info(qc_tket, title='TKet synthesized circuit')
    print('Infidelity', phoenix.utils.infidelity(u, Operator(qc_tket).to_matrix()))


if __name__ == '__main__':
    main()
