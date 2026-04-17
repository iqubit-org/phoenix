import qiskit
import phoenix
import json
from pathlib import Path
from qiskit.quantum_info import Operator

from rich.console import Console
from itertools import chain

_DIR = Path(__file__).resolve().parent

console = Console()


from phoenix.primitive.grouping import group_paulis_and_coeffs


def main():
    # ham = phoenix.Hamiltonian(['XXXZIYZI', 'YXXZIYYI', 'ZXXZIYZI'], [-0.0125, -0.0125, -0.0125])
    # u = ham.unitary_evolution()

    # with open('../benchmarks/uccsd_json/NH_frz_JW_sto3g.json', 'r') as f:
    with open(_DIR / "hams/BeH2_as_4e_4o_JW_sto3g.json", "r") as f:
        data = json.load(f)

    grouped = group_paulis_and_coeffs(data["paulis"], data["coeffs"])
    # grouped_paulis = list(chain.from_iterable(p for p, _ in grouped.values()))
    # grouped_coeffs = list(chain.from_iterable(c for _, c in grouped.values()))
    # data['paulis'] = grouped_paulis
    # data['coeffs'] = grouped_coeffs

    M = 3

    paulis = []
    coeffs = []
    for i, group in enumerate(grouped.values()):
        paulis.extend(group[0][:M])
        coeffs.extend(group[1][:M])
        if i >= 3:
            break

    paulis = data["paulis"]
    coeffs = data["coeffs"]

    # ham = phoenix.Hamiltonian(data['paulis'], data['coeffs'])
    ham = phoenix.Hamiltonian(paulis, coeffs)
    # ham = phoenix.Hamiltonian(['XXXZIYZI', 'YXXZIYYI', 'ZXXZIYZI'], [-0.0125, -0.0125, -0.0125])

    # print(ham.paulis.to_labels())
    # print(ham.coeffs)

    u = ham.unitary_evolution()

    qc_trivial = ham.generate_circuit()
    console.rule("Trivial synthesis")
    phoenix.utils.print_circ_info(qc_trivial, title="Trivial synthesized circuit")
    print("Infidelity", phoenix.utils.infidelity(u, Operator(qc_trivial).to_matrix()))

    smt_min_depth = 1
    # smt_min_depth = max(1, phoenix.utils.tket_pass(ham.paulis.to_labels(), ham.coeffs, little_endian=True).count_ops().get('cx', 0) // 2 - ham.num_nonlocal_paulis * 2)

    console.rule("Phoenix synthesis")
    # qc_phoenix = phoenix.compile_hamiltonian_simulation(ham, method='smt', smt_min_depth=smt_min_depth, smt_max_depth=50)
    qc_phoenix = phoenix.compile_hamiltonian_simulation(ham)
    # print(phoenix.utils.remove_1q_fixed_gates(qc_phoenix))
    # print(qc_phoenix)
    phoenix.utils.print_circ_info(qc_phoenix, title="Phoenix synthesized circuit")
    print("Infidelity", phoenix.utils.infidelity(u, Operator(qc_phoenix).to_matrix()))

    console.rule("Qiskit synthesis")
    qc_qiskit = phoenix.utils.qiskit_pass(ham.paulis.to_labels(), ham.coeffs)
    # print(phoenix.utils.remove_1q_fixed_gates(qc_qiskit))
    phoenix.utils.print_circ_info(qc_qiskit, title="Qiskit synthesized circuit")
    print("Infidelity", phoenix.utils.infidelity(u, Operator(qc_qiskit).to_matrix()))

    console.rule("TKet synthesis")
    qc_tket = phoenix.utils.tket_pass(ham.paulis.to_labels(), ham.coeffs, little_endian=True)
    phoenix.utils.print_circ_info(qc_tket, title="TKet synthesized circuit")
    print("Infidelity", phoenix.utils.infidelity(u, Operator(qc_tket).to_matrix()))


if __name__ == "__main__":
    main()
