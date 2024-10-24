import cirq
from phoenix import Circuit
from phoenix.utils import passes
from phoenix.utils import arch


def test_contract_1q_gate_on_dag():
    qasm_fname = '../input/cx-basis/alu/alu-v2_33.qasm'
    circ = Circuit.from_qasm(fname=qasm_fname)
    dag = circ.to_dag()
    dag = passes.contract_1q_gates_on_dag(dag)

    cirq.testing.assert_allclose_up_to_global_phase(
        passes.dag_to_circuit(dag).unitary(),
        passes.dag_to_circuit(dag).unitary(),
        atol=1e-6
    )


def test_peel_first_and_last_1q_gates():
    for _ in range(10):
        circ = arch.gene_random_circuit(8, 10)
        circ1, first_1q, last_1q = passes.peel_first_and_last_1q_gates(circ)
        circ1 = Circuit(first_1q) + circ1 + Circuit(last_1q)
        cirq.testing.assert_allclose_up_to_global_phase(
            circ.unitary(),
            circ1.unitary(),
            atol=1e-6
        )
