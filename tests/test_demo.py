import os

# Get the directory of this test file to construct absolute paths
_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_TEST_DIR)

import json
import qiskit
from phoenix import Hamiltonian
from phoenix.primitive import simplification, ordering
from phoenix.utils import infidelity, print_circ_info
from qiskit.quantum_info import Operator
from phoenix.primitive.utils import constr_circuit_from_simp_steps
from phoenix.compiler import compile_hamiltonian_simulation
import time


# DEMO_FILE = 'CH2_frz_JW_sto3g.json'
DEMO_FILE = 'LiH_frz_JW_sto3g.json'

def test_simp():
    ham = Hamiltonian(['XXXZIYZI', 'YXXZIYYI', 'ZXXZIYZI'], [-0.0125, -0.0125, -0.0125])
    ham.print_tableau()

    ham_, simp_steps = simplification.simplify_hamiltonian(ham)

    print(ham_)

    print(simp_steps)

    print(ham_.paulis.to_labels())
    print('number of simplification steps:', len(simp_steps))

    qc = constr_circuit_from_simp_steps(ham_, simp_steps)

    print('After simplification')
    print(qc)

    u = ham.unitary_evolution()
    v = Operator(qc).to_matrix()

    print('Infidelity:', infidelity(u, v))
    assert infidelity(u, v) < 1e-6


def test_simp_with_trivial_order(with_O3=False):
    with open(os.path.join(_PROJECT_ROOT, f'benchmarks/uccsd_json/{DEMO_FILE}'), 'r') as f:
        data = json.load(f)

    hamiltonian = Hamiltonian(data['paulis'], data['coeffs'])
    # u = hamiltonian.unitary_evolution()
    qc_trivial = hamiltonian.generate_circuit()
    print_circ_info(qc_trivial, title='Original circuit')

    start = time.process_time()
    qc = compile_hamiltonian_simulation(hamiltonian)
    end = time.process_time()

    # with Qiskit O3
    if with_O3:
        qc = qiskit.transpile(qc, basis_gates=['cx', 'u1', 'u2', 'u3'], optimization_level=3)

    print_circ_info(qc, title='Compiled circuit (trivial order)')
    print(qc.count_ops())

    import re
    pattern = re.compile(r'^c[xyz]{2}$')
    total = sum(
        v for k, v in qc.count_ops().items()
        if pattern.match(k)
    )
    print('Total CNOT-equivalent gates:', total)    

    # v = Operator(qc).to_matrix()
    # print('Infidelity (trivial order):', infidelity(u, v))
    print('Time (trivial order):', end - start)


def test_simp_with_greedy_order(with_O3=False):
    with open(os.path.join(_PROJECT_ROOT, f'benchmarks/uccsd_json/{DEMO_FILE}'), 'r') as f:
        data = json.load(f)

    hamiltonian = Hamiltonian(data['paulis'], data['coeffs'])
    # u = hamiltonian.unitary_evolution()
    qc_trivial = hamiltonian.generate_circuit()
    print_circ_info(qc_trivial, title='Original circuit')

    start = time.process_time()
    qc = compile_hamiltonian_simulation(hamiltonian, order_method='greedy')
    end = time.process_time()
    
    # with Qiskit O3
    if with_O3:
        qc = qiskit.transpile(qc, basis_gates=['cx', 'u1', 'u2', 'u3'], optimization_level=3)

    print_circ_info(qc, title='Compiled circuit (greedy order)')
    print(qc.count_ops())

    import re
    pattern = re.compile(r'^c[xyz]{2}$')
    total = sum(
        v for k, v in qc.count_ops().items()
        if pattern.match(k)
    )
    print('Total CNOT-equivalent gates:', total)

    # v = Operator(qc).to_matrix()
    # print('Infidelity (greedy order):', infidelity(u, v))
    print('Time (greedy order):', end - start)


def test_simp_with_greedy_multistart_order(with_O3=False):
    with open(os.path.join(_PROJECT_ROOT, f'benchmarks/uccsd_json/{DEMO_FILE}'), 'r') as f:
        data = json.load(f)

    hamiltonian = Hamiltonian(data['paulis'], data['coeffs'])
    # u = hamiltonian.unitary_evolution()
    qc_trivial = hamiltonian.generate_circuit()
    print_circ_info(qc_trivial, title='Original circuit')

    start = time.process_time()
    qc = compile_hamiltonian_simulation(hamiltonian, order_method='greedy_multistart')
    end = time.process_time()
    
    # with Qiskit O3
    if with_O3:
        qc = qiskit.transpile(qc, basis_gates=['cx', 'u1', 'u2', 'u3'], optimization_level=3)

    print_circ_info(qc, title='Compiled circuit (greedy multistart order)')
    print(qc.count_ops())

    import re
    pattern = re.compile(r'^c[xyz]{2}$')
    total = sum(
        v for k, v in qc.count_ops().items()
        if pattern.match(k)
    )
    print('Total CNOT-equivalent gates:', total)

    # v = Operator(qc).to_matrix()
    # print('Infidelity (greedy multistart order):', infidelity(u, v))
    print('Time (greedy multistart order):', end - start)


def test_simp_with_tsp_order(with_O3=False):
    with open(os.path.join(_PROJECT_ROOT, f'benchmarks/uccsd_json/{DEMO_FILE}'), 'r') as f:
        data = json.load(f)

    hamiltonian = Hamiltonian(data['paulis'], data['coeffs'])
    # u = hamiltonian.unitary_evolution()
    qc_trivial = hamiltonian.generate_circuit()
    print_circ_info(qc_trivial, title='Original circuit')

    start = time.process_time()
    qc = compile_hamiltonian_simulation(hamiltonian, order_method='tsp')
    end = time.process_time()
    
    # with Qiskit O3
    if with_O3:
        qc = qiskit.transpile(qc, basis_gates=['cx', 'u1', 'u2', 'u3'], optimization_level=3)

    print_circ_info(qc, title='Compiled circuit (tsp order)')
    print(qc.count_ops())

    import re
    pattern = re.compile(r'^c[xyz]{2}$')
    total = sum(
        v for k, v in qc.count_ops().items()
        if pattern.match(k)
    )
    print('Total CNOT-equivalent gates:', total)

    # v = Operator(qc).to_matrix()
    # print('Infidelity (tsp order):', infidelity(u, v))
    print('Time (tsp order):', end - start)


def test_simp_with_tsp_2opt_order(with_O3=False):
    with open(os.path.join(_PROJECT_ROOT, f'benchmarks/uccsd_json/{DEMO_FILE}'), 'r') as f:
        data = json.load(f)

    hamiltonian = Hamiltonian(data['paulis'], data['coeffs'])
    # u = hamiltonian.unitary_evolution()
    qc_trivial = hamiltonian.generate_circuit()
    print_circ_info(qc_trivial, title='Original circuit')

    start = time.process_time()
    qc = compile_hamiltonian_simulation(hamiltonian, order_method='tsp_2opt')
    end = time.process_time()
    
    # with Qiskit O3
    if with_O3:
        qc = qiskit.transpile(qc, basis_gates=['cx', 'u1', 'u2', 'u3'], optimization_level=3)

    print_circ_info(qc, title='Compiled circuit (tsp 2opt order)')
    print(qc.count_ops())
    import re
    pattern = re.compile(r'^c[xyz]{2}$')
    total = sum(
        v for k, v in qc.count_ops().items()
        if pattern.match(k)
    )
    print('Total CNOT-equivalent gates:', total)

    # v = Operator(qc).to_matrix()
    # print('Infidelity (tsp 2opt order):', infidelity(u, v))
    print('Time (tsp 2opt order):', end - start)




if __name__ == '__main__':
    test_simp_with_trivial_order(with_O3=True)
    test_simp_with_greedy_order(with_O3=True)
    test_simp_with_greedy_multistart_order(with_O3=True)
    test_simp_with_tsp_order(with_O3=True)
    test_simp_with_tsp_2opt_order(with_O3=True)
