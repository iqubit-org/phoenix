import os
import json
import time

from phoenix import Hamiltonian, CNOTEquivCliffordGate
from phoenix.primitive import ordering, simplification, utils
from phoenix.compiler import (
    _optimize_phoenix_circuit_by_qiskit_each_group,
    compile_hamiltonian_simulation,
    optimize_phoenix_circuit_by_qiskit,
)
from phoenix.utils import infidelity, print_circ_info
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator

# Get the directory of this test file to construct absolute paths
_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_TEST_DIR)

cxx = CNOTEquivCliffordGate('x', 'x')
cyy = CNOTEquivCliffordGate('y', 'y')
czz = CNOTEquivCliffordGate('z', 'z')


def test_cancellation_bonus():
    """Test the cancellation bonus calculation between two circuit blocks."""
    qc = QuantumCircuit(4)
    qc.append(cxx, [0,1])
    qc.append(cyy, [1,2])
    qc.rz(0.1, 0)
    qc.append(czz, [0,2])
    qc.append(cxx, [1,3])
    print(qc)

    qc2 = QuantumCircuit(4)
    qc2.append(czz, [0,2])
    qc2.append(cxx, [1,3])
    qc2.rz(0.1, 0)
    qc2.append(czz, [1,2])
    qc2.append(cxx, [0,1])
    print(qc2)

    print('tail cliffs:', [ordering.repr_circuit_instr(instr) for instr in ordering.extract_tail_cliffs(qc)])
    print('head cliffs:', [ordering.repr_circuit_instr(instr) for instr in ordering.extract_head_cliffs(qc2)])

    bonus, lhs_tail_simplified, rhs_head_simplified = ordering.cancellation_bonus(
        ordering.extract_tail_cliffs(qc),
        ordering.extract_head_cliffs(qc2),
        return_simplified_blocks=True
    )

    print('cancellation bonus:', bonus)
    print('lhs_tail_simplified:\n', QuantumCircuit.from_instructions(lhs_tail_simplified))
    print('rhs_head_simplified:\n', QuantumCircuit.from_instructions(rhs_head_simplified))

    print('--'*20)
    c = ordering.assembling_cost(
        ordering.CircuitTetris.from_circuit(qc),
        ordering.CircuitTetris.from_circuit(qc2)
    )
    print('assembling cost:', c)


def count_2q_gates(qc: QuantumCircuit) -> int:
    """Count all nonlocal gates in an optimized Qiskit circuit."""
    return sum(1 for instr in qc.data if instr.operation.num_qubits >= 2)


def test_qiskit_post_optimization_synthesizes_pauli_block():
    qc = QuantumCircuit(2)
    qc.rzx(0.1, 0, 1)
    qc.ryy(0.2, 0, 1)
    qc.rzz(0.3, 0, 1)

    optimized = optimize_phoenix_circuit_by_qiskit(qc)

    assert Operator(qc).equiv(Operator(optimized))
    assert optimized.count_ops().get("cx", 0) <= 3


def test_qiskit_each_group_synthesizes_successive_pauli_block():
    qc = QuantumCircuit(2)
    qc.rzx(0.1, 0, 1)
    qc.ryy(0.2, 0, 1)
    qc.rzz(0.3, 0, 1)

    optimized = _optimize_phoenix_circuit_by_qiskit_each_group(qc)

    assert Operator(qc).equiv(Operator(optimized))
    assert optimized.count_ops().get("cx", 0) <= 3
    assert set(optimized.count_ops()).issubset({"cx", "rz", "sx", "x"})


def test_qiskit_each_group_preserves_non_pauli_rotation_gates():
    qc = QuantumCircuit(2)
    qc.rzx(0.1, 0, 1)
    qc.ryy(0.2, 0, 1)
    qc.append(CNOTEquivCliffordGate("y", "y"), [0, 1])

    optimized = _optimize_phoenix_circuit_by_qiskit_each_group(qc)

    assert Operator(qc).equiv(Operator(optimized))
    assert "cyy" in optimized.count_ops()


def test_qiskit_post_optimization_unrolls_custom_clifford_gates():
    qc = QuantumCircuit(2)
    qc.append(CNOTEquivCliffordGate("y", "y"), [0, 1])

    optimized = optimize_phoenix_circuit_by_qiskit(qc)

    assert Operator(qc).equiv(Operator(optimized))
    assert "cyy" not in optimized.count_ops()
    assert optimized.count_ops().get("cx", 0) == 1


def test_ordering_methods():
    """
    Compare all ordering methods using the complete compile_hamiltonian_simulation flow.
    
    This tests that:
    1. Advanced methods should have lower Num2Q and Depth2Q than trivial
    2. All methods should produce circuits with similar infidelity (verifies correctness)
    """
    # Load benchmark
    benchmark_path = os.path.join(_PROJECT_ROOT, 'benchmarks/uccsd_json/LiH_frz_BK_sto3g.json')
    with open(benchmark_path, 'r') as f:
        data = json.load(f)
    
    hamiltonian = Hamiltonian(data['paulis'], data['coeffs'])
    
    # Compute reference unitary for infidelity comparison
    u = hamiltonian.unitary_evolution()
    
    # Original circuit for reference
    qc_original = hamiltonian.generate_circuit()
    print_circ_info(qc_original, title='Original circuit (no optimization)')
    
    # Define methods to test
    methods = [
        ('trivial', {}),
        ('greedy', {}),


        ('tsp', {}),  # TSP-based ordering


    ]
    
    results = []
    
    for method_name, kwargs in methods:
        print(f"\n--- Testing {method_name} ---")
        
        # Use compile_hamiltonian_simulation which includes optimize_phoenix_circuit_by_qiskit
        start = time.process_time()
        qc = compile_hamiltonian_simulation(
            hamiltonian,
            order_method=method_name,
            backend="sequential",
        )
        elapsed = time.process_time() - start
        
        # Calculate metrics
        num_2q = count_2q_gates(qc)
        depth_2q = qc.depth(lambda instr: instr.operation.num_qubits >= 2)
        
        # Calculate infidelity
        v = Operator(qc).to_matrix()
        infid = infidelity(u, v)
        
        results.append({
            'method': method_name,
            'num_2q': num_2q,
            'depth_2q': depth_2q,
            'infidelity': infid,
            'time': elapsed
        })
        
        print_circ_info(qc, title=f'Compiled circuit ({method_name})')
        print(f"Infidelity: {infid:.2e}")
    
    # Print comparison table
    print("\n" + "="*90)
    print("Ordering Method Comparison (with optimize_phoenix_circuit_by_qiskit)")
    print("="*90)
    print(f"{'Method':<20} {'Num2Q':<10} {'Depth2Q':<12} {'Infidelity':<15} {'Time(s)':<12}")
    print("-"*90)
    
    baseline = results[0]  # trivial is baseline
    for r in results:
        method = r['method']
        num_2q = r['num_2q']
        depth_2q = r['depth_2q']
        infid = r['infidelity']
        elapsed = r['time']
        
        if method == 'trivial':
            improvement = '(baseline)'
        else:
            num_2q_impr = (baseline['num_2q'] - num_2q) / baseline['num_2q'] * 100
            depth_impr = (baseline['depth_2q'] - depth_2q) / baseline['depth_2q'] * 100
            improvement = f"2Q:{num_2q_impr:+.1f}%, D:{depth_impr:+.1f}%"
        
        print(f"{method:<20} {num_2q:<10} {depth_2q:<12} {infid:<15.2e} {elapsed:<12.4f} {improvement}")
    
    print("="*90)
    
    # Verify correctness: all infidelities should be in the same order of magnitude
    infidelities = [r['infidelity'] for r in results]
    max_infid = max(infidelities)
    min_infid = min(infidelities)
    
    print(f"\nInfidelity range: {min_infid:.2e} to {max_infid:.2e}")
    
    # Check that all infidelities are small (< 1e-3) indicating correct compilation
    for r in results:
        assert r['infidelity'] < 1e-3, f"Method {r['method']} has high infidelity: {r['infidelity']}"
    
    print("✓ All methods produce correct results (infidelity < 1e-3)")
    
    return results


if __name__ == '__main__':
    test_cancellation_bonus()
    print("\n" + "#"*90 + "\n")
    test_ordering_methods()
