"""usage demo for qiskit_ibm_transpiler Pauli networkx synthesis via RL, https://quantum.cloud.ibm.com/docs/en/guides/ai-transpiler-passes"""

import os
from qiskit.circuit.library import QAOAAnsatz
from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler import PassManager
from qiskit.circuit.library import EfficientSU2  
from qiskit_ibm_transpiler.ai.routing import AIRouting
from qiskit_ibm_transpiler.ai.collection import CollectPauliNetworks
from qiskit_ibm_transpiler.ai.synthesis import AIPauliNetworkSynthesis
from qiskit_ibm_runtime import QiskitRuntimeService

TOKEN = 'arS6k_KG_M4QxzK6oDnAajNNBrneH8XTVJZXiMYSv4lk' # ! My API

ibm_torino = QiskitRuntimeService(
    channel="ibm_cloud",
    token = TOKEN
).backend("ibm_torino")

print(type(ibm_torino))


def demo_pauli_network_synthesis():
    print("Generating QAOA circuit...")
    # Create a sample Pauli network (QAOA circuit)
    # A simple Hamiltonian for demonstration: H = Z0Z1 + Z1Z2 + Z2Z0
    hamiltonian = SparsePauliOp.from_list([("ZZI", 1), ("IZZ", 1), ("ZIZ", 1)])
    ansatz = QAOAAnsatz(cost_operator=hamiltonian, reps=2)
    circuit = ansatz.decompose()
    
    print(f"Original circuit depth: {circuit.depth()}")
    print(f"Original circuit gate count: {circuit.count_ops()}")

    # Initialize the AI-driven Pauli network synthesis pass
    # This pass uses Reinforcement Learning to synthesize Pauli networks
    # It requires specifying a backend to target the connectivity/constraints
    print("\nInitializing AIPauliNetworkSynthesis pass...")
    try:
        # backend_name should be a real device name or one supported by the service
        ai_pass = AIPauliNetworkSynthesis(
            backend=ibm_torino,
            replace_only_if_better=True,
            max_threads=10,
        )
        
        # Create a PassManager with the AI pass
        pm = PassManager(ai_pass)
        
        # Run the transpilation
        print("Running transpilation (this may take a moment)...")
        transpiled_circuit = pm.run(circuit)
        
        print("\nTranspilation successful!")
        print(f"Transpiled circuit depth: {transpiled_circuit.depth()}")
        print(f"Transpiled circuit gate count: {transpiled_circuit.count_ops()}")
        
    except Exception as e:
        print(f"\nError running AI synthesis: {e}")
        print("Ensure you have 'qiskit-ibm-transpiler' installed and a valid IBM Quantum token configured.")

if __name__ == "__main__":
    # Create a quantum circuit with Pauli network structure  
    circuit = EfficientSU2(10, entanglement="full", reps=1).decompose()  
    
    # Build the AI transpilation pipeline  
    ai_passmanager = PassManager([  
        # First, route the circuit to the target backend topology  
        AIRouting(backend=ibm_torino, optimization_level=3, layout_mode="optimize"),  
        
        # Collect Pauli Network blocks (H, S, SX, CX, RX, RY, RZ gates)  
        # Supports up to 6-qubit blocks  
        CollectPauliNetworks(  
            do_commutative_analysis=True,  
            min_block_size=4,  
            max_block_size=6,  
            num_reps=10  
        ),  
        
        # Apply RL-based synthesis to optimize the collected blocks  
        # Uses reinforcement learning models trained to minimize gate count and depth  
        AIPauliNetworkSynthesis(  
            backend=ibm_torino,  # Target backend for connectivity constraints  
            replace_only_if_better=True,  # Only replace if RL synthesis improves the circuit  
            max_threads=10,  # Parallel synthesis requests  
        )  
    ])  
    
    # Run the transpilation  
    transpiled_circuit = ai_passmanager.run(circuit)  
    
    print(f"Original circuit - Gates: {circuit.num_nonlocal_gates()}, Depth: {circuit.depth()}")  
    print(f"Optimized circuit - Gates: {transpiled_circuit.num_nonlocal_gates()}, Depth: {transpiled_circuit.depth()}")



