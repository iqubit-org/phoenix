import phoenix
import json
import numpy as np
import time
import sys
import time


# json_fname = './benchmarks/hamlib/chemistry/N2-JW-22.json'
json_fname = './benchmarks/hamlib/chemistry/Na2-JW24.json'
# json_fname = './benchmarks/hamlib/condensedmatter/FH_D-2-fh-graph-2D-grid-pbc-qubitnodes_Lx-5_Ly-72_U-0_enc-parity.json'
# json_fname = './benchmarks/hamlib/discreteoptimization/gray-color02-1-fullins_5_k-5-1-fullins_5.json'


with open(json_fname, 'r') as f:
    data = json.load(f)

ham = phoenix.Hamiltonian(data['paulis'], data['coeffs'])
phoenix.utils.print_circ_info(ham.generate_circuit())
start_time = time.perf_counter()
if len(sys.argv) > 1 and sys.argv[1] == 'qiskit':
    print("Using Qiskit")
    qc = phoenix.utils.compile_by_qiskit(data['paulis'], data['coeffs'])
else:
    print("Using Phoenix")
    qc = phoenix.compile_hamiltonian_simulation(ham, grouping='holistic')
end_time = time.perf_counter()
print(f"Total time: {end_time - start_time:.2f} seconds")
phoenix.utils.print_circ_info(qc)
