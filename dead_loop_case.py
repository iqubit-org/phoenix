import phoenix
import json
import numpy as np
import time
import sys
import time


json_fname = './benchmarks/hamlib/chemistry/N2-JW-22.json'
# json_fname = './benchmarks/hamlib/chemistry/Na2-JW24.json'

with open(json_fname, 'r') as f:
    data = json.load(f)

ham = phoenix.Hamiltonian(data['paulis'], data['coeffs'])
phoenix.utils.print_circ_info(ham.generate_circuit())
start_time = time.perf_counter()
qc = phoenix.compile_hamiltonian_simulation(ham)
end_time = time.perf_counter()
print(f"Total time: {end_time - start_time:.2f} seconds")
phoenix.utils.print_circ_info(qc)

