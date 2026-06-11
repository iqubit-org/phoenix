import phoenix
import json
import numpy as np
import time


with open('./benchmarks/hamlib_json/condensedmatter/FH_D-2-fh-graph-2D-grid-pbc-qubitnodes_Lx-5_Ly-72_U-0_enc-parity.json', 'r') as f:
    data = json.load(f)

ham = phoenix.Hamiltonian(data['paulis'], data['coeffs'])
phoenix.utils.print_circ_info(ham.generate_circuit())
qc = phoenix.compile_hamiltonian_simulation(ham)
phoenix.utils.print_circ_info(qc)
