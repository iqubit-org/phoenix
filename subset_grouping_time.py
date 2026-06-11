import phoenix
import json
import numpy as np
import time


with open('./benchmarks/hamlib_json/condensedmatter/FH_D-2-fh-graph-2D-grid-pbc-qubitnodes_Lx-5_Ly-72_U-0_enc-parity.json', 'r') as f:
    data = json.load(f)

print(data['num_terms'])

for m in range(50, data['num_terms'], 50):
    ham = phoenix.Hamiltonian(data['paulis'][:m], data['coeffs'][:m])
    start = time.perf_counter()
    ham.group_same_weights(subset=True)
    end = time.perf_counter()
    print(f"Time for {m} terms: {end - start:.2f} seconds")
