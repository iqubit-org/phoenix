import sys
sys.path.append('..')

import os
import json
import networkx as nx
import numpy as np
from phoenix import models

json_dpath = './qaoa_json'
qasm_dpath = './qaoa_qasm'
text_dpath = './qaoa_text'


# only testing ordering for parameter-shift layer of QAOA

for n in [16, 20, 24]:
    g = nx.connected_watts_strogatz_graph(n, 4, 0.8, seed=123)

    def ZZ_str(n, i, j):
        I_str = 'I' * n
        res = list(I_str)
        res[i] = 'Z'
        res[j] = 'Z'
        return ''.join(res)

    paulis = [ZZ_str(n, i, j) for i, j in g.edges]
    coeffs = [np.random.rand()] * g.number_of_edges()

    ham = models.HamiltonianModel(paulis, coeffs)
    ham.generate_circuit().to_qasm(os.path.join(qasm_dpath, 'qaoa_rand_{}.qasm'.format(n)))    

    with open(os.path.join(json_dpath, 'qaoa_rand_{}.json'.format(n)), 'w') as f:
        json.dump({'num_qubits': n, 'paulis': paulis, 'coeffs': coeffs}, f, indent=4)

    lines = ['{} {}\n'.format(n, g.number_of_edges())]
    for i, j in g.edges:
        lines.append('{} {}\n'.format(i, j))
    with open(os.path.join(text_dpath, 'qaoa_rand_{}.txt'.format(n)), 'w') as f:
        f.writelines(lines)
        # f.write('paulis:\n')
        # for p in paulis:
        #     f.write(p + '\n')
        # f.write('coeffs:\n')
        # for c in coeffs:
        #     f.write(str(c) + '\n')



    print(n, len(paulis))


for n in [16, 20, 24]:
    # g = nx.connected_watts_strogatz_graph(n, 4, 0.8, seed=123)
    g = nx.random_regular_graph(3, n, seed=123)

    def ZZ_str(n, i, j):
        I_str = 'I' * n
        res = list(I_str)
        res[i] = 'Z'
        res[j] = 'Z'
        return ''.join(res)

    paulis = [ZZ_str(n, i, j) for i, j in g.edges]
    coeffs = [np.random.rand()] * g.number_of_edges()

    ham = models.HamiltonianModel(paulis, coeffs)
    ham.generate_circuit().to_qasm(os.path.join(qasm_dpath, 'qaoa_reg3_{}.qasm'.format(n)))    

    with open(os.path.join(json_dpath, 'qaoa_reg3_{}.json'.format(n)), 'w') as f:
        json.dump({'num_qubits': n, 'paulis': paulis, 'coeffs': coeffs}, f, indent=4)

    lines = ['{} {}\n'.format(n, g.number_of_edges())]
    for i, j in g.edges:
        lines.append('{} {}\n'.format(i, j))
    with open(os.path.join(text_dpath, 'qaoa_reg3_{}.txt'.format(n)), 'w') as f:
        f.writelines(lines)

    print(n, len(paulis))


