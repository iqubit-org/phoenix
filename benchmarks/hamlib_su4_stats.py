"""
Statistics of #2Q and Depth-2Q of SU(4)-based representation of hamlib benchmarks.
"""
import sys


sys.path.append('..')

import os
import json
import bqskit
from tqdm import tqdm
from natsort import natsorted
from bqskit.compiler import Compiler
from bqskit.passes import QuickPartitioner
from phoenix import Circuit, gates
from phoenix.utils.operations import is_tensor_prod

bqskit_compiler = Compiler()
workflow = QuickPartitioner(2)

stats = {
    'binaryoptimization': {},
    'discreteoptimization': {},
    'chemistry': {},
    'condensedmatter': {}
}
for dir in stats:
    print('Processing', dir)
    for fname in tqdm(natsorted(os.listdir(os.path.join('./hamlib_qasm/', dir)))):
        circ = bqskit.Circuit.from_file(os.path.join('./hamlib_qasm/', dir, fname))
        blocks = list(bqskit_compiler.compile(circ, workflow))
        # fused_2q = Circuit([gates.UnivGate(blk.get_unitary().numpy).on(list(blk.location)) for blk in blocks])

        circ_su4 = Circuit([gates.X.on(blk.location[0], blk.location[1]) for blk in blocks if
                            not is_tensor_prod(blk.get_unitary().numpy)])

        # qc_su4 = qiskit.QuantumCircuit(circ_su4.num_qubits)
        # for g in circ_su4:
        #     q0, q1 = g.tqs
        #     qc_su4.cx(q0, q1)

        stats[dir][fname.split('.')[0]] = {
            # 'num_su4': qc_su4.size(),
            # 'depth_su4': qc_su4.depth()
            'num_su4': circ_su4.num_gates,
            'depth_su4': circ_su4.depth
        }

with open('hamlib_su4_stats.json', 'w') as f:
    json.dump(stats, f)
