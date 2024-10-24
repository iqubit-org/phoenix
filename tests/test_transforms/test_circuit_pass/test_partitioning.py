import os
import cirq
from phoenix import Circuit
from functools import reduce
from operator import add
from tqdm import tqdm
from phoenix.transforms.circuit_pass import sequential_partition

circ_files_dpath = '../../input/cx-basis/'

fnames = []
for dir in os.listdir(circ_files_dpath):
    fnames.extend(
        [os.path.join(circ_files_dpath, dir, fname) for fname in os.listdir(os.path.join(circ_files_dpath, dir)) if
         fname.endswith('.qasm')])

num_gates_skip = 1000
num_qubits_skip = 8
grain = 2


def test_sequential_partition():
    for fname in tqdm(fnames):

        circ = Circuit.from_qasm(fname=fname)

        if circ.num_gates > num_gates_skip or circ.num_qubits > num_qubits_skip:
            continue

        blocks = sequential_partition(circ, grain)

        circ_merged = reduce(add, blocks)
        cirq.testing.assert_allclose_up_to_global_phase(
            circ.unitary(),
            circ_merged.unitary(),
            atol=1e-6,
        )
