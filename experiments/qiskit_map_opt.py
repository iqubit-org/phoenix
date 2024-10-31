import pytket.qasm
import pytket.passes
import os

input_dpath = './qiskit_post_opt_manhattan'
output_dpath = './qiskit_post_opt_manhattan_opt'

fnames = os.listdir(input_dpath)
import  numpy as np

np.random.shuffle(fnames)
for fname in fnames:
    print('Processing', fname)
    intput_fname = os.path.join(input_dpath, fname)
    output_fname = os.path.join(output_dpath, fname)

    if os.path.exists(output_fname):
        continue

    circ = pytket.qasm.circuit_from_qasm(intput_fname)

    print(pytket.passes.FullPeepholeOptimise(allow_swaps=False).apply(circ))
    print(pytket.passes.RemoveRedundancies().apply(circ))
    pytket.qasm.circuit_to_qasm(circ, output_fname)
