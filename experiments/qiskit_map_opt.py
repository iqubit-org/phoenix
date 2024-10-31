import pytket.qasm
import pytket.passes
import os

input_dpath = './qiskit_post_opt_manhattan'
output_dpath = './qiskit_post_opt_manhattan_opt'

fnames = os.listdir(input_dpath)
for fname in fnames:
    print('Processing', fname)
    intput_fname = os.path.join(input_dpath, fname)
    output_fname = os.path.join(output_dpath, fname)

    print(intput_fname, output_fname)

    circ = pytket.qasm.circuit_from_qasm(intput_fname)

    if circ.n_qubits != 12:
        continue

    pytket.passes.FullPeepholeOptimise(allow_swaps=False).apply(circ)
    pytket.passes.RemoveRedundancies().apply(circ)
    pytket.qasm.circuit_to_qasm(circ, output_fname)
