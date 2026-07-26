from qiskit import QuantumCircuit
import phoenix
import sys

from pathlib import Path
import sys
import json

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = PROJECT_ROOT / "experiments" / "scripts"

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SCRIPTS_DIR))

import phoenix
from bench_utils import phoenix_pass, qiskit_pass, tket_pass, quclear_pass

def get_t_cost(qc):
    return qc.count_ops()['t'], qc.depth(lambda instr: instr.operation.name == 't')

# TODO: test symphony, phoenix, qiskit, tket, quclear these four compilers with optimize=False parameter setting in <compiler>_pass() function
# benchmarks: hamlib
# call example: 
#   qc = phoenix_pass(data['paulis'], data['coeffs'], optimize=False)
#   qc = phoenix_pass(data['paulis'], data['coeffs'], optimize=False, grouping='support')
#   qc = qiskit_pass(data['paulis'], data['coeffs'], optimize=False)
# Finally Clifford+T circuit: qc_clifford_t = phoenix.utils.synth_to_clifford_t(qc)
# To get T-count and T-depth: get_t_cost

import time

json_file = sys.argv[1]
with open(json_file, 'r') as f:
    data = json.load(f)

start = time.perf_counter()
qc = phoenix_pass(data['paulis'], data['coeffs'])
# qc = tket_pass(data['paulis'], data['coeffs'], optimize=False)
print('Elapsed time:', time.perf_counter() - start)
qc = phoenix.utils.synth_to_clifford_t(qc)
print('Elapsed time:', time.perf_counter() - start)
phoenix.utils.print_circ_info(qc)

print(get_t_cost(qc))
print('Elapsed time:', time.perf_counter() - start)
