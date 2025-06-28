import sys

sys.path.append('../..')
import os
from natsort import natsorted
import json

from phoenix.models import HamiltonianModel

INPUT_JSON_DPATH = '../benchmarks/uccsd'
output_dpath = './output_uccsd/original'

json_fnames = [os.path.join(INPUT_JSON_DPATH, fname) for fname in natsorted(os.listdir(INPUT_JSON_DPATH))]

for fname in json_fnames:
    output_fname = os.path.join(output_dpath, os.path.basename(fname).replace('.json', '.qasm'))
    with open(fname, 'r') as f:
        data = json.load(f)

    HamiltonianModel(data['paulis'], data['coeffs']).generate_circuit().to_qasm(output_fname)
