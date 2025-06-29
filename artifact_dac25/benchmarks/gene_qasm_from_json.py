import sys

sys.path.append('../..')
import os
from natsort import natsorted
import json

from phoenix.models import HamiltonianModel

INPUT_JSON_DPATH = sys.argv[1]
assert os.path.isdir(INPUT_JSON_DPATH), f"Input path {INPUT_JSON_DPATH} is not a directory."
OUTPUT_QASM_PATH = os.path.dirname(INPUT_JSON_DPATH) + '_qasm'


json_fnames = [os.path.join(INPUT_JSON_DPATH, fname) for fname in natsorted(os.listdir(INPUT_JSON_DPATH))]

for fname in json_fnames:
    print(f"Processing {fname}...")
    output_fname = os.path.join(OUTPUT_QASM_PATH, os.path.basename(fname).replace('.json', '.qasm'))
    with open(fname, 'r') as f:
        data = json.load(f)
    HamiltonianModel(data['paulis'], data['coeffs']).generate_circuit().to_qasm(output_fname)
