import os
import json

fnames = os.listdir('uccsd_qasm_little')

for fname in fnames:
    qasm_fname = os.path.join('uccsd_qasm_little', fname)
    json_fname = qasm_fname.replace('qasm', 'json')

    # amplify coefficients in .qasm file
    with open(qasm_fname, 'r') as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        if '0.0001' in line:
            lines[i] = line.replace('0.0001', '0.1')
        if 'e-05' in line:
            lines[i] = line.replace('e-05', 'e-2')
    with open(os.path.join('uccsd_qasm', qasm_fname.split('/')[-1]), 'w') as f:
        f.writelines(lines)

    # amplify coefficients in .json file
    with open(json_fname, 'r') as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        if 'e-05' in line:
            lines[i] = line.replace('e-05', 'e-2')
    with open(os.path.join('uccsd_json', json_fname.split('/')[-1]), 'w') as f:
        f.writelines(lines)
