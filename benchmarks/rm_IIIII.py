import os
import json


# qasm_dpath = './hamlib_qasm'
json_dpath = './hamlib_json'

categories = [
    'binaryoptimization',
    'discreteoptimization',
    'condensedmatter',
    'chemistry',
]

def all_identities(pauli):
    if len(pauli) == pauli.count('I'):
        return True
    return False

for dir in categories:
    dpath = os.path.join(json_dpath, dir)
    fnames = os.listdir(dpath)
    for fname in fnames:
        fname = os.path.join(dpath, fname)
        # print(fname)

        with open(fname, 'r') as f:
            data = json.load(f)

        for i, (pauli, coeff) in enumerate(zip(data['paulis'], data['coeffs'])):
            # if all_identities(pauli):
            #     data['paulis'].pop(i)
            #     data['coeffs'].pop(i)
            #
            #
            #     with open(fname, 'w') as f:
            #         json.dump(data, f, indent=4)
            #
            #     break


            if all_identities(pauli):
                print(f'Found identity at index {i} in {fname}')
                print(f'Pauli: {pauli}')
                print(f'Coefficient: {coeff}')
                print()