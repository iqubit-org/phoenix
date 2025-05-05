import sys

sys.path.append('../')

from itertools import product
import pandas as pd
from phoenix.models.cliffords import PAULIS_2Q, assemble_paulistr_with_sign

from iswapclifford import Clifford2QiSWAP

_TRANSFORM_TABLE_2Q = {}
_PAULIS_2Q = [''.join(pair) for pair in product(['I', 'X', 'Y', 'Z'], ['I', 'X', 'Y', 'Z'])]
for pauli_0, pauli_1 in product(['X', 'Y', 'Z'], ['X', 'Y', 'Z']):
    cg = Clifford2QiSWAP(pauli_0, pauli_1)  # controlled gate
    _TRANSFORM_TABLE_2Q[cg.name] = [assemble_paulistr_with_sign(*cg.transform(pauli)) for pauli in _PAULIS_2Q]
TRANSFORM_TABLE_2Q = pd.DataFrame(_TRANSFORM_TABLE_2Q, index=_PAULIS_2Q)
CLIFFORD_2Q_ISWAP_SET = [Clifford2QiSWAP(pauli_0, pauli_1) for pauli_0, pauli_1 in product(['X', 'Y', 'Z'], ['X', 'Y', 'Z'])]

print(CLIFFORD_2Q_ISWAP_SET)
print(TRANSFORM_TABLE_2Q)
