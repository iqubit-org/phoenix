from phoenix.synthesis import simplification
from phoenix.models import BSF
import numpy as np
import time

from phoenix.synthesis.simplification import heuristic_bsf_cost


def random_pauli(n: int) -> str:
    return ''.join(np.random.choice(['X', 'Y', 'Z', 'I'], n, replace=True))

n = 6
paulis = [random_pauli(n) for _ in range(30)]
bsf = BSF(paulis)
print(heuristic_bsf_cost([bsf for _ in range(10)]))