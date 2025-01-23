import sys
sys.path.append('..')

import numpy as np
from phoenix.models import BSF
from rustiq import pauli_network_synthesis
from rustiq import Metric

paulis_a, coeffs_a = ['XXXZIYZI', 'YXXZIYYI', 'ZXXZIYZI'], np.array([-0.0125, -0.0125, -0.0125])

cliffs = pauli_network_synthesis(paulis_a, Metric.COUNT)
cliffs
