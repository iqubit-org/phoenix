import sys
import json
sys.path.append('../..')
from collections import Counter

from phoenix.primitive import grouping

with open(sys.argv[1], 'r') as f:
    data = json.load(f)

pauli_lists = grouping.group_paulis(data['paulis'])

print(f"Number of paulis: {len(data['paulis'])}")
print(f"Number of groups: {len(pauli_lists)}")

print('Numbers:', Counter([len(pl) for pl in pauli_lists]))
