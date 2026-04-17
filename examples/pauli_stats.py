import phoenix
import sys
import json
from collections import Counter
import time

start_time = time.time()
json_fname = sys.argv[1]
with open(json_fname, 'r') as f:
    data = json.load(f)

print(f'Loaded {json_fname} in {time.time() - start_time:.2f} seconds.')
print(f'Number of Paulis: {len(data["paulis"])}')
print(f'Number of Qubits: {data["num_qubits"]}')

start_time = time.time()
groups = phoenix.primitive.group_paulis(data['paulis'])
print(f'Grouped paulis in {time.time() - start_time:.2f} seconds.')

print(f'Number of groups: {len(groups)}')
group_weights = Counter([len(paulis[0]) - (paulis[0]).count('I') for paulis in groups.values()])
group_sizes = Counter([len(paulis) for paulis in groups.values()])
print(f'Group weights: {group_weights}')
print(f'Group sizes: {group_sizes}')
