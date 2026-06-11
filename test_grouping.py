import phoenix
import json

with open('./benchmarks/uccsd_json/LiH_frz_BK_sto3g.json', 'r') as f:
    data = json.load(f)
ham = phoenix.Hamiltonian(data['paulis'], data['coeffs'])


groups = ham.group_same_weights(subset=False)
print('Groups (exact match) (num_groups={}):'.format(len(groups)))
# for h in groups:
#     print(f"  {h.active_qubits}: {len(h.paulis)} terms")

groups = ham.group_same_weights(subset=True)
print('\nGroups (subset match) (num_groups={}):'.format(len(groups)))
for h in groups:
    print(f"  {h.active_qubits}: {len(h.paulis)} terms")
