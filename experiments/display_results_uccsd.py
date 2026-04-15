# summarize UCCSD results in text output
import pandas as pd
from scipy.stats import gmean
from prettytable import PrettyTable

result_qiskit = pd.read_csv('./results/result_uccsd_qiskit.csv')
result_tket = pd.read_csv('./results/result_uccsd_tket.csv')
result_paulihedral = pd.read_csv('./results/result_uccsd_paulihedral.csv')
result_tetris = pd.read_csv('./results/result_uccsd_tetris.csv')
result_pauliopt = pd.read_csv('./results/result_uccsd_pauliopt.csv')
result_quclear = pd.read_csv('./results/result_uccsd_quclear.csv')
result_phoenix = pd.read_csv('./results/result_uccsd_phoenix.csv')

compilers = {
    'Qiskit': result_qiskit,
    'TKet': result_tket,
    'Paulihedral': result_paulihedral,
    'Tetris': result_tetris,
    'PauliOpt': result_pauliopt,
    'QuCLEAR': result_quclear,
    'Phoenix': result_phoenix,
}

topologies = [
    ('All2all', 'all2all'),
    ('All2all O3', 'all2all_opt'),
    ('Square', 'square'),
    ('HHex', 'hhex'),
]


def opt_rate(df, col_opt, col_orig):
    if len(df) == 0:
        return float('nan')
    return gmean(df[col_opt] / df[col_orig]).round(3)


def topology_label(label):
    n = max(len(df) for df in compilers.values())
    return "{} ({})".format(label, n)


# >>> Num2Q Opt Rate
table = PrettyTable()
table.field_names = ["Num2Q Opt Rate"] + list(compilers.keys())

for label, topo in topologies:
    row = [topology_label(label)]
    for name, df in compilers.items():
        row.append(opt_rate(df, 'num_2q_gates({})'.format(topo), 'num_2q_gates'))
    table.add_row(row)

print(">>> Num2Q Opt Rate")
print(table)

# >>> Depth2Q Opt Rate
table = PrettyTable()
table.field_names = ["Depth2Q Opt Rate"] + list(compilers.keys())

for label, topo in topologies:
    row = [topology_label(label)]
    for name, df in compilers.items():
        row.append(opt_rate(df, 'depth_2q({})'.format(topo), 'depth_2q'))
    table.add_row(row)

print()
print(">>> Depth2Q Opt Rate")
print(table)
