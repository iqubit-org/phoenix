# summarize UCCSD results in text output
import os

import pandas as pd
from scipy.stats import gmean
from prettytable import PrettyTable

# display label -> results CSV key
COMPILERS = [
    ("Qiskit", "qiskit"),
    ("TKet", "tket"),
    ("Paulihedral", "paulihedral"),
    ("Tetris", "tetris"),
    ("QuCLEAR", "quclear"),
    ("Phoenix", "phoenix"),
    ("Phoenix++", "phoenixpp"),
]


def _load(key):
    """Load a compiler's results CSV, or an empty frame if it hasn't been run yet
    (so the table still renders — that column shows '-')."""
    path = "./results/result_uccsd_{}.csv".format(key)
    return pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()


compilers = {label: _load(key) for label, key in COMPILERS}

topologies = [
    ("All2all", "all2all"),
    ("Square", "square"),
    ("HHex", "hhex"),
]


def opt_rate(df, col_opt, col_orig):
    if len(df) == 0 or col_opt not in df.columns or col_orig not in df.columns:
        return "-"  # this compiler/topology has not been run yet
    return gmean(df[col_opt] / df[col_orig]).round(3)


# >>> Num2Q Opt Rate
table = PrettyTable()
table.field_names = ["Num2Q Opt Rate"] + list(compilers.keys())

for label, topo in topologies:
    row = [label]
    for name, df in compilers.items():
        row.append(opt_rate(df, "num_2q_gates({})".format(topo), "num_2q_gates"))
    table.add_row(row)

print(">>> Num2Q Opt Rate")
print(table)

# >>> Depth2Q Opt Rate
table = PrettyTable()
table.field_names = ["Depth2Q Opt Rate"] + list(compilers.keys())

for label, topo in topologies:
    row = [label]
    for name, df in compilers.items():
        row.append(opt_rate(df, "depth_2q({})".format(topo), "depth_2q"))
    table.add_row(row)

print()
print(">>> Depth2Q Opt Rate")
print(table)
