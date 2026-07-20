# summarize Hamlib results in text output
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
    path = "./results/result_hamlib_{}.csv".format(key)
    return pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()


compilers = {label: _load(key) for label, key in COMPILERS}

categories = ["binaryoptimization", "discreteoptimization", "chemistry", "condensedmatter"]


def opt_rate(df, col_opt, col_orig, category=None):
    if len(df) == 0 or col_opt not in df.columns or col_orig not in df.columns:
        return "-"  # this compiler has not been run yet
    if category is not None:
        df = df[df["category"] == category]
    if len(df) == 0:
        return "-"
    return gmean(df[col_opt] / df[col_orig]).round(3)


def category_label(cat):
    counts = [(df["category"] == cat).sum() for df in compilers.values() if "category" in df.columns]
    n = max(counts) if counts else 0
    return "{} ({})".format(cat, n)


def all_label():
    lens = [len(df) for df in compilers.values() if len(df)]
    n = max(lens) if lens else 0
    return "All ({})".format(n)


# >>> Num2Q Opt Rate
table = PrettyTable()
table.field_names = ["Num2Q Opt Rate"] + list(compilers.keys())

for cat in categories:
    row = [category_label(cat)]
    for name, df in compilers.items():
        row.append(opt_rate(df, "num_2q_gates(opt)", "num_2q_gates", category=cat))
    table.add_row(row)

row = [all_label()]
for name, df in compilers.items():
    row.append(opt_rate(df, "num_2q_gates(opt)", "num_2q_gates"))
table.add_row(row)

print(">>> Num2Q Opt Rate")
print(table)

# >>> Depth2Q Opt Rate
table = PrettyTable()
table.field_names = ["Depth2Q Opt Rate"] + list(compilers.keys())

for cat in categories:
    row = [category_label(cat)]
    for name, df in compilers.items():
        row.append(opt_rate(df, "depth_2q(opt)", "depth_2q", category=cat))
    table.add_row(row)

row = [all_label()]
for name, df in compilers.items():
    row.append(opt_rate(df, "depth_2q(opt)", "depth_2q"))
table.add_row(row)

print()
print(">>> Depth2Q Opt Rate")
print(table)
