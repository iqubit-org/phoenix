# summarize Hamlib results in text output
import pandas as pd
from scipy.stats import gmean
from prettytable import PrettyTable

result_qiskit = pd.read_csv("./results/result_hamlib_qiskit.csv")
result_tket = pd.read_csv("./results/result_hamlib_tket.csv")
result_paulihedral = pd.read_csv("./results/result_hamlib_paulihedral.csv")
result_tetris = pd.read_csv("./results/result_hamlib_tetris.csv")
result_quclear = pd.read_csv("./results/result_hamlib_quclear.csv")
result_phoenix = pd.read_csv("./results/result_hamlib_phoenix.csv")

compilers = {
    "Qiskit": result_qiskit,
    "TKet": result_tket,
    "Paulihedral": result_paulihedral,
    "Tetris": result_tetris,
    "QuCLEAR": result_quclear,
    "Phoenix": result_phoenix,
}

categories = ["binaryoptimization", "discreteoptimization", "chemistry", "condensedmatter"]


def opt_rate(df, col_opt, col_orig, category=None):
    if category is not None:
        df = df[df["category"] == category]
    if len(df) == 0:
        return float("nan")
    return gmean(df[col_opt] / df[col_orig]).round(3)


def category_label(cat):
    n = max((df["category"] == cat).sum() for df in compilers.values())
    return "{} ({})".format(cat, n)


def all_label():
    n = max(len(df) for df in compilers.values())
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
