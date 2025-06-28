# summarize results in text output
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import gmean
from prettytable import PrettyTable



result_tket = pd.read_csv('./results/result_uccsd_tket.csv')
result_paulihedral = pd.read_csv('./results/result_uccsd_paulihedral.csv')
result_tetris = pd.read_csv('./results/result_uccsd_tetris.csv')
result_phoenix = pd.read_csv('./results/result_uccsd_phoenix.csv')
result_quclear = pd.read_csv('./results/result_uccsd_quclear.csv')
result_pauliopt = pd.read_csv('./results/result_uccsd_pauliopt.csv')

num_programs = len(result_tket)
programs = result_tket['program']
num_2q_gates = result_tket['num_2q_gates']
depth_2q = result_tket['depth_2q']


table = PrettyTable()
table.field_names = ["Num2Q Opt Rate", "TKet", "Paulihedral", "Tetris", "PauliOpt", "QuCLEAR", "Phoenix"]

table.add_row([
    "All2all",
    gmean(result_tket['num_2q_gates(all2all)'] / num_2q_gates).round(3),
    gmean(result_paulihedral['num_2q_gates(all2all)'] / num_2q_gates).round(3),
    gmean(result_tetris['num_2q_gates(all2all)'] / num_2q_gates).round(3),
    gmean(result_pauliopt['num_2q_gates(all2all)'] / num_2q_gates).round(3),
    gmean(result_quclear['num_2q_gates(all2all)'] / num_2q_gates).round(3),
    gmean(result_phoenix['num_2q_gates(all2all)'] / num_2q_gates).round(3),
])

table.add_row([
    "All2all O3",
    gmean(result_tket['num_2q_gates(all2all)'] / num_2q_gates).round(3),
    gmean(result_paulihedral['num_2q_gates(all2all_opt)'] / num_2q_gates).round(3),
    gmean(result_tetris['num_2q_gates(all2all_opt)'] / num_2q_gates).round(3),
    gmean(result_pauliopt['num_2q_gates(all2all_opt)'] / num_2q_gates).round(3),
    gmean(result_quclear['num_2q_gates(all2all_opt)'] / num_2q_gates).round(3),
    gmean(result_phoenix['num_2q_gates(all2all_opt)'] / num_2q_gates).round(3),
])

table.add_row([
    "Sycamore",
    gmean(result_tket['num_2q_gates(sycamore)'] / num_2q_gates).round(3),
    gmean(result_paulihedral['num_2q_gates(sycamore)'] / num_2q_gates).round(3),
    gmean(result_tetris['num_2q_gates(sycamore)'] / num_2q_gates).round(3),
    gmean(result_pauliopt['num_2q_gates(sycamore)'] / num_2q_gates).round(3),
    gmean(result_quclear['num_2q_gates(sycamore)'] / num_2q_gates).round(3),
    gmean(result_phoenix['num_2q_gates(sycamore)'] / num_2q_gates).round(3),
])

table.add_row([
    "Manhattan",
    gmean(result_tket['num_2q_gates(manhattan)'] / num_2q_gates).round(3),
    gmean(result_paulihedral['num_2q_gates(manhattan)'] / num_2q_gates).round(3),
    gmean(result_tetris['num_2q_gates(manhattan)'] / num_2q_gates).round(3),
    gmean(result_pauliopt['num_2q_gates(manhattan)'] / num_2q_gates).round(3),
    gmean(result_quclear['num_2q_gates(manhattan)'] / num_2q_gates).round(3),
    gmean(result_phoenix['num_2q_gates(manhattan)'] / num_2q_gates).round(3),
])

print(">>> Num2Q Opt Rate")
print(table)

table = PrettyTable()
table.field_names = ["Depth2Q Opt Rate", "TKet", "Paulihedral", "Tetris", "PauliOpt", "QuCLEAR", "Phoenix"]

table.add_row([
    "All2all",
    gmean(result_tket['depth_2q(all2all)'] / depth_2q).round(3),
    gmean(result_paulihedral['depth_2q(all2all)'] / depth_2q).round(3),
    gmean(result_tetris['depth_2q(all2all)'] / depth_2q).round(3),
    gmean(result_pauliopt['depth_2q(all2all)'] / depth_2q).round(3),
    gmean(result_quclear['depth_2q(all2all)'] / depth_2q).round(3),
    gmean(result_phoenix['depth_2q(all2all)'] / depth_2q).round(3),
])

table.add_row([
    "All2all O3",
    gmean(result_tket['depth_2q(all2all)'] / depth_2q).round(3),
    gmean(result_paulihedral['depth_2q(all2all_opt)'] / depth_2q).round(3),
    gmean(result_tetris['depth_2q(all2all_opt)'] / depth_2q).round(3),
    gmean(result_pauliopt['depth_2q(all2all_opt)'] / depth_2q).round(3),
    gmean(result_quclear['depth_2q(all2all_opt)'] / depth_2q).round(3),
    gmean(result_phoenix['depth_2q(all2all_opt)'] / depth_2q).round(3),
])

table.add_row([
    "Sycamore",
    gmean(result_tket['depth_2q(sycamore)'] / depth_2q).round(3),
    gmean(result_paulihedral['depth_2q(sycamore)'] / depth_2q).round(3),
    gmean(result_tetris['depth_2q(sycamore)'] / depth_2q).round(3),
    gmean(result_pauliopt['depth_2q(sycamore)'] / depth_2q).round(3),
    gmean(result_quclear['depth_2q(sycamore)'] / depth_2q).round(3),
    gmean(result_phoenix['depth_2q(sycamore)'] / depth_2q).round(3),
])

table.add_row([
    "Manhattan",
    gmean(result_tket['depth_2q(manhattan)'] / depth_2q).round(3),
    gmean(result_paulihedral['depth_2q(manhattan)'] / depth_2q).round(3),
    gmean(result_tetris['depth_2q(manhattan)'] / depth_2q).round(3),
    gmean(result_pauliopt['depth_2q(manhattan)'] / depth_2q).round(3),
    gmean(result_quclear['depth_2q(manhattan)'] / depth_2q).round(3),
    gmean(result_phoenix['depth_2q(manhattan)'] / depth_2q).round(3),
])

print()
print(">>> Depth2Q Opt Rate")
print(table)
