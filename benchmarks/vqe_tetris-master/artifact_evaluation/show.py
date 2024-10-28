import pickle
import pandas as pd
import os

def pickle_load(filename):
    with open(filename, 'rb') as f:
        obj = pickle.load(f)
    return obj

data = {
    'Encoder': [],
    'Bench.': [],
    'PH_Total_Gate': [],
    'Tetris_Total_Gate': [],
    'Improv._Total_Gate': [],
    'PH_CNOT_Gate': [],
    'Tetris_CNOT_Gate': [],
    'Improv._CNOT_Gate': [],
    'PH_Depth': [],
    'Tetris_Depth': [],
    'Improv._Depth': [],
    'PH_duration': [],
    'Tetris_duration': [],
    'Improv._duration': [],
}

moles = ['LiH', 'BeH2', 'CH4', 'MgH2', 'LiCl', 'CO2']

prefix = 'experiment_results'

for experiment_name in ['jordan_wigner', 'bravyi_kitaev', 'random']:
    for k in range(1, 7):
        data['Encoder'].append(experiment_name)
        data['Bench.'].append(moles[k-1] if experiment_name != 'random' else f'UCC-{5+k*5}')
        if os.path.exists(os.path.join(prefix, experiment_name, f'Tetris_data_{k}.pickle')):
            ph = pickle_load(os.path.join(prefix, experiment_name, f'PH_data_{k}.pickle'))[1]
            tetris = pickle_load(os.path.join(prefix, experiment_name, f'Tetris_data_{k}.pickle'))[1]

            data['PH_Total_Gate'].append(ph['Total'])
            data['Tetris_Total_Gate'].append(tetris['Total'])
            data['Improv._Total_Gate'].append("{:.2f}%".format((tetris['Total'] - ph['Total']) / ph['Total'] * 100))
            data['PH_CNOT_Gate'].append(ph['CNOT'])
            data['Tetris_CNOT_Gate'].append(tetris['CNOT'])
            data['Improv._CNOT_Gate'].append("{:.2f}%".format((tetris['CNOT'] - ph['CNOT']) / ph['CNOT'] * 100))
            data['PH_Depth'].append(ph['Depth'])
            data['Tetris_Depth'].append(tetris['Depth'])
            data['Improv._Depth'].append(("{:.2f}%".format((tetris['Depth'] - ph['Depth']) / ph['Depth'] * 100)))
            data['PH_duration'].append(ph['duration'])
            data['Tetris_duration'].append(tetris['duration'])
            data['Improv._duration'].append(("{:.2f}%".format((tetris['duration'] - ph['duration']) / ph['duration'] * 100)))
        else:
            data['PH_Total_Gate'].append('-')
            data['Tetris_Total_Gate'].append('-')
            data['Improv._Total_Gate'].append('-')
            data['PH_CNOT_Gate'].append('-')
            data['Tetris_CNOT_Gate'].append('-')
            data['Improv._CNOT_Gate'].append('-')
            data['PH_Depth'].append('-')
            data['Tetris_Depth'].append('-')
            data['Improv._Depth'].append('-')
            data['PH_duration'].append('-')
            data['Tetris_duration'].append('-')
            data['Improv._duration'].append('-')

            
df = pd.DataFrame(data)

print(df)
df.to_csv('figs/table2.csv', index=False)  # Set index=False to exclude row indices in the output file


# Fig 14
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
def format_thousands(value, pos):
    return f'{value / 1000:.0f}K'

x_values = np.array(['LiH', 'BeH2', 'CH4', 'MgH2'])  # X-axis values
n = len(x_values)
TKet_data_list = pickle_load(f'../core/runs_final/jordan_wigner/tket_data.pickle')[:n]
PH_data_list = pickle_load(f'experiment_results/jordan_wigner/PH_data.pickle')[:n]
Tetris_data_list = pickle_load(f'../core/runs_final/jordan_wigner/Tetris_data.pickle')[:n]
Tetris_lh_data_list = pickle_load(f'experiment_results/jordan_wigner/Tetris_data.pickle')[:n]

tk_cnots = []
ph_cnots = []
tetris_cnots = []
tetris_lh_cnots = []
for i, (tk_data, ph_data, tetris_data, tetris_lh_data) in enumerate(zip(TKet_data_list, PH_data_list, Tetris_data_list, Tetris_lh_data_list)):
    mole, tk = tk_data
    mole, ph = ph_data
    mole, tetris = tetris_data
    mole, tetris_lh = tetris_lh_data
    
    tk_cnots.append(tk['CNOT'])
    ph_cnots.append(ph['CNOT'])
    tetris_cnots.append(tetris['CNOT'])
    tetris_lh_cnots.append(tetris_lh['CNOT'])


pcoast_cnots = []
for n_qubits in [12, 14, 18, 22]:
    metric = pickle_load(f'../pcoast/experiments/pcoast_{n_qubits}.pickle')
    pcoast_cnots.append(metric['CNOT'])

width = 0.15  # Width of each bar

# Create a figure and axis
fig, ax = plt.subplots(figsize=(8, 3))

# Set the bar positions
x_positions = np.arange(len(x_values))

# Create the bars
# Custom colors for bars (in hexadecimal format)
colors = ['#5ba585', '#30655f', '#e0e7c8', '#ff5733', '#3b6ea6']
bar0_plot = ax.bar(x_positions - width * 2, tk_cnots, width, label='TKet',color=colors[0],edgecolor='black')
bar1_plot = ax.bar(x_positions - width    , pcoast_cnots, width, label='PCOAST',color=colors[1],edgecolor='black')
bar1_plot = ax.bar(x_positions            , ph_cnots, width, label='PH',color=colors[2],edgecolor='black')
bar2_plot = ax.bar(x_positions + width    , tetris_cnots, width, label='Tetris',color=colors[3],edgecolor='black')
bar3_plot = ax.bar(x_positions + width * 2, tetris_lh_cnots, width, label='Tetris+lookahead',color=colors[4],edgecolor='black')

# Set the x-axis labels
ax.set_xticks(x_positions)
ax.set_xticklabels(x_values, fontsize=14, fontweight='bold')  # Set font size for x-axis tick labels

# Set the y-axis label font size
ax.set_ylabel('CNOT gate count', fontsize=14,fontweight='bold')  # Set font size for y-axis label
plt.gca().yaxis.set_major_formatter(FuncFormatter(format_thousands))

# Set font size for y-axis tick labels
ax.tick_params(axis='y', labelsize=14)
ytick_labels = ax.get_yticklabels()
for label in ytick_labels:
    label.set_fontweight('bold')

# Add labels and a legend
# ax.set_xlabel('Real Molecules with JW')

# ax.set_title('CX Gate Cancel Ratio',fontsize = 15)
legend = ax.legend(fontsize=14)
for text in legend.get_texts():
    text.set_fontweight('bold')

# Show the plot
plt.tight_layout()
plt.savefig('figs/fig14.pdf')
plt.show()

# Fig 20