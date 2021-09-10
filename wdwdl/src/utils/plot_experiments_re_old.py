import os
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.utils import shuffle

""" Creates plots of the reconstruction error for different datasets and exports them as a single page pdf file. """


def get_datasets():
    """ Returns the names of the datasets/event logs to be plotted and their respective data file names. """

    data_file_names = ["bpi2012w_converted_selection",
                       "bpi2013i_converted_selection",
                       "bpi2019_converted_selection",
                       "bpi2020_converted_selection"]

    dataset_names = []
    for dataset in data_file_names:
        dataset_names.append(dataset.split("_")[0])

    return data_file_names, dataset_names


# data
data_file_names, dataset_names = get_datasets()

# set plot theme
sns.set(style='darkgrid')

# figure size / layout
fig, axs = plt.subplots(2, 2,
                        sharey='col',
                        figsize=(11.7, 8.27))  # size of A4 paper
fig.tight_layout()

# create subplots
subplt_positions = [[0, 0], [0, 1], [1, 0], [1, 1]]

for subplot_id, dataset in enumerate(data_file_names):
    # data
    df = pd.read_csv('../../results/re_' + dataset + '.csv')
    data = pd.DataFrame()

    # data['process_instance'] = pd.Series(shuffle(df['process_instance']).tolist())
    data['process_instance'] = df['process_instance']
    data['reconstruction_error'] = df['reconstruction_error']

    data_vis = data[data['reconstruction_error'] <= 0.01]  # visualize only these data points

    threshold = df['threshold'][0]

    # hue
    hue_above = r'$s_{\sigma} > \theta$'
    hue_equal_below = r'$s_{\sigma} \leq \theta$'
    data['hue'] = np.where(data['reconstruction_error'] > threshold, hue_above, hue_equal_below)
    hue_color_palette = {hue_above: 'blue', hue_equal_below: 'grey'}

    x_pos = subplt_positions[subplot_id][0]
    y_pos = subplt_positions[subplot_id][1]


    sns.scatterplot(data=data_vis,
                    x='process_instance', y='reconstruction_error',
                    hue=data['hue'],
                    hue_order=[hue_above, hue_equal_below],
                    palette=hue_color_palette,
                    edgecolor='none',
                    alpha=0.5,  # marker opacity
                    s=5,  # marker size
                    ax=axs[x_pos][y_pos]
                    )

    # threshold
    label = r"$\theta = %s$" % str(round(threshold, 5))

    if threshold < 0.0001:
        axs[x_pos][y_pos].axhline(0.00005, label=label, linewidth=1.0, color='black')
    else:
        axs[x_pos][y_pos].axhline(threshold, label=label, linewidth=1.0, color='black')

    # legend
    handles, labels = axs[x_pos][y_pos].get_legend_handles_labels()
    axs[x_pos][y_pos].legend(loc='upper right', fontsize=9,
                             handles=[h for h in handles if
                                      handles.index(h) != 1])  # remove subtitle for hue entries from legend

    # coordinate system
    axs[x_pos][y_pos].grid(linewidth=0.5)
    axs[x_pos][y_pos].set_ylim(0, 0.005)
    axs[x_pos][y_pos].set_xticks(np.arange(0, max(df.index.tolist()), step=1000))
    axs[x_pos][y_pos].set_yticks(np.arange(0, 0.005, step=0.001))

    # text
    axs[x_pos][y_pos].set_title(dataset_names[subplot_id], fontstyle='oblique', fontweight='semibold', fontsize=12,
                                pad=15)
    axs[x_pos][y_pos].set_xlabel("Process Instance  " + r'$\sigma$', labelpad=15, fontsize=11)
    axs[x_pos][y_pos].set_ylabel("Reconstruction Error  " + r'$s_{\sigma}$', labelpad=15, fontsize=11)
    axs[x_pos][y_pos].tick_params(axis='both', which='major', labelsize=10)

plt.show()

pp = PdfPages('../../results/re_plot_experiments.pdf')
pp.savefig(fig)
pp.close()
