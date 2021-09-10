import os
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.utils import shuffle
import itertools

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


df_1 = pd.read_csv('../../results/re_' + data_file_names[0] + '.csv')
df_2 = pd.read_csv('../../results/re_' + data_file_names[1] + '.csv')
df_3 = pd.read_csv('../../results/re_' + data_file_names[2] + '.csv')
df_4 = pd.read_csv('../../results/re_' + data_file_names[3] + '.csv')

data_vis_1 = df_1[df_1['reconstruction_error'] <= 0.005]  # visualize only these data points
data_vis_2 = df_2[df_2['reconstruction_error'] <= 0.005]  # visualize only these data points
data_vis_3 = df_3[df_3['reconstruction_error'] <= 0.005]  # visualize only these data points
data_vis_4 = df_4[df_4['reconstruction_error'] <= 0.005]  # visualize only these data points

data = [data_vis_1['reconstruction_error'],
        data_vis_2['reconstruction_error'],
        data_vis_3['reconstruction_error'],
        data_vis_4['reconstruction_error']]

fig, ax = plt.subplots()

palette = itertools.cycle(sns.color_palette("viridis"))

# ax = sns.boxplot(data=data)
ax = sns.violinplot(data=data)
# ax = sns.stripplot(data=data, jitter=True)
# ax = sns.swarmplot(data=data) # color=next(palette))
# ax = plt.hist(data, density=False, bins=15)


# axis
ax.set_xlabel("Data Set", labelpad=15, fontsize=11)
ax.set_ylabel("Reconstruction Error  " + r'$s_{\sigma}$', labelpad=15, fontsize=11)
ax.set_xticklabels(["bpi2012w", "bpi2013i", "bpi2019", "bpi2020"])
ax.tick_params(axis='both', which='major', labelsize=10)



plt.show()

pp = PdfPages('../../results/re_plot_experiments.pdf')
pp.savefig(fig)
pp.close()



"""
# create subplots
subplt_positions = [[0, 0], [0, 1], [1, 0], [1, 1]]

for subplot_id, dataset in enumerate(data_file_names):
    # data
    df = pd.read_csv('../../results/re_' + dataset + '.csv')
    data = pd.DataFrame()

    data['process_instance'] = pd.Series(shuffle(df['process_instance']).tolist())
    data['reconstruction_error'] = df['reconstruction_error']

    data_vis = data[data['reconstruction_error'] <= 0.01]  # visualize only these data points


    threshold = df['threshold'][0]

    # hue
    hue_above = r'$s_{\sigma} > \theta$'
    hue_equal_below = r'$s_{\sigma} \leq \theta$'
    data['hue'] = np.where(data['reconstruction_error'] > threshold, hue_above, hue_equal_below)
    # hue_color_palette = {hue_above: 'blue', hue_equal_below: 'grey'}


    x_pos = subplt_positions[subplot_id][0]
    y_pos = subplt_positions[subplot_id][1]

    ax = sns.boxplot(data=data_vis, x='reconstruction_error', hue=data['hue']) # ax=axs[x_pos][y_pos])
    ax = sns.swarmplot(data=data_vis, x='reconstruction_error', hue=data['hue']) # ax=axs[x_pos][y_pos])

    

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
    
   
"""

