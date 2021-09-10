import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib import ticker

""" Plots f1 scores per percentile of reconstruction errors """

# retrieve data
data = pd.read_csv('../../results/f1_scores.csv', sep=';')

# set theme
sns.set_context("paper")
sns.set(style='dark')

# create figure
fig, axs = plt.subplots()

# create plot
ax = sns.lineplot(data=data,
                  x="percentile",
                  y="f1",
                  hue="event_log",  # use multiple colors
                  style="event_log",  # use multiple markers
                  markers=["o", "o", "o", "o"],
                  markeredgecolor='none',
                  legend="brief",
                  palette=['Black', 'Blue', 'Gray', 'DarkBlue'],
                  dashes=[(2, 2), (2, 2), (2, 2), (2, 2)])

# x axis and x ticks
plt.xlim(45, 95)
ax.xaxis.set_major_locator(ticker.FixedLocator(np.arange(50, 100, step=10)))
ax.xaxis.set_minor_locator(ticker.FixedLocator(np.arange(50, 95, step=5)))
ax.xaxis.set_major_formatter(ticker.FixedFormatter(['$50^{th}$', '$60^{th}$', '$70^{th}$', '$80^{th}$', '$90^{th}$']))
ax.xaxis.set_minor_formatter(ticker.NullFormatter())

# y axis and y ticks
plt.ylim(0.15, 0.95)
ax.yaxis.set_major_locator(ticker.NullLocator())
ax.yaxis.set_minor_locator(ticker.FixedLocator(np.arange(0.20, 1.1, step=0.1)))
ax.yaxis.set_major_formatter(ticker.NullFormatter())
ax.yaxis.set_minor_formatter(ticker.PercentFormatter(1.0))

# both ticks
ax.tick_params(axis='both', which='both', labelsize=10)

# coordinate system
ratio = 0.6
ax.set_aspect(1.0 / ax.get_data_ratio() * ratio)
plt.grid(which='minor', linewidth=0.5, linestyle='-')
# plt.grid(True)

# legend
handles, labels = ax.get_legend_handles_labels()
ax.legend(loc='lower left',
          fontsize=10,
          handles=[h for h in handles if handles.index(h) != 0])  # remove subtitle for entries from legend

# text
plt.xlabel("$k^{th}$ Percentile of Reconstruction Errors $\mathbf{s}$", labelpad=15, fontsize=11)
plt.ylabel("F1-Score", labelpad=15, fontsize=11)

# save and display
fig.savefig('../../results/f1_plot_experiments.pdf', bbox_inches='tight')
plt.show()
