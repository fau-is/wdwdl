from matplotlib import pyplot as plt
import plotly
import sklearn
import numpy as np
import pandas
import seaborn as sns
from functools import reduce
import matplotlib.pyplot as pyplot
import wdwdl.src.utils.general as general


def reconstruction_error(args, df, col, threshold):
    """
    Plots the reconstruction error.
    :param args:
    :param df:
    :param col:
    :param threshold:
    :return:
    """

    # data
    data = pandas.DataFrame()
    data['process_instance'] = df.index
    data['reconstruction_error'] = df[col]

    # set visual setting
    sns.set()

    # image size
    fig, axs = plt.subplots()
    fig.set_size_inches(11.7, 8.27) # size of A4 paper

    # create plot
    ax = sns.scatterplot(data=data,
                         x='process_instance', y='reconstruction_error',
                         facecolor='darkred',
                         edgecolor='none',
                         alpha=0.5,     # marker opacity
                         s=5,           # marker size
                        )

    # threshold
    ax.axhline(threshold, label="t = x", linewidth=0.75, color='black')
    ax.legend()

    # coordinate system
    ratio = 0.6
    ax.set_aspect(1.0 / ax.get_data_ratio() * ratio)
    plt.grid(linewidth=0.5)
    plt.xticks(np.arange(0, max(df.index.tolist()), step=1000))

    # text
    plt.title('Reconstruction Error for Each Process Instance', fontweight='bold', fontsize=14, pad=20)
    plt.xlabel("Process Instance", fontstyle='oblique', labelpad=20)
    plt.ylabel("Reconstruction Error", fontstyle='oblique', labelpad=20)

    # save and display
    plt.savefig('results/noise_' + args.data_set[:-4] + '.pdf')
    plt.show()


def learning_curve(history, learning_epochs):
    """
    Plots the learning curve.
    :param history:
    :param learning_epochs:
    :return:
    """

    loss = history.history['loss']
    val_loss = history.history['val_loss']
    plt.figure()
    plt.plot(range(learning_epochs), loss, 'bo', label='Training loss')
    plt.plot(range(learning_epochs), val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()


def confusion_matrix(label_ground_truth, label_prediction, args):
    """
    Plots a confusion matrix.
    :param label_ground_truth:
    :param label_prediction:
    :param args:
    :return:
    """

    label_ground_truth = np.array(label_ground_truth)
    label_prediction = np.array(label_prediction)

    classes = sklearn.utils.multiclass.unique_labels(label_ground_truth, label_prediction)
    classes_ = []
    for element in classes:
        classes_.append(general.class_names[element])
    classes = classes_

    cms = []
    cm = sklearn.metrics.confusion_matrix(label_ground_truth, label_prediction)
    cm_df = pandas.DataFrame(cm, index=classes, columns=classes)
    cms.append(cm_df)

    def prettify(n):
        """
        if n > 1000000:
            return str(np.round(n / 1000000, 1)) + 'M'
        elif n > 1000:
            return str(np.round(n / 1000, 1)) + 'K'
        else:
        """
        return str(n)

    cm = reduce(lambda x, y: x.add(y, fill_value=0), cms)
    annot = cm.applymap(prettify)
    cm = (cm.T / cm.sum(axis=1)).T
    fig, g = pyplot.subplots(figsize=(7, 4.5))
    g = sns.heatmap(cm, annot=annot, fmt='', cmap='Blues', cbar=False, rasterized=True, linewidths=0.1)
    _ = g.set(ylabel='Actual', xlabel='Prediction')

    for _, spine in g.spines.items():
        spine.set_visible(True)

    pyplot.xticks(rotation=45)
    fig.tight_layout()
    fig.savefig(str(args.result_dir + 'cm_' + args.data_set[:-4] + '.pdf'))

    general.llprint(str(label_prediction))
    general.llprint(str(label_ground_truth))
