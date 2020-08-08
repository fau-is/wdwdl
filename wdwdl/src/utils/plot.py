from matplotlib import pyplot as plt
import plotly
import sklearn
import numpy as np
import pandas
import seaborn as sns
from functools import reduce
import matplotlib.pyplot as pyplot
import wdwdl.src.utils.general as general


def reconstruction_error(df, col):
    """
    Plots the reconstruction error.
    :param df:
    :param col:
    :return:
    """

    x = df.index.tolist()
    y = df[col].tolist()

    trace = {'type': 'scatter',
             'x': x,
             'y': y,
             'mode': 'markers'
             # 'marker': {'colorscale': 'red', 'opacity': 0.5}
             }
    data = plotly.graph_objs.Data([trace])
    layout = {'title': 'Reconstruction error for each process instance',
              'titlefont': {'size': 30},
              'xaxis': {'title': 'Process instance', 'titlefont': {'size': 20}},
              'yaxis': {'title': 'Reconstruction error', 'titlefont': {'size': 20}},
              'hovermode': 'closest'
              }
    figure = plotly.graph_objs.Figure(data=data, layout=layout)

    return figure


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
