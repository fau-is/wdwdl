from matplotlib import pyplot as plt
import plotly


def reconstruction_error(df, col):

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

    loss = history.history['loss']
    val_loss = history.history['val_loss']
    plt.figure()
    plt.plot(range(learning_epochs), loss, 'bo', label='Training loss')
    plt.plot(range(learning_epochs), val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()
