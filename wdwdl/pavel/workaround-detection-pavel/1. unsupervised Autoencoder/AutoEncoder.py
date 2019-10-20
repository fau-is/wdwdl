# -*- coding: utf-8 -*-
"""
@author: Pavel Akram
"""
import numpy
import pandas
import matplotlib
import seaborn
import plotly
import tensorflow
import keras



print('Numpy version      :' , numpy.__version__)
print('Pandas version     :' ,pandas.__version__)
print('Matplotlib version :' ,matplotlib.__version__)
print('Seaborn version    :' , seaborn.__version__)
print('Plotly version     :', plotly.__version__)
print('Tensorflow version :' , tensorflow.__version__)
print('Keras version      :' , keras.__version__)


import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_row', None)
import matplotlib.pyplot as plt
plt.rcdefaults()
import seaborn as sns; sns.set()
import datetime
import plotly.offline as pyo
from plotly.graph_objs import Data, Figure
from keras.models import Model
from keras.layers import Input, Dense
from sklearn.preprocessing import   MinMaxScaler




#   Datenvorbearbeitung:
#       Die CSV wird geladen und als Dataframe 'df' gespeichert.
#       Danach werden die Spalten 'Resource' und 'Timestamp' aus dem Dataframe 
#       entfernt, jedoch kann man auch das Entfernen von der 
#       'Resource' auskommentieren, das vom 'Timestamp' jedoch nicht, 
#       da es mit dem Datentyp DateTime Komplikationen im Programm gibt.
#       Danach werden alle Zeilen mit der selben 'CaseID' in eine Zeile
#       zusammengefasst, damit das Dataframe Case-based ist und nicht 
#       mehr Event-based ist.
#       Zum Schluss werden alle leeren Werte ('NaN') ersetzt durch die Zahl -1
#       und alle Werte werden durch den MinMaxScaler durch Zahlen die zwischen
#       0 und 1 liegen ersetzt.
df = pd.read_csv("bpi2012_w_realCSV.csv")

df=df.drop(['Resource'],axis=1)  
df=df.drop(['Timestamp'],axis=1)

df=df.assign(key=df.groupby('CaseID').cumcount()+1).set_index(
        ['CaseID','key']).stack().unstack([1,2])
df.columns=df.columns.map('{0[1]}{0[0]}'.format)

df.to_csv('rapidminerwithmissing.csv')
df.fillna(-1, inplace=True)


scaler = MinMaxScaler()
df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)




#   Training- und Testset:
#       Das Dataframe 'df' wird auf geteilt auf Training- und Testset.
#       Als erstes gibt man 'train_size' einen Wert, der angibt wie viele 
#       Datensätze man dem Trainingset zu verfüung gibt.
#       Den Rest der Datensätze kriegt das Testset.
#       Danach wird jeweils der AcitivityID1-Wert enfernt, jedoch wird er 
#       zuvor gespeichert, da dieser zu Validierung wichtig ist. 
#       Anschließend werden die Dataframes zu einem numpy-Array umgewandelt.
train_size=1000

X_train=df.tail(train_size)
X_test=df.head(df.shape[0]-train_size)

y_train = X_train['ActivityID1']
X_train = X_train.drop(['ActivityID1'], axis=1)
                  
y_test  = X_test['ActivityID1']
X_test  = X_test.drop(['ActivityID1'], axis=1)

X_train = X_train.to_numpy()
X_test  = X_test.to_numpy()
print('Training data size   :', X_train.shape)
print('Validation data size :', X_test.shape)



#------------------------------------------------------------------------------
#   Veranschaulichungsfunktionen
def line_plot(df, col):
    
    x = df.index.tolist()
    y = df[col].tolist()
    
    trace = {'type':  'scatter', 
             'x'   :  x,
             'y'   :  y,
             'mode' : 'markers',
             'marker': {'colorscale':'red', 'opacity': 0.5}
            }
    data   = Data([trace])
    layout = {'title': 'Line plot of {}'.format(col),
              'titlefont': {'size': 30},
              'xaxis' : {'title' :'Data Index', 'titlefont': {'size' : 20}},
              'yaxis' : {'title': col, 'titlefont' : {'size': 20}},
              'hovermode': 'closest'
             }
    fig = Figure(data = data, layout = layout)
    return fig

def train_validation_loss(df_history):
    
    trace = []
    
    for ActivityID1, loss in zip(['Train', 'Validation'],
                                 ['loss', 'val_loss']):
        trace0 = {'type' : 'scatter', 
                  'x'    : df_history.index.tolist(),
                  'y'    : df_history[loss].tolist(),
                  'name' : ActivityID1,
                  'mode' : 'lines'
                  }
        trace.append(trace0)
    data = Data(trace)

    
    layout = {'title' : 'Model train-vs-validation loss',
              'titlefont':{'size' : 30},
              'xaxis' : {'title':  '<b> Epochs', 'titlefont':{ 'size' : 25}},
              'yaxis' : {'title':  '<b> Loss', 'titlefont':{ 'size' : 25}},
              }
    fig = Figure(data=data , layout=layout )
    
    return pyo.plot(fig)  
#------------------------------------------------------------------------------



#   AutoEncoder-Model-Aufbau und Training:
#       Der Auto besteht aus 7 verbundenen Hiddenlayers.
#       Die Anzahl der Neuronen ist 30,15,5,2,5,15,30.
#       Die erste und die letzte Schicht bestehen aus der Spaltenanzahl
#       , also 73. 
#       Die optimale 'activation'-Funktion ist in diesem Fall 'tanh', da es 
#       die geringste 'Loss' herausbringt. 'Sigmoid' und einen Mix aus 'tanh'
#       und 'sigmoid' hab ich versucht, jedoch kam es nicht ans Ergebnis 
#       'tanh' ran.
input_dim = X_train.shape[1]
encoding_dim = 30

input_layer = Input(shape=(input_dim, ))

encoder = Dense(encoding_dim, activation='tanh')(input_layer)
encoder = Dense(int(encoding_dim / 2), activation='tanh')(encoder)

encoder = Dense(int(5), activation='tanh')(encoder)

encoder = Dense(int(2), activation='tanh')(encoder)

decoder = Dense(int(5), activation='tanh')(encoder)
decoder = Dense(int(encoding_dim/ 2), activation='tanh')(decoder)
decoder = Dense(int(encoding_dim), activation='tanh')(decoder)

decoder = Dense(input_dim, activation='tanh')(decoder)

autoencoder = Model(inputs=input_layer, outputs=decoder)
autoencoder.summary()

nb_epoch = 100
batch_size = 128

autoencoder.compile(optimizer='adam', loss='mse' )

t_ini = datetime.datetime.now()
history = autoencoder.fit(X_train, X_train,
                        epochs=nb_epoch,
                        batch_size=batch_size,
                        shuffle=True,
                        validation_split=0.1,
                        )

t_fin = datetime.datetime.now()
print('Time to run the model: {} Sec.'.format((t_fin - t_ini).total_seconds()))

df_history = pd.DataFrame(history.history)
train_validation_loss(df_history)




#   Anomalien heruasfiltern:
#       Wir schauen, ob der AutoEncoder die ActivityID1 wieder herausrechnen
#       kann.  
#       Ich hab nun festgelegt, falls der AutoEncoder einen
#       Rekonstruktions-Fehler von mehr als 0.1 hat, wird es als Anomalie 
#       gekennzeichnet.
predictions = autoencoder.predict(X_test)

mse = np.mean(np.power(X_test - predictions, 2), axis=1)
df_error = pd.DataFrame({'reconstruction_error': mse, 'ActivityID1': y_test},
                        index=y_test.index)

fig = line_plot(df_error, 'reconstruction_error')
pyo.plot(fig)

outliers = df_error.index[df_error.reconstruction_error > 0.1].tolist()
print(outliers)

print('-----------')
p0bis05 = df_error.index[df_error.reconstruction_error <= 0.05  ].tolist()
print(len(p0bis05))
print(len(p0bis05)/8658)

p05bis1 = df_error.index[df_error.reconstruction_error <= 0.1   ].tolist()
print(len(p05bis1))
print(len(p05bis1)/8658)

p1 = df_error.index[df_error.reconstruction_error > 0.1  ].tolist()
print(len(p1))
print(len(p1)/8658)


dfOutliers = df[df.index.isin(outliers)]



	
# saving whole model
autoencoder.save('autoencoder_model.h5')
 






