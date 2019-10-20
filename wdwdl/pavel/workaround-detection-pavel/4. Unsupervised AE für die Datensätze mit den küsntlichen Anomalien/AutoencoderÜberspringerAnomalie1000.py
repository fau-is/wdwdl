# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 23:13:19 2019

@author: pavel
"""

# -*- coding: utf-8 -*-
"""
@author: Pavel Akram
"""

import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_row', None)
import matplotlib.pyplot as plt
plt.rcdefaults()
from pylab import rcParams
import seaborn as sns; sns.set()
import datetime
import plotly
#import plotly.plotly as py
import plotly.offline as pyo
import plotly.figure_factory as ff
from   plotly.tools import FigureFactory as FF, make_subplots
from plotly.graph_objs import Data, Figure
from   plotly.graph_objs import *
from   plotly import tools
from   plotly.offline import download_plotlyjs, init_notebook_mode, iplot
import tensorflow as tf
import keras

from keras.models import Model, load_model
from keras.layers import Input, Dense
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras import regularizers
from sklearn.preprocessing import  StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from IPython.display import display, Math, Latex




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
df = pd.read_csv("labeledÜberspringer1000.csv")
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

test_x=df






test_yl = test_x['Label'] #save the class column for the test set
test_y = test_x['ActivityID1']
test_x = test_x.drop(['Label'], axis=1) #drop the class column
test_x = test_x.drop(['ActivityID1'], axis=1)
test_x = test_x.drop(['CaseID'], axis=1)
test_x = test_x.drop([df.columns[0]], axis=1)





X_test  = test_x.to_numpy()
y_test = test_y
y_testl = test_yl




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




# loading whole model

autoencoder = load_model('autoencoder_model.h5')




predictions = autoencoder.predict(X_test)

mse = np.mean(np.power(X_test - predictions, 2), axis=1)
df_error = pd.DataFrame({'reconstruction_error': mse, 'ActivityID1': y_test},
                        index=y_test.index)

#fig = line_plot(df_error, 'reconstruction_error')
#pyo.plot(fig)

outliers = df_error.index[df_error.reconstruction_error > 0.1].tolist()

print(outliers)

dfOutliers = df[df.index.isin(outliers)]


y_testl= y_testl.to_frame()

##xrr =y_test

y_testl = y_testl[y_testl.Label ==1 ]

y_testl=  y_testl.index.tolist()

##xrr= xrr.index.tolist()
#y_test= y_test.drop(['Label'], axis=1)
#y_test=

dfOutliersbekommen= dfOutliers[dfOutliers.index.isin(y_testl)]

precision=  dfOutliersbekommen.shape[0] / len(y_testl) 
print("-----")
print("Anomalien: " )
print (len(y_testl))
print("Gefunden")
print( dfOutliersbekommen.shape[0] )
print ("wie viel Prozent wurden gefunden:")
print (precision)

print('Normale die als Anomalien gefunden wurden:')
print(len(outliers)-dfOutliersbekommen.shape[0])

print("-----")
