# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 23:31:06 2019

@author: pavel
"""


import pandas as pd
import random


df = pd.read_csv('anomalienFrei.csv')
df['Label'] = 0
print(df.shape)


#for col in df:
#   print (col)
#   print(df[col].unique())
   
# Anomaliefreier Datensatz Erkenntnisse:   
# ActivityID 1 ist nie NULL
# ca. ActivityID 38 - 74 immer -1 also NULL
# Activity37 -1 oder 1
# Entweder Null(-1) oder zwischen 0 und 5


# 100 Anomalien erstellen:
# ActivityID1 nicht vorhanden
# ca. ActivityID 38 - 74 immer -1 also NULL, wie auch bei den Anomaliefreien
# Activity37 machen wir 1

for i in range(1000):
    df.loc[df.shape[0]] = random.randint(0, 100)



size =df.shape[0]-1

for i in range(1000):
    df.loc[size-i,'Label'] = 1

    

df.to_csv('labeledRand1000.csv')
    
    
    