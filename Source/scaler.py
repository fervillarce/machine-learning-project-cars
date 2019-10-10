
# coding: utf-8

# In[ ]:


import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

def standard_scaler(X):
    scaler = StandardScaler()
    standardized_X = scaler.fit_transform(X)
    print("Se ha rescalado con StandardScaler")
    return standardized_X
    
def minmax_scaler(X):
    scaler = MinMaxScaler()
    scaled_X = scaler.fit_transform(X)
    print("Se ha rescalado con MinMaxScaler")
    return scaled_X

