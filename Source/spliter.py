
# coding: utf-8

# In[ ]:


import pandas as pd

def split_train(train):
    X_train = train.drop(['price'], axis=1)
    y_train = train['price']
    return X_train, y_train

