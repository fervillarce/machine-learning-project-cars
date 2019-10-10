
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np

def negative_to_one(price):
    if price < 0:
        price = 1
    return price

def create_submission(y_pred, test):
    data = {'Id': test['Id'], 'price': y_pred.astype(int)}
    submission = pd.DataFrame(data)    
    submission['price'] = submission['price'].apply(negative_to_one)
    return submission

def load_csv(df, csv_name):
    folder = "../Outputs/"
    df.to_csv(folder+csv_name, index=None, header=True)

