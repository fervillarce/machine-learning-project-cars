
# coding: utf-8

# In[ ]:


import pandas as pd

def extract_csv(csv_name):
    folder = "../Inputs/"
    df = pd.read_csv(folder+csv_name)
    return df

