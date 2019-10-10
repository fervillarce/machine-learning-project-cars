
# coding: utf-8

# In[ ]:


import pandas as pd

def cylinders_to_number(value):
    if value == "other":
        result = 0
    else:
        result = int(value.split()[0])
    return result

def transform_cylinders(df):
    df['cylinders'] = df['cylinders'].apply(cylinders_to_number)
    return df

def transform_manufacturer(df):
    cars['manufacturer'].replace({'alfa':'alfa-romeo','aston':'aston-martin', 'chev':'chevrolet', 'chevy':'chevrolet',
                              'harley':'harley-davidson', 'infiniti':'infinity', 'land rover':'landrover',
                              'rover':'landrover','mercedes-benz':'mercedes', 'vw':'volkswagen'}, inplace=True)
    return df

def one_hot_encode(df):
    df = pd.get_dummies(df, drop_first=True)
    return df

def unify_dimension(df1, df2, df2_dummies):
    """
    Hay valores de state_name que están en test pero no en train.
    Esto supone que al hacer get_dummies, test genera más columnas que en train.
    Tenemos que unificar la dimensión para poder entrenar el modelo.
    Solo ocurre para el campo state_name. Si pasara para más, se podría hacer una función
    que solucionara todos los casos, comparando los nunique de cada columna entre ambos dataframes.
    """
    missing_states = []
    for state in df2['state_name'].unique():
        if state not in df1['state_name'].unique():
            missing_states.append('state_name_' + state)
    df2_dummies.drop(missing_states, axis=1, inplace=True)
    return df2_dummies

