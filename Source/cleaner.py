
# coding: utf-8

# In[ ]:


import pandas as pd

def remove_cols(train, test, cols):
    """
    OJO. No se puede hacer inplace=True, porque se eliminarían las columnas en el cars_test del main.
    No quiero eliminar las columnas en ese archivo, ya que más adelante cojo el Id de cars_test.
    """
    cols.extend(['Id', 'city', 'make', 'odometer', 'lat', 'long',
                 'county_fips','county_name', 'state_code', 'state_fips'])
    train = train.drop(cols, axis=1)
    test = test.drop(cols, axis=1)
    
    print("\n\n===== Resumen de eliminación de variables de cara al modelo =====\n")
    print("""
        Eliminamos:
        - las variables no significativas resultantes del ANOVA
        - 'state_code', 'state_fips' ya que son redundantes con state_name
        - variables con muchos nulls
        - variables categóricas con muchos valores únicos
        - 'lat', 'long' no son determinantes la una sin la otra\n
        
        En definitiva, se eliminan las siguientes variables:\n
        ['Id','paint_color', 'drive', 'size', 'fuel', 'manufacturer', 'type', 'condition', 'title_status',\n
        'city', 'make', 'odometer', 'lat', 'long', 'county_fips', 'county_name', 'state_code', 'state_fips']
        """)
    
    return train, test


def remove_nans(df):
    """
    Solo se usa para el train.
    """
    df = df.dropna()
    return df


def fill_nans(df):
    """
    Solo se usa para el test.
    No podemos usar dropna porque la subimission tiene que tener el mismo número de registros que X_test.
    """
    try:
        df = df.sort_values(by=['year', 'cylinders']).fillna(method='ffill')
        df = df.sort_values(by=['year', 'cylinders']).fillna(method='bfill')
    except:
        df = df.sort_values(by=['year']).fillna(method='ffill')
        df = df.sort_values(by=['year']).fillna(method='bfill')
    df.sort_index(inplace=True)
    return df