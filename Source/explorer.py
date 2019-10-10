
# coding: utf-8

# In[ ]:


import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
import seaborn as sns
import matplotlib.pyplot as plt


def get_anova_conclusions(train):
    cols = ['year', 'manufacturer', 'condition', 'cylinders',
       'fuel', 'title_status', 'transmission', 'drive', 'size',
       'type', 'paint_color',
       'state_fips', 'state_code', 'state_name', 'weather']

    anovas = []
    print("===== Calculando ANOVAs =====\n")
    for col in cols:
        formula = "price ~ C(" + col + ")"
        print(formula)
        model = ols(formula, data=train).fit()
        anova_table = sm.stats.anova_lm(model, typ=2)
        anovas.append([col, anova_table["PR(>F)"].iloc[0]])

    print()
    print("\n===== Fin de los cálculos. Resultados =====\n")
    anova_results = pd.DataFrame(anovas, columns=["variable", "p-value"])

    insignificant_cols = list(anova_results[anova_results["p-value"] > 0.05]["variable"])
    
    print(anova_results.sort_values(by='p-value'))
    
    print("""
    \n\n===== Conclusiones de los ANOVA =====\n
    - year, cylinders, transmission, weather, state_name, state_fips, state_code: Las p_values son menores que 0.05. Rechazamos la hipótesis nula, por lo que hay diferencia significativa entre las medias de precio para cada grupo de valores de las variables . Esto significa que, en principio, deberíamos tener en cuenta estas variables para entrenar el modelo.\n
    - paint_color, drive, size, fuel, manufacturer, type, condition, title_status: Las p_values son mayores que 0.05. No hay diferencia significativa entre las medias. Si no hay diferencia, estas variables no aportan mucho para entrenar el modelo.\n
    - city, make, odometer, lat, long, county_fips, county_name: no hemos podido hacer las anovas de estos campos porque tienen muchos valores únicos y peta. Además, tiene más sentido hacerlo sobre campos que tienen un número reducido de categorías, ya sean nominales u ordinales.
    """)
    return insignificant_cols


def get_nulls_conclusions(df):
    top_nulls = df.isnull().sum().sort_values(ascending=False)
    print("\n\n===== Estos son los nulls de train =====\n")
    print(top_nulls)

def get_correlation_conclusions(df):
    """
    Sí hice alguna observación, pero no la he usado en el main.
    """
    print("\n\n===== Matriz de correlación =====\n")
    correlation_matrix = df.corr()
    sns.heatmap(correlation_matrix, annot=True)
    plt.savefig('../Outputs/correlation_matrix.png') # No funciona
    print("\nNo hay variables con alto grado de correlación (>0.9), por lo que no podemos eliminar variables por correlación.")
    return plt.show()

def pass_to_bins(train, variable):
    """
    Convierte el dataframe de train en un dataframe de dos columnas: una variable escogida en bins, y price.

    Arguments:
    train: dataframe original de train
    variable: una de las variables de train. Está pensado para aquellas variables con muchos valores únicos

    Return:
    df: dataframe con dos columnas. Variable dividida en 10 bins, y price.
    """
    df = df[[train, 'price']].copy()
    df = df.dropna()
    df[variable+'_bins'] = pd.qcut(df[variable], q=10)
    return df
          
def plot_price_outliers(df):
    """
    Se obtiene un boxplot de price.
    """
    print("\n\n===== Boxplot de la variable price =====")
    df.boxplot(column='price', figsize=(5, 10))
    plt.savefig('../Outputs/price_boxplot.png') # No funciona
    return plt.show()