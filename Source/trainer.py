
# coding: utf-8

# In[ ]:


import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor


def linear_regression(X_train, y_train, X_test):
    regression = LinearRegression().fit(X_train, y_train)
    y_pred = regression.predict(X_test)
    name = "linreg"
    print("Se ha entrenado el modelo LinearRegression")
    return y_pred, name

def random_forest(X_train, y_train, X_test):
    rand_forest = RandomForestRegressor(max_depth=2, random_state=0, n_estimators=100).fit(X_train, y_train)  
    y_pred = rand_forest.predict(X_test)
    name = "randfor"
    print("Se ha entrenado el modelo RandomForestRegressor")
    return y_pred, name

def decision_tree(X_train, y_train, X_test):
    regressor = DecisionTreeRegressor(random_state=0)
    # cross_val_score(regressor, X_train, y_train, cv=10)
    regressor.fit(X_train, y_train)
    y_pred = regressor.predict(X_test)
    name = "dectree"
    print("Se ha entrenado el modelo DecisionTreeRegressor")
    return y_pred, name

def kneighbors(X_train, y_train, X_test):
    neigh = KNeighborsRegressor(n_neighbors=2).fit(X_train, y_train) 
    y_pred = neigh.predict(X_test)
    name = "kneigh"
    print("Se ha entrenado el modelo KNeighborsRegressor")
    return y_pred, name

def gradient_boosting(X_train, y_train, X_test):
    gradient = GradientBoostingRegressor().fit(X_train, y_train) 
    y_pred = gradient.predict(X_test)
    name = "gradboost"
    print("Se ha entrenado el modelo GradientBoostingRegressor")
    return y_pred, name

