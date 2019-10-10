
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
from sklearn.decomposition import PCA


def data_to_pca(X_train, X_test):
    rows = X_train.shape[0]
    X = np.concatenate((X_train, X_test), axis=0)
    pca = PCA(n_components=2).fit(X)
    print("\n===== PCA finalizado. Varianza explicada acumulada =====\n", np.cumsum(pca.explained_variance_ratio_))
    # pca.singular_values_
    pca = pca.transform(X)
    X_train_pc, X_test_pc = np.split(pca, [rows], axis=0)
    return X_train_pc, X_test_pc

