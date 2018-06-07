import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.base import BaseEstimator, TransformerMixin # Para definição de transformadores personalizados
from sklearn.preprocessing import OneHotEncoder, Imputer, FunctionTransformer
# from category_encoders import OrdinalEncoder
from future_encoders import OrdinalEncoder
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

from sklearn.tree import DecisionTreeRegressor

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
import time
from IPython.display import display

import os.path

from preprocessing import PreProcessor

p = PreProcessor()

X_train_processed, y_train = p.loadData('./data/train.csv')

dt_regressor = DecisionTreeRegressor(max_depth=10, random_state=1)
dt_regressor.fit(X_train_processed, y_train)
y_pred = dt_regressor.predict(X_train_processed)
dt_rmse = np.sqrt(mean_squared_error(y_train, y_pred))

print(dt_rmse)

scores = cross_val_score(dt_regressor, X_train_processed, y_train, scoring='neg_mean_squared_error', cv=10)
rmse_scores = np.sqrt(-scores)
print('Mean:', rmse_scores.mean())
print('std:', rmse_scores.std())

print(y_pred[np.where(np.logical_or(y_pred < 0, y_pred > 1))].size)
