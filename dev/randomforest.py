import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
import time
from IPython.display import display

import os.path

from preprocessing import PreProcessor

p = PreProcessor()

X_train_processed, y_train = p.loadData('./data/train.csv')
y_train = y_train.values.ravel()

start = time.time()

gb_classifier = RandomForestRegressor(n_estimators=40, max_depth=10)
gb_classifier.fit(X_train_processed, y_train)
y_pred = gb_classifier.predict(X_train_processed)

end = time.time()
print('Fitting & Predicting time:', end - start)

dt_rmse = np.sqrt(mean_squared_error(y_train, y_pred))

print(dt_rmse)

scores = cross_val_score(gb_classifier, X_train_processed, y_train, scoring='neg_mean_squared_error', cv=10)
rmse_scores = np.sqrt(-scores)
print('Mean:', rmse_scores.mean())
print('std:', rmse_scores.std())

print(y_pred[np.where(np.logical_or(y_pred < 0, y_pred > 1))].size)