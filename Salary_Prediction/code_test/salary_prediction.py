# coding: utf-8
import matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor

data_all = pd.read_csv("data_pg_v2.0.csv")
X_all = data_all.iloc[:,4:]
Y_salary = data_all['salary']

min_max_scaler = preprocessing.MinMaxScaler(feature_range = (0,1))
X_data = min_max_scaler.fit_transform(X_all)
Y_data = min_max_scaler.fit_transform(Y_salary[:,np.newaxis])

forest = RandomForestRegressor(n_estimators=1000, random_state=1, n_jobs=-1, min_samples_leaf = 10)
forest.fit(X_data,Y_data.flatten())

'''
# This part is the code for entering new data and get the prediction
input = np.array([55,1727,11.3,1.7,1.8,0.5,1.2,204,532,115,323,89,209,45,60,170,192,59,83,568])
X = min_max_scaler.fit_transform(input.reshape(-1, 1))
X_input = X.reshape(1,20)
Y_predict = forest.predict(X_input)

Y_max = np.max(Y_salary)
Y_min = np.min(Y_salary)
Y = Y_predict * (Y_max-Y_min) + Y_min
Y
'''