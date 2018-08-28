import pandas as pd
import numpy as np
from scipy import stats
from scipy import stats, special
from sklearn import model_selection, metrics, linear_model, datasets, feature_selection
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler



df_train = pd.read_csv('..\Data\TrainingData.csv', index_col =0)
df_test = pd.read_csv('..\Data\TestingData.csv',index_col =0)

## Model Training Data

X = df_train.drop(columns='total_runs')
y = df_train['total_runs']

scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)
reg = linear_model.LinearRegression()
reg.fit(
     X,
     y
);

# Predict Total Runs and Error Analysis
print('Training Score : ',reg.score(X, df_train['total_runs']))


X_test = df_test.drop(columns="total_runs")
X_test = scaler.transform(X_test)
predicted_total_runs = reg.predict(X_test)
df_test["Predicted Runs"] = predicted_total_runs
df_test["Error"] = abs(df_test["Predicted Runs"] - df_test["total_runs"])
df_test["Perc_Err"] = (df_test["Error"])/(df_test["total_runs"])
mean_err_perc = (df_test["Perc_Err"]).mean()
mean_err = (df_test["Error"]).mean()
print(mean_err_perc)
print('Testing Score :' ,reg.score(X_test, df_test['total_runs']))

print('No Scaling!')



X = df_train.drop(columns='total_runs')
y = df_train['total_runs']

#scaler = StandardScaler()
#scaler.fit(X)
#X = scaler.transform(X)
reg = linear_model.LinearRegression()
reg.fit(
     X,
     y
);

# Predict Total Runs and Error Analysis
print('Training Score : ',reg.score(X, df_train['total_runs']))


X_test = df_test.drop(columns=["total_runs", 'Error', 'Predicted Runs', 'Perc_Err'])
#X_test = scaler.transform(X_test)
#predicted_total_runs = reg.predict(X_test)
#df_test["Predicted Runs"] = predicted_total_runs
#df_test["Error"] = abs(df_test["Predicted Runs"] - df_test["total_runs"])
#df_test["Perc_Err"] = (df_test["Error"])/(df_test["total_runs"])
#mean_err_perc = (df_test["Perc_Err"]).mean()
#mean_err = (df_test["Error"]).mean()
print(mean_err_perc)
print('Testing Score :' ,reg.score(X_test, df_test['total_runs']))

