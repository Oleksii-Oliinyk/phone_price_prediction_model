import pickle
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)

# Import custom files
import model_best_hyperparameters
import columns

import warnings
warnings.simplefilter('ignore')

dtype_spec = {'primary_camera_front': str, 'num_front_cameras': str,'primary_camera_rear': str,}
# Read train data
ds = pd.read_csv("test_data.csv", dtype=dtype_spec)

ds_pred = ds.copy()

# Loading features
param_dict = pickle.load(open('param_dict.pickle', 'rb'))

# Missing data imputation function
def impute_na(df, variable, value):
    return df[variable].fillna(value)

for column in ds.columns:
    if ds[column].isnull().values.any():
        ds[column] = impute_na(ds, column, param_dict['impute_values'][column])


# encoding boolean data 
ds["has_ir_blaster"] = ds["has_ir_blaster"].map({True: 1, False: 0})
ds["has_nfc"] = ds["has_nfc"].map({True: 1, False: 0})
ds["has_5g"] = ds["has_5g"].map({True: 1, False: 0})    

# Categorial encoding 
for column in columns.cat_columns:
    ds[column+'_freq_encoded'] = ds[column].map(param_dict['map_dicts'][column])
    unknown_frequency = param_dict['map_dicts'][column].get('Unknown')
    if unknown_frequency is not None:
        ds[column+'_freq_encoded'] = ds[column+'_freq_encoded'].fillna(unknown_frequency)
    else:
        ds[column+'_freq_encoded'].fillna(0.0, inplace=True)

# Finding Anomalies

for column in columns.outlier_columns:
    upper_limit = param_dict['upper_lower_limits'][column+'_upper_limit']
    lower_limit = param_dict['upper_lower_limits'][column+'_lower_limit']
    ds.loc[ds[column] > upper_limit, column] = upper_limit
    ds.loc[ds[column] < lower_limit, column] = lower_limit

# Scaling 
scaler = MinMaxScaler()

scaler.fit(ds[columns.scaling_columns])

train_scaled = scaler.transform(ds[columns.scaling_columns])
ds[columns.scaling_columns] = train_scaled

# Define target and features columns
X = ds[columns.X_columns]

# Loading of the model
rf = pickle.load(open('finalized_model.sav', 'rb'))

# Prediction
y_pred = rf.predict(X)

# Writing binary and categorial prediction
ds_pred['price_pred'] = y_pred

ds_pred.to_csv("prediction_results.csv", index=False)

print("Done!")