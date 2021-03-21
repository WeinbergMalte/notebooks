# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.8.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # SHAP minimal example
# Plot customized SHAP feature importances and summary-plot.

# +
# %load_ext lab_black
# %load_ext autoreload
# %autoreload 2

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import shap
from xgboost import XGBRegressor

from nb.plotting import create_shap_plot
# -

# # Load sample data: Boston Housing

# +
from sklearn.datasets import load_boston

boston = load_boston()

ffrmt = "feature_{:02d}".format

X = pd.DataFrame(boston.data)
X = X.rename(columns={i: ffrmt(i + 1) for i in range(X.shape[0])})

n_cols = X.shape[1]
cols = X.columns.tolist()
y = pd.Series(boston.target)

X_train, X_test, y_train, y_test = train_test_split(X, y)
# -

# # Fit dummy model

model = XGBRegressor()
model.fit(X_train, y_train)

# # SHAP Plots

# +
MAX_SHAP_SAMPLE = 5000

n_shap = min([MAX_SHAP_SAMPLE, X_train.shape[0]])
df_features = X_train.sample(n_shap)

shap_idx = df_features.index
shap_vals = shap.TreeExplainer(model).shap_values(df_features)
df_shap = pd.DataFrame(shap_vals, columns=X_train.columns)
# -

create_shap_plot(df_shap, df_features)

# # Example without feature data

create_shap_plot(df_shap)


