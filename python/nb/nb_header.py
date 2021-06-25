%load_ext lab_black
%load_ext autoreload
%autoreload 2
%matplotlib inline
%config InlineBackend.figure_format = 'retina'
%load_ext google.cloud.bigquery

import warnings

warnings.simplefilter("ignore")

import os
import sys
import math
import pathlib
import numpy as np
import re
import json
import datetime as dt


import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt
import shap
from cycler import cycler

plt.style.use("seaborn-colorblind")
PLT_COLOR = plt.rcParams["axes.prop_cycle"].by_key()["color"]
PLT_LINES = ["-", "--", ":", "-."]
PLT_CYCLER = cycler(color=PLT_COLOR * 2) + cycler(linestyle=PLT_LINES * 3)

import pandas as pd

pd.set_option("display.max_rows", 120)
pd.set_option("display.max_columns", 120)
pd.plotting.register_matplotlib_converters()

import logging

logging.basicConfig(level=logging.INFO, stream=sys.stdout)

import xgboost as xgb
import lightgbm as lgbm

from sklearn.base import TransformerMixin, BaseEstimator, clone
from sklearn.pipeline import Pipeline, make_pipeline, FeatureUnion
from sklearn import set_config

set_config(display="diagram")
