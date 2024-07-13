"""
Python library which is a one-stop-shop for time series analysis forecasting, detecting patterns, multi-variate forecasting, etc.

Kats is released by Facebook’s infrastructure Data Science team and is a result of development and research efforts in the past two years.
https://engineering.fb.com/2021/06/21/open-source/kats/

!pip install kats
"""

import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt
import warnings
from kats.consts
from kats.models.sarima import SARIMAModel, SARIMAParams
from kats.models.holtwinters import HoltWintersParams, HoltWintersModel
warnings.simplefilter(action='ignore')

df = pd.read_csv('sales.csv')

df.columns = ["time", "value"]
ts = TimeSeriesData(df)

params = SARIMAParams(p=2, d=1, q=1, trend = 'ct', seasonal_order=(1,0,1,12))
sarima = SARIMAModel(data=ts, params=params)
sarima.fit()
forecast = sarima.predict(steps=10)
sarima.plot()

params = HoltWintersParams(trend="add",seasonal="mul",seasonal_periods=12)

holt_model = HoltWintersModel(data=ts,params=params)
holt_model.fit()
forecast = holt_model.predict(steps=30, alpha = 0.1)
holt_model.plot()
