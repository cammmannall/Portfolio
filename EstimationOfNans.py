# -*- coding: utf-8 -*-
"""
Created on Sun Mar  9 19:31:21 2025

@author: emreg
"""

import yfinance as yf
import numpy as np
import datetime
import pandas as pd
#%%
def data_with_nans(ticker, today=datetime.datetime.now()):
    end = today.replace(hour=0, minute=0, second=0, microsecond=0)
    stock = yf.download(ticker, start='1950-01-01', end=today).dropna()
    full_dates = pd.date_range(stock.index.min(), end, freq='D')
    stock = stock.reindex(full_dates)
    stock.columns = [col[0] for col in stock.columns]
    return stock

def estimate_nans(data, column):
    nan_position = data[column].isna()
    nonnan_position = ~nan_position
    y = data[column][nonnan_position].values
    x = np.where(nonnan_position)[0]
    degree = len(y) // 2
    while degree > 0:
        try:
            coefficients = np.polyfit(x, y, degree)
            poly = np.poly1d(coefficients)
            data.loc[nan_position, column] = poly(np.where(nan_position)[0])
            break
        except Exception:
            degree -= 100
            degree = max(degree, 1)
    if degree <= 0:
        while degree < 100:
            try:
                degree = 1
                coefficients = np.polyfit(nonnan_position, y, degree)
                poly = np.poly1d(coefficients)
                for i in nan_position.index[nan_position]:
                    data.loc[i, column] = poly(i)
                degree += 1
            except Exception:
                break
    return data