# -*- coding: utf-8 -*-
"""
Created on Mon Mar 10 02:30:36 2025

@author: emreg
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from dateutil.relativedelta import relativedelta
import EstimationOfNans as Estimation
import datetime

def create_dataset(data, time_step=6):
    X, y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i:(i + time_step), 0])
        y.append(data[i + time_step, 0])
    return np.array(X), np.array(y)

def build_model(dropout):
    model = Sequential([
        LSTM(units=50, return_sequences=True, input_shape=(6, 1)),
        Dropout(dropout),
        LSTM(units=50, return_sequences=False),
        Dropout(dropout),
        Dense(units=1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def predict_with_mc_dropout(model, X, n_iter):
    predictions = np.zeros((n_iter, X.shape[0]))
    for i in range(n_iter):
        predictions[i] = model(X, training=True).numpy().flatten()
    return predictions

def predict_future(model, last_window, scaler, months, n_iter):
    future_predictions = []
    future_intervals = []
    input_seq = last_window.copy()

    for _ in range(months):
        mc_preds = predict_with_mc_dropout(model, input_seq.reshape(1, 6, 1), n_iter)
        mean_pred = mc_preds.mean(axis=0)[0]
        lower_pred = np.percentile(mc_preds, 2.5, axis=0)[0]
        upper_pred = np.percentile(mc_preds, 97.5, axis=0)[0]

        future_predictions.append(mean_pred)
        future_intervals.append((lower_pred, upper_pred))

        input_seq = np.roll(input_seq, -1)
        input_seq[-1] = mean_pred

    return (
        scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1)),
        scaler.inverse_transform(np.array([x[0] for x in future_intervals]).reshape(-1, 1)),
        scaler.inverse_transform(np.array([x[1] for x in future_intervals]).reshape(-1, 1))
    )



def interpolate_final_prediction(future_dates, future_pred, future_lower, future_upper, investment_end_date):
    """Interpolates last prediction and confidence intervals if the last date is before investment_end_date"""
    last_two_dates = future_dates[-2:]
    x = np.array([(date - last_two_dates[0]).days for date in last_two_dates])

    target_x = (investment_end_date - last_two_dates[0]).days

    y_pred = np.array(future_pred[-2:]).flatten()
    interpolated_pred = np.interp(target_x, x, y_pred)

    y_lower = np.array(future_lower[-2:]).flatten()
    interpolated_lower = np.interp(target_x, x, y_lower)

    y_upper = np.array(future_upper[-2:]).flatten()
    interpolated_upper = np.interp(target_x, x, y_upper)

    return interpolated_pred, interpolated_lower, interpolated_upper

def predict_stock_prices_gui(stock_tickers, investment_end_date, target_day=None, drop=0.4, n_iter=10):
    predictions_dict = {}
    final_conf_intervals = []
    future_dates = None

    investment_end_date = pd.to_datetime(investment_end_date)

    for key, ticker in stock_tickers.items():
        print(f'Processing {ticker}...')

        df = Estimation.estimate_nans(Estimation.data_with_nans(ticker), 'Close')

        if target_day is None:
            target_day = datetime.datetime.now().day

        df = df[df.index.day == target_day]
        df = df[['Close']]
        df.dropna(inplace=True)

        investment_start_date = df.index[-1]
        months = (investment_end_date.year - investment_start_date.year) * 12 + (investment_end_date.month - investment_start_date.month)


        scaler = MinMaxScaler(feature_range=(0, 1))
        data_scaled = scaler.fit_transform(df)

        time_step = 6
        X, y = create_dataset(data_scaled, time_step)
        X = X.reshape(X.shape[0], X.shape[1], 1)

        train_size = int(len(X) * 0.8)
        X_train, y_train = X[:train_size], y[:train_size]

        model = build_model(drop)
        model.fit(X_train, y_train, epochs=50, batch_size=132, verbose=0)

        future_pred, future_lower, future_upper = predict_future(model, X[-1], scaler, months, n_iter)

        if future_dates is None:
            future_dates = [investment_start_date + relativedelta(months=i) for i in range(1, months + 1)]

        if future_dates[-1] > investment_end_date:
            interpolated_pred, interpolated_lower, interpolated_upper = interpolate_final_prediction(
                future_dates, future_pred, future_lower, future_upper, investment_end_date
            )

            future_dates[-1] = investment_end_date
            future_pred[-1] =  interpolated_pred
            future_lower[-1] =  interpolated_lower
            future_upper[-1] = interpolated_upper

        predictions_dict[key] = future_pred.flatten()
        last_pred = future_pred[-1]
        last_lower = future_lower[-1]
        last_upper = future_upper[-1]
        last_date = future_dates[-1]

        final_conf_intervals.append({
            'Ticker': key,
            'Final Prediction Date': last_date.strftime('%Y-%m-%d'),
            'Final Predicted Price': last_pred,
            '95% Lower Bound': last_lower,
            '95% Upper Bound': last_upper
        })

        print(f"Finished processing {ticker}")

    predictions_df = pd.DataFrame(predictions_dict, index=future_dates)
    final_conf_intervals_df = pd.DataFrame(final_conf_intervals)


    return predictions_df, final_conf_intervals_df