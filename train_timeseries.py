from .preprocess import preprocess_main
from tqdm.contrib import itertools
from sklearn.metrics import mean_squared_error
from math import sqrt

import pandas as pd
import numpy as np
import statsmodels.api as sm


def train_model(train, test):
    all_result = []
    p_values = range(0, 4)
    d_values = range(0, 4)
    q_values = range(0, 4)
    P_values = range(0, 4)
    D_values = range(0, 4)
    Q_values = range(0, 4)
    s_values = [24]  # Seasonal period, adjust as needed

    best_rmse = float("inf")
    best_order = None
    best_seasonal_order = None

    for order in itertools.product(p_values, d_values, q_values):
        for seasonal_order in itertools.product(P_values, D_values, Q_values, s_values):
            try:
                model = sm.tsa.SARIMAX(
                    train, order=order, seasonal_order=seasonal_order, exog=None
                )
                results = model.fit()
                forecast = results.get_forecast(steps=len(test))
                predicted = forecast.predicted_mean
                rmse = sqrt(mean_squared_error(test, predicted))
                temp_dict = {
                    "rmse": rmse,
                    "order": order,
                    "seasonal_order": seasonal_order,
                }
                all_result.append(temp_dict)
                if rmse < best_rmse:
                    best_rmse = rmse
                    best_order = order
                    best_model = results
                    best_seasonal_order = seasonal_order
            except Exception as e:
                print(e)

    print(f"Best Order: {best_order}")
    print(f"Best Seasonal Order: {best_seasonal_order}")
    print(f"Best RMSE: {best_rmse}")
    best_model.save("best_model.np")

    return best_rmse, best_order, best_model, best_seasonal_order


def train_main():
    train, test = preprocess_main()
    train_model(train, test)
