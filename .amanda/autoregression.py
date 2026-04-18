# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 17:55:41 2026

@author: akayj
"""

import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.ar_model import AutoReg

path = r"D:\Senior Project\Cost Data\panel_costs.csv"
df = pd.read_csv(path)

#select school
school_id = 100654
school_df = df[df["unitid"] == school_id].copy()

# sort by chronologically
school_df = school_df.sort_values("Year")
school_df = school_df.reset_index(drop=True)


is_series = school_df["ISPrice"]
oos_series = school_df["OOSPrice"]

#AR models, lag=3 (3 year eval for furutr forecast)
is_model = AutoReg(is_series, lags=4, old_names=False).fit()
oos_model = AutoReg(oos_series, lags=4, old_names=False).fit()

#forecast undergrad tuition (year=4)
is_forecast = is_model.predict(start=len(is_series), end=len(is_series) + 3)
oos_forecast = oos_model.predict(start=len(oos_series), end=len(oos_series) + 3)

last_year = school_df["Year"].max()
future_years = [last_year + 1, last_year + 2, last_year + 3, last_year + 4]

#forecast table, round
forecast_df = pd.DataFrame({
    "Year": future_years,
    "Forecast_ISPrice": is_forecast.values,
    "Forecast_OOSPrice": oos_forecast.values
})

forecast_df["Forecast_ISPrice"] = forecast_df["Forecast_ISPrice"].round(2)
forecast_df["Forecast_OOSPrice"] = forecast_df["Forecast_OOSPrice"].round(2)

print("Undergraduate Price Forecast:")
print(forecast_df)

#plot in state
plt.figure()
plt.plot(school_df["Year"], school_df["ISPrice"], marker="o", label="Actual ISPrice")
plt.plot(forecast_df["Year"], forecast_df["Forecast_ISPrice"], marker="o", linestyle="--", label="Forecast ISPrice")
plt.title("In-State Tuition Forecast for UNITID " + str(school_id))
plt.xlabel("Year")
plt.ylabel("ISPrice")
plt.legend()
plt.grid(True)
plt.show()

#plot out of state
plt.figure()
plt.plot(school_df["Year"], school_df["OOSPrice"], marker="o", label="Actual OOSPrice")
plt.plot(forecast_df["Year"], forecast_df["Forecast_OOSPrice"], marker="o", linestyle="--", label="Forecast OOSPrice")
plt.title("Out-of-State Tuition Forecast for UNITID " + str(school_id))
plt.xlabel("Year")
plt.ylabel("OOSPrice")
plt.legend()
plt.grid(True)
plt.show()

#eval/test AR model

is_pred = is_model.predict(start=3, end=len(is_series)-1)
is_actual = is_series[3:]


oos_pred = oos_model.predict(start=3, end=len(oos_series)-1)
oos_actual = oos_series[3:]

is_error = is_actual - is_pred

is_mad = abs(is_error).mean()
is_mse = (is_error**2).mean()
is_rmse = (is_mse)**0.5
is_mape = (abs(is_error / is_actual)).mean() * 100


oos_error = oos_actual - oos_pred

oos_mad = abs(oos_error).mean()
oos_mse = (oos_error**2).mean()
oos_rmse = (oos_mse)**0.5
oos_mape = (abs(oos_error / oos_actual)).mean() * 100

#print("\nISPrice Performance")
#print("MAD:", round(is_mad,2))
#print("MSE:", round(is_mse,2))
#print("RMSE:", round(is_rmse,2))
#print("MAPE:", round(is_mape,2), "%")

#print("\nOOSPrice Performance")
#print("MAD:", round(oos_mad,2))
#print("MSE:", round(oos_mse,2))
#print("RMSE:", round(oos_rmse,2))
#print("MAPE:", round(oos_mape,2), "%")