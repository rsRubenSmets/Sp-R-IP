import pandas as pd
import numpy as np
import pytz
from datetime import datetime, timedelta
from entsoe import EntsoePandasClient
# Running this script requires an ENTSOE API Key. We did not include ours here, but reviewers interested in executing the script can find info on obtaining one here:
# https://thesmartinsights.com/how-to-query-data-from-the-entso-e-transparency-platform-using-python/.


# To get the last hour of the current day
def rounded_to_the_DA_FC_end():
    tz = pytz.timezone('Europe/Brussels')
    now = datetime.now(tz).replace(tzinfo=None)
    rounded = now - (now - datetime.min) % timedelta(minutes=1440) + timedelta(hours=23)
    return rounded


first = rounded_to_the_DA_FC_end() - timedelta(days=365 * 4)
last = rounded_to_the_DA_FC_end() - timedelta(hours=48)  # add a 48 hours lag to make sure all data is avaialble
last_fc = last + timedelta(hours=24)
start = pd.Timestamp(first, tz='Europe/Brussels')
end = pd.Timestamp(last, tz='Europe/Brussels')
end_fc = pd.Timestamp(last_fc, tz='Europe/Brussels')
client = EntsoePandasClient(api_key='ENTSOE_API_KEY HERE')
country_code = 'BE'  # Hungary

df_DA = client.query_day_ahead_prices(country_code, start=start, end=end)  # different than the forecasted load end
df_FR = client.query_load_forecast('FR', start=start, end=end_fc).resample('60T').mean()
df_NL = client.query_load_forecast('NL', start=start, end=end_fc).resample('60T').mean()

# In case of missing values, interpolate
df_DA.interpolate(method='linear', inplace=True)
df_FR.interpolate(method='linear', inplace=True)
df_NL.interpolate(method='linear', inplace=True)

df_DA.index = df_DA.index.tz_convert(None)
df_FR.index = df_FR.index.tz_convert(None)
df_NL.index = df_NL.index.tz_convert(None)

df_LOAD = client.query_load_forecast(country_code, start=start, end=end_fc).resample('60T').mean()
df_GEN = client.query_generation_forecast(country_code, start=start, end=end_fc).resample('60T').mean()

# In case of missing values, interpolate
df_LOAD.interpolate(method='linear', inplace=True)
df_GEN.interpolate(method='linear', inplace=True)

# Creating the dataframe for the model
X_df = pd.DataFrame()
X_df['ds'] = df_LOAD.index
X_df['unique_id'] = 'BE'
len_price = len(df_DA.values)
X_df["y"] = 0
X_df["y"][0:len_price] = df_DA.values  # The remaining yero values will be the forecasted ones
X_df['LOAD_FC'] = df_LOAD.values
X_df['GEN_FC'] = df_GEN.values
X_df['FR'] = df_FR.values
X_df['NL'] = df_NL.values
# Adding the temporal features
X_df['week_day'] = X_df['ds'].dt.day_of_week
X_df['hour-of-day'] = X_df['ds'].dt.hour

# For local purpose
X_df = X_df[:-24]  # Remove the last 24 hours of the dataframe

# Get names of X_df columns except 'ds' and 'y' and 'unique_id'
features = X_df.columns[~X_df.columns.isin(['ds', 'y', 'unique_id'])].to_numpy()

import logging

logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)
from neuralforecast import NeuralForecast

# Split X_df to test and training
split = pd.Timestamp(datetime(2023, 1, 1, 0, 0), tz='Europe/Brussels')
X_df_train = X_df[(X_df['ds'] < split)]
X_df_test = X_df[(X_df['ds'] >= split)]

horizon = 24
epochs = 100

from neuralforecast.models import NBEATSx
from neuralforecast.losses.pytorch import MQLoss

levels = [90]
models = [
    NBEATSx(h=horizon,
            input_size=5 * horizon,
            futr_exog_list=features,
            max_epochs=epochs,
            activation='ReLU',
            scaler_type='robust')
]

val_size = 24 * 10
test_size = 24 * 10

fcst = NeuralForecast(
    models=models,
    freq='H')

X_df_train.ds = X_df_train.ds.dt.tz_localize(None)

fcst_df = fcst.cross_validation(df=X_df_train, val_size=val_size,
                                test_size=test_size, n_windows=None)


def smape(A, F):
    return 100 / len(A) * np.sum(2 * np.abs(F - A) / (np.abs(A) + np.abs(F)))


def rmse(y, y_hat):
    return np.sqrt(np.mean(np.square(y - y_hat)))


##### TEST #####
X_df_test.ds = X_df_test.ds.dt.tz_localize(None)

for d in range(1, 120):
    if d == 1:
        forecasts = fcst.predict(futr_df=X_df_test.iloc[0:24])
        forecast_df = forecasts
        forecast_df.ds = X_df_test.ds.iloc[0:24].values
        print(forecast_df.ds.iloc[-1])
    if d == 2:
        start = 24
        end = start + 24
        forecasts = fcst.predict(futr_df=X_df_test.iloc[start:end])
        forecasts.ds = X_df_test.ds.iloc[start:end].values
        forecast_df = pd.concat([forecast_df, forecasts])
        print(forecast_df.ds.iloc[-1])
    if d > 2:
        start = (d - 1) * 24
        end = start + 24
        forecasts = fcst.predict(futr_df=X_df_test.iloc[0:end])
        forecasts.ds = X_df_test.ds.iloc[start:end].values
        forecast_df = pd.concat([forecast_df, forecasts])
        print(forecast_df.ds.iloc[-1])

# Chose a month of the year

df_slice_true = X_df_test[(X_df_test['ds'].dt.month == 3)]
df_slice_pred = forecast_df[(forecast_df['ds'].dt.month == 3)]

pred_Y = df_slice_pred["NBEATSx"]
true_Y = df_slice_true["y"]

smape(true_Y.values, pred_Y.values)
rmse(true_Y.values, pred_Y.values)

import matplotlib.pyplot as plt

# for the last 240 steps
plt.plot(df_slice_true.ds, true_Y, label='Actual', alpha=0.5)
plt.plot(df_slice_pred.ds, pred_Y, label='Forecasted', alpha=0.5)
# plt.plot(X_df_test.ds, X_df_test['DA_MEAN'], label='Forecasted', color='red', alpha=0.8)
plt.legend()
plt.show()