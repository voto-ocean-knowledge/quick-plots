import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
import datetime
from sklearn import datasets, linear_model
import logging
_log = logging.getLogger(__name__)


def battery_plots(combined_nav_file, out_dir):
    parts = combined_nav_file.parts
    glider = parts[-4][3:]
    mission = parts[-3][1:]
    title = f"SEA{glider.zfill(3)} M{mission}"
    ds = xr.open_dataset(combined_nav_file)
    df = ds.Voltage.to_dataframe()
    df = df[df.index > datetime.datetime(1980, 1, 1)]

    df_min = df.rolling(window=datetime.timedelta(hours=6)).min()
    df_mean = df.rolling(window=datetime.timedelta(hours=6)).mean()
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(df_mean.index, df_mean.Voltage, label="6 hour mean")
    ax.plot(df_min.index, df_min.Voltage, label="6 hour min")
    ax.legend()
    ax.grid()
    ax.set(ylabel="Voltage (v)", title=title)
    plt.xticks(rotation=45)
    plt.tight_layout()
    filename = f"{out_dir}/battery.png"
    _log.info(f'writing figure to {filename}')
    fig.savefig(filename, format='png', transparent=True)

    # Prediction plot
    df_3day = df[df.index > df.index.max() - datetime.timedelta(days=3)]
    regr = linear_model.LinearRegression()
    regr.fit(df.index.values.reshape(-1, 1), df['Voltage'].values.reshape(-1, 1))

    datetime_pred = pd.date_range(df_3day.index[0], df_3day.index[0] + datetime.timedelta(days=30), 31)
    y_forward = regr.predict(datetime_pred.values.astype(float).reshape(-1, 1))
    end = datetime_pred[y_forward[:, 0] > 23][-1]

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.scatter(df_3day.index, df_3day.Voltage, label="Voltage last 3 days")
    ax.plot(datetime_pred, y_forward, label="Linear prediction")
    ax.set(xlim=(df_3day.index[0], end), ylim=(22.9, df_3day.Voltage.max() + 0.1))
    ax.grid()
    ax.set(ylabel="Voltage (v)", title=title)
    plt.xticks(rotation=45)
    ax.legend()
    plt.tight_layout()
    filename = f"{out_dir}/battery_prediction.png"
    _log.info(f'writing figure to {filename}')
    fig.savefig(filename, format='png', transparent=True)

