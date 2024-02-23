import pandas as pd
import numpy as np
import polars as pl
import matplotlib.pyplot as plt
import datetime
from sklearn import linear_model
import logging
_log = logging.getLogger(__name__)


def battery_plots(combined_nav_file, out_dir):
    parts = combined_nav_file.parts
    glider = parts[-4][3:]
    mission = parts[-3][1:]
    title = f"SEA{glider.zfill(3)} M{mission}"
    df_polar = pl.read_parquet(combined_nav_file)
    df = pd.read_parquet(combined_nav_file)
    df.index = df_polar.select("time").to_numpy()[:, 0]
    df = df[["Voltage"]]
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
    df = df.dropna()
    df_3day = df[df.index > df.index.max() - datetime.timedelta(days=3)]
    regr = linear_model.LinearRegression()
    regr.fit(df_3day.index.values.reshape(-1, 1), df_3day['Voltage'].values.reshape(-1, 1))
    # Create time array of one point every hour for 60 days starting three days ago
    datetime_pred = pd.date_range(df_3day.index[0], df_3day.index[0] + datetime.timedelta(days=60), 60*24)
    y_forward = regr.predict(datetime_pred.values.astype(float).reshape(-1, 1))
    v_per_ns = regr.coef_[0][0]
    v_per_day_3 = v_per_ns * 24 * 60 * 60 * 1e9

    df_5day = df[df.index > df.index.max() - datetime.timedelta(days=5)]
    regr = linear_model.LinearRegression()
    regr.fit(df_5day.index.values.reshape(-1, 1), df_5day['Voltage'].values.reshape(-1, 1))
    # Create time array of one point every hour for 60 days starting three days ago
    datetime_pred_5 = pd.date_range(df_5day.index[0], df_5day.index[0] + datetime.timedelta(days=60), 60*24)
    y_forward_5 = regr.predict(datetime_pred_5.values.astype(float).reshape(-1, 1))
    end = datetime_pred_5[y_forward_5[:, 0] > 23][-1]

    recover = datetime_pred[y_forward[:, 0] > 24][-1]
    v_per_ns = regr.coef_[0][0]
    v_per_day = v_per_ns * 24 * 60 * 60 * 1e9
    fig, ax = plt.subplots(figsize=(12, 8))
    if df_mean.Voltage.min() > 28:
        _log.info("voltage too high to make prediction plot")
        ax.text(0.3, 0.5, "Voltage too high \nfor battery prediction", fontsize="24")
        ax.axis("off")
    else:
        ax.scatter(df_5day.index, df_5day.Voltage, label="Voltage last 5 days", s=3)
        ax.plot(datetime_pred, y_forward, label="last 3 days prediction")
        ax.set(xlim=(df_5day.index[0], end), ylim=(22.9, df_5day.Voltage.max() + 0.1))
        loss = np.round(np.abs(v_per_day), 2)
        loss_3 = np.round(np.abs(v_per_day_3), 2)
        recover_3 = datetime_pred[np.argmin(np.abs(24 - y_forward))]
        ax.axvline(recover_3, color="red")
        ax.plot(datetime_pred_5, y_forward_5, label="last 5 days prediction")
        recover_5 = datetime_pred_5[np.argmin(np.abs(24 - y_forward_5))]
        if recover_3 > recover_5:
            r_string = f"{str(recover_5)[5:10]} - {str(recover_3)[5:10]}"
        else:
            r_string = f"{str(recover_3)[5:10]} - {str(recover_5)[5:10]}"
        ax.axhline(24, color="red")
        ax.axvline(recover, color="red")
        ax.axvspan(recover_3, recover, alpha=0.1, color='red')
        v_string = f"Voltage: {df['Voltage'].values[-1]} V\n{loss}-{loss_3} V/day\nrecover {r_string}"
        ax.text(0.05, 0.08, v_string, transform=ax.transAxes)
        ax.grid()
        ax.set(ylabel="Voltage (v)", title=title)
        plt.xticks(rotation=45)
        ax.legend(loc=1)
    plt.tight_layout()
    filename = f"{out_dir}/battery_prediction.png"
    _log.info(f'writing figure to {filename}')
    fig.savefig(filename, format='png', transparent=True)
    dline = f"{datetime.datetime.now()},{glider},{mission},{v_per_day},{recover},{end}\n"
    with open("/data/plots/nrt/battery_prediction.csv", "a") as file:
        file.write(dline)


if __name__ == '__main__':
    from pathlib import Path
    battery_plots(Path('/data/data_l0_pyglider/nrt/SEA44/M85/rawnc/Martorn-rawgli.parquet'), '.')
    