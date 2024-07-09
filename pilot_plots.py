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
    fig, ax = plt.subplots(figsize=(12, 8))
    if df_mean.Voltage.min() > 28:
        _log.info("voltage too high to make prediction plot")
        ax.text(0.3, 0.5, "Voltage too high \nfor battery prediction", fontsize="24")
        ax.axis("off")
    else:
        # Prediction plot
        df = df.dropna()
        df_3day = df[df.index > df.index.max() - datetime.timedelta(days=3)]
        regr_3 = linear_model.LinearRegression()
        regr_3.fit(df_3day.index.values.astype('datetime64[us]').astype(float).reshape(-1, 1), df_3day['Voltage'].values.reshape(-1, 1))
        # Create time array of one point every hour for 60 days starting three days ago
        datetime_pred_3 = pd.date_range(df_3day.index[0], df_3day.index[0] + datetime.timedelta(days=60), 60 * 24).values.astype('datetime64[us]')
        y_forward_3 = regr_3.predict(datetime_pred_3.astype(float).reshape(-1, 1))
        v_per_ns_3 = regr_3.coef_[0][0]
        v_per_day_3 = v_per_ns_3 * 24 * 60 * 60 * 1e6

        df_sub_28v = df[df.index > df_mean[df_mean.Voltage < 28].index.min()]
        df_5day = df[df.index > df.index.max() - datetime.timedelta(days=5)]

        regr_5 = linear_model.LinearRegression()
        regr_5.fit(df_sub_28v.index.values.astype('datetime64[us]').astype(float).reshape(-1, 1), df_sub_28v['Voltage'].values.reshape(-1, 1))
        # Create time array of one point every hour for 60 days starting three days ago
        datetime_pred_5 = pd.date_range(df_sub_28v.index[0], df_sub_28v.index[0] + datetime.timedelta(days=60), 60 * 24).values.astype('datetime64[us]')
        y_forward_5 = regr_5.predict(datetime_pred_5.astype(float).reshape(-1, 1))

        end = datetime_pred_5[y_forward_5[:, 0] > 23][-1]

        v_per_ns_5 = regr_5.coef_[0][0]
        v_per_day_5 = v_per_ns_5 * 24 * 60 * 60 * 1e6
        v_per_days = np.sort(np.array((v_per_day_3, v_per_day_5)))

        loss_5 = np.round(np.abs(v_per_day_5), 2)
        loss_3 = np.round(np.abs(v_per_day_3), 2)
        losses = np.sort(np.array((loss_3, loss_5)))
        recover_3 = datetime_pred_3[np.argmin(np.abs(24 - y_forward_3))]
        recover_5 = datetime_pred_5[np.argmin(np.abs(24 - y_forward_5))]
        recoveries = np.sort(np.array((recover_3, recover_5)))

        ax.scatter(df_sub_28v.index, df_sub_28v.Voltage, label="Voltage last 5 days", s=3)
        ax.plot(datetime_pred_3, y_forward_3, label="last 3 days prediction")
        ax.set(xlim=(df_5day.index[0], end), ylim=(22.9, df_5day.Voltage.max() + 0.1))
        ax.axvline(recover_3, color="red")
        ax.plot(datetime_pred_5, y_forward_5, label="Entire mission prediction")
        r_string = f"{str(recoveries[0])[:10]} - {str(recoveries[1])[:10]}"
        ax.axhline(24, color="red")
        ax.axvline(recover_3, color="red")
        ax.axvline(recover_5, color="red")
        ax.axvspan(recoveries[0], recoveries[1], alpha=0.1, color='red')
        v_string = f"Voltage: {np.round(df['Voltage'].values[-1], 1)} V\n{losses[0]}-{losses[1]} V/day\nrecover {r_string}"
        ax.text(0.05, 0.08, v_string, transform=ax.transAxes)
        ax.grid()
        ax.set(ylabel="Voltage (v)", title=title)
        plt.xticks(rotation=45)
        ax.legend(loc=1)
        dline = f"{datetime.datetime.now()},{glider},{mission},{v_per_days[1]},{recoveries[0]},{end}\n"
        with open("/data/plots/nrt/battery_prediction.csv", "a") as file:
            file.write(dline)
    plt.tight_layout()
    filename = f"{out_dir}/battery_prediction.png"
    _log.info(f'writing figure to {filename}')
    fig.savefig(filename, format='png', transparent=True)


if __name__ == '__main__':
    from pathlib import Path
    battery_plots(Path('/data/data_l0_pyglider/nrt/SEA63/M73/rawnc/Fibbla-rawgli.parquet'), '.')
