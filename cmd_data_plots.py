import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import geopy.distance
import math
import logging
from pathlib import Path

_log = logging.getLogger(__name__)
if __name__ == '__main__':
    logf = 'cmdconsole_plots.log'
    logging.basicConfig(filename=logf,
                        filemode='a',
                        format='%(asctime)s %(levelname)-8s %(message)s',
                        level=logging.WARNING,
                        datefmt='%Y-%m-%d %H:%M:%S')
    
def basic_load(path):
    df = pd.read_csv(path, sep=";", usecols=range(0, 6), header=0)
    a = df['LOG_MSG'].str.split(',', expand=True)
    data = pd.concat([df, a], axis=1)
    return data


def load_cmd(path):
    cmd = basic_load(path)

    # Transform time from object to datetime
    cmd.DATE_TIME = pd.to_datetime(cmd.DATE_TIME, dayfirst=True, yearfirst=False, )
    # Add cycle
    cmd['cycle'] = cmd.where(cmd[0] == '$SEAMRS').dropna(how='all')[3]
    # create lat lon columns in decimal degrees
    cmd['lat'] = cmd.where(cmd[0] == '$SEAMRS').dropna(how='all')[8].str.rsplit('*').str[0]
    cmd['lon'] = cmd.where(cmd[0] == '$SEAMRS').dropna(how='all')[9].str.rsplit('*').str[0]
    cmd['lat'] = cmd.where(cmd[0] == '$SEAMRS').dropna(how='all').lat.replace('', np.nan).dropna(how='all').astype(
        float)
    cmd['lon'] = cmd.where(cmd[0] == '$SEAMRS').dropna(how='all').lon.replace('', np.nan).dropna(how='all').astype(
        float)

    # The SEAMRS nmea sentence prints the coordinates in ddmm.mmm so we what to transform them into dd.dddd
    def dd_coord(x):
        degrees = x // 100
        minutes = x - 100 * degrees
        res = degrees + minutes / 60
        return res

    df_glider = pd.DataFrame({"time": pd.to_datetime(cmd.dropna(subset=['lon', 'lat']).DATE_TIME),
                              "lon": dd_coord(cmd['lon'].dropna().astype(float).values),
                              "lat": dd_coord(cmd['lat'].dropna().astype(float).values),
                              "Cycle": cmd.dropna(subset=['lon', 'lat']).cycle.astype(int)})
    return df_glider


def measure_drift(cmd):
    drift_dist = np.zeros([len(cmd['Cycle'].unique())])
    drift_x = np.zeros([len(cmd['Cycle'].unique())])
    drift_y = np.zeros([len(cmd['Cycle'].unique())])
    u_vel = np.zeros([len(cmd['Cycle'].unique())])
    v_vel = np.zeros([len(cmd['Cycle'].unique())])
    speed = np.zeros([len(cmd['Cycle'].unique())])
    theta = np.zeros([len(cmd['Cycle'].unique())])
    time = np.zeros([len(cmd['Cycle'].unique())]).astype(pd._libs.tslibs.timestamps.Timestamp)
    total_time = np.zeros([len(cmd['Cycle'].unique())])

    for i in tqdm(range(len(cmd['Cycle'].unique()))):
        cycle_num = cmd['Cycle'].unique()
        loc_data = cmd.where(cmd['Cycle'] == cycle_num[i]).dropna(how='all')

        drift_dist[i] = geopy.distance.distance((float(loc_data.lat.iloc[0]), float(loc_data.lon.iloc[0])),
                                                (float(loc_data.lat.iloc[-1]), float(loc_data.lon.iloc[-1]))).m
        drift_x[i] = 111 * 1000 * (float(loc_data.lon.iloc[0]) - float(loc_data.lon.iloc[-1])) * np.cos(
            np.deg2rad(float(loc_data.lon.iloc[0])))
        drift_y[i] = 111 * 1000 * (float(loc_data.lat.iloc[0]) - float(loc_data.lat.iloc[-1]))
        speed[i] = drift_dist[i] / float(
            (loc_data.time.iloc[-1] - loc_data.time.iloc[0]) / np.timedelta64(1000000000, 'ns'))
        u_vel[i] = drift_x[i] / float(
            (loc_data.time.iloc[-1] - loc_data.time.iloc[0]) / np.timedelta64(1000000000, 'ns'))
        v_vel[i] = drift_y[i] / float(
            (loc_data.time.iloc[-1] - loc_data.time.iloc[0]) / np.timedelta64(1000000000, 'ns'))
        theta[i] = (270 - np.rad2deg(math.atan2(drift_y[i], drift_x[i]))) % 360
        time[i] = (loc_data.time.iloc[-1])
        total_time[i] = float((loc_data.time.iloc[-1] - loc_data.time.iloc[0]) / np.timedelta64(1000000000, 'ns')) / 60

    return drift_dist, drift_y, drift_x, speed, u_vel, v_vel, theta, time, total_time


def dst_data(path):
    nmea_sep = basic_load(path)
    time = nmea_sep.where(nmea_sep[0] == '$SEADST').dropna(how='all').DATE_TIME
    dst_info = pd.DataFrame({"time": pd.to_datetime(time, dayfirst=True, yearfirst=False),
                             "pitch": nmea_sep.where(nmea_sep[0] == '$SEADST').dropna(how='all')[4].astype(float),
                             "surf_depth": nmea_sep.where(nmea_sep[0] == '$SEADST').dropna(how='all')[6].astype(float)
                             })
    return dst_info


def time_connections(path):
    cmd = basic_load(path)

    # Transform time from object to datetime
    cmd.DATE_TIME = pd.to_datetime(cmd.DATE_TIME, dayfirst=True, yearfirst=False, )
    cmd['Cycle'] = np.nan
    cycle = cmd.where(cmd[0] == '$SEAMRS')

    cyclenum = cycle[3].dropna(how='all').unique()

    hang_conn = cmd.where(cmd.LOG_LEVEL == 'INFO').dropna(how='all')

    # INFO has both hang and connect but also file transferring time !!!!

    a = hang_conn.iloc[np.where((hang_conn.LOG_MSG == 'Glider Connected !') | (hang_conn.LOG_MSG == 'Glider Hang !'))]

    for i in range(len(cyclenum)):
        start = cycle.where(cycle[3] == cyclenum[i]).dropna(how='all').index[0]
        end = cycle.where(cycle[3] == cyclenum[i]).dropna(how='all').index[-1]

        i_start = a.index[int(np.abs(a.index - start).argmin())]
        i_end = a.index[int(np.abs(a.index - end).argmin())]

        cmd.iloc[i_start:i_end + 1, -1] = int(cyclenum[i])

    hang_conn = cmd.where(cmd.LOG_LEVEL == 'INFO').dropna(how='all')
    a = hang_conn.iloc[np.where((hang_conn.LOG_MSG == 'Glider Connected !') | (hang_conn.LOG_MSG == 'Glider Hang !'))]
    bla = np.diff(pd.to_datetime(a.DATE_TIME))
    minutes = (bla / np.timedelta64(1000000000, 'ns')) / 60
    a.loc[:, 0] = pd.to_datetime(a.DATE_TIME)
    return a, minutes


def make_all_plots(path_to_cmdlog):
    active_m1 = load_cmd(path_to_cmdlog)
    _log.warning("Command console data loaded. Starting with drift and velocities computation")
    drift_dist, drift_y, drif_x, speed, u_vel, v_vel, theta, time, tot_time = measure_drift(active_m1)
    _log.warning("Drift computed. Starting with DST")
    dst = dst_data(path_to_cmdlog)
    _log.warning('DST analysed. Starting with time')
    cut, mins = time_connections(path_to_cmdlog)
    _log.warning('All variables computed. Starting to create plots')
    # Prepare subplot
    gridsize = (12, 4)  # rows-cols
    fig = plt.figure(figsize=(15, 10))
    ax1 = plt.subplot2grid(gridsize, (0, 0), colspan=2, rowspan=2)
    ax2 = plt.subplot2grid(gridsize, (0, 2), colspan=2, rowspan=2)
    ax3 = plt.subplot2grid(gridsize, (2, 0), colspan=4, rowspan=2)
    ax4 = plt.subplot2grid(gridsize, (4, 0), colspan=4, rowspan=2)
    ax5 = plt.subplot2grid(gridsize, (6, 0), colspan=4, rowspan=2)
    ax6 = plt.subplot2grid(gridsize, (8, 0), colspan=2, rowspan=2)
    ax7 = plt.subplot2grid(gridsize, (10, 0), colspan=2, rowspan=2)
    ax8 = plt.subplot2grid(gridsize, (8, 2), colspan=2, rowspan=2)
    ax9 = plt.subplot2grid(gridsize, (10, 2), colspan=2, rowspan=2)

    # Surface pitch and depth
    ax1.scatter(dst.time, dst.pitch, s=10)
    ax1.set(ylabel='Surface pitch (deg)')
    ax2.scatter(dst.time, dst.surf_depth, s=10)
    ax2.set(ylabel='Surface depth (m)')

    ax3.plot(time, drift_dist)
    ax3.scatter(time, drift_dist, s=10)
    ax3.set(ylabel='Surface drfit \n(m)')

    ax4.plot(time, theta)
    ax4.scatter(time, theta, s=10)
    ax4.set(ylabel='Drift direction \n(deg)')

    ax5.plot(time, speed)
    ax5.scatter(time, speed, s=10)
    ax5.set(ylabel='Surface currents velocity \n(m/s)')

    conn = np.round(cut.groupby('Cycle').count().MODULE.mean(), 1)
    ax6.scatter(cut.groupby('Cycle').count().index, cut.groupby('Cycle').count().MODULE / 2,
                label=f'Average num of connections is {conn} ', s=10)

    ax7.scatter(cut.groupby('Cycle').count().index, tot_time,
                label=f'Average time at surface is {np.round(np.nanmean(tot_time), 1)} min', s=10)
    ax8.scatter(cut['DATE_TIME'][:-1], mins,
                label=f'Average time between surfacings {np.round(np.nanmean(mins[np.where(mins >= 10)]), 1)} min ',
                s=10)
    ax8.set_ylim(bottom=30)

    ax9.scatter(cut['DATE_TIME'][:-1], mins, s=10,
                label=f'Average time between GLIDERHANG on the same cycle {np.round(np.nanmean(mins[np.where(mins <= 10)]), 1)} min')
    ax9.set_ylim(-2, 15)

    [a.grid() for a in [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9]]
    [a.legend(loc=9) for a in [ax6, ax7, ax8, ax9]]
    [a.set(ylabel='Minutes') for a in [ax7, ax8, ax9]]
    [a.set(xlabel='Cycle') for a in [ax6, ax7]]
    ax6.set_ylabel('N of connections')
    plt.tight_layout()
    _log.warning('All plots have been created')
    return fig


def command_cosole_log_plots(glider, mission, plots_dir):
    cmd_log = Path(f"/data/data_raw/nrt/SEA{str(glider).zfill(3)}/{str(mission).zfill(6)}/G-Logs/sea{str(glider).zfill(3)}.{mission}.com.raw.log")
    make_all_plots(cmd_log)
    filename = plots_dir / f'SEA{glider}_M{mission}_cmd_log.png'
    plt.savefig(filename, format='png', transparent=True)
