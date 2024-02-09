import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
from tqdm import tqdm
import geopy.distance
import math
import logging
from pathlib import Path
import datetime

_log = logging.getLogger(__name__)


def load_all_cmd(path):
    df = pd.read_csv(path, sep=";", usecols=range(0, 6), header=0)
    if("Message" in df.columns):
        new_cmd = pd.DataFrame({"DATE_TIME": pd.to_datetime(df['Date'] + ' ' + df['Time'], dayfirst=True, yearfirst=False,),
                              "LOG_MSG": df.Message,
                              })
        a = new_cmd['LOG_MSG'].str.split(',', expand=True)
        data = pd.concat([new_cmd, a], axis=1)
        data['DATE_TIME'] = pd.to_datetime(data.DATE_TIME, dayfirst=True, yearfirst=False, )
        data['Cycle'] = df.Cycle
    else:
        df = pd.read_csv('C:/Users/monfo/OneDrive/Desktop/VOTO/CMD_data/glimpse-data/SEA044/000083/G-Logs/sea044.83.com.raw.log', sep=";", usecols=range(0, 6), header=0)
        new_cmd = pd.DataFrame({"DATE_TIME": pd.to_datetime(df.DATE_TIME, dayfirst=True, yearfirst=False, ),
                                  "LOG_MSG": df.LOG_MSG})
        a = new_cmd['LOG_MSG'].str.split(',', expand=True)
        data = pd.concat([new_cmd, a], axis=1)
        data['Cycle'] = data.where(data[0] == '$SEAMRS')[3]
        glider_hang= data.where(data.LOG_MSG == 'Glider Hang !').dropna(how='all').index
        data.loc[glider_hang, 'Cycle']=9999
        data['Cycle'] = data['Cycle'].backfill()
        data.loc[data.Cycle==9999, 'Cycle']=np.nan
        data['Cycle'] = data['Cycle'].ffill()
        data['Cycle'] = data['Cycle'].astype(int)
    # Remove data from the first and last 2h of the mission as we generally spend a lot of time at surface 
    sub_data = data.where((data.DATE_TIME > data.DATE_TIME.min() + datetime.timedelta(hours=2)) & (data.DATE_TIME < data.DATE_TIME.max() - datetime.timedelta(hours=2))).dropna(how='all')
    return sub_data

def load_cmd(path):
    cmd = load_all_cmd(path)
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

    df_glider = pd.DataFrame({"time": cmd.dropna(subset=['lon', 'lat']).DATE_TIME,
                              "lon": dd_coord(cmd['lon'].dropna().astype(float).values),
                              "lat": dd_coord(cmd['lat'].dropna().astype(float).values),
                              "Cycle": cmd.dropna(subset=['lon', 'lat']).Cycle.astype(int)})
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
    nmea_sep = load_all_cmd(path)
    time = nmea_sep.where(nmea_sep[0] == '$SEADST').dropna(how='all').DATE_TIME
    surf_z = nmea_sep.where(nmea_sep[0] == '$SEADST').dropna(how='all')[6]
    surf_z = surf_z[~surf_z.astype(str).str.lower().str.contains('nan')]
    dst_info = pd.DataFrame({"time": time,
                             "pitch": nmea_sep.where(nmea_sep[0] == '$SEADST').dropna(how='all')[4].astype(float),
                             "surf_depth":surf_z[surf_z.str.contains('\d', regex=True).astype(bool)].astype(float),
                             "Cycle": nmea_sep.where(nmea_sep[0] == '$SEADST').dropna(how='all').Cycle})
    return dst_info


def time_connections(path):
    cmd = load_all_cmd(path)
    cmd['LOG_MSG'] = cmd.LOG_MSG.str.replace('GLIDERCONNECT','Glider Connected !')
    cmd['LOG_MSG'] = cmd.LOG_MSG.str.replace('GLIDERHANG','Glider Hang !')
    a = cmd.iloc[np.where((cmd.LOG_MSG == 'Glider Connected !') | (cmd.LOG_MSG == 'Glider Hang !'))]
    bla = np.diff(a.DATE_TIME)
    minutes = (bla / np.timedelta64(1000000000, 'ns')) / 60
    a.loc[:, 0] = a.DATE_TIME
    return a, minutes



def make_all_plots(path_to_cmdlog):
    fig = plt.figure(figsize=(15, 12))
    active_m1 = load_cmd(path_to_cmdlog)
    if len(active_m1) < 1:
        _log.debug("Command console data missing or too few data points as the mission just started")
        return fig        
    _log.debug("Command console data loaded. Starting with drift and velocities computation")
    drift_dist, drift_y, drif_x, speed, u_vel, v_vel, theta, time, tot_time = measure_drift(active_m1)
    _log.debug("Drift computed. Starting with DST")
    dst = dst_data(path_to_cmdlog)
    cycle_df = dst.copy()
    cycle_df.time = mdates.date2num(cycle_df.time)
    cycle_av = cycle_df.groupby('Cycle').mean()
    cycle_av.time = mdates.num2date(cycle_av.time)
    _log.debug('DST analysed. Starting with time')
    cut, mins = time_connections(path_to_cmdlog)
    _log.debug('All variables computed. Starting to create plots')
    # Prepare subplot
    gridsize = (12, 4)  # rows-cols
    with plt.style.context('default'):
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
        if len(dst) <1:
            ax1.text(0.3, 0.5, "DST NMEA sentence not enabled \nSurface pitch unavailable", fontsize="12",transform=ax1.transAxes)
            #ax1.axis("off")
            ax2.text(0.3, 0.5, "DST NMEA sentence not enabled \nSurface depth unavailable", fontsize="12", transform=ax2.transAxes)
            #ax2.axis("off")
        else: 
            ax1.plot( cycle_av.time,cycle_av.pitch,c='r', label='Cycle average')
            ax1.scatter(dst.time, dst.pitch, s=10)
            ax1.set(ylabel='Surface pitch (deg)')

            ax2.plot( cycle_av.time,cycle_av.surf_depth,c='r', label='Cycle average')
            ax2.scatter(dst.time, dst.surf_depth, s=10)
            ax2.set(ylabel='Surface depth (m)')
            [a.legend(loc=2) for a in [ax1,ax2]]
            [a.grid() for a in [ax1,ax2]]

        ax3.plot(time, drift_dist)
        ax3.scatter(time, drift_dist, s=10)
        ax3.set(ylabel='Surface drfit \n(m)')

        ax4.plot(time, theta)
        ax4.scatter(time, theta, s=10)
        ax4.set(ylabel='Drift direction \n(deg)')

        ax5.plot(time, speed)
        ax5.scatter(time, speed, s=10)
        ax5.set(ylabel='Surface currents velocity \n(m/s)')

        conn = np.round(cut.groupby('Cycle').count().LOG_MSG.mean(), 1)
        ax6.scatter(cut.groupby('Cycle').count().index, cut.groupby('Cycle').count().LOG_MSG / 2,
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

        [a.grid() for a in [ax3,ax4,ax5,ax6,ax7,ax8,ax9]]
        [a.legend(loc=9) for a in [ax6,ax7,ax8,ax9]]
        [a.set( ylabel='Minutes') for a in [ax7,ax8, ax9]]
        [a.set(xlabel='Cycle') for a in [ax6, ax7]]
        [a.tick_params(axis='x', labelrotation=20) for a in [ax1, ax2, ax8, ax9]]
        ax6.set_ylabel('N of connections')
        [a.set_xlim(active_m1.time.min(), active_m1.time.max()) for a in [ax1,ax2,ax3,ax4,ax5,ax8,ax9]]
        [a.set_xlim(active_m1.Cycle.min(), active_m1.Cycle.max()) for a in [ax6,ax7]]
        plt.tight_layout()
    _log.debug('All plots have been created')
    return fig


def command_cosole_log_plots(glider, mission, plots_dir):
    _log.info(f"Make command console plots for SEA{glider} M{mission}")
    cmd_log = Path(f"/data/data_raw/nrt/SEA{str(glider).zfill(3)}/{str(mission).zfill(6)}/G-Logs/sea{str(glider).zfill(3)}.{mission}.com.raw.log")
    if not cmd_log.exists():
        _log.warning(f"No command log found at {cmd_log}")
        return
    make_all_plots(cmd_log)
    filename = plots_dir / f'SEA{glider}_M{mission}_cmd_log.png'
    plt.savefig(filename, format='png', transparent=True)
    _log.debug(f"saved figure to {filename}")


if __name__ == '__main__':
    logf = 'cmdconsole_plots.log'
    logging.basicConfig(filename=logf,
                        filemode='a',
                        format='%(asctime)s %(levelname)-8s %(message)s',
                        level=logging.WARNING,
                        datefmt='%Y-%m-%d %H:%M:%S')
    command_cosole_log_plots(44, 85, Path("."))
