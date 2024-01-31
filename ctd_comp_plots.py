import datetime
import numpy as np
import pytz
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 14})
from pathlib import Path
from erddapy import ERDDAP
import logging
_log = logging.getLogger(__name__)
logging.basicConfig(filename='/data/log/ctd_plots.log',
                    filemode='a',
                    format='%(asctime)s %(levelname)-8s %(message)s',
                    level=logging.INFO,
                    datefmt='%Y-%m-%d %H:%M:%S')


def init_erddap(protocol="tabledap"):
    # Setup initial ERDDAP connection
    erddap_instance = ERDDAP(
        server="https://erddap.observations.voiceoftheocean.org/erddap",
        protocol=protocol,
    )
    return erddap_instance


def comp_plot(glider, ctd):
    df_max = glider.groupby("dive_num").max()
    pressure_target = np.nanpercentile(ctd.pressure.values, 50)
    first_deep_dive = df_max[df_max.pressure > pressure_target].index.values[0]
    glider_start = glider[glider.dive_num == first_deep_dive]
    df_max = glider_start.groupby("dive_num").max()
    deepest_dive = df_max[df_max.pressure == df_max.max().pressure].index.values[0]
    glider_start = glider_start[glider_start.dive_num == deepest_dive]
    dlon = glider_start.longitude.values[0] - ctd.longitude.values[0]
    dlat = glider_start.latitude.values[0] - ctd.latitude.values[0]
    dx = dlon * 111 * np.cos(np.deg2rad(glider_start.latitude.values[0]))
    dy = dlat * 111
    distance = np.round(np.sqrt(dx ** 2 + dy ** 2), 2)
    dtime = abs(np.round(int(glider_start.time.values[0] - ctd.time.values[0]) / (1e9 * 60 ** 2), 2))
    fig, ax = plt.subplots(2, 2, figsize=(16, 12), sharey="row",)
    ax = ax.ravel()
    for i, variable in enumerate(("temperature", "salinity", "oxygen_concentration", "chlorophyll")):
        if variable == "oxygen_concentration":
            cutoff = 9
        else:
            cutoff = 3
        glider_sub = glider.copy()
        if f"{variable}_qc" in list(glider_sub):
            glider_sub = glider_sub[glider_sub[f"{variable}_qc"] < cutoff]
        ctd_sub = ctd.copy()
        if f"{variable}_qc" in list(ctd_sub):
            ctd_sub = ctd_sub[ctd_sub[f"{variable}_qc"] < cutoff]
        ax[i].plot(glider_sub[variable], glider_sub.pressure, label="glider")
        ax[i].plot(ctd_sub[variable], ctd_sub.pressure, label="ctd")
        if len(glider_sub) > 0:
            prop = int(len(ctd_sub)/len(glider_sub))
            pool = list(ctd_sub[variable])[::prop] + list(glider_sub[variable])
        else:
            pool = list(ctd_sub[variable])
        min = np.nanpercentile(pool, 5)
        max = np.nanpercentile(pool, 95)
        vmin = min - (max - min) * 0.05
        vmax = max + (max - min) * 0.05
        ax[i].set(xlabel=variable, xlim=(vmin, vmax))
    ax[0].legend()
    ax[0].invert_yaxis()
    ax[0].set(ylabel="Pressure (dbar)")
    ax[1].set(title=f"Separation: {distance} km, {dtime} hours")
    ax[2].invert_yaxis()
    ax[2].set(ylabel="Pressure (dbar)")
    return fig, ax


e = init_erddap()
e.dataset_id = "ctd_deployment"
df_ctd = e.to_xarray().drop_dims("timeseries").to_pandas()
df_ctd.index = df_ctd["time"]
df_ctd = df_ctd.sort_index()


def nearby_ctd(ds_glider, comparison_plots=False, max_dist=0.5, max_days=2, num_dives=5):

    name = f'SEA0{ds_glider.attrs["glider_serial"]}_M{ds_glider.attrs["deployment_id"]}'
    df_glider = ds_glider.to_pandas()
    df_glider["time"] = df_glider.index

    start = np.nanpercentile(df_glider.time.values, 1)
    end = np.nanpercentile(df_glider.time.values, 99)

    # Look for nearby CTDs at start and end of deployment
    dives = list(set(df_glider.dive_num))
    dives.sort()
    ind_start = 1
    ind_end = min(num_dives, len(dives) - 1)
    glider_start = df_glider[np.logical_and(df_glider.dive_num > dives[ind_start], df_glider.dive_num < dives[ind_end])]
    glider_end = df_glider[np.logical_and(df_glider.dive_num > dives[-ind_end], df_glider.dive_num < dives[-ind_start])]

    lon_start = glider_start.longitude.mean()
    lat_start = glider_start.latitude.mean()
    df_near_start = df_ctd[
        np.logical_and(abs(df_ctd.longitude - lon_start) < max_dist, abs(df_ctd.latitude - lat_start) < max_dist)]
    df_start = df_near_start[abs(df_near_start.time - start) < datetime.timedelta(days=max_days)]

    lon_end = glider_end.longitude.mean()
    lat_end = glider_end.latitude.mean()
    df_near_end = df_ctd[np.logical_and(abs(df_ctd.longitude - lon_end) < max_dist, abs(df_ctd.latitude - lat_end) < max_dist)]
    df_end = df_near_end[abs(df_near_end.time - end) < datetime.timedelta(days=max_days)]

    ctds = {}
    if not Path(f'/data/plots/nrt/SEA{ds_glider.attrs["glider_serial"]}/M{ds_glider.attrs["deployment_id"]}').is_dir():
        Path(f'/data/plots/nrt/SEA{ds_glider.attrs["glider_serial"]}/M{ds_glider.attrs["deployment_id"]}').mkdir(parents=True)
        
    if not df_start.isnull().all().all():
        ctds["deployment"] = df_start
        if comparison_plots:
            fig, ax = comp_plot(glider_start, df_start)
            ax[0].set(title=f"{name} deployment")
            fig.savefig(f'/data/plots/nrt/SEA{ds_glider.attrs["glider_serial"]}/M{ds_glider.attrs["deployment_id"]}/ctd_deployment.png')
            _log.info(f"processed {name} deployment")

    if not df_end.isnull().all().all():
        ctds["recovery"] = df_end
        if comparison_plots:
            fig, ax = comp_plot(glider_end, df_end)
            ax[0].set(title=f"{name} recovery")
            fig.savefig(f'/data/plots/nrt/SEA{ds_glider.attrs["glider_serial"]}/M{ds_glider.attrs["deployment_id"]}/ctd_recovery.png')
            _log.info(f"processed {name} recovery")

    return ctds


def download_glider_datasets(dataset_ids):
    dataset_dict = {}
    for dataset_id in dataset_ids:
        e.dataset_id = dataset_id
        ds = e.to_xarray()
        if "timeseries" in ds.dims.keys() and "obs" in ds.dims.keys():
            ds = ds.drop_dims("timeseries")
        if "obs" in list(ds.dims):
            ds = ds.swap_dims({"obs": "time"})
        dataset_dict[dataset_id] = ds
    return dataset_dict


def recent_ctds():
    _log.info("start to process all CTDs")
    ctd_casts = df_ctd.groupby("cast_no").first()
    e.dataset_id = "allDatasets"
    df_datasets = e.to_pandas(parse_dates=['minTime (UTC)', 'maxTime (UTC)'])

    df_datasets.set_index("datasetID", inplace=True)
    df_datasets.drop("allDatasets", inplace=True)
    df_datasets = df_datasets[df_datasets.index.str[:3] == "nrt"]
    df_relevant = df_datasets
    mintime = ctd_casts.time.min().replace(tzinfo=pytz.utc) - datetime.timedelta(days=1)
    maxtime = ctd_casts.time.max().replace(tzinfo=pytz.utc) + datetime.timedelta(days=1)

    df_relevant = df_relevant[df_relevant["minTime (UTC)"] > mintime]
    df_relevant = df_relevant[df_relevant["maxTime (UTC)"] < maxtime]
    df_relevant["longitude"] = (df_relevant["minLongitude (degrees_east)"] + df_relevant[
        "maxLongitude (degrees_east)"]) / 2
    df_relevant["latitude"] = (df_relevant["minLatitude (degrees_north)"] + df_relevant[
        "maxLatitude (degrees_north)"]) / 2

    df_relevant = df_relevant[['longitude',
                               'latitude',
                               'minAltitude (m)',
                               'maxAltitude (m)',
                               'minTime (UTC)',
                               'maxTime (UTC)',
                               ]]
    df_relevant = df_relevant.sort_values('minTime (UTC)')
    nrt_dict = download_glider_datasets(df_relevant.index)
    i = 0
    summary_plot(df_relevant, ctd_casts, nrt_dict)
    for mission, ds in nrt_dict.items():
        _log.info(f"process: {mission}")
        try:
            ctds = nearby_ctd(ds, comparison_plots=True, num_dives=4)
        except:
            _log.warning("4 dives insufficient. Expanding to 8")
            ctds = nearby_ctd(ds, comparison_plots=True, num_dives=8)

        found = list(ctds.keys())
        if found == ['deployment', 'recovery']:
            continue
        missing = {'deployment', 'recovery'}.difference(set(found))
        _log.warning(f'{mission[4:]}, missing {missing}, {ds.attrs["basin"]}')
        i += 1
    _log.info(f"total bad: {i}")
    _log.info("completed process all CTDs")


def summary_plot(df_relevant, ctd_casts, nrt_dict):
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    fig, ax = plt.subplots(figsize=(12, 8))
    diff = 0.15
    for i, (name, row) in enumerate(df_relevant.iterrows()):
        ds = nrt_dict[name]
        ctds = nearby_ctd(ds)
        found = list(ctds.keys())
        if found == ['deployment', 'recovery']:
            continue
        # if len(found)>0:
        #    continue
        diff = diff * -1
        a = row
        plt.plot([row["minTime (UTC)"], row["maxTime (UTC)"]], [row["longitude"], row["longitude"]])
        # plt.scatter([row["minTime (UTC)"], row["maxTime (UTC)"]], [row["longitude"], row["longitude"]])
        if i > 5:
            i = i % len(colors)
        plt.text(row["minTime (UTC)"], row["longitude"] - 0.4 - diff, name[7:], fontsize=10, color=colors[i])
    ax.scatter(ctd_casts.time, ctd_casts.longitude, color="k", marker="s", label="CTD casts")
    for lon, name in zip((12, 15, 18.5, 20), ("Skagerrak", "Bornholm", "Gotland W", "Gotland E")):
        plt.text(ctd_casts.time.min(), lon, name)
    for lon in (13, 17, 19.2):
        ax.axhline(lon, color="k")
    ax.legend()
    ax.set(ylabel="longitude")
    fig.savefig("/data/plots/ctds_gant.png")


if __name__ == '__main__':
    recent_ctds()
