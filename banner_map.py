import xarray as xr
import pandas as pd
import json
import datetime
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import pathlib
import cmocean.cm as cmo
import matplotlib as mpl
import logging
_log = logging.getLogger(__name__)

glider_names = {
    '44': 'Martorn',
    '45': 'Kvanne',
    '55': 'Kaprifol',
    '56': 'Trift',
    '57': 'King Tubby',
    '61': 'Vass',
    '63': 'Ljung',
    '66': 'Saltarv',
    '67': 'Marviol',
    '68': 'Aster',
    '69': 'Kalmus',
    '70': 'Scratch',
}


def create_map():
    # prepare data
    _log.info("starting banner map creation")
    with open(f"/data/plots/glider_locs.json") as json_to_load:
        glider_locs = json.load(json_to_load)
    to_process = pd.read_csv('/home/pipeline/to_process.csv', dtype=int)
    df = pd.read_csv('/data/plots/glider_stats.csv', parse_dates=['datetime'], index_col='datetime')
    total_dives = df.iloc[-1].total_dives
    idx_yesterday = df.index.get_loc(df.index[-1] - datetime.timedelta(hours=24), method='nearest')
    dives_yesterday = df.iloc[idx_yesterday].total_dives
    _log.debug("looking for smhi data files")
    smhi_data_dir = pathlib.Path("/data/third_party/smhi/model_output")
    grib_files = list(smhi_data_dir.glob("*"))
    grib_files.sort()
    _log.debug(f"found {len(grib_files)} files in {smhi_data_dir}")
    good_gribs = []
    for grib_name in grib_files:
        if str(grib_name)[-8:] == '+000H00M':
            good_gribs.append(grib_name)
    _log.debug(f"found {len(good_gribs)} good files")
    _log.info(f"using {good_gribs[-1]}")
    dat = xr.open_dataset(good_gribs[-1], engine='cfgrib')

    _log.debug("start figure creation")
    fig = plt.figure()
    ax = plt.axes(projection=ccrs.AzimuthalEquidistant(central_longitude=13, central_latitude=58))
    ax.set_extent([-4, 30, 50, 66])

    cs = dat.lccpg60.plot(transform=ccrs.PlateCarree(), cmap=cmo.haline, robust=True, add_colorbar=False)

    ### this fills out white space
    ax.set_title('')
    ax.spines['geo'].set_visible(False)

    cax = ax.inset_axes([0.45, 0.65, 0.1, 0.02], transform=ax.transAxes)
    cbar = fig.colorbar(cs, ax=ax, cax=cax, orientation='horizontal')
    cbar.ax.tick_params(labelsize=4)
    cbar.set_label('Salinity (psu)', fontsize=4)

    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.margins(0, 0)
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=colors[1:])
    last_update = datetime.datetime(1970, 1, 1)
    to_plot = to_process.glider
    for glider_num in to_plot:
        if str(glider_num) not in glider_locs.keys():
            continue
        glider = glider_locs[str(glider_num)]
        lon = glider['lon'][-1]
        lat = glider['lat'][-1]
        name = glider_names[str(glider_num)]
        time = datetime.datetime.strptime(glider['time'][-1][:19], "%Y-%m-%dT%H:%M:%S")
        if time > last_update:
            last_update = time
        label = f"SEA{str(glider_num).zfill(3)} {name}"
        ax.scatter(lon, lat, s=8, transform=ccrs.PlateCarree(),  label=label)
    mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=colors)

    ax.text(0.1, 0.95, f'Total dives completed: {total_dives}\nActive gliders:', transform=ax.transAxes, fontsize=8)
    ax.text(0.3, 0.05, 'Sea surface salinity last updated {}'.format(
        dat.time.valid_time.dt.strftime("%d %B, %Y").values), transform=ax.transAxes, fontsize=5)
    ax.text(0.3, 0.01, 'Glider locations last updated {}'.format(last_update.strftime("%I %p %d %B, %Y")),
            transform=ax.transAxes, fontsize=5)
    ax.legend(loc=(0.1, 0.87), frameon=False, fontsize=6)
    fig_path = '/data/plots/maps/salinity_gliders.png'
    fig.savefig(fig_path, transparent=True, dpi=300, bbox_inches='tight', pad_inches=0)
    return fig_path
