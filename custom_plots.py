import matplotlib.pyplot as plt
import matplotlib
import matplotlib.dates as mdates
import cmocean.cm as cmo
import pathlib
import numpy as np
import xarray as xr
from gridded_plots import cmap_dict, prepare_for_plotting, label_replace
import logging
_log = logging.getLogger(__name__)

cmap_dict["oxygen_concentration"] = cmo.oxy


def single_plot(dataset, variable, img_file_path):
    fig, ax = plt.subplots(figsize=(12, 5))
    colormap = cmap_dict[variable]
    if 'cdom' in variable:
        std_devs = 4
        percentile = 2
    elif 'backs' in variable:
        std_devs = 2
        percentile = 0.5
    elif 'chlor' in variable or 'DOWN' in variable:
        std_devs = 0
        percentile = 0.2
    else:
        std_devs = 4
        percentile = 0.1
    dataset = prepare_for_plotting(dataset, variable, std_devs=std_devs, percentile=percentile)
    ds = dataset[variable]
    vmin = np.nanmin(dataset[variable])
    if "oxy" in variable:
        vmin = 0
    if 'DOWN' in variable:
        vals = ds.values
        vals[vals < 0] = 0
        # Hack to replace bad PAR values that have been averaged in. Bad values are 9999 in og data
        vals[vals > 500] = np.nan
        pcol = ax.pcolor(ds.time.values, ds.depth, ds.values, cmap=colormap, shading='auto',
                         norm=matplotlib.colors.LogNorm(vmin=np.nanmin(vals), vmax=np.nanmax(vals)))
    else:
        pcol = ax.pcolor(ds.time.values, ds.depth, ds.values, vmin=vmin, cmap=colormap, shading='auto')
    var_sum = np.nansum(dataset[variable].data, 1)
    valid_depths = dataset[variable].depth.data[var_sum != 0.0]
    ax.set_ylim(valid_depths.max(), valid_depths.min())
    ax.set_title(label_replace(str(variable)))
    days = (1, 5, 10, 15, 20, 25)
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_minor_locator(mdates.DayLocator(days))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("\n%b %Y"))
    ax.xaxis.set_minor_formatter(mdates.DateFormatter("%d"))
    ax.tick_params(axis="x", which="both", length=4)
    plt.setp(ax.get_xticklabels(), rotation=0, ha="center")
    ymin, ymax = ax.get_ylim()
    ax.set(xlabel='', ylabel='Depth (m)', ylim=(ymin, 0))
    plt.colorbar(mappable=pcol, ax=ax, label=label_replace(ds.units), aspect=13, pad=0.02)
    plt.tight_layout()
    _log.info(f'writing figure to {img_file_path}')
    fig.savefig(img_file_path, format='png', transparent=True)


if __name__ == '__main__':
    outdir = pathlib.Path(f'/data/plots/custom/')
    if not outdir.exists():
        outdir.mkdir(parents=True)
    img_name = "dissolved_oxygen_baltic_cmo"
    glider = 45
    mission = 62
    infile = f"/data/data_l0_pyglider/complete_mission/SEA{glider}/M{mission}/gridfiles/mission_grid.nc"
    ds = xr.open_dataset(infile)
    _log.info(f'start plotting {img_name} from SEA{glider} M{mission}')
    img_path = outdir / f'{img_name}.png'
    single_plot(ds, "oxygen_concentration", img_path)
