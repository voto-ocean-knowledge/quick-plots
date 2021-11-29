"""
Basic plotting functions to operate on gridded netCDF output by pyglider https://github.com/c-proof/pyglider
Based on work by Elizabeth Siddle and Callum Rollo https://github.com/ESiddle/basestation_plotting
To use as a standaline:
$ python gridded-plots.py path/to/nc/file path/to/plotting/dir
"""

import sys
import xarray as xr
from cmocean import cm as cmo
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from pathlib import Path
import shutil

# list of variables we want to plot
glider_variables = (
    'conductivity',
    'temperature',
    'salinity',
    'chlorophyll',
    'cdom',
    'oxygen_concentration',
    'temperature_oxygen',
    'potential_density',
    'density',
    'potential_temperature')

# create dictionary to match each variable in glider_variables to a colourmap
default_cmap = "viridis"
cmap_dict = {}
for j in range(len(glider_variables)):
    cmap_dict[glider_variables[j]] = default_cmap

# update colourmaps for certain variables
cmap_dict['temperature_oxygen'] = cmo.thermal
cmap_dict['temperature'] = cmo.thermal
cmap_dict['conductivity'] = cmo.haline
cmap_dict['salinity'] = cmo.haline
cmap_dict['potential_density'] = cmo.dense
cmap_dict['oxygen_concentration'] = cmo.oxy
cmap_dict['chlorophyll'] = cmo.algae
cmap_dict['cdom'] = cmo.algae


def create_plots(nc, output_dir):
    if Path.exists(Path.absolute(output_dir)):
        shutil.rmtree(output_dir)
    output_dir.mkdir()
    ds = xr.open_dataset(nc)
    a = list(ds.keys())  # list data variables in ds
    to_plot = list(set(a).intersection(glider_variables))  # find elements in glider_variables relevant to this dataset
    for i in range(len(to_plot)):
        plotter(ds, to_plot[i], cmap_dict[to_plot[i]], to_plot[i], output_dir)


def prepare_for_plotting(dataset, variable, std_devs=2):
    """Prepare variable for plotting by:
    1. Removing outliers more than 3 std dev from the mean
    2. Interpolating over nans
    """
    data = dataset[variable].data
    # Copy some of the cleaning functionality from GliderTools and use it here
    # https://github.com/GliderToolsCommunity/GliderTools/blob/master/glidertools/cleaning.py
    # e.g. remove data more than 2 standard deviations from the mean

    arr = data
    # standard deviation
    mean = np.nanmean(arr)
    std = np.nanstd(arr)

    ll = mean - std * std_devs
    ul = mean + std * std_devs

    mask = (arr < ll) | (arr > ul)
    arr[mask] = np.nan
    dataset[variable].data = data

    # nanpercentile
    ll = np.nanpercentile(arr, 0.5)
    ul = np.nanpercentile(arr, 99.5)
    # tbi

    return dataset


# define a basic plotting function for the profiles
def plotter(dataset, variable, colourmap, title, plots_dir, glider='', mission=''):
    """Create time depth profile coloured by desired variable

    Input:
    dataset: the name of the xarray dataset
    variable: the name of the data variable to be plotted
    colourmap: name of the colourmap to be used in profile
    title: variable name included as title to easily identify plot

    The intended use of the plotter function is to iterate over a list of variables,
    plotting a pcolormesh style plot for each variable, where each variable has a colourmap assigned using a dictionary"""

    # find max depth the given variable was measures to
    var_sum = np.nansum(dataset[variable].data, 1)
    valid_depths = dataset[variable].depth.data[var_sum != 0.0]

    fig, ax = plt.subplots()
    if 'cdom' in variable:
        std_devs = 4
    else:
        std_devs = 2
    dataset = prepare_for_plotting(dataset, variable, std_devs=std_devs)
    dataset[variable].T.plot(yincrease=False, y="depth", x="time", cmap=colourmap)
    ax.set_ylim(valid_depths.max(), valid_depths.min())
    ax.set_title(str(title))
    plt.tight_layout()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m/%y'))  # sets x tick format
    fig.savefig(plots_dir / f'{variable}.jpg', format='jpeg')
    return fig, ax


if __name__ == '__main__':
    netcdf = Path(sys.argv[1])
    if len(sys.argv) > 2:
        outdir = Path(sys.argv[2])
    else:
        outdir = Path('plots')
    create_plots(netcdf, outdir)
