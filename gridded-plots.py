"""
Basic plotting functions to operate on gridded netCDF output by pyglider https://github.com/c-proof/pyglider
Based on work by Elizabeth Siddle and Callum Rollo https://github.com/ESiddle/basestation_plotting
To use as a standaline:
$ python gridded-plots.py path/to/nc/file path/to/plotting/dir
"""

import sys
import gsw
import xarray as xr
from cmocean import cm as cmo
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib
import numpy as np
from pathlib import Path
import shutil
from collections import defaultdict
import boto3
from botocore.exceptions import ClientError
from matplotlib import style
style.use('presentation.mplstyle')

def sort_by_priority_list(values, priority):
    priority_dict = defaultdict(
        lambda: len(priority), zip(priority, range(len(priority)),),
    )
    priority_getter = priority_dict.__getitem__
    return sorted(values, key=priority_getter)

# list of variables we want to plot in order
glider_variables = (
    'potential_temperature',
    'absolute_salinity',
    'potential_density',
    'oxygen_concentration',
    'chlorophyll',
    'cdom',
    'backscatter_700',
    'DOWNWELLING_PAR',
    'DOWN_IRRADIANCE380',
    'DOWN_IRRADIANCE490',
    'DOWN_IRRADIANCE532',
    'molar_nitrate'
)

# create dictionary to match each variable in glider_variables to a colourmap
default_cmap = "viridis"
cmap_dict = {}
for j in range(len(glider_variables)):
    cmap_dict[glider_variables[j]] = default_cmap

# update colourmaps for certain variables
cmap_dict['temperature_oxygen'] = cmo.thermal
cmap_dict['temperature'] = cmo.thermal
cmap_dict['potential_temperature'] = cmo.thermal
cmap_dict['conductivity'] = cmo.haline
cmap_dict['salinity'] = cmo.haline
cmap_dict['absolute_salinity'] = cmo.haline
cmap_dict['density'] = cmo.dense
cmap_dict['potential_density'] = cmo.dense
cmap_dict['chlorophyll'] = cmo.algae
cmap_dict['cdom'] = cmo.turbid

labels_dict = {'Celsius': '$^{\circ$C'}

def create_plots(nc, output_dir):
    if not Path.exists(Path.absolute(output_dir)):
        output_dir.mkdir()
    ds = xr.open_dataset(nc)
    ds = additional_vars(ds)
    a = list(ds.keys())  # list data variables in ds
    to_plot_unsort = list(set(a).intersection(glider_variables))  # find elements in glider_variables relevant to this dataset
    to_plot = sort_by_priority_list(to_plot_unsort, glider_variables)
    #for i in range(len(to_plot)):
    #    plotter(ds, to_plot[i], cmap_dict[to_plot[i]], to_plot[i], output_dir)
    image_file = multiplotter(ds, to_plot, output_dir, glider=ds.glider_serial, mission=ds.deployment_id)
    # image_file = tempsal_scatter(ds, output_dir)
    return image_file


def additional_vars(ds):
    if 'absolute_salinity' not in list(ds):
        ab_sal = gsw.SA_from_SP(ds['salinity'], ds['pressure'].values, ds['longitude'].values, ds['latitude'].values)
        attrs = ab_sal.attrs
        attrs['long_name'] = 'absolute salinity'
        attrs['standard_name'] = 'sea_water_absolute_salinity'
        attrs['sources'] = 'conductivity temperature pressure longitude latitude'
        attrs['units'] = 'g kg^-1'
        attrs['comment'] = 'uncorrected absolute salinity'
        ab_sal.attrs = attrs
        ds['absolute_salinity'] = ab_sal
    return ds

def tempsal_scatter(dataset, plots_dir):
    ds_temp = prepare_for_plotting(dataset, 'temperature', std_devs=0, percentile=1)['temperature']
    ds_sal = prepare_for_plotting(dataset, 'salinity', std_devs=0, percentile=1)['salinity']

    sal_lim = [np.nanmin(ds_sal.values), np.nanmax(ds_sal.values)]
    temp_lim = [np.nanmin(ds_temp.values), np.nanmax(ds_temp.values)]
    sal_ex, temp_ex = np.meshgrid(np.arange(sal_lim[0], sal_lim[1], 0.01), np.arange(temp_lim[0], temp_lim[1], 0.01))
    seawater_density = gsw.rho(sal_ex, temp_ex, 0)
    den_conts = np.arange(0, 35, 1)

    fig, ax = plt.subplots(1, 1, figsize=(14, 14))
    mappable0 = ax.contour(sal_ex[0, :], temp_ex[:, 0], seawater_density - 1000, den_conts, colors='k', zorder=-10)
    ax.scatter(ds_sal.values, ds_temp.values)
    ax.clabel(mappable0, inline=1, inline_spacing=0, fmt='%i', fontsize=12)
    ax.set(xlim=sal_lim, ylim=temp_lim, xlabel=f'Salinity (psu)', ylabel='Temperature ($^{\circ}$C)')
    figpath = plots_dir / f'tempsal_scatter'
    fig.savefig(figpath, format='jpeg')
    return figpath


def prepare_for_plotting(dataset, variable, std_devs=2, percentile=0.5):
    """Prepare variable for plotting by:
    1. Removing outliers more than 3 std dev from the mean
    2. Interpolating over nans
    """
    arr = dataset[variable].data
    # Copy some of the cleaning functionality from GliderTools and use it here
    # https://github.com/GliderToolsCommunity/GliderTools/blob/master/glidertools/cleaning.py
    # e.g. remove data more than 2 standard deviations from the mean

    arr_in = arr.copy()
    arr_min = np.empty(np.shape(arr), dtype=bool)
    arr_min[:] = False
    arr_max = np.empty(np.shape(arr), dtype=bool)
    arr_max[:] = False
    # standard deviation
    if std_devs:
        mean = np.nanmean(arr_in)
        std = np.nanstd(arr_in)

        ll = mean - std * std_devs
        ul = mean + std * std_devs

        mask = (arr < ll) | (arr > ul)
        arr[mask] = np.nan

        arr_max[arr_in>ul] = True
        arr_min[arr_in<ll] = True

    # nanpercentile
    if percentile:
        ll = np.nanpercentile(arr_in, percentile)
        ul = np.nanpercentile(arr_in, 100-percentile)

        mask = (arr < ll) | (arr > ul)
        arr[mask] = np.nan

        arr_max[arr_in>ul] = True
        arr_min[arr_in<ll] = True
    # tbi
    arr_in[arr_min] = np.nanmin(arr)
    arr_in[arr_max] = np.nanmax(arr)
    dataset[variable].data = arr_in

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


def multiplotter(dataset, variables, plots_dir, glider='', mission=''):
    """Create time depth profile coloured by desired variable

    Input:
    dataset: the name of the xarray dataset
    variable: the name of the data variable to be plotted
    colourmap: name of the colourmap to be used in profile
    title: variable name included as title to easily identify plot

    The intended use of the plotter function is to iterate over a list of variables,
    plotting a pcolormesh style plot for each variable, where each variable has a colourmap assigned using a dictionary"""
    num_variables = len(variables)
    fig, axs = plt.subplots(num_variables, 1, figsize=(12, 3 * num_variables))
    axs = axs.ravel()
    for i, ax in enumerate(axs):
        variable = variables[i]
        colormap = cmap_dict[variable]
        title = variable
        var_sum = np.nansum(dataset[variable].data, 1)
        valid_depths = dataset[variable].depth.data[var_sum != 0.0]
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
        if 'DOWN' in variable:
            vals = ds.values
            vals[vals < 0] = 0
            pcol = ax.pcolor(ds.time.values, ds.depth, ds.values, cmap=colormap, shading='auto',
                             norm=matplotlib.colors.LogNorm(vmin=np.nanmin(vals), vmax=np.nanmax(vals)))
        else:
            pcol = ax.pcolor(ds.time.values, ds.depth, ds.values, cmap=colormap, shading='auto')
        ax.set_ylim(valid_depths.max(), valid_depths.min())
        ax.set_title(str(title))
        if i != num_variables-1:
            ax.tick_params(labelbottom=False)
        else:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m/%y'))
            plt.xticks(rotation=45)
        ymin, ymax = ax.get_ylim()
        ax.set(xlabel='', ylabel='Depth (m)', ylim=(ymin, 0))
        plt.colorbar(mappable=pcol, ax=ax, label=f'{ds.units}', aspect=13, pad=0.02)
    plt.tight_layout()
    filename = plots_dir / f'SEA{glider}_M{mission}.jpg'
    print(filename)
    fig.savefig(filename, format='jpeg')
    return filename


def upload_to_s3(file_name, bucket, object_name=None, profile_name='voto:prod'):
    """Upload a file to an S3 bucket
    Original function by Isabelle Giddy

    :param file_name: File to upload
    :param bucket: Bucket to upload to
    :param object_name: S3 object name. If not specified then file_name is used
    :return: True if file was uploaded, else False
    """
    boto3.setup_default_session(profile_name=profile_name)

    # If S3 object_name was not specified, use file_name
    if object_name is None:
        object_name = file_name

    # Upload the file
    s3_client = boto3.client('s3')
    try:
        s3_client.upload_file(file_name, bucket, object_name)
    except ClientError as e:
        return False
    return True


if __name__ == '__main__':
    netcdf = Path(sys.argv[1])
    if len(sys.argv) > 2:
        outdir = Path(sys.argv[2])
    else:
        outdir = Path('plots')
    image_file = create_plots(netcdf, outdir)
    if 'upload' in sys.argv:
        path_parts = str(image_file).split('/')
        s3_filename = f'complete_mission_grid/{path_parts[-1]}'
        upload_to_s3(str(image_file), 'voto-figures', object_name=s3_filename, profile_name='produser')
