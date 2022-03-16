"""
Basic plotting functions to operate on gridded netCDF output by pyglider https://github.com/c-proof/pyglider
Based on work by Elizabeth Siddle and Callum Rollo https://github.com/ESiddle/basestation_plotting
"""
import sys
import os
import logging
import json
import gsw
import pandas as pd
import xarray as xr
from cmocean import cm as cmo
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib
import cartopy
import numpy as np
from pathlib import Path
import pandas
import datetime
from collections import defaultdict
import boto3
from botocore.exceptions import ClientError
from matplotlib import style
import pathlib
_log = logging.getLogger(__name__)

script_dir = pathlib.Path(__file__).parent.absolute()
os.chdir(script_dir)
style.use('presentation.mplstyle')


def sort_by_priority_list(values, priority):
    priority_dict = defaultdict(
        lambda: len(priority), zip(priority, range(len(priority)), ),
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
    #    'backscatter_700',
    'DOWNWELLING_PAR',
    #    'DOWN_IRRADIANCE380',
    #    'DOWN_IRRADIANCE490',
    #    'DOWN_IRRADIANCE532',
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

labels_dict = {'Celsius': '$^{\\circ}$C',
               'None': '',
               'ug m-3': 'µg m$^{-1}$',
               'umol l-1': 'µmol l$^{-1}$',
               'W/m^2/nm': 'W m$^{-2}$ nm$^{-1}$',
               'μE/m^2/s': 'µE m$^{-1}$ s$^{-1}$',
               '1': '',
               'kg m-3': 'kg m$^{-3}$',
               'g kg^-1': 'g kg$^{-1}$',
               'arbitrary': '',
               'backscatter_700': 'backscatter 700 nm',
               'oxygen_concentration': 'oxygen concentration',
               'DOWNWELLING_PAR': 'photosynthetically active radiation',
               'cdom': 'colored dissolved organic matter',
               'potential_density': 'potential density',
               'potential_temperature': 'potential temperature',
               'absolute_salinity': 'absolute salinity'
               }


def label_replace(lab):
    if lab in labels_dict.keys():
        lab = labels_dict[lab]
    return lab


def create_plots(nc, output_dir, grid):
    if not Path.exists(Path.absolute(output_dir)):
        output_dir.mkdir(parents=True)
    _log.info(f'opening {nc}')
    ds = xr.open_dataset(nc)
    ds = additional_vars(ds)
    a = list(ds.keys())  # list data variables in ds
    to_plot_unsort = list(
        set(a).intersection(glider_variables))  # find elements in glider_variables relevant to this dataset
    to_plot = sort_by_priority_list(to_plot_unsort, glider_variables)
    _log.info(f'will plot {to_plot}')
    # for i in range(len(to_plot)):
    #    plotter(ds, to_plot[i], cmap_dict[to_plot[i]], to_plot[i], output_dir)
    image_file = multiplotter(ds, to_plot, output_dir, glider=ds.glider_serial, mission=ds.deployment_id, grid=grid)
    # image_file = tempsal_scatter(ds, output_dir)
    return image_file


def additional_vars(ds):
    if 'absolute_salinity' not in list(ds):
        _log.info(f'adding absolute salinity variable')
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
    ax.set(xlim=sal_lim, ylim=temp_lim, xlabel=f'Salinity (psu)', ylabel='Temperature ($^{\\circ}$C)')
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
    _log.info(f'preparing {variable} for plot. std dev = {std_devs}, percentile = {percentile}')

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

        arr_max[arr_in > ul] = True
        arr_min[arr_in < ll] = True

    # nanpercentile
    if percentile:
        ll = np.nanpercentile(arr_in, percentile)
        ul = np.nanpercentile(arr_in, 100 - percentile)

        mask = (arr < ll) | (arr > ul)
        arr[mask] = np.nan

        arr_max[arr_in > ul] = True
        arr_min[arr_in < ll] = True
    # tbi
    arr_in[arr_min] = np.nanmin(arr)
    arr_in[arr_max] = np.nanmax(arr)
    dataset[variable].data = arr_in

    return dataset


# define a basic plotting function for the profiles
def plotter(dataset, variable, colourmap, title, plots_dir):
    """Create time depth profile coloured by desired variable

    Input:
    dataset: the name of the xarray dataset
    variable: the name of the data variable to be plotted
    colourmap: name of the colourmap to be used in profile
    title: variable name included as title to easily identify plot

    The intended use of the plotter function is to iterate over a list of variables,
    plotting a pcolormesh style plot for each variable, where each variable has a colourmap assigned using a dict"""

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


def multiplotter(dataset, variables, plots_dir, glider='', mission='', grid=True):
    """Create time depth profile coloured by desired variable

    Input:
    dataset: the name of the xarray dataset
    variable: the name of the data variable to be plotted
    colourmap: name of the colourmap to be used in profile
    title: variable name included as title to easily identify plot

    The intended use of the plotter function is to iterate over a list of variables,
    plotting a pcolormesh style plot for each variable, where each variable has a colourmap assigned using a dict"""
    if not grid:
        dataset = dataset.where(dataset.profile_direction < 0.)
        end = pandas.to_datetime(dataset.time.values[-1])
        dataset = dataset.sel(time=slice(end - datetime.timedelta(days=7), end))
    num_variables = len(variables)
    fig, axs = plt.subplots(num_variables, 1, figsize=(12, 3.5 * num_variables))
    axs = axs.ravel()
    for i, ax in enumerate(axs):
        variable = variables[i]
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
        if 'DOWN' in variable:
            vals = ds.values
            vals[vals < 0] = 0
            # Hack to replace bad PAR values that have been averaged in. Bad values are 9999 in og data
            vals[vals > 500] = np.nan
            if grid:
                pcol = ax.pcolor(ds.time.values, ds.depth, ds.values, cmap=colormap, shading='auto',
                                 norm=matplotlib.colors.LogNorm(vmin=np.nanmin(vals), vmax=np.nanmax(vals)))
            else:
                time = ds.time.values
                depth = ds.depth.values
                depth_grid = np.tile(depth, (len(time), 1)).T
                time_grid = np.tile(time, (len(depth), 1))
                pcol = ax.scatter(time_grid, depth_grid, c=vals, cmap=colormap,
                                  norm=matplotlib.colors.LogNorm(vmin=np.nanmin(vals), vmax=np.nanmax(vals)))
        else:
            if grid:
                pcol = ax.pcolor(ds.time.values, ds.depth, ds.values, cmap=colormap, shading='auto')
            else:
                time = ds.time.values
                depth = ds.depth.values
                depth_grid = np.tile(depth, (len(time), 1)).T
                time_grid = np.tile(time, (len(depth), 1))
                pcol = ax.scatter(time_grid, depth_grid, c=ds.values, cmap=colormap)

        var_sum = np.nansum(dataset[variable].data, 1)
        valid_depths = dataset[variable].depth.data[var_sum != 0.0]
        ax.set_ylim(valid_depths.max(), valid_depths.min())
        ax.set_title(label_replace(str(variable)))
        if grid:
            days = (1, 5, 10, 15, 20, 25)
        else:
            days = np.arange(1, 31)
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
    filename = plots_dir / f'SEA{glider}_M{mission}.png'
    _log.info(f'writing figure to {filename}')
    fig.savefig(filename, format='png', transparent=True)
    return filename


def scale_bar(ax, length=None, location=(0.5, 0.05), linewidth=3, coord=cartopy.crs.PlateCarree()):
    """
    ax is the axes to draw the scalebar on.
    length is the length of the scalebar in km.
    location is center of the scalebar in axis coordinates.
    (ie. 0.5 is the middle of the plot)
    linewidth is the thickness of the scalebar.
    """
    # Get the limits of the axis in lat long
    llx0, llx1, lly0, lly1 = ax.get_extent(coord)
    # Make tmc horizontally centred on the middle of the map,
    # vertically at scale bar location
    sbllx = (llx1 + llx0) / 2
    sblly = lly0 + (lly1 - lly0) * location[1]
    tmc = cartopy.crs.TransverseMercator(sbllx, sblly)
    # Get the extent of the plotted area in coordinates in metres
    x0, x1, y0, y1 = ax.get_extent(tmc)
    # Turn the specified scalebar location into coordinates in metres
    sbx = x0 + (x1 - x0) * location[0]
    sby = y0 + (y1 - y0) * location[1]

    # Calculate a scale bar length if none has been given
    # (Theres probably a more pythonic way of rounding the number but this works)
    if not length:
        length = (x1 - x0) / 5000  # in km
        ndim = int(np.floor(np.log10(length)))  # number of digits in number
        length = round(length, -ndim)  # round to 1sf

        # Returns numbers starting with the list
        def scale_number(x):
            if str(x)[0] in ['1', '2', '5']:
                return int(x)
            else:
                return scale_number(x - 10 ** ndim)

        length = scale_number(length)

    # Generate the x coordinate for the ends of the scalebar
    bar_xs = [sbx - length * 500, sbx + length * 500]
    # Plot the scalebar
    ax.plot(bar_xs, [sby, sby], transform=tmc, color='k', linewidth=linewidth)
    # Plot the scalebar label
    ax.text(sbx, sby, str(length) + ' km', transform=tmc,
            horizontalalignment='center', verticalalignment='bottom')


def make_map(nc, filename):
    dataset = xr.open_dataset(nc)
    lats = dataset.latitude.values
    lons = dataset.longitude.values
    times = dataset.time.values
    coord = cartopy.crs.AzimuthalEquidistant(central_longitude=np.mean(lons),
                                      central_latitude=np.mean(lats))
    pc = cartopy.crs.PlateCarree()
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111, projection=coord)
    ax.scatter(lons, lats, transform=pc, s=10)
    transform = cartopy.crs.PlateCarree()._as_mpl_transform(ax)

    ax.annotate(f'{dataset.glider_name}\n{str(times[-1])[:10]}',
                xy=(lons[-1], lats[-1]), xytext=(lons[-1]+1.2, lats[-1]+0.5),
                arrowprops=dict(facecolor='black',
                                arrowstyle="simple",
                                alpha=0.5),
                xycoords=transform,
                ha='right', va='top')

    lon_extend = 3
    lat_extend = 1
    lims = (np.nanmin(lons) - lon_extend, np.nanmax(lons) + lon_extend,
            np.nanmin(lats) - lat_extend, np.nanmax(lats) + lat_extend)
    ax.set_extent(lims, crs=pc)

    feature = cartopy.feature.NaturalEarthFeature(name='land', category='physical',
                                           scale='10m', edgecolor='black', facecolor='lightgreen')
    ax.add_feature(feature)
    gl = ax.gridlines(draw_labels=True,
                      linewidth=2, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = None
    gl.right_labels = None
    scale_bar(ax, location=(0.41, 0.05))

    fn_root, fn_ext = str(filename).split('.')
    filename_map = f"{fn_root}_map.{fn_ext}"
    _log.info(f"writing mapt to {filename_map}")
    fig.savefig(filename_map, format='png', transparent=True)
    return filename_map


def upload_to_s3(file_name, bucket, object_name=None, profile_name='voto:prod', image=True):
    """Upload a file to an S3 bucket
    Original function by Isabelle Giddy

    :param file_name: File to upload
    :param bucket: Bucket to upload to
    :param object_name: S3 object name. If not specified then file_name is used
    :param profile_name: name of AWS profile to use for credentials
    :return: True if file was uploaded, else False
    """
    boto3.setup_default_session(profile_name=profile_name)

    # If S3 object_name was not specified, use file_name
    if object_name is None:
        object_name = file_name

    # Upload the file
    s3_client = boto3.client('s3')
    try:
        if image:
            s3_client.upload_file(file_name, bucket, object_name, ExtraArgs={'Metadata': {'Content-Type': ' image/png'}})
        else:
            s3_client.upload_file(file_name, bucket, object_name)

        _log.info(f'uploading to s3 bucket {bucket} as {object_name}')
    except ClientError:
        _log.warning(f'could not upload {file_name} to S3')
        return False
    return True


def glider_locs_to_json(ds_grid, glider_locs_file="/data/plots/glider_locs.json"):
    profs = ds_grid.time.values[-10::2]
    ds = ds_grid.sel(time=profs)
    if Path(glider_locs_file).exists():
        with open(glider_locs_file, 'r') as openfile:
            locs_dict = json.load(openfile)
    else:
        locs_dict = {}
    latest_locs = {'lat': list(ds.latitude.values), 'lon': list(ds.latitude.values),
                   'time': list(ds.time.values.astype(str))}
    locs_dict[ds.attrs['glider_serial']] = latest_locs
    _log.info(f'Writing {ds.glider_serial} locations to {glider_locs_file}')
    with open(glider_locs_file, "w") as outfile:
        json.dump(locs_dict, outfile)

def count_dives():
    try:
        df_dives = pd.read_csv("/data/plots/glider_stats.csv")
    except FileNotFoundError:
        df_dives = pd.DataFrame({"datetime": [], "total_dives": []})
    data_dir = pathlib.Path("/data/data_raw")
    all_files = list(data_dir.glob("**/*.gli.sub.*"))
    print(len(all_files))
    all_dives = []
    for f in all_files:
        name = f.name
        parts = name.split('.')
        if parts[-1] == 'gz':
            suffix = parts[-2]
        else:
            suffix = parts[-1]
        try:
            dive_num = int(suffix)
            all_dives.append(name)
        except ValueError:
            continue
    all_dives.sort()
    uniq_dives = np.unique(all_dives)
    total_dives = len(uniq_dives)
    now = datetime.datetime.now()
    df2 = pd.DataFrame({"datetime": [now], "total_dives": [total_dives]})
    df_now = df_dives.append(df2)
    df_now['total_dives'] = df_now['total_dives'].astype(int)
    df_now.to_csv("/data/plots/glider_stats.csv", index=False)
    return total_dives
