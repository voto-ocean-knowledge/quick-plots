import sys
import numpy as np
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import datetime
import pandas as pd
import xarray as xr
from gridded_plots import additional_vars, glider_variables, sort_by_priority_list, cmap_dict, label_replace

sys.path.append("/home/pipeline/GliderTools")
# noinspection PyUnresolvedReferences
import glidertools as gt


def public_plots(nc, plots_dir):
    ds = xr.open_dataset(nc)
    # Clean out any bad times
    end = pd.to_datetime(ds.time.max().values)
    ds = ds.sel(time=slice(end - datetime.timedelta(days=60), end))
    # Prepare a variable of averaged time per profile. This is used in plotting later
    profile_time = ds.time.values.copy()
    profile_index = ds.profile_index
    for profile in np.unique(profile_index.values):
        if np.isnan(profile):
            continue
        mean_time = ds.time[profile_index == profile].mean().values
        new_times = np.empty((len(ds.time[profile_index == profile])), dtype='datetime64[ns]')
        new_times[:] = mean_time
        profile_time[profile_index == profile] = new_times
    ds["profile_mean_time"] = profile_time
    # Add additional derived variables for plotting
    ds = additional_vars(ds)
    a = list(ds.keys())  # list data variables in ds
    to_plot_unsort = list(
        set(a).intersection(glider_variables))  # find elements in glider_variables relevant to this dataset
    variables = sort_by_priority_list(to_plot_unsort, glider_variables)
    num_variables = len(variables)
    fig, axs = plt.subplots(num_variables, 1, figsize=(12, 3.5 * num_variables), sharex="col")
    axs = axs.ravel()
    for i, ax in enumerate(axs):
        variable = variables[i]
        # GliderTools cleaning step. Ideally this would be a wrapper, but wrapper doesn't do both IQR and std dev atm
        var = ds[variable]
        var_iqr = gt.cleaning.outlier_bounds_iqr(var, multiplier=1.5)
        var_std = gt.cleaning.outlier_bounds_std(var, multiplier=1.5)
        mask = np.isnan(var_iqr) + np.isnan(var_std)
        var_qc = var.copy()
        var_qc.values[mask] = np.nan
        colormap = cmap_dict[variable]
        ax, im = gt.plot(ds.profile_mean_time, ds.depth, var_qc, cmap=colormap, robust=True, ax=ax)
        valid_depths = ds.depth.values[~np.isnan(ds[variable])]
        ax.set_ylim(valid_depths.max(), 0)
        ax.set_title(label_replace(str(variable)))
        extent = ds.time.max() - ds.time.min()
        total_days = int(extent.values) / (24 * 60 * 60 * 1e9)
        if total_days > 30:
            days = np.arange(0, 40, 5)
        elif total_days > 14:
            days = np.arange(0, 40, 2)
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
        ax.collections[-1].colorbar.remove()
        plt.colorbar(mappable=im, ax=ax, label=label_replace(ds[variable].units), aspect=13, pad=0.02)
    plt.tight_layout()
    filename = plots_dir / f'SEA{ds.glider_serial}_M{ds.deployment_id}_gt.png'
    plt.savefig(filename, format='png', transparent=True)
