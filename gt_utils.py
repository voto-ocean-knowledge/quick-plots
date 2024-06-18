import inspect
from inspect import currentframe as getframe


def transfer_nc_attrs(frame, input_xds, output_arr, output_name, **attrs):

    import xarray as xr

    not_dataarray = not isinstance(input_xds, xr.DataArray)
    no_parent_frame = inspect.getmodule(frame.f_back) is None
    if not_dataarray:
        if no_parent_frame:
            return output_arr
    else:
        if output_name is None:
            output_name = input_xds.name
        elif output_name.startswith("_"):
            output_name = input_xds.name + output_name

        attributes = input_xds.attrs.copy()
        history = "" if "history" not in attributes else attributes["history"]

        attributes.update({"history": history})
        attributes.update(attrs)

        keys = list(attributes.keys())
        for key in keys:
            if attributes[key] == "":
                attributes.pop(key)

        xds = xr.DataArray(
            data=output_arr,
            coords=input_xds.coords,
            dims=input_xds.dims,
            name=output_name,
            attrs=attributes,
        )

        return xds


def get_optimal_bins(depth, chunk_depth=50, round_up=True):
    """
    Uses depth data to estimate the optimal bin depths for gridding.

    Data is grouped in 50 m chunks (default for chunk_depth) where the
    average sequential depth difference is used to estimate the binning
    resolution for that chunk. The chunk binning resolution is rounded to
    the upper/lower 0.5 metres (specified by user).

    Parameters
    ----------
    depth : array
        A sequential array of depth (concatenated dives)
    chunk_depth : float=50
        chunk depth over which the bin sizes will be calculated
    round_up : True
        if True, rounds up to the nearest 0.5 m, else rounds down.

    Returns
    -------
    bins : array
    bin_avg_sampling_freq : float
        un-rounded depth weighted depth sampling frequency (for verbose use)

    """

    from numpy import abs, arange, array, ceil, diff, floor, isnan, nanmax, nanmedian

    y = array(depth)
    bins = []
    bin_avg_sampling_freq = []

    if round_up:
        round_func = ceil
    else:
        round_func = floor

    d0 = 0
    d1 = chunk_depth
    last_freq = 0.5
    while d0 <= nanmax(depth):
        i = (y > d0) & (y < d1)

        bin_avg_sampling_freq += (nanmedian(abs(diff(y[i]))),)
        bin_freq = round_func(bin_avg_sampling_freq[-1] * 2) / 2
        if bin_freq == 0:
            bin_freq = 0.5
        elif isnan(bin_freq):
            bin_freq = last_freq
        bin_step = arange(d0, d1, bin_freq).tolist()
        bins += bin_step

        d0 = bin_step[-1] + bin_freq
        d1 = d0 + chunk_depth

        last_freq = bin_freq

    return array(bins), nanmedian(bin_avg_sampling_freq)


def grid_data(
    x,
    y,
    var,
    bins=None,
    how="mean",
    interp_lim=6,
    verbose=False,
    return_xarray=True,
):
    """
    Grids the input variable to bins for depth/dens (y) and time/dive (x).
    The bins can be specified to be non-uniform to adapt to variable sampling
    intervals of the profile. It is useful to use the ``gt.plot.bin_size``
    function to identify the sampling intervals. The bins are averaged (mean)
    by default but can also be the ``median, std, count``,

    Parameters
    ----------
    x : np.array, dtype=float, shape=[n, ]
        The horizontal values by which to bin need to be in a psudeo discrete
        format already. Dive number or ``time_average_per_dive`` are the
        standard inputs for this variable. Has ``p`` unique values.
    y : np.array, dtype=float, shape=[n, ]
        The vertical values that will be binned; typically depth, but can also
        be density or any other variable.
    bins : np.array, dtype=float; shape=[q, ], default=[0 : 1 : max_depth ]
        Define the bin edges for y with this function. If not defined, defaults
        to one meter bins.
    how : str, defualt='mean'
        the string form of a function that can be applied to pandas.Groupby
        objects. These include ``mean, median, std, count``.
    interp_lim : int, default=6
        sets the maximum extent to which NaNs will be filled.

    Returns
    -------
    glider_section : xarray.DataArray, shape=[p, q]
        A 2D section in the format specified by ``ax_xarray`` input.

    Raises
    ------
    Userwarning
        Triggers when ``x`` does not have discrete values.
    """
    from numpy import array, c_, diff, unique
    from pandas import Series, cut
    from xarray import DataArray

    xvar, yvar = x.copy(), y.copy()
    z = Series(var)
    y = array(y)
    x = array(x)

    u = unique(x).size
    s = x.size
    if (u / s) > 0.2:
        raise UserWarning(
            "The x input array must be psuedo discrete (dives or dive_time). "
            "{:.0f}% of x is unique (max 20% unique)".format(u / s * 100)
        )

    chunk_depth = 50
    # -DB this might not work if the user uses anything other than depth, example
    # density. Chunk_depth would in that case apply to density, which will
    # probably have a range that is much smaller than 50.
    optimal_bins, avg_sample_freq = get_optimal_bins(y, chunk_depth)
    if bins is None:
        bins = optimal_bins

    # warning if bin average is smaller than average bin size
    # -DB this is not being raised as a warning. Instead just seems like useful
    # information conveyed to user. Further none of this works out if y is not
    # depth, since avg_sample freq will not make sense otherwise.
    if verbose:
        avg_bin_size = diff(bins).mean()
        print(
            (
                "Mean bin size = {:.2f}\n"
                "Mean depth binned ({} m) vertical sampling frequency = {:.2f}"
            ).format(avg_bin_size, chunk_depth, avg_sample_freq)
        )

    labels = c_[bins[:-1], bins[1:]].mean(axis=1)  # -DB creates the mean bin values
    bins = cut(y, bins, labels=labels)
    # -DB creates a new variable where instead of variable the bin category
    # is mentioned (sort of like a discretization)

    grp = Series(z).groupby([x, bins], observed=False)  # -DB put z into the many bins (like 2D hist)
    grp_agg = getattr(
        grp, how
    )()  # -DB basically does grp.how() or in this case grp.mean()
    gridded = grp_agg.unstack(level=0)
    gridded = gridded.reindex(labels.astype(float))

    if interp_lim > 0:
        gridded = gridded.interpolate(limit=interp_lim).bfill(limit=interp_lim)

    if not return_xarray:
        return gridded

    if return_xarray:
        dummy = transfer_nc_attrs(getframe(), var, var, "_vert_binned")

        xda = DataArray(gridded)
        if isinstance(var, DataArray):
            xda.attrs = dummy.attrs
            xda.name = dummy.name

        if isinstance(yvar, DataArray):
            y = xda.dims[0]
            xda[y].attrs = yvar.attrs
            xda = xda.rename({y: yvar.name})

        if isinstance(xvar, DataArray):
            x = xda.dims[1]
            xda[x].attrs = xvar.attrs
            xda = xda.rename({x: xvar.name})

        return xda
    
