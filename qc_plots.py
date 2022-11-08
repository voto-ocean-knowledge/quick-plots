import numpy as np
import xarray as xr
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd


def plot_qc(nc, plots_dir):
    ds = xr.open_dataset(nc)
    attrs = ds.attrs
    fig_name_base = f'SEA{attrs["glider_serial"]}_M{attrs["deployment_id"]}'
    vars = list(ds)
    time = ds["time"]
    for var_name in vars:
        if var_name[-2:] == "qc":
            flag = ds[var_name]
            var = var_name[:-3]
            data = ds[var]
            meaning = np.empty(len(time), dtype=object)
            meaning[:] = "UNKNOWN"
            meaning[flag == 1] = "GOOD"
            meaning[flag == 9] = "MISSING"
            meaning[flag == 3] = "SUSPECT"
            meaning[flag == 4] = "FAIL"
            df = pd.DataFrame({"time": time, var: data, "flag": flag, "depth": ds.depth, "quality control": meaning})
            fig1 = px.line(df, x="time", y=var)
            fig1.update_traces(line=dict(color='rgba(50,50,50,0.2)'))
            fig2 = px.scatter(df, x="time", y=var, color="quality control", size="flag",
                              hover_data=['quality control'], symbol="flag",
                              color_discrete_sequence=["red", "green", "blue"])

            fig3 = go.Figure(data=fig1.data + fig2.data)
            fig3.write_html(plots_dir / f"{fig_name_base}_time_{var}_qc.html")
            fig = px.scatter(df, x="time", y="depth", color=var, size="flag",
                             hover_data=['quality control'], symbol="flag")
            fig.write_html(plots_dir / f"{fig_name_base}_depth_{var}_qc.html")
