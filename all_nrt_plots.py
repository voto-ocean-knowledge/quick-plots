import pathlib
from cmd_data_plots import command_cosole_log_plots
from glidertools_plots import public_plots
import matplotlib.pyplot as plt

if __name__ == '__main__':
    base_dir = '/data/data_l0_pyglider/nrt/'
    ts_files = list(pathlib.Path('/data/data_l0_pyglider/nrt/').rglob("mission_timeseries.nc"))
    for ts_file in ts_files:
        parts = ts_file.parts
        glider = int(parts[-4][3:])
        mission = int(parts[-3][1:])
        outdir = pathlib.Path(f"/data/plots/nrt/SEA{glider}/M{mission}/")
        if not outdir.exists():
            outdir.mkdir(parents=True)
        public_plots(ts_file, outdir)
        command_cosole_log_plots(glider, mission, outdir)
        plt.close('all')
