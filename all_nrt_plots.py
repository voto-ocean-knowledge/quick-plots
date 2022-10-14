import pathlib
from glidertools_plots import public_plots

if __name__ == '__main__':
    base_dir = '/data/data_l0_pyglider/nrt/'
    ts_files = list(pathlib.Path('/data/data_l0_pyglider/nrt/').rglob("mission_timeseries.nc"))
    for ts_file in ts_files:
        parts = ts_file.parts
        glider = int(parts[-4][3:])
        mission = int(parts[-3][1:])
        outdir = pathlib.Path(f"/data/plots/nrt/SEA{glider}/M{mission}/")
        public_plots(ts_file, outdir)
