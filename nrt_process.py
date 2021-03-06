import sys
import os
import pathlib
import pandas as pd
import xarray as xr
import logging

script_dir = pathlib.Path(__file__).parent.absolute()
sys.path.append(str(script_dir))
os.chdir(script_dir)
from gridded_plots import glider_locs_to_json, create_plots, make_map
from pilot_plots import battery_plots

_log = logging.getLogger(__name__)
logging.basicConfig(filename='/data/log/nrt_plots.log',
                    filemode='a',
                    format='%(asctime)s %(levelname)-8s %(message)s',
                    level=logging.INFO,
                    datefmt='%Y-%m-%d %H:%M:%S')


def main():
    _log.info('Starting plot creation')
    try:
        to_process = pd.read_csv('/home/pipeline/to_process.csv', dtype=int)
    except FileNotFoundError:
        _log.error("to_process.csv not found")
        return
    for i, row in to_process.iterrows():
        glider = str(row.glider)
        mission = str(row.mission)
        mission_dir = f'/data/data_l0_pyglider/nrt/SEA{glider}/M{mission}/gridfiles/'
        try:
            nc_file = list(pathlib.Path(mission_dir).glob('*.nc'))[0]
        except IndexError:
            _log.error(f"nc file in {mission_dir} not found")
            continue
        nc_time = nc_file.lstat().st_mtime
        infile_time = 0
        in_files = list(pathlib.Path(f'/data/plots/nrt/SEA{glider}/M{mission}/').glob('*.png'))
        for file in in_files:
            if file.lstat().st_mtime > infile_time:
                infile_time = file.lstat().st_mtime
        if nc_time < infile_time:
            _log.info(f"SEA{glider} M{mission} unchanged. No plotting")
            continue
        _log.info(f"Processing SEA{glider} M{mission}")
        ds = xr.open_dataset(nc_file)
        glider_locs_to_json(ds)
        outdir = pathlib.Path(f'/data/plots/nrt/SEA{glider}/M{mission}/')
        if not outdir.exists():
            outdir.mkdir(parents=True)
        _log.info(f'start plotting {nc_file} ')
        image_file = create_plots(nc_file, outdir, False)
        make_map(nc_file, image_file)
        _log.info("start pilot plots")
        combi_nav_files = list(pathlib.Path(f'/data/data_l0_pyglider/nrt/SEA{glider}/M{mission}/rawnc/').glob("*rawgli.nc"))
        if combi_nav_files:
            battery_plots(combi_nav_files[0], outdir)
            _log.info("Finished pilot plots")
        else:
            _log.info("No combi nav file found for piloting plots")
    _log.info('End plot creation')


if __name__ == '__main__':
    main()
