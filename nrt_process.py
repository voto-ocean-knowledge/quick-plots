import sys
import os
import pathlib
import logging

from glidertools_plots import public_plots

script_dir = pathlib.Path(__file__).parent.absolute()
sys.path.append(str(script_dir))
os.chdir(script_dir)
from gridded_plots import create_plots, make_map
from pilot_plots import battery_plots
from qc_plots import plot_qc

_log = logging.getLogger(__name__)
logging.basicConfig(filename='/data/log/nrt_plots.log',
                    filemode='a',
                    format='%(asctime)s %(levelname)-8s %(message)s',
                    level=logging.INFO,
                    datefmt='%Y-%m-%d %H:%M:%S')
glider_no_proc = []


def main():
    _log.info("Start nrt processing")
    all_glider_paths = pathlib.Path(f"/data/data_raw/nrt").glob("SEA*")
    for glider_path in all_glider_paths:
        glider = str(glider_path)[-3:].lstrip("0")
        if int(glider) in glider_no_proc:
            _log.info(f"SEA{glider} is not to be processed. Skipping")
            continue
        _log.info(f"Checking SEA{glider}")
        mission_paths = list(glider_path.glob("00*"))
        if not mission_paths:
            _log.warning(f"No missions found for SEA{glider}. Skipping")
            continue
        mission_paths.sort()
        mission = str(mission_paths[-1])[-3:].lstrip("0")
        _log.info(f"Checking SEA{glider} M{mission}")
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
        outdir = pathlib.Path(f'/data/plots/nrt/SEA{glider}/M{mission}/')
        if not outdir.exists():
            outdir.mkdir(parents=True)
        _log.info(f'start plotting {nc_file} ')
        image_file = create_plots(nc_file, outdir, False)
        make_map(nc_file, image_file)
        _log.info("start glidertools plots")
        ts_dir = f'/data/data_l0_pyglider/nrt/SEA{glider}/M{mission}/timeseries/'
        ts_file = list(pathlib.Path(ts_dir).glob('*.nc'))[0]
        public_plots(ts_file, outdir)
        #plot_qc(ts_file, outdir)
        _log.info("start pilot plots")
        combi_nav_files = list(pathlib.Path(f'/data/data_l0_pyglider/nrt/SEA{glider}/M{mission}/rawnc/').glob("*rawgli.parquet"))
        if combi_nav_files:
            battery_plots(combi_nav_files[0], outdir)
            _log.info("Finished pilot plots")
        else:
            _log.info("No combi nav file found for piloting plots")
    _log.info('End plot creation')


if __name__ == '__main__':
    main()
