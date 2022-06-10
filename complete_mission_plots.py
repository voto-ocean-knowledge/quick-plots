import sys
import os
import pathlib
import logging

script_dir = pathlib.Path(__file__).parent.absolute()
sys.path.append(str(script_dir))
os.chdir(script_dir)
from gridded_plots import create_plots, make_map

_log = logging.getLogger(__name__)


def complete_plots(glider, mission):
    logging.basicConfig(filename='/data/log/complete_mission/plots.log',
                        filemode='a',
                        format='%(asctime)s %(levelname)-8s %(message)s',
                        level=logging.INFO,
                        datefmt='%Y-%m-%d %H:%M:%S')
    mission_dir = f'/data/data_l0_pyglider/complete_mission/SEA{glider}/M{mission}/gridfiles/'
    try:
        netcdf = list(pathlib.Path(mission_dir).glob('*.nc'))[0]
    except IndexError:
        _log.error(f"nc file in {mission_dir} not found")
        return
    outdir = pathlib.Path(f'/data/plots/complete_mission/SEA{glider}/M{mission}/')
    if not outdir.exists():
        outdir.mkdir(parents=True)
    if 'scatter' in sys.argv:
        grid = False
    else:
        grid = True
    _log.info(f'start plotting {netcdf} grid = {grid}')
    image_file = create_plots(netcdf, outdir, grid)
    make_map(netcdf, image_file)


if __name__ == '__main__':
    complete_plots(sys.argv[1], sys.argv[2])
