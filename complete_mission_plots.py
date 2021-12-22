import sys
import os
import pathlib
import logging

script_dir = pathlib.Path(__file__).parent.absolute()
sys.path.append(str(script_dir))
os.chdir(script_dir)
from gridded_plots import upload_to_s3, create_plots, make_map

_log = logging.getLogger(__name__)
logging.basicConfig(filename='/data/log/complete_mission/plots.log',
                    filemode='a',
                    format='%(asctime)s %(levelname)-8s %(message)s',
                    level=logging.INFO,
                    datefmt='%Y-%m-%d %H:%M:%S')


def main():
    glider = sys.argv[1]
    mission = sys.argv[2]
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
    map_file = make_map(netcdf, image_file)
    if 'upload' in sys.argv:
        path_parts = str(image_file).split('/')
        if grid:
            root_dir = 'complete_mission_grid'
        else:
            root_dir = 'complete_mission_scatter'
        s3_filename = f'{root_dir}/{path_parts[-1]}'
        upload_to_s3(str(image_file), 'voto-figures', object_name=s3_filename, profile_name='produser')
        path_parts = str(map_file).split('/')
        s3_filename = f'{root_dir}/{path_parts[-1]}'
        upload_to_s3(str(map_file), 'voto-figures', object_name=s3_filename, profile_name='produser')


if __name__ == '__main__':
    main()
