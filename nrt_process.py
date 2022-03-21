import sys
import os
import json
import pathlib
import pandas as pd
import xarray as xr
import logging

script_dir = pathlib.Path(__file__).parent.absolute()
sys.path.append(str(script_dir))
os.chdir(script_dir)
from gridded_plots import glider_locs_to_json, upload_to_s3, create_plots, make_map, count_dives
from banner_map import create_map

_log = logging.getLogger(__name__)
logging.basicConfig(filename='/data/log/nrt_plots.log',
                    filemode='a',
                    format='%(asctime)s %(levelname)-8s %(message)s',
                    level=logging.INFO,
                    datefmt='%Y-%m-%d %H:%M:%S')


def main():
    _log.info('Starting plot creation')
    glider_locs_file = '/data/plots/glider_locs.json'
    try:
        to_process = pd.read_csv('/home/pipeline/to_process.csv', dtype=int)
    except FileNotFoundError:
        _log.error("to_process.csv not found")
        return
    try:
        with open(glider_locs_file, 'r') as openfile:
            locs_dict_og = json.load(openfile)
    except FileNotFoundError:
        _log.warning("glider_locs.json not found. Making a new one")
        locs_dict_og = {}
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
        map_file = make_map(nc_file, image_file)
        path_parts = str(image_file).split('/')
        root_dir = 'nrt_mission_scatter'
        s3_filename = f'{root_dir}/{path_parts[-1]}'
        upload_to_s3(str(image_file), 'voto-figures', object_name=s3_filename, profile_name='produser')
        path_parts = str(map_file).split('/')
        s3_filename = f'{root_dir}/{path_parts[-1]}'
        upload_to_s3(str(map_file), 'voto-figures', object_name=s3_filename, profile_name='produser')

    try:
        with open(glider_locs_file, 'r') as openfile:
            locs_dict_new = json.load(openfile)
    except FileNotFoundError:
        _log.warning("glider_locs.json not found.")
        locs_dict_new = {}
    if locs_dict_new != locs_dict_og:
        upload_to_s3(glider_locs_file, 'voto-figures',
                     object_name='nrt_glider_locs.json', profile_name='produser', image=False)
    try:
        total_dives = count_dives()
        _log.info(f"total dives: {total_dives}")
    except:
        _log.warning("count of total dives failed")
    map_file = create_map()
    upload_to_s3(map_file, 'voto-figures', object_name='banner-map.png', profile_name='produser', image=True)
    _log.info('End plot creation')


if __name__ == '__main__':
    main()
