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
from gridded_plots import glider_locs_to_json, upload_to_s3

_log = logging.getLogger(__name__)
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')


def main():
    glider_locs_file = '/data/plots/glider_locs.json'
    try:
        to_process = pd.read_csv('../to_process.csv', dtype=int)
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
        mission_dir = f'/data/nrt-pyglider/SEA{glider}/M{mission}/gridfiles'
        try:
            nc_file = list(pathlib.Path(mission_dir).glob('*.nc'))[0]
        except IndexError:
            _log.error(f"nc file in {mission_dir} not found")
            continue
        nc_time = nc_file.lstat().st_mtime
        infile_time = 0
        in_files = list(pathlib.Path(f'/data/raw-nrt/SEA0{glider}/0000{mission}/C-Csv/').glob('*'))
        for file in in_files:
            if file.lstat().st_mtime > infile_time:
                infile_time = file.lstat().st_mtime
        if nc_time > infile_time:
            _log.info(f"No new SEA{glider} M{mission} files")
            continue
        _log.info(f"Processing SEA{glider} M{mission}")
        ds = xr.open_dataset(nc_file)
        glider_locs_to_json(ds)

    try:
        with open(glider_locs_file, 'r') as openfile:
            locs_dict_new = json.load(openfile)
    except FileNotFoundError:
        _log.warning("glider_locs.json not found.")
        locs_dict_new = {}
    if locs_dict_new != locs_dict_og:
        upload_to_s3(glider_locs_file, 'voto-figures',
                     object_name='nrt_glider_locs.json', profile_name='produser', image=False)


if __name__ == '__main__':
    main()
