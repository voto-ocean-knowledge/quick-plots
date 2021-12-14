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
    locs_dict_new = locs_dict_og.copy()
    for i, row in to_process.iterrows():
        glider = str(row.glider)
        mission = str(row.mission)
        nc_file = f'/data/nrt-pyglider/SEA{glider}/M{mission}/gridfiles/sea{glider}_m{mission}_realtime_grid.nc'
        try:
            ds = xr.open_dataset(nc_file)
        except FileNotFoundError:
            _log.error(f"File {nc_file} not found")
            continue
        locs_dict_new = glider_locs_to_json(ds)
    if locs_dict_new != locs_dict_og:
        upload_to_s3(locs_dict_new, 'voto-figures', object_name='nrt_glider_locs.json', profile_name='produser')


if __name__ == '__main__':
    main()
