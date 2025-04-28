import pathlib
from cmd_data_plots import command_cosole_log_plots
from glidertools_plots import public_plots
import matplotlib.pyplot as plt
import logging

from gridded_plots import create_plots, make_map

_log = logging.getLogger(__name__)
bad_missions = ((57, 75), (70, 29), (66, 45))

if __name__ == '__main__':
    logging.basicConfig(filename='/data/log/nrt_plots.log',
                        filemode='a',
                        format='%(asctime)s %(levelname)-8s %(message)s',
                        level=logging.INFO,
                        datefmt='%Y-%m-%d %H:%M:%S')
    reprocess = False
    base_dir = '/data/data_l0_pyglider/nrt/'
    ts_files = list(pathlib.Path('/data/data_l0_pyglider/nrt/').rglob("mission_timeseries.nc"))
    for ts_file in ts_files:
        parts = ts_file.parts
        platform_serial = parts[-4]
        mission = int(parts[-3][1:])
        if (int(platform_serial[-3:]), mission) in bad_missions:
            _log.info(f"this missions is to be skipped {platform_serial} M{mission}")
            continue
        outdir = pathlib.Path(f"/data/plots/nrt/{platform_serial}/M{mission}/")
        _log.info(f"Check {platform_serial}, M{mission}")
        if not outdir.exists():
            outdir.mkdir(parents=True)
        files_parts = list(ts_file.parts)
        files_parts[-1] = "gridded.nc"
        files_parts[-2] = "gridfiles"
        nc_file = pathlib.Path("/".join(files_parts))
        filename = outdir / f'{platform_serial}_M{mission}_gt.png'
        if not filename.exists() or reprocess:
            _log.info(f"Process {platform_serial}, M{mission}")
            public_plots(ts_file, outdir)
            _log.info(f'start plotting {nc_file} ')
            image_file = create_plots(nc_file, outdir, False)
            make_map(nc_file, image_file)
        filename = outdir / f'{platform_serial}_M{mission}_cmd_log.png'
        if not filename.exists() or reprocess:
            _log.info(f"Process command console plots {platform_serial}, M{mission}")
            command_cosole_log_plots(platform_serial, mission, outdir)
        plt.close('all')
