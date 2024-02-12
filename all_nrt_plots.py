import pathlib
from cmd_data_plots import command_cosole_log_plots
from glidertools_plots import public_plots
import matplotlib.pyplot as plt
import logging
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
        glider = int(parts[-4][3:])
        mission = int(parts[-3][1:])
        if (glider, mission) in bad_missions:
            _log.info(f"this missions is to be skipped SEA{glider} M{mission}")
            continue
        outdir = pathlib.Path(f"/data/plots/nrt/SEA{glider}/M{mission}/")
        _log.info(f"process for SEA{glider}, M{mission}")
        if not outdir.exists():
            outdir.mkdir(parents=True)
        filename = outdir / f'SEA{glider}_M{mission}_gt.png'
        if not filename.exists() or reprocess:
            public_plots(ts_file, outdir)
        filename = outdir / f'SEA{glider}_M{mission}_cmd_log.png'
        if not filename.exists() or reprocess:
            command_cosole_log_plots(glider, mission, outdir)
        plt.close('all')
