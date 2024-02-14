
import pathlib
import sys
import pytest

script_dir = pathlib.Path(__file__).parent.absolute()

parent_dir = script_dir.parents[0]
sys.path.append(str(parent_dir))

import cmd_data_plots
cmd_data_list = list((parent_dir / 'tests/data').glob('*.log'))
@pytest.mark.parametrize("path_to_cmdlog", cmd_data_list)
def test_cmd_run(path_to_cmdlog):
  active_m1 = cmd_data_plots.load_cmd(path_to_cmdlog)
  drift_df, tot_time = cmd_data_plots.measure_drift(path_to_cmdlog)
  dst = cmd_data_plots.dst_data(path_to_cmdlog)
  cut, mins = cmd_data_plots.time_connections(path_to_cmdlog)
  assert len(cut.groupby('Cycle').count()) == len(tot_time)

