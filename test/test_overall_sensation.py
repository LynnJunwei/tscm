# -*- coding: utf-8 -*-
# @Time    : 2024/7/5
# @Author  : Eric
import pandas as pd
from tscm.local_sensation import LocalSensationCalculator
from tscm.overall_sensation import OverallSensationCalculator
from tscm.config import HumanConfig, LocalSensationConfig, OverallSensationConfig

tsk_df = pd.read_csv('test_tsk.csv', index_col=0)

tsk_ser = tsk_df.iloc[1, :]

human_config = HumanConfig(sex='male', met=1, clo=0.5)
local_sensation_config = LocalSensationConfig(dynamic=False, setpoint_type='null_zone', mean_skin_temperature_approach=7)

local_ts_model = LocalSensationCalculator(skin_temperature=tsk_ser,
                                          human_config=human_config,
                                          local_sensation_config=local_sensation_config)
local_ts = local_ts_model.local_sensation
print(local_ts)

overall_sensation_config = OverallSensationConfig(smooth=False, smooth_alpha=2)
overall_ts_model = OverallSensationCalculator(local_sensation=local_ts,
                                              overall_sensation_config=overall_sensation_config)
whole_ts = overall_ts_model.overall_sensation
print(whole_ts)