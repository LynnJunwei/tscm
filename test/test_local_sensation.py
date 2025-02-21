# -*- coding: utf-8 -*-
# @Time    : 2024/7/3
# @Author  : Eric
import pandas as pd
from tscm.local_sensation import LocalSensationCalculator
from tscm.config import HumanConfig, LocalSensationConfig

tsk_df = pd.read_csv('test_tsk.csv', index_col=0)

tsk_ser = tsk_df.iloc[-1, :]

human_config = HumanConfig(sex='male', met=1, clo=0.5)
local_sensation_config = LocalSensationConfig(dynamic=False, setpoint_type='setpoint', mean_skin_temperature_approach=7)

local_ts_model = LocalSensationCalculator(skin_temperature=tsk_ser,
                                          human_config=human_config,
                                          local_sensation_config=local_sensation_config)
local_ts = local_ts_model.local_sensation
print(pd.concat([tsk_ser, local_ts], axis=1))
print(local_ts_model.setpoint)
print(local_ts.index.str.endswith('Hand').sum())