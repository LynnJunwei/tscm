# -*- coding: utf-8 -*-
# @Time    : 2024/7/3
# @Author  : Eric
import pandas as pd
import numpy as np

setpoint_null_zone = pd.read_csv('setpoint_null_zone.csv', index_col=0)

neutral_male = setpoint_null_zone.iloc[[0, 1], :].mean().to_frame().T
neutral_male.index = ['neutral_male']
neutral_female = setpoint_null_zone.iloc[[2, 3], :].mean().to_frame().T
neutral_female.index = ['neutral_female']
setpoint_null_zone = pd.concat([setpoint_null_zone, neutral_male, neutral_female])

setpoint_null_zone['SetpointType'] = 1
setpoint_null_zone['SetpointIndex'] = [0, 0, 1, 1, 0, 1]
setpoint_null_zone['LimitType'] = [-1, 1, -1, 1, 0, 0]
setpoint_null_zone.set_index(['SetpointType', 'SetpointIndex', 'LimitType'], inplace=True)
setpoint_null_zone.sort_index(inplace=True)
print(setpoint_null_zone)
setpoint_null_zone.to_csv('setpoint_null_zone_t.csv')