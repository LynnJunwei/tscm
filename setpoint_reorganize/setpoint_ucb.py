# -*- coding: utf-8 -*-
# @Time    : 2024/7/3
# @Author  : Eric
import pandas as pd

setpoint_ucb = pd.read_csv('setpoint_ucb.csv', index_col=0)
setpoint_range_ucb = pd.read_csv('setpoint_range_ucb.csv', index_col=0)
lower = setpoint_ucb - setpoint_range_ucb.iloc[0, :]
upper = setpoint_ucb + setpoint_range_ucb.iloc[1, :]
neutral = setpoint_ucb
print(setpoint_range_ucb)
lower['LimitType'] = -1
lower['SetpointType'] = 0
lower['SetpointIndex'] = list(range(len(lower.index)))
upper['LimitType'] = 1
upper['SetpointType'] = 0
upper['SetpointIndex'] = list(range(len(upper.index)))
neutral['LimitType'] = 0
neutral['SetpointType'] = 0
neutral['SetpointIndex'] = list(range(len(neutral.index)))

setpoint = pd.concat([lower, upper, neutral])
setpoint.set_index(['SetpointType', 'SetpointIndex', 'LimitType'], inplace=True)
setpoint.sort_index(inplace=True)

print(setpoint)
setpoint.to_csv('setpoint_ucb_t.csv')