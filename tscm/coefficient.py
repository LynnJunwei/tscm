#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2023/9/29 15:38
# @Author  : Eric
import pandas as pd
from pathlib import Path

"""
coefficient_ucb: Add "Abdomen" coefficient same as "Pelvis".
setpoint_range_ucb: Add "Neck" range same as "Head". Add "Abdomen" range same as "Pelvis".
setpoint_ucb: Add "Neck" values same as "Head". Add "Abdomen" values same as "Pelvis".
"""
data_folder_path = (Path(__file__) / '../../data').resolve()
coefficient_ucb_file = data_folder_path / 'coefficient_ucb.csv'
setpoint_ucb_file = data_folder_path / 'setpoint_ucb.csv'
setpoint_range_ucb_file = data_folder_path / 'setpoint_range_ucb.csv'
null_zone_file = data_folder_path / 'setpoint_null_zone.csv'
setpoint_jos3_file = data_folder_path / 'setpoint_jos3.csv'

coefficient_ucb = pd.read_csv(coefficient_ucb_file, index_col=0)
setpoint_ucb = pd.read_csv(setpoint_ucb_file, index_col=0)
setpoint_range_ucb = pd.read_csv(setpoint_range_ucb_file, index_col=0)
null_zone_range = pd.read_csv(null_zone_file, index_col=0)
setpoint_jos3 = pd.read_csv(setpoint_jos3_file, index_col=0)
setpoint_jos3.replace({'LArm': 'LLowerArm', 'RArm': 'RLowerArm',
                       'LShoulder': 'LUpperArm', 'RShoulder': 'RUpperArm'},
                      inplace=True)


if __name__ == '__main__':
    print(coefficient_ucb)
    print(setpoint_ucb)
    print(setpoint_range_ucb)
    print(null_zone_range)