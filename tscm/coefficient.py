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
DATA_FOLDER_PATH = (Path(__file__) / '../../data').resolve()
SETPOINT_FILE = DATA_FOLDER_PATH / 'setpoint.csv'
SETPOINT = pd.read_csv(SETPOINT_FILE, index_col=[0, 1, 2])

# coefficient_ucb_file = data_folder_path / 'coefficient.csv'


if __name__ == '__main__':
    print(SETPOINT)