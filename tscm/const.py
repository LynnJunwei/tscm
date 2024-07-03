# -*- coding: utf-8 -*-
# @Time    : 2024/7/2
# @Author  : Eric
import pandas as pd
from pathlib import Path


BODY_NAMES = [
    "Head", "Neck", "Chest", "Back", "Pelvis",
    "LUpperArm", "LLowerArm", "LHand",
    "RUpperArm", "RLowerArm", "RHand",
    "LThigh", "LLeg", "LFoot",
    "RThigh", "RLeg", "RFoot"]

MEAN_TSK_BODY_PARTS = {
    3: ["Chest", "Leg", "LowerArm"],
    4: ["Chest", "Arm", "Thigh", "Leg"],
    7: ["Head", "Pelvis", "LowerArm", "Hand", "Thigh", "Leg", "Foot"],
    8: ["Head", "Chest", "Back", "UpperArm", "LowerArm", "Hand", "Thigh", "Leg"],
}

MEAN_TSK_COEFFICIENT = {
    3: [0.50, 0.36, 0.14],
    4: [0.30, 0.20, 0.20, 0.20],
    7: [0.07, 0.35, 0.14, 0.05, 0.19, 0.13, 0.07],
    8: [0.07, 0.175, 0.175, 0.07, 0.07, 0.05, 0.19, 0.20],
}

SETPOINT_TYPE_DICT = {
    "setpoint": 0,
    "null_zone": 1
}


SETPOINT_INDEX_DICT = {
    "setpoint": {
        ("Summer", "0.8"): 0,
        ("Summer", "1.0"): 1,
        ("Summer", "1.1"): 2,
        ("Summer", "1.2"): 3,
        ("Summer", "1.3"): 4,
        ("WinterIndoor", "0.8"): 5,
        ("WinterIndoor", "1.0"): 6,
        ("WinterIndoor", "1.1"): 7,
        ("WinterIndoor", "1.2"): 8,
        ("WinterIndoor", "1.3"): 9,
    },
    'null_zone': {
        'male': 0,
        'female': 1,
    }
}


LIMIT_TYPE_DICT = {
    'lower': -1,
    'upper': 1,
    'neutral': 0
}


"""
coefficient_ucb: Add "Abdomen" coefficient same as "Pelvis".
setpoint_range_ucb: Add "Neck" range same as "Head". Add "Abdomen" range same as "Pelvis".
setpoint_ucb: Add "Neck" values same as "Head". Add "Abdomen" values same as "Pelvis".
"""
DATA_FOLDER_PATH = (Path(__file__) / '../../data').resolve()

SETPOINT_FILE = DATA_FOLDER_PATH / 'setpoint.csv'
SETPOINT = pd.read_csv(SETPOINT_FILE, index_col=[0, 1, 2])

COEFFICIENT_FILE = DATA_FOLDER_PATH / 'coefficient.csv'
COEFFICIENT = pd.read_csv(COEFFICIENT_FILE, index_col=0)


if __name__ == '__main__':
    print(SETPOINT_INDEX_DICT['setpoint'][('Summer', '1.0')])

    import numpy as np
    import pandas as pd

    a = pd.Series([36]*5, index=['Head', 'Neck', 'Chest', 'Back', 'Pelvis'])
    for b in MEAN_TSK_BODY_PARTS[7]:
        n = a[a.index.str.endswith(b)]
        print(n.isna())
        if all(n.isna()):
            raise ValueError('Body part: {} are missing!'.format(b))

    clo = 0.8
    print(np.where(abs(clo - 0.60) <= abs(clo - 1.27), 'Summer', 'WinterIndoor'))

    met = 1.23
    clo = 2
    met_value = np.piecewise(met,
                             [met < 0.9,
                              0.9 <= met < 1.05,
                              1.05 <= met < 1.15,
                              1.15 <= met < 1.25,
                              1.25 <= met],
                             [0.8, 1.0, 1.1, 1.2, 1.3]).item()
    clo_type = np.where(abs(clo - 0.60) <= abs(clo - 1.27), 'Summer', 'WinterIndoor').item()
    print()
    print(SETPOINT_INDEX_DICT['setpoint'][('Summer', '1.0')])

    print(SETPOINT.loc[(0, 1)].loc[LIMIT_TYPE_DICT['lower']])
