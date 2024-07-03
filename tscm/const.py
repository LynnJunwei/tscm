# -*- coding: utf-8 -*-
# @Time    : 2024/7/2
# @Author  : Eric
import pandas as pd

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

SETPOINT_UCB = {
    "Summer": {
        0.8: {
            "Head": 35.34080045,
            "Neck": 35.34080045,
            "Chest": 35.00115051,
            "Back": 34.82662898,
            "Pelvis": 34.88948793,
            "LUpperArm": 33.02792105,
            "RUpperArm": 33.04651133,
            "LLowerArm": 31.29705059,
            "RLowerArm": 31.37314583,
            "LHand": 31.14142947,
            "RHand": 31.37301919,
            "LThigh": 33.64975231,
            "RThigh": 33.65118092,
            "LLeg": 30.91952386,
            "RLeg": 30.9407924,
            "LFoot": 31.06892236,
            "RFoot": 31.08127289
        },
        1.0: {
            "Head": 34.0, "Neck": 34.0, "Chest": 34.0, "Back": 34.0, "Pelvis": 34.0,
            "LUpperArm": 34.0, "LLowerArm": 34.0, "LHand": 34.0,
            "RUpperArm": 34.0, "RLowerArm": 34.0, "RHand": 34.0,
            "LThigh": 34.0, "LLeg": 34.0, "LFoot": 34.0,
            "RThigh": 34.0, "RLeg": 34.0, "RFoot": 34.0,
        },
        1.1: {
            "Head": 34.0, "Neck": 34.0, "Chest": 34.0, "Back": 34.0, "Pelvis": 34.0,
            "LUpperArm": 34.0, "LLowerArm": 34.0, "LHand": 34.0,
            "RUpperArm": 34.0, "RLowerArm": 34.0, "RHand": 34.0,
            "LThigh": 34.0, "LLeg": 34.0, "LFoot": 34.0,
            "RThigh": 34.0, "RLeg": 34.0, "RFoot": 34.0,
        },
        1.2: {
            "Head": 34.0, "Neck": 34.0, "Chest": 34.0, "Back": 34.0, "Pelvis": 34.0,
            "LUpperArm": 34.0, "LLowerArm": 34.0, "LHand": 34.0,
            "RUpperArm": 34.0, "RLowerArm": 34.0, "RHand": 34.0,
            "LThigh": 34.0, "LLeg": 34.0, "LFoot": 34.0,
            "RThigh": 34.0, "RLeg": 34.0, "RFoot": 34.0,
        },
    }
}

if __name__ == '__main__':
    import numpy as np

    a = pd.Series([36] * 17, index=BODY_NAMES)
    for b in MEAN_TSK_BODY_PARTS[7]:
        n = a[a.index.str.endswith(b)]
        print(n.isna())
        if all(n.isna()):
            raise ValueError('Body part: {} are missing!'.format(b))

    clo = 0.8
    print(np.where(abs(clo - 0.60) <= abs(clo - 1.27), 'Summer', 'WinterIndoor'))
