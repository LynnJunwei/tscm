# -*- coding: utf-8 -*-
# @Time    : 2025/3/19
# @Author  : Eric
import math

import pandas as pd
import numpy as np

from tscm.const import COEFFICIENT


class LocalComfortCalculator:
    """
    A class for calculating local comfort based on local sensation and overall sensation.
    """
    def __init__(self,
                 local_sensation: pd.Series,
                 overall_sensation: float):
        """
        Args:
            local_sensation: A series of local sensations.
            overall_sensation: A float of overall sensation.
        """
        self.local_sensation = local_sensation
        self.overall_sensation = overall_sensation

        self.body_parts = self.local_sensation.index

    def get_local_comfort(self) -> pd.Series:
        """
        Calculate the local comfort based on local sensation and overall sensation.

        Returns:
            A series of local comfort.
        """
        local_comfort = pd.Series(index=self.body_parts)

        for body_part in self.body_parts:
            s1 = self.local_sensation[body_part]
            s0 = self.overall_sensation
            s0_abs = abs(s0)
            c31 = COEFFICIENT.loc['C31', body_part]
            c32 = COEFFICIENT.loc['C32', body_part]
            c6 = COEFFICIENT.loc['C6', body_part]
            c71 = COEFFICIENT.loc['C71', body_part]
            c72 = COEFFICIENT.loc['C72', body_part]
            c8 = COEFFICIENT.loc['C8', body_part]
            n = COEFFICIENT.loc['n', body_part]

            c3 = c32 if s0 >= 0 else c31
            c7 = c72 if s0 >= 0 else c71

            max_comfort = c6 + c7 * s0_abs
            offset = c3 * s0_abs + c8
            left_slope = (-4 - max_comfort) / (abs(-4 + offset) ** n)
            right_slope = (-4 - max_comfort) / (abs(4 + offset) ** n)
            s1_offset = s1 + offset

            local_comfort_i = (
                    (
                            (left_slope - right_slope) / (1 + math.exp(25 * s1_offset))
                            + right_slope
                    )
                    * (abs(s1_offset) ** n)
                    + max_comfort
            )
            local_comfort[body_part] = np.clip(local_comfort_i, -4, 4)

        return local_comfort

    @property
    def local_comfort(self) -> pd.Series:
        """
        Return the local comfort based on local sensation and overall sensation.

        Returns:
            A series of local comfort.
        """
        return self.get_local_comfort()

if __name__ == '__main__':
    local_sensation = pd.Series({
        'Head': 4,
        'Chest': -1,
        'Back': -1,
        'Pelvis': -1,
        'LUpperArm': -4,
        'LThigh': -1,
        'LLeg': 0.3,
        'LFoot': -4,
    })
    overall_sensation = -3
    local_comfort_calculator = LocalComfortCalculator(local_sensation=local_sensation,
                                                      overall_sensation=overall_sensation)
    local_comfort1 = local_comfort_calculator.local_comfort
    print(local_comfort1)

    local_comforts = []
    for i in np.arange(-4, 4.1, 0.5):
        local_sensation = pd.Series({
            'Head': 0.5,
            'Chest': 1,
            'Back': 0,
            'Pelvis': 0,
            'LUpperArm': i,
            'LThigh': 0.5,
            'LLeg': 0.3,
            'LFoot': 0,
        })
        overall_sensation = local_sensation.mean()
        local_comfort_calculator = LocalComfortCalculator(local_sensation=local_sensation,
                                                          overall_sensation=overall_sensation)
        local_comfort = local_comfort_calculator.local_comfort
        local_comforts.append(local_comfort['Pelvis'])
        print((local_comfort.sort_values(ascending=True)[0] + local_comfort.sort_values(ascending=True)[1])/2)
    print(local_comforts)

    print((8) ** (7/5))