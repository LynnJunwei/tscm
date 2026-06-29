# -*- coding: utf-8 -*-
import math

import pandas as pd
import numpy as np

from tscm.const import COEFFICIENT
from tscm.config import LocalComfortConfig


class LocalComfortCalculator:
    """
    A class for calculating local comfort based on local sensation and overall sensation.
    """
    def __init__(self,
                 local_sensation: pd.Series,
                 overall_sensation: float,
                 local_comfort_config: LocalComfortConfig = LocalComfortConfig()):
        """
        Args:
            local_sensation: A series of local sensations.
            overall_sensation: A float of overall sensation.
            local_comfort_config: Configuration of local comfort calculation. Refer to class LocalComfortConfig.
        """
        self.local_sensation = local_sensation
        self.overall_sensation = overall_sensation
        self.config = local_comfort_config

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
            if not self.config.exponential:
                n = 1

            c3 = c32 if s0 >= 0 else c31
            c7 = c72 if s0 >= 0 else c71

            max_comfort = c6 + c7 * s0_abs
            offset = c3 * s0_abs + c8
            left_slope = (max_comfort + 4) / ((-offset + 4) ** n)
            right_slope = (-max_comfort - 4) / ((offset + 4) ** n)
            s1_offset = s1 + offset
            local_comfort_i = (
                    (
                            (left_slope - right_slope) / (1 + math.exp(25 * s1_offset))
                            + right_slope
                    )
                    * (abs(s1_offset) ** n) * np.sign(s1_offset)
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