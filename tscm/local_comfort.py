# -*- coding: utf-8 -*-
# @Time    : 2025/3/19
# @Author  : Eric
import math

import pandas as pd
import numpy as np


class LocalComfortCalculator:
    """
    A class for processing local skin temperature data. The class contains three read-only properties:

    - **mean_skin_temperature:** The mean skin temperature of a group of skin temperatures.
    - **setpoint:** The neutral skin temperature ranges for different body parts.
    - **local_sensation_sorted:** The local sensations calculated based on skin temperatures and their setpoints.
    """
    def __init__(self,
                 local_sensation: pd.Series,
                 overall_sensation: float):
        """
        Args:
            local_sensation: A series of local sensations.
        """
        self.local_sensation = local_sensation
        self.overall_sensation = overall_sensation

    def get_local_comfort(self) -> pd.Series:
        """
        Calculate the local comfort based on local sensation and overall sensation.

        Returns:
            A series of local comfort.
        """

        return self.local_sensation + self.overall_sensation