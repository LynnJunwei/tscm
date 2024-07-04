#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2023/10/16 15:43
# @Author  : Eric
from typing import Optional
import math

import pandas as pd
import numpy as np

from tscm.const import SETPOINT, SETPOINT_TYPE_DICT, SETPOINT_INDEX_DICT, LIMIT_TYPE_DICT, COEFFICIENT
from tscm.config import HumanConfig, LocalSensationConfig
from tscm.const import MEAN_TSK_BODY_PARTS, MEAN_TSK_COEFFICIENT


class SkinTemperatureProcessor:
    """
    A class for processing local skin temperature data. The class contains three read-only properties:

    - **mean_skin_temperature:** The mean skin temperature of a group of skin temperatures.
    - **setpoint:** The neutral skin temperature ranges for different body parts.
    - **local_sensation_sorted:** The local sensations calculated based on skin temperatures and their setpoints.
    """
    def __init__(self, skin_temperature: pd.Series,
                 delta_skin_temperature: Optional[pd.Series] = None,
                 delta_core_temperature: Optional[float] = None,
                 human_config: HumanConfig = HumanConfig(),
                 local_sensation_config: LocalSensationConfig = LocalSensationConfig()):
        """
        Args:
            skin_temperature: A series of skin temperatures.
            delta_skin_temperature: A series of delta skin temperatures.
            delta_core_temperature: A series of delta core temperatures.
            human_config: Configurations of human. Refer to class HumanConfig.
            local_sensation_config: Configurations for local local_sensation_sorted calculation. Refer to class LocalSensationConfig.
        """
        self.skin_temperature = skin_temperature
        self.delta_skin_temperature = delta_skin_temperature
        self.delta_core_temperature = delta_core_temperature

        self.body_parts = self.skin_temperature.index

        self.mean_skin_temperature_approach = local_sensation_config.mean_skin_temperature_approach
        self.dynamic = local_sensation_config.dynamic
        self.setpoint_type = local_sensation_config.setpoint_type

        self.met = human_config.met
        self.sex = human_config.sex
        self.clo = human_config.clo

    def get_mean_skin_temperature(self, skin_temperature: pd.Series) -> float:
        """
        Calculate the mean skin temperature of a group of skin temperatures.

        Args:
            skin_temperature: A series of skin temperatures.

        Returns:
            The mean skin temperature of a group of skin temperatures.

        Raises:
            ValueError: Skin temperature data needed not include in input.
        """
        body_parts_needed = MEAN_TSK_BODY_PARTS[self.mean_skin_temperature_approach]
        body_parts_coefficient = MEAN_TSK_COEFFICIENT[self.mean_skin_temperature_approach]

        body_parts_coefficient = pd.Series(body_parts_coefficient, index=body_parts_needed)

        skin_temperature_for_mean = pd.Series(index=body_parts_needed)
        for body_part_needed in body_parts_needed:
            skin_temperature_needed = skin_temperature[skin_temperature.index.str.endswith(body_part_needed)]

            if not all(skin_temperature.isna()):
                skin_temperature_for_mean[body_part_needed] = np.mean(skin_temperature_needed)
            else:
                raise ValueError('Body part: {} are missing!'.format(body_part_needed) +
                                 'Please change the approach for mean skin temperature calculation.')

        return (body_parts_coefficient * skin_temperature_for_mean).sum()

    def get_setpoint(self) -> tuple[pd.Series, pd.Series, pd.Series]:
        """
        Select the skin temperature setpoints (ranges) for different body parts.

        Returns:
            A tuple of skin temperature setpoints for neutral, upper and lower limits.
        """
        if self.setpoint_type == 'setpoint':
            met_class = min([1.3, 1.2, 1.1, 1.0, 0.8], key=lambda x: round(abs(x - self.met), 3))
            clo_type = np.where(abs(self.clo - 0.60) <= abs(self.clo - 1.27), 'Summer', 'WinterIndoor').item()
            setpoint = SETPOINT.loc[(SETPOINT_TYPE_DICT[self.setpoint_type],
                                     SETPOINT_INDEX_DICT[self.setpoint_type][(clo_type, '{:.1f}'.format(met_class))])]

        if self.setpoint_type == 'null_zone':
            setpoint = SETPOINT.loc[(SETPOINT_TYPE_DICT[self.setpoint_type],
                                     SETPOINT_INDEX_DICT[self.setpoint_type][self.sex])]

        setpoint_lower = setpoint.loc[LIMIT_TYPE_DICT['lower']]
        setpoint_upper = setpoint.loc[LIMIT_TYPE_DICT['upper']]
        setpoint_neutral = setpoint.loc[LIMIT_TYPE_DICT['neutral']]
        return setpoint_neutral, setpoint_upper, setpoint_lower

    def get_local_sensation(self) -> pd.Series:
        """
        Calculate the local sensations for different body parts.

        Returns:
            A series of local sensations for different body parts.
        """
        setpoint_neutral, setpoint_upper, setpoint_lower = self.get_setpoint()

        mean_skin_temperature = self.get_mean_skin_temperature(self.skin_temperature)
        mean_setpoint_upper = self.get_mean_skin_temperature(setpoint_upper)
        mean_setpoint_lower = self.get_mean_skin_temperature(setpoint_lower)

        if mean_skin_temperature >= mean_setpoint_upper:
            mean_skin_temperature_diff = mean_skin_temperature - mean_setpoint_upper
        elif mean_skin_temperature <= mean_setpoint_lower:
            mean_skin_temperature_diff = mean_skin_temperature - mean_setpoint_lower
        else:
            mean_skin_temperature_diff = 0

        local_sensation = pd.Series(index=self.body_parts)

        for body_part in self.body_parts:
            skin_temperature = self.skin_temperature[body_part]
            skin_temperature_diff = skin_temperature - setpoint_neutral[body_part]

            if skin_temperature_diff >= 0:
                c1 = COEFFICIENT.loc['C1_warm', body_part]
                k1 = COEFFICIENT.loc['K1_warm', body_part]
                skin_temperature_diff = skin_temperature - setpoint_upper[body_part]
                skin_temperature_diff = skin_temperature_diff if skin_temperature_diff >= 0 else 0

            else:
                c1 = COEFFICIENT.loc['C1_cool', body_part]
                k1 = COEFFICIENT.loc['K1_cool', body_part]
                skin_temperature_diff = skin_temperature - setpoint_lower[body_part]
                skin_temperature_diff = skin_temperature_diff if skin_temperature_diff <= 0 else 0

            exponent = -(c1 + k1) * skin_temperature_diff + k1 * mean_skin_temperature_diff
            local_sensation_i = 4 * (2 / (1 + math.exp(exponent)) - 1)

            # dynamic local_sensation_sorted
            if self.dynamic:
                delta_skin_temperature = self.delta_skin_temperature[body_part]

                c21 = COEFFICIENT.loc['C21', body_part]
                c22 = COEFFICIENT.loc['C22', body_part]
                c3 = COEFFICIENT.loc['C3', body_part]

                if delta_skin_temperature >= 0:
                    local_sensation_i += c22 * delta_skin_temperature
                else:
                    local_sensation_i += c21 * delta_skin_temperature
                local_sensation_i += c3 * self.delta_core_temperature

            # Set local_sensation_sorted limits
            local_sensation[body_part] = np.clip(local_sensation_i, -4, 4)

        return local_sensation

    @property
    def mean_skin_temperature(self) -> float:
        """The mean skin temperature."""
        return self.get_mean_skin_temperature(self.skin_temperature)

    @property
    def setpoint(self) -> pd.DataFrame:
        """A dataframe of skin temperature setpoints for neutral, upper and lower limits."""
        return pd.DataFrame(self.get_setpoint(), index=['setpoint_neutral', 'setpoint_upper', 'setpoint_lower'])

    @property
    def local_sensation(self) -> pd.Series:
        """A series of local sensations for different body parts."""
        return self.get_local_sensation()


if __name__ == '__main__':
    a = [1.3, 1.2, 1.1, 1.0, 0.8]
    b = 1.05
    print([round(abs(i - b), 3) for i in a])
    m = min(a, key=lambda x: round(abs(x - b), 3))
    print(m)
