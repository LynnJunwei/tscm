#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2023/10/16 15:43
# @Author  : Eric
from typing import Optional
import math

import pandas as pd
import numpy as np

from tscm.coefficient import coefficient_ucb, setpoint_ucb, setpoint_range_ucb, null_zone_range, setpoint_jos3
from tscm.config import HumanConfig, LocalSensationConfig
from tscm.const import MEAN_TSK_BODY_PARTS, MEAN_TSK_COEFFICIENT


class SkinTemperatureProcessor:
    """Calculate local sensations for different body parts based on their corresponding skin temperature."""
    def __init__(self, skin_temperature: pd.Series,
                 delta_skin_temperature: Optional[pd.Series],
                 delta_core_temperature: Optional[float],
                 human_config: HumanConfig,
                 local_sensation_config: LocalSensationConfig):
        """
        Args:
            skin_temperature: A series of skin temperatures.
            delta_skin_temperature: A series of delta skin temperatures.
            delta_core_temperature: A series of delta core temperatures.
            human_config: Configurations of human. Refer to class HumanConfig.
            local_sensation_config: Configurations for local sensation calculation. Refer to class LocalSensationConfig.
        """
        self.skin_temperature = skin_temperature
        self.delta_skin_temperature = delta_skin_temperature
        self.delta_core_temperature = delta_core_temperature

        self.body_parts = skin_temperature.index

        self.mean_skin_temperature_approach = local_sensation_config.mean_skin_temperature_approach
        self.mean_skin_temperature_substitute = local_sensation_config.mean_skin_temperature_substitute
        self.dynamic = local_sensation_config.dynamic
        self.setpoint_type = local_sensation_config.setpoint_type

        self.human_config = human_config

    def get_mean_skin_temperature(self, skin_temperature: pd.Series):
        """
        Return the mean skin temperature of a group of skin temperatures.

        Args:
            skin_temperature: A series of skin temperatures.

        Raises:
            ValueError: Index number of mean skin temperature calculation method not in {3, 4, 7, 8}.
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
                raise ValueError('Body part: {} are missing!'.format(body_part_needed))

        return (body_parts_coefficient * skin_temperature_for_mean).sum()

    def get_setpoint(self):
        if self.setpoint_type == 'setpoint':
            met_value = np.piecewise(self.human_config.met_,
                                     [self.human_config.met_ < 0.9,
                                      0.9 <= self.human_config.met_ < 1.05,
                                      1.05 <= self.human_config.met_ < 1.15,
                                      1.15 <= self.human_config.met_ < 1.25,
                                      1.25 <= self.human_config.met_],
                                     [0.8, 1.0, 1.1, 1.2, 1.3])
            clo_type = np.where(abs(self.human_config.clo - 0.60) <= abs(self.human_config.clo - 1.27),
                                'Summer', 'WinterIndoor')
            setpoint_neutral = setpoint_ucb.loc['{}-Met_{:.1f}'.format(self.human_config.clo_type, met_value), :]
            setpoint_upper = setpoint_neutral + setpoint_range_ucb.loc['upper_limit_delta', :]
            setpoint_lower = setpoint_neutral - setpoint_range_ucb.loc['lower_limit_delta', :]
            return setpoint_neutral, setpoint_upper, setpoint_lower

        if self.setpoint_type == 'null_zone':
            setpoint_upper = null_zone_range.loc['upper_limit_{}'.format(self.human_config.sex), :]
            setpoint_lower = null_zone_range.loc['lower_limit_{}'.format(self.human_config.sex), :]
            setpoint_neutral = (setpoint_upper + setpoint_lower) / 2
            return setpoint_neutral, setpoint_upper, setpoint_lower

        if self.setpoint_type == 'setpoint_modified':
            con1 = setpoint_jos3['met'] == np.round(self.human_config.met_, 1)
            con2 = setpoint_jos3['clo'] == np.round(self.human_config.clo * 2, 1) / 2
            con3 = setpoint_jos3['sex'] == self.human_config.sex
            setpoint_df = setpoint_jos3[con1 & con2 & con3]
            setpoint_neutral = pd.Series(setpoint_df['setpoint_neutral'].tolist(), index=setpoint_df['body_part'])
            setpoint_upper = pd.Series(setpoint_df['setpoint_upper'].tolist(), index=setpoint_df['body_part'])
            setpoint_lower = pd.Series(setpoint_df['setpoint_lower'].tolist(), index=setpoint_df['body_part'])
            return setpoint_neutral, setpoint_upper, setpoint_lower

    def get_local_sensation(self):
        mean_skin_temperature = self.get_mean_skin_temperature(self.skin_temperature)
        setpoint_neutral, setpoint_upper, setpoint_lower = self.get_setpoint()
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
                c1 = coefficient_ucb.loc['C1_warm', body_part]
                k1 = coefficient_ucb.loc['K1_warm', body_part]
                skin_temperature_diff = skin_temperature - setpoint_upper[body_part]
                skin_temperature_diff = skin_temperature_diff if skin_temperature_diff >= 0 else 0

            else:
                c1 = coefficient_ucb.loc['C1_cool', body_part]
                k1 = coefficient_ucb.loc['K1_cool', body_part]
                skin_temperature_diff = skin_temperature - setpoint_lower[body_part]
                skin_temperature_diff = skin_temperature_diff if skin_temperature_diff <= 0 else 0

            exponent = -(c1 + k1) * skin_temperature_diff + k1 * mean_skin_temperature_diff
            local_sensation_i = 4 * (2 / (1 + math.exp(exponent)) - 1)

            # dynamic sensation
            if self.dynamic:
                delta_skin_temperature = self.delta_skin_temperature[body_part]

                c21 = coefficient_ucb.loc['C21', body_part]
                c22 = coefficient_ucb.loc['C22', body_part]
                c3 = coefficient_ucb.loc['C3', body_part]

                if delta_skin_temperature >= 0:
                    local_sensation_i += c22 * delta_skin_temperature
                else:
                    local_sensation_i += c21 * delta_skin_temperature
                local_sensation_i += c3 * self.delta_core_temperature

            local_sensation_i = 4 if local_sensation_i > 4 else local_sensation_i
            local_sensation_i = -4 if local_sensation_i < -4 else local_sensation_i
            local_sensation[body_part] = local_sensation_i

        return local_sensation
