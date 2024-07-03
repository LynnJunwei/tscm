#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2023/11/1 12:31
# @Author  : Eric
from typing import Optional, Literal

import pandas as pd
import numpy as np


class LocalSensationConfig:
    """Adjust configuration used to calculate local sensation from skin temperature."""
    def __init__(self, dynamic: bool = True,
                 mean_skin_temperature_approach: Literal[3, 4, 7, 8] = 7,
                 mean_skin_temperature_substitute: Optional[dict[str, str]] = None,
                 setpoint_type: Literal['setpoint', 'null_zone'] = 'setpoint'):
        """
        Args:
            dynamic:
                A boolean indicating whether to calculate dynamic sensation. Default is True.

            mean_skin_temperature_approach:
                An int represents the approach used to calculate mean skin temperature. Default is 7, which means
                7-Site calculation approach is used.

                The details refer to the function get_mean_skin_temperature in LocalSensationCalculator. Currently,
                only 3-Site (3), 4-Site (4), 7-Site (7), and 8-Site (8) approaches are supported.

            mean_skin_temperature_substitute:
                A substitute dictionary used to calculate mean skin temperature. Default is None.

                When the skin temperature of specific body part does not include in inputs but is needed in mean
                skin temperature calculation, substitute is allowed by using a substitute dictionary. The format of
                substitute dictionary should follow: **{body_part_needed: body_part_substituted}**.

            setpoint_type:
                The type of setpoint (skin temperatures in neutral state) used in local sensation calculation.
                Default is 'setpoint'.

                Three types of setpoint are supported including 'setpoint', 'null_zone', and 'setpoint_modified'.

                *'setpoint': Date from UCB software. It includes setpoints with different clothing types (Summer and
                WinterIndoor) and different metabolic rates (0.8, 1.0, 1.1, 1.2, and 1.3), and their corresponding
                range for different body parts.

                *'null_zone': A set of skin temperature ranges from experiments conducted in hot weather.
                Refer to Xie, Y., Niu, J., Zhang, H., Liu, S., Liu, J., Huang, T., Li, J., & Mak, C. M.
                (2020). Development of a multi-nodal thermal regulation and comfort model for the outdoor environment
                assessment. Building and Environment, 176, 106809. https://doi.org/10.1016/j.buildenv.2020.106809.

                *'setpoint_modified': A set of skin temperature ranges from JOS-3 simulation with different ambient
                temperatures (PMV from -0.5 to 0.5).
        """
        self.dynamic = dynamic
        self.mean_skin_temperature_approach = mean_skin_temperature_approach
        self.mean_skin_temperature_substitute = mean_skin_temperature_substitute
        self.setpoint_type = setpoint_type


class WholeSensationConfig:
    """
    Adjust configuration used to calculate whole body sensation from local sensation. Currently, all configurations
    are related to the smoothing method. The smooth function refers to Zhao, Y., Zhang, H., Arens, E. A., & Zhao, Q.
    (2014). Thermal sensation and comfort models for non-uniform and transient environments, part IV: Adaptive neutral
    setpoints and smoothed whole-body sensation model. Building and Environment, 72, 300–308.
    https://doi.org/10.1016/j.buildenv.2013.11.004
    """
    def __init__(self, smooth: bool = False,
                 smooth_alpha: int = 5,
                 smooth_adjusted: bool = True,
                 dominant_parts: list[str] = ('Chest', 'Back', 'Pelvis')):
        """
        Args:
            smooth:
                A boolean to control whether to use the smoothing method. Default is False.
            smooth_alpha:
                The value of alpha used to smooth the model. Default is 5.
            smooth_adjusted:
                A boolean to control whether to use the adjusted smoothing method. After adjusting, the fluctuation
                happens in critical state could be fixed. Only work when smooth is set to True. Default is True.
            dominant_parts:
                A list of names of dominant body parts used in calculation. Default includes Chest, Back and Pelvis.
        """
        self.smooth_alpha = smooth_alpha
        self.smooth = smooth
        self.smooth_adjusted = smooth_adjusted
        self.dominant_parts = dominant_parts


class HumanConfig:
    """
    Adjust physiological features of human object.

    Attributes:
        clo_type:
            The clothing type only used when LocalSensationConfig.setpoint_type is set to 'setpoint'. The choices
            include only 'Summer' and 'WinterIndoor'. The value is based on input clothing level.
        met_:
            A float of temporary metabolic rate for each iteration.
    """
    def __init__(self, met: float = 1.0,
                 clo: float = 0.5,
                 sex: Literal['male', 'female'] = 'male'):
        """
        Args:
            met:
                A single float of Metabolic rate of human object. Default is 1.0.

                If metabolic rate is changing in the calculation, please use a series as input and make sure the index
                of series is same as that of input skin temperature data.
            clo:
                A float of clothing level of human object. Default is 0.5.
            sex:
                Sex of human object. The value should be 'male' or 'female'. Default is 'male'.
        """
        self.met = met
        self.clo = clo
        self.sex = sex
