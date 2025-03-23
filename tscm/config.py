#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2023/11/1 12:31
# @Author  : Eric
from typing import Optional, Literal
import warnings


class LocalSensationConfig:
    """Adjust configuration used to calculate local local_sensation_sorted from skin temperature."""
    def __init__(self,
                 dynamic: bool = False,
                 mean_skin_temperature_approach: Literal[3, 4, 7, 8] = 7,
                 setpoint_type: Literal['setpoint', 'null_zone'] = 'setpoint'):
        """
        Args:
            dynamic:
                A boolean indicating whether to calculate dynamic local_sensation_sorted. Default is True.

            mean_skin_temperature_approach:
                An int represents the approach used to calculate mean skin temperature. Default is 7, which means
                7-Site calculation approach is used.

                The details refer to the function get_mean_skin_temperature in LocalSensationCalculator. Currently,
                only 3-Site (3), 4-Site (4), 7-Site (7), and 8-Site (8) approaches are supported.

            setpoint_type:
                The type of setpoint (skin temperatures in neutral state) used in local local_sensation_sorted calculation.
                Default is 'setpoint'.

                Three types of setpoint are supported including 'setpoint', 'null_zone', and 'setpoint_modified'.

                - **'setpoint':**
                  Date from UCB software. It includes setpoints with different clothing types (Summer and
                  WinterIndoor) and different metabolic rates (0.8, 1.0, 1.1, 1.2, and 1.3), and their corresponding
                  range for different body parts.

                - **'null_zone':**
                  A set of skin temperature ranges from experiments conducted in hot weather.
                  Refer to Xie, Y., Niu, J., Zhang, H., Liu, S., Liu, J., Huang, T., Li, J., & Mak, C. M.
                  (2020). Development of a multi-nodal thermal regulation and comfort model for the outdoor
                  environment assessment. Building and Environment, 176, 106809.
                  https://doi.org/10.1016/j.buildenv.2020.106809.
        """
        self.dynamic = dynamic
        self.mean_skin_temperature_approach = mean_skin_temperature_approach
        self.setpoint_type = setpoint_type


class OverallSensationConfig:
    """
    Adjust configuration used to calculate overall sensation from local sensation. Currently, all configurations
    are related to the smoothing method. The external_smooth function refers to Zhao, Y., Zhang, H., Arens, E. A., & Zhao, Q.
    (2014). Thermal local_sensation_sorted and comfort models for non-uniform and transient environments, part IV:
    Adaptive neutral setpoints and smoothed whole-body local_sensation_sorted model. Building and Environment,
    72, 300–308. https://doi.org/10.1016/j.buildenv.2013.11.004
    """
    def __init__(self,
                 external_smooth: bool = False,
                 external_smooth_alpha: int = 5,
                 external_smooth_adjusted: bool = False,
                 external_smooth_simplified: bool = False,
                 internal_smooth: bool = False,
                 internal_smooth_alpha: int = 30,
                 original_model: bool = True):
        """
        Args:
            external_smooth:
                A boolean to control whether to use the smoothing method between models. Default is False.
            external_smooth_alpha:
                The value of alpha used to external model smoothing. Default is 5.
            external_smooth_adjusted:
                A boolean to control whether to use the adjusted smoothing method between models. After adjusting,
                the fluctuation happens in critical state could be fixed. Only work when external_smooth is set to True.
                Default is False.
            external_smooth_simplified:
                A boolean to control whether to use the simplified smoothing method between models.
                Only work when external_smooth is set to True. Default is False.
            internal_smooth:
                A boolean to control whether to use the internal smoothing method. Default is False.
            internal_smooth_alpha:
                The value of alpha used to internal model smoothing. Default is 30.
            original_model:
                A boolean to define whether the original model is applied. If True, the calculation will be based on
                the original model, otherwise it will be based on the modified model. Default is True.
        """
        self.external_smooth_alpha = external_smooth_alpha
        self.external_smooth = external_smooth
        self.external_smooth_adjusted= external_smooth_adjusted
        self.external_smooth_simplified = external_smooth_simplified

        self.internal_smooth = internal_smooth
        self.internal_smooth_alpha = internal_smooth_alpha

        self.original_model = original_model

        if (not self.external_smooth) and self.external_smooth_adjusted:
            warnings.warn('The external_smooth_adjusted is only available when external_smooth is set to True.',
                          RuntimeWarning)
            self.external_smooth_adjusted = False
        if (not self.external_smooth) and self.external_smooth_simplified:
            warnings.warn('The external_smooth_simplified is only available when external_smooth is set to True.',
                          RuntimeWarning)
            self.external_smooth_simplified = False
        if self.original_model and self.external_smooth_simplified:
            warnings.warn('The external_smooth_simplified is only available when modified models are used.',
                              RuntimeWarning)
            self.external_smooth_simplified = False

class HumanConfig:
    """Adjust physiological features of human object."""
    def __init__(self,
                 met: float = 1.0,
                 clo: float = 0.5,
                 sex: Literal['male', 'female'] = 'male'):
        """
        Args:
            met:
                A single float of Metabolic rate of human object. Default is 1.0.
            clo:
                A float of clothing level of human object. Default is 0.5.
            sex:
                Sex of human object. The value should be 'male' or 'female'. Default is 'male'.
        """
        self.met = met
        self.clo = clo
        self.sex = sex

class LocalComfortConfig:
    """
    Adjust configuration used to calculate local comfort from local sensation and overall sensation.
    """
    def __init__(self,
                 exponential: bool = True):
        """
        Args:
            exponential:
                A boolean to control whether to use exponential function to calculate local comfort. Default is True.
                If set to False, n for all body parts will be set to 1.
        """
        self.exponential = exponential

class OverallComfortConfig:
    """
    Adjust configuration used to calculate overall comfort from local comfort.
    """
    def __init__(self,
                 is_transient: bool = False,
                 is_controlled: bool = False):
        """
        Args:
            is_transient:
                A boolean to control whether to calculate transient comfort. Default is False.
            is_controlled:
                A boolean to control whether to calculate controlled comfort. Default is False.
        """
        self.is_transient = is_transient
        self.is_controlled = is_controlled
