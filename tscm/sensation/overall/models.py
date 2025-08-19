# -*- coding: utf-8 -*-
# @Time    : 2024/8/22
# @Author  : Eric
import pandas as pd
import numpy as np
from functools import wraps

from tscm.const import COEFFICIENT, DOMINANT_BODY_PARTS
import tscm.sensation.overall.utils as utils


# ------------------------------Utilities for Sensation Model-----------------------------------------------

def _get_interval(local_sensation):
    """
            Calculate the value of interval for low-level sensation models.

            The interval value is equal to 2 divided by total body number. In low-level sensation models,
            sensations will be divided into equal intervals (scale from 2 to 0) by the number of body parts
            (counting hands and feet as only two body parts).
            """
    hands_num = local_sensation.index.str.endswith("Hand").sum()
    feet_num = local_sensation.index.str.endswith("Foot").sum()
    hands_corr = 1 if hands_num == 2 else 0
    feet_corr = 1 if feet_num == 2 else 0
    body_parts_num = len(local_sensation) - hands_corr - feet_corr
    return 2 / body_parts_num


def sig(x, a, t):
    return 1 / (1 + np.exp(-a * (x - t)))


def _opposite_force(local_sensation, baseline_sensation) -> float:
    """
            Calculate combined force of opposite sensations as modifier for opposite sensation model.

            Args:
                local_sensation:
                baseline_sensation: Value of overall sensation of bigger group calculated by no-opposite model.

            Returns:
                Value of modifier for extreme sensation model.
            """
    if len(local_sensation) == 0:
        return 0

    individual_force = pd.Series().reindex_like(local_sensation)

    for body_part, sensation in local_sensation.items():
        delta_sensation = sensation - baseline_sensation
        coefficient_suffix_map = {  # condition: suffix
            lambda x: x <= -2: -2,
            lambda x: -2 < x < 2: 0,
            lambda x: x >= 2: 2
        }
        coefficient_suffix = np.piecewise(
            delta_sensation,
            [cond_func(delta_sensation) for cond_func in coefficient_suffix_map.keys()],
            [value for value in coefficient_suffix_map.values()]
        ).astype(int)

        a = COEFFICIENT.loc['a_{}'.format(coefficient_suffix), body_part]
        b = COEFFICIENT.loc['b_{}'.format(coefficient_suffix), body_part]
        c = COEFFICIENT.loc['c_{}'.format(coefficient_suffix), body_part]

        individual_force[body_part] = (a * (delta_sensation - c) + b)

    individual_force_sorted = individual_force.sort_values(ascending=False, key=np.abs)

    combined_force = individual_force_sorted.iloc[0]
    if len(individual_force_sorted) > 1:
        combined_force += 0.1 * individual_force_sorted.iloc[1]

    return combined_force


def _extreme_force(local_sensation, baseline_sensation) -> float:
    """
            Calculate combined force of extreme sensations as modifier for all sensation model.

            Args:
                local_sensation:
                baseline_sensation: Value of precalculated overall sensation by using no-opposite model.

            Returns:
                Value of modifier for extreme sensation model.
            """
    if len(local_sensation) == 0:
        return 0

    individual_force = pd.Series().reindex_like(local_sensation)

    for body_part, sensation in local_sensation.items():
        delta_sensation = sensation - baseline_sensation
        coefficient_suffix_map = {  # condition: suffix
            lambda x: x <= -2: -2,
            lambda x: -2 < x < 2: 0,
            lambda x: x >= 2: 2
        }
        coefficient_suffix = np.piecewise(
            delta_sensation,
            [cond_func(delta_sensation) for cond_func in coefficient_suffix_map.keys()],
            [value for value in coefficient_suffix_map.values()]
        ).astype(int)

        a = COEFFICIENT.loc['a_{}'.format(coefficient_suffix), body_part]
        b = COEFFICIENT.loc['b_{}'.format(coefficient_suffix), body_part]
        c = COEFFICIENT.loc['c_{}'.format(coefficient_suffix), body_part]

        individual_force[body_part] = (a * (delta_sensation - c) + b)

    individual_force_sorted = individual_force.sort_values(ascending=False, key=np.abs)

    combined_force = individual_force_sorted.iloc[0]
    if len(individual_force_sorted) > 1:
        combined_force += 0.1 * individual_force_sorted.iloc[1]

    most_extreme_sensation_abs = np.abs(max(local_sensation, key=abs))

    correction_factor_map = {  # condition: correction factor
        lambda x: x < 1: 0,
        lambda x: 1 <= x < 2: round(most_extreme_sensation_abs - 1, 3),
        lambda x: x >= 2: 1
    }
    correction_factor = np.piecewise(
        most_extreme_sensation_abs,
        [cond_func(most_extreme_sensation_abs) for cond_func in correction_factor_map.keys()],
        [value for value in correction_factor_map.values()]
    )

    return combined_force * correction_factor


def _extreme_modifier(local_sensation, baseline_sensation) -> float:
    # sensations larger than 1 and overall sensation are considered for modifier
    threshold_warm = max([baseline_sensation, 1])
    modifier_warm = _extreme_force(local_sensation[local_sensation > threshold_warm], baseline_sensation)
    # sensations less than -1 and overall sensation are considered for modifier
    threshold_cold = min([baseline_sensation, -1])
    modifier_cold = _extreme_force(local_sensation[local_sensation < threshold_cold], baseline_sensation)
    return modifier_warm + modifier_cold

def check_input_sensation(model):
    @wraps(model)
    def wrapper(local_sensation, *args, **kwargs):
        body_parts_num = len(local_sensation)
        if body_parts_num == 0:
            return 0
        if body_parts_num == 1:
            return local_sensation.iloc[0]
        if body_parts_num == 2 and utils.are_hands_feet_most_extreme(local_sensation):
            return local_sensation.mean()
        return model(local_sensation, *args, **kwargs)
    return wrapper


class OriginalModel:
    """
    A class for calculating overall sensation based on local sensation using original no-smoothed model.
    """

    @staticmethod
    @check_input_sensation
    def high_level_warm(local_sensation) -> float:
        """Returns the overall sensation calculated by no-opposite high-level warm (complaint warm) model."""
        local_sensation_descending = local_sensation.sort_values(ascending=False)
        if utils.are_hands_feet_most_extreme(local_sensation_descending):
            return 0.5 * local_sensation_descending.iloc[0] + 0.5 * local_sensation_descending.iloc[2]
        else:
            return 0.5 * local_sensation_descending.iloc[0] + 0.5 * local_sensation_descending.iloc[1]


    @staticmethod
    @check_input_sensation
    def high_level_cold(local_sensation) -> float:
        """Returns the overall sensation calculated by no-opposite high-level cold (complaint cold) model."""
        local_sensation_ascending = local_sensation.sort_values(ascending=True)
        if utils.are_hands_feet_most_extreme(local_sensation_ascending):
            return 0.38 * local_sensation_ascending.iloc[0] + 0.62 * local_sensation_ascending.iloc[2]
        else:
            return 0.38 * local_sensation_ascending.iloc[0] + 0.62 * local_sensation_ascending.iloc[1]


    @staticmethod
    @check_input_sensation
    def low_level_warm(local_sensation) -> float:
        """Returns the overall sensation calculated by no-opposite low-level warm (gradual warm) model."""
        local_sensation_descending = local_sensation.sort_values(ascending=False)
        interval = _get_interval(local_sensation_descending)

        if utils.are_hands_feet_most_extreme(local_sensation_descending):
            local_sensation_descending = local_sensation_descending.drop(local_sensation_descending.index[1])
        local_sensation_selected = list(local_sensation_descending)[:2]
        for i in range(2, len(local_sensation_descending)):
            local_sensation_selected.append(local_sensation_descending.iloc[i])
            if local_sensation_descending.iloc[i] > 2 - interval * (i - 1):
                break  # Stop when sensation meets the condition.

        return np.mean(local_sensation_selected, dtype=float)


    @staticmethod
    @check_input_sensation
    def low_level_cold(local_sensation) -> float:
        """Returns the overall sensation calculated by no-opposite low-level cold (gradual cold) model."""
        local_sensation_ascending = local_sensation.sort_values(ascending=True)
        interval = _get_interval(local_sensation_ascending)

        if utils.are_hands_feet_most_extreme(local_sensation_ascending):
            local_sensation_ascending = local_sensation_ascending.drop(local_sensation_ascending.index[1])
        local_sensation_selected = list(local_sensation_ascending)[:2]
        for i in range(2, len(local_sensation_ascending)):
            local_sensation_selected.append(local_sensation_ascending.iloc[i])
            if local_sensation_ascending.iloc[i] < -2 + interval * (i - 1):
                break

        return np.mean(local_sensation_selected, dtype=float)


    @staticmethod
    @check_input_sensation
    def dominated_cold(local_sensation) -> float:
        """Return the overall sensation calculated by opposite-dominated cold model."""
        return min(local_sensation[[body_part for body_part in local_sensation.index
                                    if body_part in DOMINANT_BODY_PARTS]])


    @staticmethod
    @check_input_sensation
    def opposite_warm(local_sensation) -> float:
        """Return the overall sensation calculated by opposite warm model."""
        sensation_level = utils.get_sensation_level(local_sensation, "warm")
        if sensation_level == "high":
            overall_sensation_bigger = OriginalModel.high_level_warm(local_sensation[local_sensation >= 0])
            overall_sensation_bigger_ex = OriginalModel.high_level_warm(local_sensation[local_sensation >= -1])
        else:
            overall_sensation_bigger = OriginalModel.low_level_warm(local_sensation[local_sensation >= 0])
            overall_sensation_bigger_ex = OriginalModel.low_level_warm(local_sensation[local_sensation >= 1])
        modifier = _opposite_force(local_sensation[local_sensation < 0], overall_sensation_bigger)

        return overall_sensation_bigger_ex + modifier


    @staticmethod
    @check_input_sensation
    def opposite_cold(local_sensation) -> float:
        """Return the overall sensation calculated by opposite cool model."""
        sensation_level = utils.get_sensation_level(local_sensation, "cold")
        if sensation_level == "high":
            overall_sensation_bigger = OriginalModel.high_level_cold(local_sensation[local_sensation <= 0])
            overall_sensation_bigger_ex = OriginalModel.high_level_cold(local_sensation[local_sensation <= 1])
        else:
            overall_sensation_bigger = OriginalModel.low_level_cold(local_sensation[local_sensation <= 0])
            overall_sensation_bigger_ex = OriginalModel.low_level_cold(local_sensation[local_sensation <= -1])
        modifier = _opposite_force(local_sensation[local_sensation > 0], overall_sensation_bigger)

        return overall_sensation_bigger_ex + modifier


class ModifiedModel:
    """
    A class for calculating overall sensation based on local sensation using modified no-smoothed model.
    This model is an extension of the OriginalModel with additional modifications for extreme sensations.
    """

    @staticmethod
    @check_input_sensation
    def high_level_warm(local_sensation) -> float:
        """Returns the overall sensation calculated by modified no-opposite high-level warm (complaint warm) model."""
        sensation = OriginalModel.high_level_warm(local_sensation)
        modifier = _extreme_modifier(local_sensation, sensation)
        return sensation + modifier


    @staticmethod
    @check_input_sensation
    def high_level_cold(local_sensation) -> float:
        """Returns the overall sensation calculated by modified no-opposite high-level cold (complaint cold) model."""
        sensation = OriginalModel.high_level_cold(local_sensation)
        modifier = _extreme_modifier(local_sensation, sensation)
        return sensation + modifier


    @staticmethod
    @check_input_sensation
    def low_level_warm(local_sensation) -> float:
        """Returns the overall sensation calculated by modified no-opposite low-level warm (gradual warm) model."""
        sensation = OriginalModel.low_level_warm(local_sensation)
        modifier = _extreme_modifier(local_sensation, sensation)
        return sensation + modifier


    @staticmethod
    @check_input_sensation
    def low_level_cold(local_sensation) -> float:
        """Returns the overall sensation calculated by modified no-opposite low-level cold (gradual cold) model."""
        sensation = OriginalModel.low_level_cold(local_sensation)
        modifier = _extreme_modifier(local_sensation, sensation)
        return sensation + modifier


    @staticmethod
    @check_input_sensation
    def dominated_cold(local_sensation) -> float:
        """Return the overall sensation calculated by modified dominated cold model."""
        bigger_group = utils.get_bigger_group(local_sensation)
        sensation_level = utils.get_sensation_level(local_sensation, bigger_group)
        # lower value of local sensation in the dominant parts and from no-opposite model
        sensation = OriginalModel.dominated_cold(local_sensation)
        if bigger_group == "cold" and sensation_level == "high":
            sensation = min([OriginalModel.dominated_cold(local_sensation),
                             OriginalModel.high_level_cold(local_sensation)])
        if bigger_group == "cold" and sensation_level == "low":
            sensation = min([OriginalModel.dominated_cold(local_sensation),
                             OriginalModel.low_level_cold(local_sensation)])

        modifier = _extreme_modifier(local_sensation, sensation)
        return sensation + modifier


    @staticmethod
    @check_input_sensation
    def opposite_warm(local_sensation) -> float:
        """Return the overall sensation calculated by modified opposite warm model."""
        sensation_level = utils.get_sensation_level(local_sensation, "warm")
        if sensation_level == "high":
            return ModifiedModel.high_level_warm(local_sensation)
        if sensation_level == "low":
            return ModifiedModel.low_level_warm(local_sensation)


    @staticmethod
    @check_input_sensation
    def opposite_cold(local_sensation) -> float:
        """Return the overall sensation calculated by modified opposite cold model."""
        sensation_level = utils.get_sensation_level(local_sensation, "cold")
        if sensation_level == "high":
            return ModifiedModel.high_level_cold(local_sensation)
        if sensation_level == "low":
            return ModifiedModel.low_level_cold(local_sensation)


class SmoothedLowLevelModel:
    """
    A class for calculating overall sensation based on local sensation using internally smoothed low-level sensation models.
    This model is an extension of the OriginalModel and ModifiedModel with additional smoothing for low-level sensations.
    """

    @staticmethod
    @check_input_sensation
    def warm(local_sensation, alpha) -> float:
        """Returns the overall sensation calculated by internally smoothed low-level warm model."""
        local_sensation_descending = local_sensation.sort_values(ascending=False)
        interval = _get_interval(local_sensation_descending)

        if utils.are_hands_feet_most_extreme(local_sensation_descending):
            local_sensation_descending = local_sensation_descending.drop(local_sensation_descending.index[1])

        opposite_gamma = []
        beta = []
        beta_x = []

        # gamma_i = sig(x_i, alpha, 2 - interval * (i - 1))
        # beta_i = gamma_i * (1 - gamma_i-1) * (1 - gamma_i-2) * ... * (1 - gamma_2)
        # beta_N = 1 - [(1-beta_N-1) + (1-beta_N-2) + ... + (1-beta_2)]
        for i in range(2, len(local_sensation_descending)):
            x_i = local_sensation_descending.iloc[i]
            x_i_sum = np.mean(local_sensation_descending.iloc[:i + 1])

            if i != len(local_sensation_descending) - 1:
                gamma_i = sig(x_i, alpha, 2 - interval * (i - 1))
                beta_i = gamma_i * np.prod(opposite_gamma)  # np.prod([]) = 1
                beta_x_i = x_i_sum * beta_i

                opposite_gamma.append(1 - gamma_i)
                beta.append(beta_i)

            else:
                beta_i = 1 - np.sum(beta)
                beta_x_i = x_i_sum * beta_i

            beta_x.append(beta_x_i)
        return np.sum(beta_x)


    @staticmethod
    @check_input_sensation
    def warm_bigger(local_sensation, alpha) -> float:
        """Returns the overall sensation calculated by internally smoothed low-level warm model for bigger group (x_i > 0)."""
        local_sensation_descending = local_sensation.sort_values(ascending=False)
        interval = _get_interval(local_sensation_descending)

        if utils.are_hands_feet_most_extreme(local_sensation_descending):
            local_sensation_descending = local_sensation_descending.drop(local_sensation_descending.index[1])

        opposite_gamma = []
        beta = []
        beta_x = []

        # gamma_i = 1 / (1 + exp(-alpha * (x_i - 2 + interval * (i - 1)))
        # beta_i = (gamma_i + (1 - gamma_i) * sig(x_i, alpha, 0) * sig(-x_i+1, alpha, 0)) *
        #          (1 - gamma_i-1) * (1 - gamma_i-2) * ... * (1 - gamma_2)
        # beta_N = 1 - [(1-beta_N-1) + (1-beta_N-2) + ... + (1-beta_2)]
        for i in range(2, len(local_sensation_descending)):
            x_i = local_sensation_descending.iloc[i]
            x_i_sum = np.mean(local_sensation_descending.iloc[:i + 1])

            if i != len(local_sensation_descending) - 1:
                x_i_plus_1 = local_sensation_descending.iloc[i + 1]
                gamma_i = sig(x_i, alpha, 2 - interval * (i - 1))
                beta_i = (gamma_i + (1 - gamma_i) * sig(x_i, alpha, 0) * sig(-x_i_plus_1, alpha, 0)) * np.prod(opposite_gamma)
                beta_x_i = x_i_sum * beta_i

                opposite_gamma.append(1 - gamma_i)
                beta.append(beta_i)

            else:
                beta_i = 1 - np.sum(beta)
                beta_x_i = x_i_sum * beta_i

            beta_x.append(beta_x_i)
        return np.sum(beta_x)


    @staticmethod
    @check_input_sensation
    def warm_extended_bigger(local_sensation, alpha) -> float:
        """Returns the overall sensation calculated by internally smoothed low-level warm model for extended bigger group (x_i > -1)."""
        local_sensation_descending = local_sensation.sort_values(ascending=False)
        interval = _get_interval(local_sensation_descending)

        if utils.are_hands_feet_most_extreme(local_sensation_descending):
            local_sensation_descending = local_sensation_descending.drop(local_sensation_descending.index[1])

        opposite_gamma = []
        beta = []
        beta_x = []

        # gamma_i = 1 / (1 + exp(-alpha * (x_i - 2 + interval * (i - 1)))
        # beta_i = (gamma_i + (1 - gamma_i) * sig(x_i, alpha, -1) * sig(-x_i+1, alpha, 1)) *
        #          (1 - gamma_i-1) * (1 - gamma_i-2) * ... * (1 - gamma_2)
        # beta_N = 1 - [(1-beta_N-1) + (1-beta_N-2) + ... + (1-beta_2)]
        for i in range(2, len(local_sensation_descending)):
            x_i = local_sensation_descending.iloc[i]
            x_i_sum = np.mean(local_sensation_descending.iloc[:i + 1])

            if i != len(local_sensation_descending) - 1:
                x_i_plus_1 = local_sensation_descending.iloc[i + 1]
                gamma_i = sig(x_i, alpha, 2 - interval * (i - 1))
                beta_i = (gamma_i + (1 - gamma_i) * sig(x_i, alpha, -1) * sig(-x_i_plus_1, alpha, 1)) * np.prod(opposite_gamma)
                beta_x_i = x_i_sum * beta_i

                opposite_gamma.append(1 - gamma_i)
                beta.append(beta_i)

            else:
                beta_i = 1 - np.sum(beta)
                beta_x_i = x_i_sum * beta_i

            beta_x.append(beta_x_i)
        return np.sum(beta_x)


    @staticmethod
    @check_input_sensation
    def cold(local_sensation, alpha) -> float:
        """Returns the overall sensation calculated by internally smoothed low-level cold model."""
        local_sensation_ascending = local_sensation.sort_values(ascending=True)
        interval = _get_interval(local_sensation_ascending)

        if utils.are_hands_feet_most_extreme(local_sensation_ascending):
            local_sensation_ascending = local_sensation_ascending.drop(local_sensation_ascending.index[1])

        opposite_gamma = []
        beta = []
        beta_x = []

        # gamma_i = sig(-x_i, alpha, 2 - interval * (i - 1))
        # beta_i = gamma_i * (1 - gamma_i-1) * (1 - gamma_i-2) * ... * (1 - gamma_2)
        # beta_N = 1 - [(1-beta_N-1) + (1-beta_N-2) + ... + (1-beta_2)]
        for i in range(2, len(local_sensation_ascending)):
            x_i = local_sensation_ascending.iloc[i]
            x_i_sum = np.mean(local_sensation_ascending.iloc[:i + 1])

            if i != len(local_sensation_ascending) - 1:
                gamma_i = sig(-x_i, alpha, 2 - interval * (i - 1))
                beta_i = gamma_i * np.prod(opposite_gamma)  # np.prod([]) = 1
                beta_x_i = x_i_sum * beta_i

                opposite_gamma.append(1 - gamma_i)
                beta.append(beta_i)

            else:
                beta_i = 1 - np.sum(beta)
                beta_x_i = x_i_sum * beta_i

            beta_x.append(beta_x_i)
        return np.sum(beta_x)


    @staticmethod
    @check_input_sensation
    def cold_bigger(local_sensation, alpha) -> float:
        """Returns the overall sensation calculated by internally smoothed low-level warm model for bigger group (x_i < 0)."""
        local_sensation_ascending = local_sensation.sort_values(ascending=True)
        interval = _get_interval(local_sensation_ascending)

        if utils.are_hands_feet_most_extreme(local_sensation_ascending):
            local_sensation_ascending = local_sensation_ascending.drop(local_sensation_ascending.index[1])

        opposite_gamma = []
        beta = []
        beta_x = []

        # gamma_i = 1 / (1 + exp(-alpha * (x_i - 2 + interval * (i - 1)))
        # beta_i = (gamma_i + (1 - gamma_i) * sig(x_i, alpha, 0) * sig(-x_i+1, alpha, 0)) *
        #          (1 - gamma_i-1) * (1 - gamma_i-2) * ... * (1 - gamma_2)
        # beta_N = 1 - [(1-beta_N-1) + (1-beta_N-2) + ... + (1-beta_2)]
        for i in range(2, len(local_sensation_ascending)):
            x_i = local_sensation_ascending.iloc[i]
            x_i_sum = np.mean(local_sensation_ascending.iloc[:i + 1])

            if i != len(local_sensation_ascending) - 1:
                x_i_plus_1 = local_sensation_ascending.iloc[i + 1]
                gamma_i = sig(-x_i, alpha, 2 - interval * (i - 1))
                beta_i = (gamma_i + (1 - gamma_i) * sig(-x_i, alpha, 0) * sig(x_i_plus_1, alpha, 0)) * np.prod(opposite_gamma)
                beta_x_i = x_i_sum * beta_i

                opposite_gamma.append(1 - gamma_i)
                beta.append(beta_i)

            else:
                beta_i = 1 - np.sum(beta)
                beta_x_i = x_i_sum * beta_i

            beta_x.append(beta_x_i)
        return np.sum(beta_x)


    @staticmethod
    @check_input_sensation
    def cold_extended_bigger(local_sensation, alpha) -> float:
        """Returns the overall sensation calculated by internally smoothed low-level warm model for extended bigger group (x_i < 1)."""
        local_sensation_ascending = local_sensation.sort_values(ascending=True)
        interval = _get_interval(local_sensation_ascending)

        if utils.are_hands_feet_most_extreme(local_sensation_ascending):
            local_sensation_ascending = local_sensation_ascending.drop(local_sensation_ascending.index[1])

        opposite_gamma = []
        beta = []
        beta_x = []

        # gamma_i = 1 / (1 + exp(-alpha * (x_i - 2 + interval * (i - 1)))
        # beta_i = (gamma_i + (1 - gamma_i) * sig(x_i, alpha, -1) * sig(-x_i+1, alpha, 1)) *
        #          (1 - gamma_i-1) * (1 - gamma_i-2) * ... * (1 - gamma_2)
        # beta_N = 1 - [(1-beta_N-1) + (1-beta_N-2) + ... + (1-beta_2)]
        for i in range(2, len(local_sensation_ascending)):
            x_i = local_sensation_ascending.iloc[i]
            x_i_sum = np.mean(local_sensation_ascending.iloc[:i + 1])

            if i != len(local_sensation_ascending) - 1:
                x_i_plus_1 = local_sensation_ascending.iloc[i + 1]
                gamma_i = sig(-x_i, alpha, 2 - interval * (i - 1))
                beta_i = (gamma_i + (1 - gamma_i) * sig(-x_i, alpha, -1) * sig(x_i_plus_1, alpha, 1)) * np.prod(opposite_gamma)
                beta_x_i = x_i_sum * beta_i

                opposite_gamma.append(1 - gamma_i)
                beta.append(beta_i)

            else:
                beta_i = 1 - np.sum(beta)
                beta_x_i = x_i_sum * beta_i

            beta_x.append(beta_x_i)
        return np.sum(beta_x)


class SmoothedOriginalModel(OriginalModel):
    """
    A class for calculating overall sensation based on local sensation using internally smoothed original model.
    This model is an extension of the OriginalModel with additional smoothing for low-level sensations.
    """

    @staticmethod
    @check_input_sensation
    def low_level_warm(local_sensation, alpha) -> float:
        """Return the overall sensation calculated by internally smoothed low-level warm model."""
        return SmoothedLowLevelModel.warm(local_sensation, alpha)


    @staticmethod
    @check_input_sensation
    def low_level_cold(local_sensation, alpha) -> float:
        """Return the overall sensation calculated by internally smoothed low-level cold model."""
        return SmoothedLowLevelModel.cold(local_sensation, alpha)


    @staticmethod
    @check_input_sensation
    def opposite_warm(local_sensation, alpha) -> float:
        """Return the overall sensation calculated by internally smoothed opposite warm model."""
        sensation_level = utils.get_sensation_level(local_sensation, "warm")
        if sensation_level == "high":
            overall_sensation_bigger = OriginalModel.high_level_warm(local_sensation[local_sensation >= 0])
            overall_sensation_bigger_ex = OriginalModel.high_level_warm(local_sensation[local_sensation >= -1])
        else:
            overall_sensation_bigger = SmoothedLowLevelModel.warm_bigger(local_sensation, alpha)
            overall_sensation_bigger_ex = SmoothedLowLevelModel.warm_extended_bigger(local_sensation, alpha)

        modifier = _opposite_force(local_sensation[local_sensation < 0], overall_sensation_bigger)

        return overall_sensation_bigger_ex + modifier


    @staticmethod
    @check_input_sensation
    def opposite_cold(local_sensation, alpha) -> float:
        """Return the overall sensation calculated by internally smoothed opposite warm model."""
        sensation_level = utils.get_sensation_level(local_sensation, "cold")
        if sensation_level == "high":
            overall_sensation_bigger = OriginalModel.high_level_cold(local_sensation[local_sensation <= 0])
            overall_sensation_bigger_ex = OriginalModel.high_level_cold(local_sensation[local_sensation <= 1])
        else:
            overall_sensation_bigger = SmoothedLowLevelModel.cold_bigger(local_sensation, alpha)
            overall_sensation_bigger_ex = SmoothedLowLevelModel.cold_extended_bigger(local_sensation, alpha)

        modifier = _opposite_force(local_sensation[local_sensation > 0], overall_sensation_bigger)

        return overall_sensation_bigger_ex + modifier


class SmoothedModifiedModel(ModifiedModel):
    """
    A class for calculating overall sensation based on local sensation using modified smoothed model.
    This model is an extension of the ModifiedModel with additional modifications for extreme sensations.
    """

    @staticmethod
    @check_input_sensation
    def low_level_warm(local_sensation, alpha) -> float:
        """Returns the overall sensation calculated by internally smoothed modified low-level warm model."""
        sensation = SmoothedLowLevelModel.warm(local_sensation, alpha)
        modifier = _extreme_modifier(local_sensation, sensation)
        return sensation + modifier


    @staticmethod
    @check_input_sensation
    def low_level_cold(local_sensation, alpha) -> float:
        """Returns the overall sensation calculated by internally smoothed modified low-level cold model."""
        sensation = SmoothedLowLevelModel.cold(local_sensation, alpha)
        modifier = _extreme_modifier(local_sensation, sensation)
        return sensation + modifier


    @staticmethod
    @check_input_sensation
    def opposite_warm(local_sensation, alpha) -> float:
        """Returns the overall sensation calculated by internally smoothed modified opposite warm model."""
        sensation_level = utils.get_sensation_level(local_sensation, "warm")
        if sensation_level == "high":
            return SmoothedModifiedModel.high_level_warm(local_sensation)
        if sensation_level == "low":
            return SmoothedModifiedModel.low_level_warm(local_sensation, alpha)


    @staticmethod
    @check_input_sensation
    def opposite_cold(local_sensation, alpha) -> float:
        """Returns the overall sensation calculated by internally smoothed modified opposite cold model."""
        sensation_level = utils.get_sensation_level(local_sensation, "cold")
        if sensation_level == "high":
            return SmoothedModifiedModel.high_level_cold(local_sensation)
        if sensation_level == "low":
            return SmoothedModifiedModel.low_level_cold(local_sensation, alpha)

if __name__ == '__main__':
    test_sensation = pd.Series({
        "Head": 1.5,
        "LArm": 1.0,
        "RArm": 1.0,
        "LHand": -1.5,
        "RHand": -1.5,
        "Chest": 2.5,
    })
    print("OriginalModel.high_level_warm:", OriginalModel.high_level_warm(test_sensation))
    print("OriginalModel.high_level_cold:", OriginalModel.high_level_cold(test_sensation))
    print("OriginalModel.low_level_warm:", OriginalModel.low_level_warm(test_sensation))
    print("OriginalModel.low_level_cold:", OriginalModel.low_level_cold(test_sensation))
    print("OriginalModel.dominated_cold:", OriginalModel.dominated_cold(test_sensation))
    print("SmoothedModifiedModel.low_level_warm:", SmoothedModifiedModel.low_level_warm(test_sensation, 0.5))
