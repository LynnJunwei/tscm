# -*- coding: utf-8 -*-
# @Time    : 2024/8/22
# @Author  : Eric
import pandas as pd
import numpy as np

from tscm.const import COEFFICIENT, DOMINANT_BODY_PARTS
import tscm.overall_sensation.utilities as utils


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


# ------------------------------Original Sensation Model-----------------------------------------------

def _opposite_modifier(local_sensation, baseline_sensation):
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


def high_level_warm(local_sensation) -> float:
    """Returns the overall sensation calculated by no-opposite high level warm (complaint warm) model."""
    local_sensation_descending = local_sensation.sort_values(ascending=False)
    if utils.are_hands_feet_most_extreme(local_sensation_descending):
        return 0.5 * local_sensation_descending.iloc[0] + 0.5 * local_sensation_descending.iloc[2]
    else:
        return 0.5 * local_sensation_descending.iloc[0] + 0.5 * local_sensation_descending.iloc[1]


def high_level_cold(local_sensation) -> float:
    """Returns the overall sensation calculated by no-opposite high level cold (complaint cold) model."""
    local_sensation_ascending = local_sensation.sort_values(ascending=True)
    if utils.are_hands_feet_most_extreme(local_sensation_ascending):
        return 0.38 * local_sensation_ascending.iloc[0] + 0.62 * local_sensation_ascending.iloc[2]
    else:
        return 0.38 * local_sensation_ascending.iloc[0] + 0.62 * local_sensation_ascending.iloc[1]


def low_level_warm(local_sensation) -> float:
    """Returns the overall sensation calculated by no-opposite low level warm (gradual warm) model."""
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


def low_level_cold(local_sensation) -> float:
    """Returns the overall sensation calculated by no-opposite low level cold (gradual cold) model."""
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


def no_opposite_model(local_sensation, bigger_group):
    """Return the overall sensation calculated by no-opposite model."""
    body_parts_num = len(local_sensation)
    # If no sensation in input series, return 0.
    if body_parts_num == 0:
        return 0
    # If only one sensation in input series, return itself.
    if body_parts_num == 1:
        return local_sensation.iloc[0]
    # If only two sensation in input series and both come from hands or feet, return the average value.
    if body_parts_num == 2 and utils.are_hands_feet_most_extreme(local_sensation):
        return local_sensation.mean()

    sensation_level = utils.get_sensation_level(local_sensation, bigger_group)
    if sensation_level == "high":
        return high_level_warm(local_sensation) if bigger_group == "warm" else high_level_cold(local_sensation)
    if sensation_level == "low":
        return low_level_warm(local_sensation) if bigger_group == "warm" else low_level_cold(local_sensation)


def opposite_dominated_cold(local_sensation):
    """Return the overall sensation calculated by opposite dominated cold model."""
    return min(local_sensation[DOMINANT_BODY_PARTS])


def opposite_warm(local_sensation):
    """Return the overall sensation calculated by opposite warm model."""
    overall_sensation_bigger = no_opposite_model(local_sensation[local_sensation >= 0], "warm")
    overall_sensation_bigger_ex = no_opposite_model(local_sensation[local_sensation >= -1], "warm")
    modifier = _opposite_modifier(local_sensation[local_sensation < 0], overall_sensation_bigger)

    return overall_sensation_bigger_ex + modifier


def opposite_cold(local_sensation):
    """Return the overall sensation calculated by opposite cool model."""
    overall_sensation_bigger = no_opposite_model(local_sensation[local_sensation <= 0], "cold")
    overall_sensation_bigger_ex = no_opposite_model(local_sensation[local_sensation <= 1], "cold")
    modifier = _opposite_modifier(local_sensation[local_sensation > 0], overall_sensation_bigger)

    return overall_sensation_bigger_ex + modifier


# ------------------------------Modified Sensation Model-----------------------------------------------

def _extreme_modifier(local_sensation, baseline_sensation):
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


def modified_high_level_warm(local_sensation) -> float:
    sensation = high_level_warm(local_sensation)
    # sensations larger than 1 and overall sensation are considered for modifier
    threshold = max([sensation, 1])
    modifier = _extreme_modifier(local_sensation[local_sensation > threshold], sensation)
    return sensation + modifier


def modified_high_level_cold(local_sensation) -> float:
    sensation = high_level_cold(local_sensation)
    # sensations less than -1 and overall sensation are considered for modifier
    threshold = min([sensation, -1])
    modifier = _extreme_modifier(local_sensation[local_sensation < threshold], sensation)
    return sensation + modifier


def modified_low_level_warm(local_sensation) -> float:
    sensation = low_level_warm(local_sensation)
    # sensations larger than 1 and overall sensation are considered for modifier
    threshold = max([sensation, 1])
    modifier = _extreme_modifier(local_sensation[local_sensation > threshold], sensation)
    return sensation + modifier


def modified_low_level_cold(local_sensation) -> float:
    sensation = low_level_cold(local_sensation)
    # Limit sensation by dominant parts
    threshold = min([sensation, -1])
    modifier = _extreme_modifier(local_sensation[local_sensation < threshold], sensation)
    return sensation + modifier


def modified_dominated_cold(local_sensation) -> float:
    bigger_group = utils.get_bigger_group(local_sensation)
    # lower value of local sensation in the dominant parts and from no-opposite model
    sensation = min([opposite_dominated_cold(local_sensation), no_opposite_model(local_sensation, bigger_group)])

    modifier_warm = _extreme_modifier(local_sensation[local_sensation > 0], 0)
    threshold = min([sensation, -1])
    modifier_cold = _extreme_modifier(local_sensation[local_sensation < threshold], sensation)
    return sensation + modifier_warm + modifier_cold


def modified_opposite_warm(local_sensation) -> float:
    sensation = no_opposite_model(local_sensation, 'warm')

    modifier_cold = _extreme_modifier(local_sensation[local_sensation < -1], sensation)
    threshold = max([sensation, 1])
    modifier_warm = _extreme_modifier(local_sensation[local_sensation > threshold], sensation)
    return sensation + modifier_warm + modifier_cold


def modified_opposite_cold(local_sensation) -> float:
    sensation = no_opposite_model(local_sensation, 'cold')

    modifier_warm = _extreme_modifier(local_sensation[local_sensation > 1], sensation)
    threshold = min([sensation, -1])
    modifier_cold = _extreme_modifier(local_sensation[local_sensation < threshold], sensation)
    return sensation + modifier_warm + modifier_cold


# ------------------------------Internally Smoothed Sensation Model-----------------------------------------------

def smoothed_low_level_warm(local_sensation) -> float:
    """Returns the overall sensation calculated by internally smoothed low level warm model."""
    local_sensation_descending = local_sensation.sort_values(ascending=False)
    interval = _get_interval(local_sensation_descending)

    if utils.are_hands_feet_most_extreme(local_sensation_descending):
        local_sensation_descending = local_sensation_descending.drop(local_sensation_descending.index[1])

    opposite_gamma = []
    opposite_beta = []
    beta_x = []
    alpha = 10
    # gamma_i = sig(x_i, alpha, 2 - interval * (i - 1))
    # beta_i = gamma_i * (1 - gamma_i-1) * (1 - gamma_i-2) * ... * (1 - gamma_2)
    # beta_N = 1 - (1-beta_N-1) * (1-beta_N-2) * ... * (1-beta_2)
    for i in range(2, len(local_sensation_descending)):
        x_i = local_sensation_descending.iloc[i]
        x_i_sum = np.mean(local_sensation_descending.iloc[:i + 1])

        if i != len(local_sensation_descending) - 1:
            gamma_i = sig(x_i, alpha, 2 - interval * (i - 1))
            beta_i = gamma_i * np.prod(opposite_gamma)  # np.prod([]) = 1
            beta_x_i = x_i_sum * beta_i

            opposite_gamma.append(1 - gamma_i)
            opposite_beta.append(1 - beta_i)

        else:
            beta_i = 1 - np.prod(opposite_beta)
            beta_x_i = x_i_sum * beta_i

        beta_x.append(beta_x_i)
    return np.sum(beta_x)


def smoothed_low_level_warm_0(local_sensation) -> float:
    """Returns the overall sensation calculated by internally smoothed low level warm model."""
    local_sensation_descending = local_sensation.sort_values(ascending=False)
    interval = _get_interval(local_sensation_descending)

    if utils.are_hands_feet_most_extreme(local_sensation_descending):
        local_sensation_descending = local_sensation_descending.drop(local_sensation_descending.index[1])

    opposite_gamma = []
    opposite_beta = []
    beta_x = []
    alpha = 10
    # gamma_i = 1 / (1 + exp(-alpha * (x_i - 2 + interval * (i - 1)))
    # beta_i = (gamma_i + (1 - gamma_i) * sig(x_i, alpha, 0) * sig(-x_i+1, alpha, 0)) *
    #          (1 - gamma_i-1) * (1 - gamma_i-2) * ... * (1 - gamma_2)
    # beta_N = 1 - (1-beta_N-1) * (1-beta_N-2) * ... * (1-beta_2)
    for i in range(2, len(local_sensation_descending)):
        x_i = local_sensation_descending.iloc[i]
        x_i_sum = np.mean(local_sensation_descending.iloc[:i + 1])

        if i != len(local_sensation_descending) - 1:
            x_i_plus_1 = local_sensation_descending.iloc[i + 1]
            gamma_i = sig(x_i, alpha, 2 - interval * (i - 1))
            beta_i = (gamma_i + (1 - gamma_i) * sig(x_i, alpha, 0) * sig(-x_i_plus_1, alpha, 0)) * np.prod(opposite_gamma)
            beta_x_i = x_i_sum * beta_i

            opposite_gamma.append(1 - gamma_i)
            opposite_beta.append(1 - beta_i)

        else:
            beta_i = 1 - np.prod(opposite_beta)
            beta_x_i = x_i_sum * beta_i

        beta_x.append(beta_x_i)
    return np.sum(beta_x)


def smoothed_low_level_warm_minus_1(local_sensation) -> float:
    """Returns the overall sensation calculated by internally smoothed low level warm model."""
    local_sensation_descending = local_sensation.sort_values(ascending=False)
    interval = _get_interval(local_sensation_descending)

    if utils.are_hands_feet_most_extreme(local_sensation_descending):
        local_sensation_descending = local_sensation_descending.drop(local_sensation_descending.index[1])

    opposite_gamma = []
    opposite_beta = []
    beta_x = []
    alpha = 10
    # gamma_i = 1 / (1 + exp(-alpha * (x_i - 2 + interval * (i - 1)))
    # beta_i = (gamma_i + (1 - gamma_i) * sig(x_i, alpha, -1) * sig(-x_i+1, alpha, 1)) *
    #          (1 - gamma_i-1) * (1 - gamma_i-2) * ... * (1 - gamma_2)
    # beta_N = 1 - (1-beta_N-1) * (1-beta_N-2) * ... * (1-beta_2)
    for i in range(2, len(local_sensation_descending)):
        x_i = local_sensation_descending.iloc[i]
        x_i_sum = np.mean(local_sensation_descending.iloc[:i + 1])

        if i != len(local_sensation_descending) - 1:
            x_i_plus_1 = local_sensation_descending.iloc[i + 1]
            gamma_i = sig(x_i, alpha, 2 - interval * (i - 1))
            beta_i = (gamma_i + (1 - gamma_i) * sig(x_i, alpha, -1) * sig(-x_i_plus_1, alpha, 1)) * np.prod(opposite_gamma)
            beta_x_i = x_i_sum * beta_i

            opposite_gamma.append(1 - gamma_i)
            opposite_beta.append(1 - beta_i)

        else:
            beta_i = 1 - np.prod(opposite_beta)
            beta_x_i = x_i_sum * beta_i

        beta_x.append(beta_x_i)
    return np.sum(beta_x)


def smoothed_opposite_warm(local_sensation):
    """Return the overall sensation calculated by opposite warm model."""
    sensation_level = utils.get_sensation_level(local_sensation, "warm")
    if sensation_level == "high":
        overall_sensation_bigger = high_level_warm(local_sensation[local_sensation >= 0])
        overall_sensation_bigger_ex = high_level_warm(local_sensation[local_sensation >= -1])
    else:
        overall_sensation_bigger = smoothed_low_level_warm_0(local_sensation)
        overall_sensation_bigger_ex = smoothed_low_level_warm_minus_1(local_sensation)

    modifier = _extreme_modifier(local_sensation[local_sensation < 0], overall_sensation_bigger)

    return overall_sensation_bigger_ex + modifier


def smoothed_low_level_cold(local_sensation) -> float:
    """Returns the overall sensation calculated by internally smoothed low level cold model."""
    local_sensation_ascending = local_sensation.sort_values(ascending=True)
    interval = _get_interval(local_sensation_ascending)

    if utils.are_hands_feet_most_extreme(local_sensation_ascending):
        local_sensation_ascending = local_sensation_ascending.drop(local_sensation_ascending.index[1])

    opposite_gamma = []
    opposite_beta = []
    beta_x = []
    alpha = 10
    # gamma_i = sig(-x_i, alpha, 2 - interval * (i - 1))
    # beta_i = gamma_i * (1 - gamma_i-1) * (1 - gamma_i-2) * ... * (1 - gamma_2)
    # beta_N = 1 - (1-beta_N-1) * (1-beta_N-2) * ... * (1-beta_2)
    for i in range(2, len(local_sensation_ascending)):
        x_i = local_sensation_ascending.iloc[i]
        x_i_sum = np.mean(local_sensation_ascending.iloc[:i + 1])

        if i != len(local_sensation_ascending) - 1:
            gamma_i = sig(-x_i, alpha, 2 - interval * (i - 1))
            beta_i = gamma_i * np.prod(opposite_gamma)  # np.prod([]) = 1
            beta_x_i = x_i_sum * beta_i

            opposite_gamma.append(1 - gamma_i)
            opposite_beta.append(1 - beta_i)

        else:
            beta_i = 1 - np.prod(opposite_beta)
            beta_x_i = x_i_sum * beta_i

        beta_x.append(beta_x_i)
    return np.sum(beta_x)


def smoothed_low_level_cold_0(local_sensation) -> float:
    """Returns the overall sensation calculated by internally smoothed low level warm model."""
    local_sensation_ascending = local_sensation.sort_values(ascending=True)
    interval = _get_interval(local_sensation_ascending)

    if utils.are_hands_feet_most_extreme(local_sensation_ascending):
        local_sensation_ascending = local_sensation_ascending.drop(local_sensation_ascending.index[1])

    opposite_gamma = []
    opposite_beta = []
    beta_x = []
    alpha = 10
    # gamma_i = 1 / (1 + exp(-alpha * (x_i - 2 + interval * (i - 1)))
    # beta_i = (gamma_i + (1 - gamma_i) * sig(x_i, alpha, 0) * sig(-x_i+1, alpha, 0)) *
    #          (1 - gamma_i-1) * (1 - gamma_i-2) * ... * (1 - gamma_2)
    # beta_N = 1 - (1-beta_N-1) * (1-beta_N-2) * ... * (1-beta_2)
    for i in range(2, len(local_sensation_ascending)):
        x_i = local_sensation_ascending.iloc[i]
        x_i_sum = np.mean(local_sensation_ascending.iloc[:i + 1])

        if i != len(local_sensation_ascending) - 1:
            x_i_plus_1 = local_sensation_ascending.iloc[i + 1]
            gamma_i = sig(-x_i, alpha, 2 - interval * (i - 1))
            beta_i = (gamma_i + (1 - gamma_i) * sig(-x_i, alpha, 0) * sig(x_i_plus_1, alpha, 0)) * np.prod(opposite_gamma)
            beta_x_i = x_i_sum * beta_i

            opposite_gamma.append(1 - gamma_i)
            opposite_beta.append(1 - beta_i)

        else:
            beta_i = 1 - np.prod(opposite_beta)
            beta_x_i = x_i_sum * beta_i

        beta_x.append(beta_x_i)
    return np.sum(beta_x)


def smoothed_low_level_cold_plus_1(local_sensation) -> float:
    """Returns the overall sensation calculated by internally smoothed low level warm model."""
    local_sensation_ascending = local_sensation.sort_values(ascending=True)
    interval = _get_interval(local_sensation_ascending)

    if utils.are_hands_feet_most_extreme(local_sensation_ascending):
        local_sensation_ascending = local_sensation_ascending.drop(local_sensation_ascending.index[1])

    opposite_gamma = []
    opposite_beta = []
    beta_x = []
    alpha = 10
    # gamma_i = 1 / (1 + exp(-alpha * (x_i - 2 + interval * (i - 1)))
    # beta_i = (gamma_i + (1 - gamma_i) * sig(x_i, alpha, -1) * sig(-x_i+1, alpha, 1)) *
    #          (1 - gamma_i-1) * (1 - gamma_i-2) * ... * (1 - gamma_2)
    # beta_N = 1 - (1-beta_N-1) * (1-beta_N-2) * ... * (1-beta_2)
    for i in range(2, len(local_sensation_ascending)):
        x_i = local_sensation_ascending.iloc[i]
        x_i_sum = np.mean(local_sensation_ascending.iloc[:i + 1])

        if i != len(local_sensation_ascending) - 1:
            x_i_plus_1 = local_sensation_ascending.iloc[i + 1]
            gamma_i = sig(-x_i, alpha, 2 - interval * (i - 1))
            beta_i = (gamma_i + (1 - gamma_i) * sig(-x_i, alpha, -1) * sig(x_i_plus_1, alpha, 1)) * np.prod(opposite_gamma)
            beta_x_i = x_i_sum * beta_i

            opposite_gamma.append(1 - gamma_i)
            opposite_beta.append(1 - beta_i)

        else:
            beta_i = 1 - np.prod(opposite_beta)
            beta_x_i = x_i_sum * beta_i

        beta_x.append(beta_x_i)
    return np.sum(beta_x)


def smoothed_opposite_cold(local_sensation):
    """Return the overall sensation calculated by opposite warm model."""
    sensation_level = utils.get_sensation_level(local_sensation, "cold")
    if sensation_level == "high":
        overall_sensation_bigger = high_level_cold(local_sensation[local_sensation <= 0])
        overall_sensation_bigger_ex = high_level_cold(local_sensation[local_sensation <= 1])
    else:
        overall_sensation_bigger = smoothed_low_level_cold_0(local_sensation)
        overall_sensation_bigger_ex = smoothed_low_level_cold_plus_1(local_sensation)

    modifier = _extreme_modifier(local_sensation[local_sensation > 0], overall_sensation_bigger)

    return overall_sensation_bigger_ex + modifier


# ------------------------------Internally Smoothed and Modified Sensation Model------------------------------------

def smoothed_modified_low_level_warm(local_sensation) -> float:
    sensation = smoothed_low_level_warm(local_sensation)
    # sensations larger than 1 and overall sensation are considered for modifier
    threshold = max([sensation, 1])
    modifier = _extreme_modifier(local_sensation[local_sensation > threshold], sensation)
    return sensation + modifier


def smoothed_modified_low_level_cold(local_sensation) -> float:
    sensation = smoothed_low_level_cold(local_sensation)
    # Limit sensation by dominant parts
    sensation = min([min(local_sensation[DOMINANT_BODY_PARTS]), sensation])
    # sensations less than -1 and overall sensation are considered for modifier
    threshold = min([sensation, -1])
    modifier = _extreme_modifier(local_sensation[local_sensation < threshold], sensation)
    return sensation + modifier


def smoothed_modified_no_opposite_model(local_sensation, bigger_group):
    """Return the overall sensation calculated by no-opposite model."""
    sensation_level = utils.get_sensation_level(local_sensation, bigger_group)
    if sensation_level == "high":
        return high_level_warm(local_sensation) if bigger_group == "warm" else high_level_cold(local_sensation)
    if sensation_level == "low":
        return smoothed_low_level_warm(local_sensation) if bigger_group == "warm" else smoothed_low_level_cold(local_sensation)


def smoothed_modified_opposite_warm(local_sensation) -> float:
    sensation = smoothed_modified_no_opposite_model(local_sensation.where(local_sensation >= -1, -1), 'warm')
    # sensation = no_opposite_model(local_sensation)
    modifier_cold = _extreme_modifier(local_sensation[local_sensation < 0], 0)
    threshold = max([sensation, 0])
    modifier_warm = _extreme_modifier(local_sensation[local_sensation > threshold], sensation)
    return sensation + modifier_warm + modifier_cold


def smoothed_modified_opposite_cold(local_sensation) -> float:
    sensation = smoothed_modified_no_opposite_model(local_sensation.where(local_sensation <= 1, 1), 'cold')
    # sensation = no_opposite_model(local_sensation)
    # Limit sensation by dominant parts
    sensation = min([min(local_sensation[DOMINANT_BODY_PARTS]), sensation])
    modifier_warm = _extreme_modifier(local_sensation[local_sensation > 0], 0)
    threshold = min([sensation, 0])
    modifier_cold = _extreme_modifier(local_sensation[local_sensation < threshold], sensation)
    return sensation + modifier_warm + modifier_cold
