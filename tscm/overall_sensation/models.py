# -*- coding: utf-8 -*-
# @Time    : 2024/8/22
# @Author  : Eric
import pandas as pd
import numpy as np

from tscm.const import COEFFICIENT, DOMINANT_BODY_PARTS
import tscm.overall_sensation.utilities as util


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


def _extreme_modifier(local_sensation, baseline_sensation):
    """
    Calculate combined force of extreme sensations as modifier for opposite sensation model.

    For the original sensation model, the modifier is for the opposite sensations.

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


# ------------------------------Original Sensation Model-----------------------------------------------

def high_level_warm(local_sensation) -> float:
    """Returns the overall sensation calculated by no-opposite high level warm (complaint warm) model."""
    local_sensation_descending = local_sensation.sort_values(ascending=False)
    if util.are_hands_feet_most_extreme(local_sensation_descending):
        return 0.5 * local_sensation_descending.iloc[0] + 0.5 * local_sensation_descending.iloc[2]
    else:
        return 0.5 * local_sensation_descending.iloc[0] + 0.5 * local_sensation_descending.iloc[1]


def high_level_cold(local_sensation) -> float:
    """Returns the overall sensation calculated by no-opposite high level cold (complaint cold) model."""
    local_sensation_ascending = local_sensation.sort_values(ascending=True)
    if util.are_hands_feet_most_extreme(local_sensation_ascending):
        return 0.38 * local_sensation_ascending.iloc[0] + 0.62 * local_sensation_ascending.iloc[2]
    else:
        return 0.38 * local_sensation_ascending.iloc[0] + 0.62 * local_sensation_ascending.iloc[1]


def low_level_warm(local_sensation) -> float:
    """Returns the overall sensation calculated by no-opposite low level warm (gradual warm) model."""
    local_sensation_descending = local_sensation.sort_values(ascending=False)
    interval = _get_interval(local_sensation_descending)

    if util.are_hands_feet_most_extreme(local_sensation_descending):
        local_sensation_descending = local_sensation_descending.drop(local_sensation_descending.index[1])
    local_sensation_selected = list(local_sensation_descending)[:2]
    for i in range(2, len(local_sensation_descending)):
        local_sensation_selected.append(local_sensation_descending.iloc[i])
        if local_sensation_descending.iloc[i] > 2 - interval * (i - 1):
            break  # Stop when sensation meets the condition.

    return np.mean(local_sensation_selected)


def low_level_cold(local_sensation) -> float:
    """Returns the overall sensation calculated by no-opposite low level cold (gradual cold) model."""
    local_sensation_ascending = local_sensation.sort_values(ascending=True)
    interval = _get_interval(local_sensation_ascending)

    if util.are_hands_feet_most_extreme(local_sensation_ascending):
        local_sensation_ascending = local_sensation_ascending.drop(local_sensation_ascending.index[1])
    local_sensation_selected = list(local_sensation_ascending)[:2]
    for i in range(2, len(local_sensation_ascending)):
        local_sensation_selected.append(local_sensation_ascending.iloc[i])
        if local_sensation_ascending.iloc[i] < -2 + interval * (i - 1):
            break

    return np.mean(local_sensation_selected)


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
    if body_parts_num == 2 and util.are_hands_feet_most_extreme(local_sensation):
        return local_sensation.mean()

    sensation_level = util.get_sensation_level(local_sensation, bigger_group)
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
    modifier = _extreme_modifier(local_sensation[local_sensation < 0], overall_sensation_bigger)

    return overall_sensation_bigger_ex + modifier


def opposite_cool(local_sensation):
    """Return the overall sensation calculated by opposite cool model."""
    overall_sensation_bigger = no_opposite_model(local_sensation[local_sensation <= 0], "cold")
    overall_sensation_bigger_ex = no_opposite_model(local_sensation[local_sensation <= 1], "cold")
    modifier = _extreme_modifier(local_sensation[local_sensation > 0], overall_sensation_bigger)

    return overall_sensation_bigger_ex + modifier


# ------------------------------Modified Sensation Model-----------------------------------------------

def modified_high_level_warm(local_sensation) -> float:
    sensation = high_level_warm(local_sensation)
    # sensations larger than 1 and overall sensation are considered for modifier
    threshold = max([sensation, 1])
    modifier = _extreme_modifier(local_sensation[local_sensation > threshold], sensation)
    return sensation + modifier


def modified_high_level_cold(local_sensation) -> float:
    sensation = high_level_cold(local_sensation)
    # Limit sensation by dominant parts
    sensation = min([min(local_sensation[DOMINANT_BODY_PARTS]), sensation])
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
    sensation = min([min(local_sensation[DOMINANT_BODY_PARTS]), sensation])
    # sensations less than -1 and overall sensation are considered for modifier
    threshold = min([sensation, -1])
    modifier = _extreme_modifier(local_sensation[local_sensation < threshold], sensation)
    return sensation + modifier


def modified_opposite_dominated_cold(local_sensation) -> float:
    sensation = opposite_dominated_cold(local_sensation)
    modifier_warm = _extreme_modifier(local_sensation[local_sensation > 0], 0)
    threshold = min([sensation, -1])
    modifier_cold = _extreme_modifier(local_sensation[local_sensation < threshold], sensation)
    return sensation + modifier_warm + modifier_cold


def modified_opposite_warm(local_sensation) -> float:
    sensation = no_opposite_model(local_sensation.where(local_sensation >= -1, -1), 'warm')
    # sensation = no_opposite_model(local_sensation)
    modifier_cold = _extreme_modifier(local_sensation[local_sensation < 0], 0)
    threshold = max([sensation, 0])
    modifier_warm = _extreme_modifier(local_sensation[local_sensation > threshold], sensation)
    return sensation + modifier_warm + modifier_cold


def modified_opposite_cool(local_sensation) -> float:
    sensation = no_opposite_model(local_sensation.where(local_sensation <= 1, 1), 'cold')
    # sensation = no_opposite_model(local_sensation)
    # Limit sensation by dominant parts
    sensation = min([min(local_sensation[DOMINANT_BODY_PARTS]), sensation])
    modifier_warm = _extreme_modifier(local_sensation[local_sensation > 0], 0)
    threshold = min([sensation, 0])
    modifier_cold = _extreme_modifier(local_sensation[local_sensation < threshold], sensation)
    return sensation + modifier_warm + modifier_cold
