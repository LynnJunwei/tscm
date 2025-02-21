# -*- coding: utf-8 -*-
# @Time    : 2024/7/5
# @Author  : Eric
import pandas as pd
import numpy as np
from tscm.const import COEFFICIENT


def _opposite_modifier(local_sensation, overall_sensation):
    """
    Calculate combined force of opposite sensations as modifier for opposite sensation model.

    Args:
        local_sensation:
        overall_sensation: Value of overall sensation of bigger group calculated by no-opposite model.

    Returns:
        Value of modifier for opposite sensation model.
    """
    if len(local_sensation) == 0:
        return 0

    individual_force = pd.Series().reindex_like(local_sensation)

    for body_part, sensation in local_sensation.items():
        delta_sensation = sensation - overall_sensation
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

    most_extreme_sensation_abs = np.abs(local_sensation[individual_force_sorted.index[0]])

    correction_factor_map = {  # condition: correction factor
        lambda x: x < 1: 0,
        lambda x: 1 <= x < 2: round(most_extreme_sensation_abs - 1, 3),
        lambda x: x > 2: 1
    }
    correction_factor = np.piecewise(
        most_extreme_sensation_abs,
        [cond_func(most_extreme_sensation_abs) for cond_func in correction_factor_map.keys()],
        [value for value in correction_factor_map.values()]
    )

    return combined_force * correction_factor


if __name__ == '__main__':
    local_sensation = pd.Series({"Head": -1.2, "Neck": -1., "Chest": 0.5, "Back": 0.5, "Pelvis": 0.5,
                                 "LUpperArm": 0.5, "LLowerArm": 3, "LHand": 0.5,
                                 "RUpperArm": 0.5, "RLowerArm": 0.5, "RHand": 0.5,
                                 "LThigh": 0.5, "LLeg": 0.5, "LFoot": 0.5,
                                 "RThigh": 0.5, "RLeg": 3, "RFoot": 3})
    overall_sensation = 1
    _opposite_modifier(local_sensation, overall_sensation)
