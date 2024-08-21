# -*- coding: utf-8 -*-
# @Time    : 2024/7/4
# @Author  : Eric
from typing import Literal

import pandas as pd
import numpy as np

from tscm.const import COEFFICIENT
from tscm.config import WholeSensationConfig
import tscm.utility as util


class LocalSensationProcessor:
    """
    Calculate overall local_sensation_sorted for a set of local sensations.
    """

    def __init__(self, local_sensation, whole_sensation_config: WholeSensationConfig):
        """
        Args:
            whole_sensation_config:
                Configurations for overall sensation calculation. Refer to class WholeSensationConfig.
        """
        self.local_sensation = local_sensation

        self.body_parts = self.local_sensation.index

        self.smooth_alpha = whole_sensation_config.smooth_alpha
        self.smooth = whole_sensation_config.smooth
        self.smooth_adjusted = whole_sensation_config.smooth_adjusted
        self.model_type = whole_sensation_config.model_type
        self.dominant_parts = list(whole_sensation_config.dominant_parts)

    def _is_cold_dominated(self) -> bool:
        """
        Determine if the cool or cold sensation has potential to dominate overall sensation.

        In this function, the situation that the minimum value of the sensations come from dominant body parts
        (chest, back and pelvis) is less or equal to -1 will be considered as that has potential to dominate overall
        sensation. Noted that it does not mean that the cool or cold sensation could actually dominate overall
        sensation when the condition is satisfied. An exception is that when bigger group of input sensations is
        in the cold / negative side, the cool or cold sensation comes from dominant body part will not dominate
        overall sensation.

        Returns:
            Return True if the cool or cold sensation has potential to dominate overall sensation.
            Otherwise, return False.
        """
        return True if min(self.local_sensation[self.dominant_parts]) <= -1 else False

    def _are_sensations_no_opposite(self) -> bool:
        """
        Determine if the input local sensation group does not contain opposite sensations.

        For input sensation group, if all sensations are warm or cold / positive or negative, the group will be
        considered as no opposite. Even though there exist opposite sensations in the sensation group, as long as
        the opposite sensations are at low level (minimum sensation value larger or equal to -1 for whose bigger
        group is in the warm / positive side; maximum value less or equal to 1 for whose bigger group is in the
        cold / negative side), the group will be treated as no opposite. Noted that for input sensation group where
        bigger group is in the warm / positive side, if the cool or cold sensation has potential to dominate overall
        sensation (less or equal to -1), it will not be no opposite.

        Returns:
            Return True if the input group is considered as no opposite. Otherwise, return False.
        """
        if self.bigger_group == "warm":
            return True if min(self.local_sensation) >= -1 and not self.is_cold_dominated else False
        if self.bigger_group == "cold":
            return True if max(self.local_sensation) <= 1 else False

    def _get_sensation_model_num(self) -> Literal[1, 2, 3, 4, 5, 6, 7]:
        """
        Determine the index number of sensation model for input sensations.

        The sensation models and corresponding index numbers are as follows:
            1. No-opposite high level warm (complaint warm)
            2. No-opposite high level cold (complaint cold)
            3. No-opposite low level warm (gradual warm)
            4. No-opposite low level cold (gradual cold)
            5. Opposite dominated cold
            6. Opposite warm
            7. Opposite cool

        Returns:
            The number of whole-body local_sensation_sorted calculation model.
        """
        if self.bigger_group == "warm" and self.is_cold_dominated:
            return 5

        if not self.are_sensations_no_opposite:
            if self.bigger_group == "warm":
                return 6
            if self.bigger_group == "cold":
                return 7

        if self.are_sensations_no_opposite:
            if self.bigger_group == "warm":
                sensation_level = util.get_sensation_level(self.local_sensation, self.bigger_group)
                if sensation_level == "high":
                    return 1
                if sensation_level == "low":
                    return 3

            if self.bigger_group == "cold":
                sensation_level = util.get_sensation_level(self.local_sensation, self.bigger_group)
                if sensation_level == "high":
                    return 2
                if sensation_level == "low":
                    return 4

    @property
    def bigger_group(self) -> Literal["warm", "cold"]:
        """Name of bigger group which is either "warm" or "cold"."""
        return util.get_bigger_group(self.local_sensation)

    @property
    def are_hands_feet_warmest(self) -> bool:
        """Whether the warmest sensations are from the hands or feet."""
        local_sensation_descending = self.local_sensation.sort_values(ascending=False)
        return util.are_hands_feet_most_extreme(local_sensation_descending)

    @property
    def are_hands_feet_coldest(self) -> bool:
        """Whether the coldest sensations are from the hands or feet."""
        local_sensation_ascending = self.local_sensation.sort_values(ascending=True)
        return util.are_hands_feet_most_extreme(local_sensation_ascending)

    @property
    def is_cold_dominated(self) -> bool:
        """Whether the cool or cold sensation has potential to dominate overall sensation."""
        return self._is_cold_dominated()

    @property
    def are_sensations_no_opposite(self) -> bool:
        """Whether the input local sensation group does not contain opposite sensations."""
        return self._are_sensations_no_opposite()

    @property
    def model_num(self) -> Literal[1, 2, 3, 4, 5, 6, 7]:
        """The number of whole-body sensation calculation model."""
        return self._get_sensation_model_num()

    def _get_sensation_models(self) -> dict:
        def high_level_warm(local_sensation) -> float:
            """Returns the whole-body sensation calculated by no-opposite high level warm (complaint warm) model."""
            local_sensation_descending = local_sensation.sort_values(ascending=False)
            if util.are_hands_feet_most_extreme(local_sensation_descending):
                return 0.5 * local_sensation_descending.iloc[0] + 0.5 * local_sensation_descending.iloc[2]
            else:
                return 0.5 * local_sensation_descending.iloc[0] + 0.5 * local_sensation_descending.iloc[1]

        def high_level_cold(local_sensation) -> float:
            """Returns the whole-body sensation calculated by no-opposite high level cold (complaint cold) model."""
            local_sensation_ascending = local_sensation.sort_values(ascending=True)
            if util.are_hands_feet_most_extreme(local_sensation_ascending):
                return 0.38 * local_sensation_ascending.iloc[0] + 0.62 * local_sensation_ascending.iloc[2]
            else:
                return 0.38 * local_sensation_ascending.iloc[0] + 0.62 * local_sensation_ascending.iloc[1]

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

        def low_level_warm(local_sensation) -> float:
            """Returns the whole-body sensation calculated by no-opposite low level warm (gradual warm) model."""
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
            """Returns the whole-body sensation calculated by no-opposite low level cold (gradual cold) model."""
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
            """Return the whole-body sensation calculated by no-opposite model."""
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
            """Return the whole-body sensation calculated by opposite dominated cold model."""
            return min(local_sensation[self.dominant_parts])

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
                lambda x: x >= 2: 1
            }
            correction_factor = np.piecewise(
                most_extreme_sensation_abs,
                [cond_func(most_extreme_sensation_abs) for cond_func in correction_factor_map.keys()],
                [value for value in correction_factor_map.values()]
            )

            return combined_force * correction_factor

        def opposite_warm(local_sensation):
            """Return the whole-body sensation calculated by opposite warm model."""
            overall_sensation_bigger = no_opposite_model(local_sensation[local_sensation >= 0], "warm")
            overall_sensation_bigger_ex = no_opposite_model(local_sensation[local_sensation >= -1], "warm")
            modifier = _opposite_modifier(local_sensation[local_sensation < 0], overall_sensation_bigger)

            return overall_sensation_bigger_ex + modifier

        def opposite_cool(local_sensation):
            """Return the whole-body sensation calculated by opposite cool model."""
            overall_sensation_bigger = no_opposite_model(local_sensation[local_sensation <= 0], "cold")
            overall_sensation_bigger_ex = no_opposite_model(local_sensation[local_sensation <= 1], "cold")
            modifier = _opposite_modifier(local_sensation[local_sensation > 0], overall_sensation_bigger)

            return overall_sensation_bigger_ex + modifier

        sensation_model_map = {
            1: high_level_warm,
            2: high_level_cold,
            3: low_level_warm,
            4: low_level_cold,
            5: opposite_dominated_cold,
            6: opposite_warm,
            7: opposite_cool
        }

        return sensation_model_map

    def _get_modified_sensation_models(self) -> dict:

        sensation_model_map = self._get_sensation_models()
        high_level_warm = sensation_model_map[1]
        high_level_cold = sensation_model_map[2]
        low_level_warm = sensation_model_map[3]
        low_level_cold = sensation_model_map[4]
        opposite_dominated_cold = sensation_model_map[5]

        def _extreme_modifier(local_sensation, overall_sensation):
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

        def modified_no_opposite_model(local_sensation, bigger_group):
            """Return the whole-body sensation calculated by no-opposite model."""
            sensation_level = util.get_sensation_level(local_sensation, bigger_group)
            if sensation_level == "high":
                return high_level_warm(local_sensation) if bigger_group == "warm" else high_level_cold(local_sensation)
            if sensation_level == "low":
                return low_level_warm(local_sensation) if bigger_group == "warm" else low_level_cold(local_sensation)

        def modified_high_level_warm(local_sensation) -> float:
            sensation = high_level_warm(local_sensation)
            # sensations larger than 1 and overall sensation are considered for modifier
            threshold = max([sensation, 1])
            modifier = _extreme_modifier(local_sensation[local_sensation > threshold], sensation)
            return sensation + modifier

        def modified_high_level_cold(local_sensation) -> float:
            sensation = high_level_cold(local_sensation)
            # Limit sensation by dominant parts
            sensation = min([min(local_sensation[self.dominant_parts]), sensation])
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
            sensation = min([min(local_sensation[self.dominant_parts]), sensation])
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
            sensation = modified_no_opposite_model(local_sensation.where(local_sensation >= -1, -1), 'warm')
            # sensation = no_opposite_model(local_sensation)
            modifier_cold = _extreme_modifier(local_sensation[local_sensation < 0], 0)
            threshold = max([sensation, 0])
            modifier_warm = _extreme_modifier(local_sensation[local_sensation > threshold], sensation)
            return sensation + modifier_warm + modifier_cold

        def modified_opposite_cool(local_sensation) -> float:
            sensation = modified_no_opposite_model(local_sensation.where(local_sensation <= 1, 1), 'cold')
            # sensation = no_opposite_model(local_sensation)
            # Limit sensation by dominant parts
            sensation = min([min(local_sensation[self.dominant_parts]), sensation])
            modifier_warm = _extreme_modifier(local_sensation[local_sensation > 0], 0)
            threshold = min([sensation, 0])
            modifier_cold = _extreme_modifier(local_sensation[local_sensation < threshold], sensation)
            return sensation + modifier_warm + modifier_cold

        modified_sensation_model_map = {
            1: modified_high_level_warm,
            2: modified_high_level_cold,
            3: modified_low_level_warm,
            4: modified_low_level_cold,
            5: modified_opposite_dominated_cold,
            6: modified_opposite_warm,
            7: modified_opposite_cool
        }

        return modified_sensation_model_map

    @property
    def overall_sensations_dict(self) -> dict:
        """A series of overall sensations for different whole-body sensation models."""
        if self.model_type == "origin":
            return {model_num: model(self.local_sensation)
                    for model_num, model in self._get_sensation_models().items()}
        if self.model_type == "modified":
            return {model_num: model(self.local_sensation)
                    for model_num, model in self._get_modified_sensation_models().items()}

    def _get_overall_sensation(self):
        if not self.smooth:
            return self.overall_sensations_dict[self.model_num]

        def sig(x, a, t): return 1 / (1 + np.exp(-a * (x - t)))

        x1 = min(self.local_sensation[self.dominant_parts])
        x2 = self.local_sensation.sort_values(ascending=False).iloc[0]
        x3 = self.local_sensation.sort_values(ascending=True).iloc[0]
        x4 = self.local_sensation.sort_values(ascending=False).iloc[1 if not self.are_hands_feet_warmest else 2]
        x5 = self.local_sensation.sort_values(ascending=True).iloc[1 if not self.are_hands_feet_coldest else 2]
        x6 = self.local_sensation.median()

        y_dict = self.overall_sensations_dict

        y_i = y_dict[self.model_num]

        y_k_dict = {
            1: [y_dict[i] for i in [3, 5, 6, 7]],
            2: [y_dict[i] for i in [4, 5, 6, 7]],
            3: [y_dict[i] for i in [1, 4, 5, 6, 7]],
            4: [y_dict[i] for i in [2, 3, 5, 6, 7]],
            5: [y_dict[i] for i in [1, 2, 3, 4, 6, 7]],
            6: [y_dict[i] for i in [1, 2, 3, 4, 5, 7]],
            7: [y_dict[i] for i in [1, 2, 3, 4, 5, 6]]
        }

        alpha = self.smooth_alpha

        w_ik_dict = {
            1: [
                sig(-x4, alpha, -2),
                sig(-x1, alpha, 1),
                sig(x1, alpha, -1) * sig(-x3, alpha, 1),
                sig(-x6, alpha, 0),
            ],
            2: [
                sig(x5, alpha, -2),
                sig(-x1, alpha, 1) * sig(x2, alpha, 0) * sig(x6, alpha, 0),
                sig(x6, alpha, 0) * sig(x1, alpha, -1),
                sig(x2, alpha, 1) * sig(x1, alpha, -1),
            ],
            3: [
                sig(x4, alpha, 2),
                sig(-x6, alpha, 0) * sig(-x2, alpha, -1),
                sig(-x1, alpha, 1),
                sig(x1, alpha, -1) * sig(-x3, alpha, 1),
                sig(-x6, alpha, 0) * sig(x2, alpha, 1),
            ],
            4: [
                sig(-x5, alpha, 2),
                sig(x6, alpha, 0) * sig(x3, alpha, -1),
                sig(-x1, alpha, 1) * sig(x2, alpha, 0) * sig(x6, alpha, 0),
                sig(x6, alpha, 0) * sig(x1, alpha, -1) * sig(-x3, alpha, -1),
                sig(x2, alpha, 1) * sig(x1, alpha, -1),
            ],
            5: [
                sig(x6, alpha, 0) * sig(x4, alpha, 2) * sig(x3, alpha, -1),
                sig(-x6, alpha, 0) * sig(-x2, alpha, -1) * sig(-x5, alpha, 2) * sig(x1, alpha, -1),
                sig(x6, alpha, 0) * sig(-x4, alpha, -2) * sig(x3, alpha, -1),
                sig(-x6, alpha, 0) * sig(-x2, alpha, -1) * sig(x5, alpha, -2) * sig(x1, alpha, -1),
                sig(x6, alpha, 0) * sig(x1, alpha, -1) * sig(-x3, alpha, 1),
                sig(-x6, alpha, 0) * sig(x1, alpha, -1) * sig(x2, alpha, 1),
            ],
            6: [
                sig(x4, alpha, 2) * sig(x3, alpha, -1),
                sig(-x6, alpha, 0) * sig(-x2, alpha, -1) * sig(-x5, alpha, 2),
                sig(-x4, alpha, -2) * sig(x3, alpha, -1),
                sig(-x6, alpha, 0) * sig(-x2, alpha, -1) * sig(x5, alpha, -2),
                sig(-x1, alpha, 1),
                sig(-x6, alpha, 0) * sig(x2, alpha, 1),
            ],
            7: [
                sig(x6, alpha, 0) * sig(x4, alpha, 2) * sig(x3, alpha, -1),
                sig(-x2, alpha, -1) * sig(-x5, alpha, 2),
                sig(x6, alpha, 0) * sig(-x4, alpha, -2) * sig(x3, alpha, -1),
                sig(-x2, alpha, -1) * sig(x5, alpha, -2),
                sig(-x1, alpha, 1),
                sig(x6, alpha, 0) * sig(-x3, alpha, 1),
            ]
        }

        y_k = y_k_dict[self.model_num]
        w_ik = w_ik_dict[self.model_num]

        y_ik = [y - y_i for y in y_k]
        w_y_ik = [float(w * y) for w, y in zip(w_ik, y_ik)]

        if not self.smooth_adjusted:
            return y_i + np.sum(w_y_ik)

        w_y_ik_max_index = np.argmax(np.abs(w_y_ik))
        print(w_y_ik)
        print(w_ik)
        y_i_modified = y_i + w_y_ik[w_y_ik_max_index]
        y_ik_modified = [y_k[i] - y_i if i == w_y_ik_max_index else y_k[i] - y_i_modified for i in range(len(y_k))]
        w_y_ik_modified = [float(w * y) for w, y in zip(w_ik, y_ik_modified)]
        print(w_y_ik_modified)

        return y_i + np.sum(w_y_ik_modified)

    @property
    def overall_sensation(self) -> float:
        """The overall sensation for the input local sensations."""
        return self._get_overall_sensation()
