# -*- coding: utf-8 -*-
# @Time    : 2024/7/4
# @Author  : Eric
from typing import Literal

import pandas as pd
import numpy as np

from tscm.const import COEFFICIENT
from tscm.config import WholeSensationConfig


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
        self.dominant_parts = list(whole_sensation_config.dominant_parts)

    @staticmethod
    def _get_bigger_group(local_sensation) -> Literal["warm", "cold"]:
        """
        Determine which sensation group (warm or cold / positive or negative) is bigger group. The group including
        larger number of sensations will be considered as bigger group. If the numbers are the same, bigger group is
        warm / positive sensation group.

        Returns:
            Name of bigger group which is either "warm" or "cold".
        """
        return "warm" if sum(local_sensation.gt(0)) >= sum(local_sensation.gt(0)) else "cold"

    def _get_sensation_level(self, local_sensation, bigger_group) -> Literal["high", "low"]:
        """
        Determine the level of sensation for input local sensation group.

        Args:
            local_sensation : A series containing thermal sensations for various body parts.
            bigger_group : Name of bigger group which is either "warm" or "cold".

        Returns:
            The level of sensation for input local sensation group.
        """
        if bigger_group == "warm":
            second_max_index = 1 if not self._are_hands_feet_warmest(local_sensation) else 2
            second_max = local_sensation.iloc[second_max_index]
            return 'high' if second_max >= 2 else 'low'
        if bigger_group == "cold":
            second_min_index = 1 if not self._are_hands_feet_coldest(local_sensation) else 2
            second_min = local_sensation.iloc[second_min_index]
            return 'high' if second_min <= -2 else 'low'

    @staticmethod
    def _are_hands_feet_most_extreme(local_sensation_sorted: pd.Series) -> bool:
        """
        Determine if the most extreme sensations in a given sorted series are from the hands or feet.

        This function examines the input series of sensations to determine if the most extreme sensations
        come from the hands or feet. The input series should be sorted in advanced.
        For sensations in ascending order, this function will determine whether hands or feet sensations are
        maximum. For those in descending order, it will determine whether hands or feet sensations are minimum.

        Args:
            local_sensation_sorted : A series containing sorted thermal sensations for various body parts.

        Returns:
            Return True if the most extreme sensations (either the two largest or two smallest)
            correspond to both hands or both feet. Otherwise, return False.
        """
        if len(local_sensation_sorted) < 2:
            return False

        body_parts_sorted = local_sensation_sorted.index
        are_hands_most_extreme = body_parts_sorted[0].endswith("Hand") and body_parts_sorted[1].endswith("Hand")
        are_feet_most_extreme = body_parts_sorted[0].endswith("Foot") and body_parts_sorted[1].endswith("Foot")
        return are_hands_most_extreme or are_feet_most_extreme

    def _are_hands_feet_warmest(self, local_sensation) -> bool:
        """
        Determine if the warmest sensations are from the hands or feet.

        Returns:
            Return True if the warmest sensations are from the hands or feet. Otherwise, return False.
        """
        local_sensation_descending = local_sensation.sort_values(ascending=False)
        return self._are_hands_feet_most_extreme(local_sensation_descending)

    def _are_hands_feet_coldest(self, local_sensation) -> bool:
        """
        Determine if the coldest sensations are from the hands or feet.

        Returns:
            Return True if the coldest sensations are from the hands or feet. Otherwise, return False.
        """
        local_sensation_ascending = local_sensation.sort_values(ascending=True)
        return self._are_hands_feet_most_extreme(local_sensation_ascending)

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
        if not self.are_sensations_no_opposite:
            if self.bigger_group == "cold":
                return 7
            if self.is_cold_dominated:
                return 5
            if self.bigger_group == "warm":
                return 6

        if self.are_sensations_no_opposite:
            if self.bigger_group == "warm":
                sensation_level = self._get_sensation_level(self.local_sensation, self.bigger_group)
                if sensation_level == "high":
                    return 1
                if sensation_level == "low":
                    return 3

            if self.bigger_group == "cold":
                sensation_level = self._get_sensation_level(self.local_sensation, self.bigger_group)
                if sensation_level == "high":
                    return 2
                if sensation_level == "low":
                    return 4

    @property
    def bigger_group(self) -> Literal["warm", "cold"]:
        """Name of bigger group which is either "warm" or "cold"."""
        return self._get_bigger_group(self.local_sensation)

    @property
    def are_hands_feet_warmest(self) -> bool:
        """Whether the warmest sensations are from the hands or feet."""
        return self._are_hands_feet_warmest(self.local_sensation)

    @property
    def are_hands_feet_coldest(self) -> bool:
        """Whether the coldest sensations are from the hands or feet."""
        return self._are_hands_feet_coldest(self.local_sensation)

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

    def _get_overall_sensations_dict(self) -> dict:
        def high_level_warm(local_sensation) -> float:
            """Returns the whole-body sensation calculated by no-opposite high level warm (complaint warm) model."""
            local_sensation_descending = local_sensation.sort_values(ascending=False)
            if self._are_hands_feet_most_extreme(local_sensation_descending):
                return 0.5 * local_sensation_descending.iloc[0] + 0.5 * local_sensation_descending.iloc[2]
            else:
                return 0.5 * local_sensation_descending.iloc[0] + 0.5 * local_sensation_descending.iloc[1]

        def high_level_cold(local_sensation) -> float:
            """Returns the whole-body sensation calculated by no-opposite high level cold (complaint cold) model."""
            local_sensation_ascending = local_sensation.sort_values(ascending=True)
            if self._are_hands_feet_most_extreme(local_sensation_ascending):
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

            if self._are_hands_feet_most_extreme(local_sensation_descending):
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

            if self._are_hands_feet_most_extreme(local_sensation_ascending):
                local_sensation_ascending = local_sensation_ascending.drop(local_sensation_ascending.index[1])
            local_sensation_selected = list(local_sensation_ascending)[:2]
            for i in range(2, len(local_sensation_ascending)):
                local_sensation_selected.append(local_sensation_ascending.iloc[i])
                if local_sensation_ascending.iloc[i] < -2 + interval * (i - 1):
                    break

            return np.mean(local_sensation_selected)

        def no_opposite_model(local_sensation):
            """Return the whole-body sensation calculated by no-opposite model."""
            body_parts_num = len(local_sensation)
            # If no sensation in input series, return 0.
            if body_parts_num == 0:
                return 0
            # If only one sensation in input series, return itself.
            if body_parts_num == 1:
                return local_sensation.iloc[0]
            # If only two sensation in input series and both come from hands or feet, return the average value.
            if body_parts_num == 2 and self._are_hands_feet_most_extreme(local_sensation):
                return local_sensation.mean()

            bigger_group = self._get_bigger_group(local_sensation)

            if bigger_group == "warm":
                sensation_level = self._get_sensation_level(local_sensation, bigger_group)
                if sensation_level == "high":
                    return high_level_warm(local_sensation)
                if sensation_level == "low":
                    return low_level_warm(local_sensation)

            if bigger_group == "cold":
                sensation_level = self._get_sensation_level(local_sensation, bigger_group)
                if sensation_level == "high":
                    return high_level_cold(local_sensation)
                if sensation_level == "low":
                    return low_level_cold(local_sensation)

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
                lambda x: x > 2: 1
            }
            correction_factor = np.piecewise(
                most_extreme_sensation_abs,
                [cond_func(most_extreme_sensation_abs) for cond_func in correction_factor_map.keys()],
                [value for value in correction_factor_map.values()]
            )

            return combined_force * correction_factor

        def opposite_warm(local_sensation):
            """Return the whole-body sensation calculated by opposite warm model."""
            overall_sensation_bigger = no_opposite_model(local_sensation[local_sensation >= 0])
            overall_sensation_bigger_ex = no_opposite_model(local_sensation[local_sensation >= -1])
            modifier = _opposite_modifier(local_sensation[local_sensation < 0], overall_sensation_bigger)

            return overall_sensation_bigger_ex + modifier

        def opposite_cool(local_sensation):
            """Return the whole-body sensation calculated by opposite cool model."""
            overall_sensation_bigger = no_opposite_model(local_sensation[local_sensation <= 0])
            overall_sensation_bigger_ex = no_opposite_model(local_sensation[local_sensation <= 1])
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

        return {model_num: model(self.local_sensation) for model_num, model in sensation_model_map.items()}

    def _get_modified_overall_sensations_dict(self) -> dict:
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

        def high_level_warm(local_sensation) -> float:
            """Returns the whole-body sensation calculated by no-opposite high level warm (complaint warm) model."""
            local_sensation_descending = local_sensation.sort_values(ascending=False)
            if self._are_hands_feet_most_extreme(local_sensation_descending):
                return 0.5 * local_sensation_descending.iloc[0] + 0.5 * local_sensation_descending.iloc[2]
            else:
                return 0.5 * local_sensation_descending.iloc[0] + 0.5 * local_sensation_descending.iloc[1]

        def high_level_cold(local_sensation) -> float:
            """Returns the whole-body sensation calculated by no-opposite high level cold (complaint cold) model."""
            local_sensation_ascending = local_sensation.sort_values(ascending=True)
            if self._are_hands_feet_most_extreme(local_sensation_ascending):
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

            if self._are_hands_feet_most_extreme(local_sensation_descending):
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

            if self._are_hands_feet_most_extreme(local_sensation_ascending):
                local_sensation_ascending = local_sensation_ascending.drop(local_sensation_ascending.index[1])
            local_sensation_selected = list(local_sensation_ascending)[:2]
            for i in range(2, len(local_sensation_ascending)):
                local_sensation_selected.append(local_sensation_ascending.iloc[i])
                if local_sensation_ascending.iloc[i] < -2 + interval * (i - 1):
                    break

            return np.mean(local_sensation_selected)

        def no_opposite_model(local_sensation):
            """Return the whole-body sensation calculated by no-opposite model."""
            body_parts_num = len(local_sensation)
            # If no sensation in input series, return 0.
            if body_parts_num == 0:
                return 0
            # If only one sensation in input series, return itself.
            if body_parts_num == 1:
                return local_sensation.iloc[0]
            # If only two sensation in input series and both come from hands or feet, return the average value.
            if body_parts_num == 2 and self._are_hands_feet_most_extreme(local_sensation):
                return local_sensation.mean()

            bigger_group = self._get_bigger_group(local_sensation)

            if bigger_group == "warm":
                sensation_level = self._get_sensation_level(local_sensation, bigger_group)
                if sensation_level == "high":
                    return high_level_warm(local_sensation)
                if sensation_level == "low":
                    return low_level_warm(local_sensation)

            if bigger_group == "cold":
                sensation_level = self._get_sensation_level(local_sensation, bigger_group)
                if sensation_level == "high":
                    return high_level_cold(local_sensation)
                if sensation_level == "low":
                    return low_level_cold(local_sensation)

        def opposite_dominated_cold(local_sensation):
            """Return the whole-body sensation calculated by opposite dominated cold model."""
            return min(local_sensation[self.dominant_parts])

        def opposite_warm(local_sensation):
            """Return the whole-body sensation calculated by opposite warm model."""
            overall_sensation_bigger = no_opposite_model(local_sensation[local_sensation >= 0])
            overall_sensation_bigger_ex = no_opposite_model(local_sensation[local_sensation >= -1])
            modifier = _extreme_modifier(local_sensation[local_sensation < 0], overall_sensation_bigger)

            return overall_sensation_bigger_ex + modifier

        def opposite_cool(local_sensation):
            """Return the whole-body sensation calculated by opposite cool model."""
            overall_sensation_bigger = no_opposite_model(local_sensation[local_sensation <= 0])
            overall_sensation_bigger_ex = no_opposite_model(local_sensation[local_sensation <= 1])
            modifier = _extreme_modifier(local_sensation[local_sensation > 0], overall_sensation_bigger)

            return overall_sensation_bigger_ex + modifier

        def modified_low_level_warm(local_sensation) -> float:
            sensation = low_level_warm(local_sensation)
            modifier = _extreme_modifier(local_sensation[local_sensation > 1], sensation)
            return sensation + modifier

        def modified_low_level_cold(local_sensation) -> float:
            sensation = low_level_cold(local_sensation)
            modifier = _extreme_modifier(local_sensation[local_sensation < -1], sensation)
            return sensation + modifier

        sensation_model_map = {
            1: high_level_warm,
            2: high_level_cold,
            3: low_level_warm,
            4: low_level_cold,
            5: opposite_dominated_cold,
            6: opposite_warm,
            7: opposite_cool
        }

        return {model_num: model(self.local_sensation) for model_num, model in sensation_model_map.items()}

    @property
    def overall_sensations_dict(self) -> dict:
        """A series of overall sensations for different whole-body sensation models."""
        return self._get_overall_sensations_dict()

    def _get_overall_sensation(self):
        if not self.smooth:
            return self.overall_sensations_dict[self.model_num]

        def sig(x, a, t): return 1 / (1 + np.exp(-a * (x - t)))

        x1 = min(self.local_sensation[self.dominant_parts])
        x2 = self.local_sensation.sort_values(ascending=False).iloc[0]
        x3 = self.local_sensation.sort_values(ascending=True).iloc[0]
        x4 = self.local_sensation.iloc[1 if not self._are_hands_feet_warmest(self.local_sensation) else 2]
        x5 = self.local_sensation.iloc[1 if not self._are_hands_feet_coldest(self.local_sensation) else 2]
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
                sig(x4, alpha, 2) * sig(x3, alpha, -1),
                sig(-x6, alpha, 0) * sig(-x2, alpha, -1) * sig(-x5, alpha, 2),
                sig(-x4, alpha, -2) * sig(x3, alpha, -1),
                sig(-x6, alpha, 0) * sig(-x2, alpha, -1) * sig(x5, alpha, -2),
                sig(-x1, alpha, 1),
                sig(-x6, alpha, 0) * sig(x2, alpha, 1),
            ]
        }

        y_k = y_k_dict[self.model_num]
        w_ik = w_ik_dict[self.model_num]

        y_ik = [y - y_i for y in y_k]
        w_y_ik = [float(w * y) for w, y in zip(w_ik, y_ik)]

        if not self.smooth_adjusted:
            return y_i + np.sum(w_y_ik)

        w_y_ik_max_index = np.argmax(np.abs(w_y_ik))
        y_i_modified = y_i + w_y_ik[w_y_ik_max_index]
        y_ik_modified = [y_k[i] - y_i if i == w_y_ik_max_index else y_k[i] - y_i_modified for i in range(len(y_k))]
        w_y_ik_modified = [float(w * y) for w, y in zip(w_ik, y_ik_modified)]

        return y_i + np.sum(w_y_ik_modified)

    @property
    def overall_sensation(self) -> float:
        """The overall sensation for the input local sensations."""
        return self._get_overall_sensation()
