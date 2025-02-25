# -*- coding: utf-8 -*-
# @Time    : 2024/7/4
# @Author  : Eric
from typing import Literal

import pandas as pd
import numpy as np

from tscm.const import DOMINANT_BODY_PARTS
from tscm.config import OverallSensationConfig
import tscm.overall_sensation.utilities as utils
import tscm.overall_sensation.models as models


class OverallSensationCalculator:
    """
    Calculate overall local_sensation_sorted for a set of local sensations.
    """

    def __init__(self, local_sensation: pd.Series, overall_sensation_config: OverallSensationConfig):
        """
        Args:
            local_sensation: A series of local sensations.
            overall_sensation_config: Configurations for overall sensation calculation.
                Refer to class OverallSensationConfig.
        """
        self.local_sensation = local_sensation
        self.body_parts = self.local_sensation.index
        self.config = overall_sensation_config

    @property
    def bigger_group(self) -> Literal["warm", "cold"]:
        """Name of bigger group which is either "warm" or "cold"."""
        return utils.get_bigger_group(self.local_sensation)

    @property
    def are_hands_feet_warmest(self) -> bool:
        """Whether the warmest sensations are from the hands or feet."""
        local_sensation_descending = self.local_sensation.sort_values(ascending=False)
        return utils.are_hands_feet_most_extreme(local_sensation_descending)

    @property
    def are_hands_feet_coldest(self) -> bool:
        """Whether the coldest sensations are from the hands or feet."""
        local_sensation_ascending = self.local_sensation.sort_values(ascending=True)
        return utils.are_hands_feet_most_extreme(local_sensation_ascending)

    @property
    def is_cold_dominated(self) -> bool:
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
        return True if min(self.local_sensation[DOMINANT_BODY_PARTS]) <= -1 else False

    @property
    def are_sensations_no_opposite(self) -> bool:
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

    @property
    def model_num(self) -> Literal[1, 2, 3, 4, 5, 6, 7]:
        """
        Determine the index number of sensation model for input sensations.

        The sensation models and corresponding index numbers are as follows:
            1. No-opposite high level warm (complaint warm)
            2. No-opposite high level cold (complaint cold)
            3. No-opposite low level warm (gradual warm)
            4. No-opposite low level cold (gradual cold)
            5. Opposite-dominated cold / Dominated cold (for modified models)
            6. Opposite warm
            7. Opposite cool

        Returns:
            The number of overall sensation calculation model.
        """
        if self.is_cold_dominated:
            if self.config.original_model and self.local_sensation.max() > 0:
                return 5
            if not self.config.original_model:
                # modified models
                return 5

        if not self.are_sensations_no_opposite:
            if self.bigger_group == "warm":
                return 6
            if self.bigger_group == "cold":
                return 7

        if self.are_sensations_no_opposite:
            if self.bigger_group == "warm":
                sensation_level = utils.get_sensation_level(self.local_sensation, self.bigger_group)
                if sensation_level == "high":
                    return 1
                if sensation_level == "low":
                    return 3

            if self.bigger_group == "cold":
                sensation_level = utils.get_sensation_level(self.local_sensation, self.bigger_group)
                if sensation_level == "high":
                    return 2
                if sensation_level == "low":
                    return 4

    @property
    def overall_sensations_dict(self) -> dict:
        """A series of overall sensations for different overall sensation models."""
        sensation_model = {}

        alpha = self.config.internal_smooth_alpha
        if self.config.original_model:
            if self.config.internal_smooth:
                sensation_model = {
                    1: models.high_level_warm(self.local_sensation),
                    2: models.high_level_cold(self.local_sensation),
                    3: models.smoothed_low_level_warm(self.local_sensation, alpha),
                    4: models.smoothed_low_level_cold(self.local_sensation, alpha),
                    5: models.opposite_dominated_cold(self.local_sensation),
                    6: models.smoothed_opposite_warm(self.local_sensation, alpha),
                    7: models.smoothed_opposite_cold(self.local_sensation, alpha)
                }
            if not self.config.internal_smooth:
                sensation_model = {
                    1: models.high_level_warm(self.local_sensation),
                    2: models.high_level_cold(self.local_sensation),
                    3: models.low_level_warm(self.local_sensation),
                    4: models.low_level_cold(self.local_sensation),
                    5: models.opposite_dominated_cold(self.local_sensation),
                    6: models.opposite_warm(self.local_sensation),
                    7: models.opposite_cold(self.local_sensation)
                }

        if not self.config.original_model:
            # modified models
            if self.config.internal_smooth:
                sensation_model = {
                    1: models.modified_high_level_warm(self.local_sensation),
                    2: models.modified_high_level_cold(self.local_sensation),
                    3: models.smoothed_modified_low_level_warm(self.local_sensation, alpha),
                    4: models.smoothed_modified_low_level_cold(self.local_sensation, alpha),
                    5: models.modified_dominated_cold(self.local_sensation),
                    6: models.smoothed_modified_opposite_warm(self.local_sensation, alpha),
                    7: models.smoothed_modified_opposite_cold(self.local_sensation, alpha)
                }
            if not self.config.internal_smooth:
                sensation_model = {
                    1: models.modified_high_level_warm(self.local_sensation),
                    2: models.modified_high_level_cold(self.local_sensation),
                    3: models.modified_low_level_warm(self.local_sensation),
                    4: models.modified_low_level_cold(self.local_sensation),
                    5: models.modified_dominated_cold(self.local_sensation),
                    6: models.modified_opposite_warm(self.local_sensation),
                    7: models.modified_opposite_cold(self.local_sensation)
                }
        return sensation_model

    def _get_overall_sensation(self):
        if not self.config.external_smooth:
            return self.overall_sensations_dict[self.model_num]


        y_dict = self.overall_sensations_dict
        y_i = y_dict[self.model_num]

        y_k_dict = self._get_y_k_dict(y_dict)
        w_ik_dict = self._get_w_ik_dict()

        y_k = y_k_dict[self.model_num]
        w_ik = w_ik_dict[self.model_num]

        y_ik = [y - y_i for y in y_k]
        w_y_ik = [float(w * y) for w, y in zip(w_ik, y_ik)]

        if not self.config.external_smooth_adjusted:
            return y_i + np.sum(w_y_ik)

        w_y_ik_max_index = np.argmax(np.abs(w_ik))
        y_i_modified = y_i + w_y_ik[w_y_ik_max_index]
        y_ik_modified = [y_k[i] - y_i if i == w_y_ik_max_index else y_k[i] - y_i_modified for i in range(len(y_k))]
        w_y_ik_modified = [float(w * y) for w, y in zip(w_ik, y_ik_modified)]

        return y_i + np.sum(w_y_ik_modified)


    def _get_y_k_dict(self, y_dict) -> dict:
        # for original and modified models
        y_k_dict = {
            1: [y_dict[i] for i in [3, 5, 6, 7]],
            2: [y_dict[i] for i in [4, 5, 6, 7]],
            3: [y_dict[i] for i in [1, 4, 5, 6, 7]],
            4: [y_dict[i] for i in [2, 3, 5, 6, 7]],
            5: [y_dict[i] for i in [1, 2, 3, 4, 6, 7]],
            6: [y_dict[i] for i in [1, 2, 3, 4, 5, 7]],
            7: [y_dict[i] for i in [1, 2, 3, 4, 5, 6]]
        }

        if self.config.external_smooth_simplified:
            y_k_dict = {
                1: [y_dict[i] for i in [3, 5, 7]],
                2: [y_dict[i] for i in [4, 6]],
                3: [y_dict[i] for i in [1, 4, 5, 7]],
                4: [y_dict[i] for i in [2, 3, 6]],
                5: [y_dict[i] for i in [1, 3, 6]],
                6: [y_dict[i] for i in [2, 4, 5, 7]],
                7: [y_dict[i] for i in [1, 3, 6]]
            }
        return y_k_dict

    def _get_w_ik_dict(self) -> dict:
        def sig(x, a, t):
            return 1 / (1 + np.exp(-a * (x - t)))
        alpha = self.config.external_smooth_alpha

        x1 = min(self.local_sensation[DOMINANT_BODY_PARTS])
        x2 = self.local_sensation.sort_values(ascending=False).iloc[0]
        x3 = self.local_sensation.sort_values(ascending=True).iloc[0]
        x4 = self.local_sensation.sort_values(ascending=False).iloc[1 if not self.are_hands_feet_warmest else 2]
        x5 = self.local_sensation.sort_values(ascending=True).iloc[1 if not self.are_hands_feet_coldest else 2]
        x6 = self.local_sensation.median()

        w_ik_dict = {
            1: [
                sig(-x4, alpha, -2),
                sig(-x1, alpha, 1),
                sig(x1, alpha, -1) * sig(-x3, alpha, 1),
                sig(-x6, alpha, 0),
            ],
            2: [
                sig(x5, alpha, -2),
                sig(-x1, alpha, 1) * sig(x2, alpha, 0),
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
                sig(-x1, alpha, 1) * sig(x2, alpha, 0),
                sig(x6, alpha, 0) * sig(x1, alpha, -1) * sig(-x3, alpha, 1),
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

        if (not self.config.original_model) and (not self.config.external_smooth_simplified):
            w_ik_dict = {
                1: [
                    sig(-x4, alpha, -2),
                    sig(-x1, alpha, 1),
                    sig(x1, alpha, -1) * sig(-x3, alpha, 1),
                    sig(-x6, alpha, 0),
                ],
                2: [
                    sig(x5, alpha, -2),
                    sig(-x1, alpha, 1),
                    sig(x6, alpha, 0),
                    sig(x2, alpha, 1),
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
                    sig(-x1, alpha, 1),
                    sig(x6, alpha, 0) * sig(-x3, alpha, 1),
                    sig(x2, alpha, 1),
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
                    sig(-x6, alpha, 0) * sig(-x2, alpha, -1) * sig(-x5, alpha, 2) * sig(x1, alpha, -1),
                    sig(-x4, alpha, -2) * sig(x3, alpha, -1),
                    sig(-x6, alpha, 0) * sig(-x2, alpha, -1) * sig(x5, alpha, -2),
                    sig(-x1, alpha, 1),
                    sig(-x6, alpha, 0) * sig(x2, alpha, 1) * sig(x1, alpha, -1),
                ],
                7: [
                    sig(x6, alpha, 0) * sig(x4, alpha, 2) * sig(x3, alpha, -1),
                    sig(-x2, alpha, -1) * sig(-x5, alpha, 2),
                    sig(x6, alpha, 0) * sig(-x4, alpha, -2) * sig(x3, alpha, -1),
                    sig(-x2, alpha, -1),
                    sig(-x1, alpha, 1),
                    sig(x6, alpha, 0) * sig(-x3, alpha, 1),
                ]
            }

        if (not self.config.original_model) and self.config.external_smooth_simplified:
            w_ik_dict = {
                1: [
                    sig(-x4, alpha, -2),
                    sig(-x1, alpha, 1),
                    sig(-x6, alpha, 0),
                ],
                2: [
                    sig(x5, alpha, -2),
                    sig(x6, alpha, 0),
                ],
                3: [
                    sig(x4, alpha, 2),
                    sig(-x6, alpha, 0) * sig(-x2, alpha, -1),
                    sig(-x1, alpha, 1),
                    sig(-x6, alpha, 0) * sig(x2, alpha, 1),
                ],
                4: [
                    sig(-x5, alpha, 2),
                    sig(x6, alpha, 0) * sig(x3, alpha, -1),
                    sig(x6, alpha, 0) * sig(-x3, alpha, 1),
                ],
                5: [
                    sig(x6, alpha, 0) * sig(x4, alpha, 2) * sig(x3, alpha, -1),
                    sig(x6, alpha, 0) * sig(-x4, alpha, -2) * sig(x3, alpha, -1),
                    sig(x6, alpha, 0) * sig(x1, alpha, -1) * sig(-x3, alpha, 1),
                ],
                6: [
                    sig(-x6, alpha, 0) * sig(-x2, alpha, -1) * sig(-x5, alpha, 2),
                    sig(-x6, alpha, 0) * sig(-x2, alpha, -1) * sig(x5, alpha, -2),
                    sig(-x1, alpha, 1),
                    sig(-x6, alpha, 0) * sig(x2, alpha, 1),
                ],
                7: [
                    sig(x6, alpha, 0) * sig(x4, alpha, 2) * sig(x3, alpha, -1),
                    sig(x6, alpha, 0) * sig(-x4, alpha, -2) * sig(x3, alpha, -1),
                    sig(x6, alpha, 0) * sig(-x3, alpha, 1),
                ]
            }

        return w_ik_dict



    @property
    def overall_sensation(self) -> float:
        """The overall sensation for the input local sensations."""
        return self._get_overall_sensation()


